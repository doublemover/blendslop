"""Input-supported structural programs; every proposal requires geometry scoring."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from .spatial_regions import pca_box, cuboid_levels, connected_point_regions


def box_node(points, node_id, operation='add'):
    from primitives.shape_program import ShapeNode
    center, radii, rotation = pca_box(points)
    return ShapeNode(node_id, operation, 'box', parameters={
        **dict(zip(('x', 'y', 'z'), center.tolist())),
        **dict(zip(('width_world', 'depth_world', 'height_world'), (2*radii).tolist())),
        'rotation': rotation.tolist(), 'input_points': len(points)}, name=node_id)


def cuboid_programs(target, seed, max_parts=24):
    from reconstruction.point_cloud import target_surface_points
    points, _ = target_surface_points(target, resolution=32, max_points=2048)
    groups = cuboid_levels(points, fine_parts=min(24, max_parts), levels=(12, 6, 3, 1))
    return [replace(seed, root_nodes=tuple(box_node(p, 'cuboid_'+str(i)) for i, p in enumerate(level)),
        constraints=(), residual_patches=(), metadata={**seed.metadata, 'proposal': 'adjacency_merged_pca_cuboids', 'parts': len(level)}) for level in groups]


def whole_primitive_programs(target, seed, *, max_evaluations=96, max_elapsed_s=3., points=None):
    """Complementary one-part structural hypotheses fitted to observed supports."""
    from primitives.shape_program import ShapeNode
    from .point_cloud import target_surface_points
    from .projection_contract import observed_holes
    from .oriented_support import support_evidence, fit_whole_support
    import time
    if any(pixels >= 4 for pixels in observed_holes(target).values()):
        return []  # These four convex families cannot preserve a known hole.
    if points is None:
        points, _ = target_surface_points(target, resolution=24, max_points=1024)
    if len(points) < 8:
        return []
    center, radii, frame = pca_box(points)
    evidence = support_evidence(target)
    # PCA's longest axis seeds cylinder local Z, with a proper handed frame.
    axial_frame = frame[:, [1, 2, 0]]
    circular_radius = float(np.sqrt(radii[1]*radii[2]))
    hypotheses = (("box", radii, frame), ("ellipsoid", radii, frame),
                  ("cylinder", [circular_radius, radii[0]], axial_frame),
                  ("frustum", [circular_radius, circular_radius, radii[0]], axial_frame))
    proposals, started = [], time.perf_counter()
    for family, dimensions, rotation in hypotheses:
        remaining = max_elapsed_s - (time.perf_counter()-started) if max_elapsed_s is not None else None
        if remaining is not None and remaining <= 0.:
            break
        fit = fit_whole_support(family, evidence, center=center, dimensions=dimensions,
                                rotation=rotation, max_evaluations=max_evaluations,
                                max_elapsed_s=remaining)
        dimensions = fit["dimensions"]
        if family in {"box", "ellipsoid"}:
            sizes = 2.*dimensions
        else:
            radius = float(max(dimensions[:1] if family == "cylinder" else dimensions[:2]))
            sizes = np.array([2.*radius, 2.*radius, 2.*dimensions[-1]])
        parameters = {**dict(zip(('x','y','z'), fit["center"].tolist())),
                      **dict(zip(('width_world','depth_world','height_world'), sizes.tolist())),
                      'rotation': fit['rotation'].tolist()}
        if family in {"cylinder", "frustum"}:
            parameters.update(radius_bottom=float(dimensions[0]),
                              radius_top=float(dimensions[0] if family == "cylinder" else dimensions[1]))
        node = ShapeNode('whole_'+family, 'add', family, parameters=parameters)
        diagnostics = {key:value for key,value in fit.items() if key not in {'center','dimensions','rotation'}}
        proposals.append(replace(seed, root_nodes=(node,), constraints=(), residual_patches=(),
            metadata={**seed.metadata, 'proposal':'fitted_whole_oriented_'+family,
                      'support_fit':diagnostics, 'parts':1}))
    return proposals


def structural_programs(target, program, geometry, limit=6):
    from reconstruction.projected_metrics import projected_mesh_masks
    from reconstruction.projection_contract import project_vertices
    from reconstruction.point_cloud import target_surface_points
    from reconstruction.visibility import valid_evidence
    from primitives.shape_program import ShapeNode
    points, _ = target_surface_points(target, resolution=32, max_points=2048)
    masks = projected_mesh_masks(target, geometry.vertices, geometry.faces)
    missing = np.zeros(len(points), bool)
    for c in target.constraints:
        xy = np.rint(project_vertices(target, c, points)).astype(int)
        h, w = np.asarray(c.mask).shape
        valid = (xy[:, 0]>=0)&(xy[:, 0]<w)&(xy[:, 1]>=0)&(xy[:, 1]<h)
        ids = np.flatnonzero(valid); x, y = xy[ids].T
        missing[ids] |= np.asarray(c.mask, bool)[y, x] & valid_evidence(c)[y, x] & ~masks[c.view][y, x]
    proposals = []
    regions = connected_point_regions(points[missing], minimum=8, max_regions=3)
    for i, region in enumerate(regions):
        node = box_node(region, 'residual_add_'+str(i))
        proposals.append(replace(program, root_nodes=(*program.root_nodes, node), metadata={**program.metadata, 'proposal_operation': 'add'}))
        # Mirror is an actual positioned instance of the residual part.
        if len(region) and len(program.root_nodes) > 0:
            from .program_transforms import reflect_parameters
            params = reflect_parameters(node.parameters, axis=0, plane=target.bounds.center[0])
            mirror = replace(node, node_id=node.node_id+'_mirror', parameters=params)
            proposals.append(replace(program, root_nodes=(*program.root_nodes, node, mirror), metadata={**program.metadata, 'proposal_operation': 'mirror'}))
    if len(regions) >= 3:
        templates = [box_node(region, 'repeat_'+str(i)) for i, region in enumerate(regions)]
        dimensions = ('width_world', 'depth_world', 'height_world')
        shared = {key: float(np.median([node.parameters[key] for node in templates])) for key in dimensions}
        instances = tuple(replace(node, parameters={**node.parameters, **shared,
            'rotation': templates[0].parameters['rotation'], 'instance_template': 'residual_repeat'}) for node in templates)
        proposals.append(replace(program, root_nodes=(*program.root_nodes, *instances),
            metadata={**program.metadata, 'proposal_operation': 'repeat'}))
    largest = max(program.root_nodes, key=lambda n: np.prod([float(n.parameters.get(k, 1.)) for k in ('width_world', 'depth_world', 'height_world')]), default=None)
    if largest is not None and largest.primitive_type in {'box', 'rounded_box', 'ellipsoid', 'superquadric'}:
        p = dict(largest.parameters); axis = int(np.argmax([float(p.get(k, 1.)) for k in ('width_world', 'depth_world', 'height_world')]))
        dimension = ('width_world', 'depth_world', 'height_world')[axis]
        extent = float(p.get(dimension, 1.))
        first, second = dict(p), dict(p)
        first[dimension] = second[dimension] = extent*.52
        from .program_transforms import local_pose_edit
        displacement = np.zeros(3); displacement[axis] = extent*.24
        first = local_pose_edit(first, translation=-displacement)
        second = local_pose_edit(second, translation=displacement)
        nodes = [n for n in program.root_nodes if n.node_id != largest.node_id]
        nodes += [replace(largest, node_id=largest.node_id+'_a', parameters=first), replace(largest, node_id=largest.node_id+'_b', parameters=second)]
        proposals.append(replace(program, root_nodes=tuple(nodes), constraints=(), metadata={**program.metadata, 'proposal_operation': 'split'}))
    # Keep each structural operation reachable under the small proposal cap.
    ordered = []
    for kind in ('add', 'split', 'mirror', 'repeat'):
        ordered += [p for p in proposals if p.metadata.get('proposal_operation') == kind][:1]
    used = {id(p) for p in ordered}
    ordered += [p for p in proposals if id(p) not in used]
    return ordered[:limit]


def subtractive_programs(target, program, positive_geometry, limit=2):
    """Intersect known empty rays inside positive projections across >=2 views."""
    from reconstruction.projected_metrics import projected_mesh_masks
    from reconstruction.projection_contract import project_vertices
    from reconstruction.visibility import valid_evidence
    from scipy.ndimage import label
    lo, hi = np.asarray(target.bounds.to_min_max(), float)
    n = 24
    axes = [np.linspace(lo[a], hi[a], n) for a in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    positives = projected_mesh_masks(target, positive_geometry.vertices, positive_geometry.faces)
    empty_count = np.zeros(len(points), int); observed_count = np.zeros(len(points), int)
    for c in target.constraints:
        xy = np.rint(project_vertices(target, c, points)).astype(int)
        h, w = np.asarray(c.mask).shape
        inside = (xy[:, 0]>=0)&(xy[:, 0]<w)&(xy[:, 1]>=0)&(xy[:, 1]<h)
        ids = np.flatnonzero(inside); x, y = xy[ids].T
        known = valid_evidence(c)[y, x]
        observed_count[ids] += known
        empty_count[ids] += known & ~np.asarray(c.mask, bool)[y, x] & positives[c.view][y, x]
    cavities = ((empty_count >= 2)&(observed_count >= 2)).reshape((n, n, n))
    labels, count = label(cavities)
    proposals = []
    for index in sorted(range(1, count+1), key=lambda i: -np.count_nonzero(labels == i)):
        selected = points[labels.ravel() == index]
        if len(selected) < 8: continue
        node = box_node(selected, 'negative_empty_rays_'+str(index), operation='subtract')
        node = replace(node, parameters={**node.parameters, 'support': 'known_empty_inside_positive_projection', 'required_views': 2})
        proposals.append(replace(program, root_nodes=(*program.root_nodes, node)))
        if len(proposals) >= limit: break
    return proposals


def signed_csg_field(positive_fields, negative_fields=()):
    """Negative-inside SDF: positive union=min, subtraction=max(base,-cut)."""
    field = np.min(np.asarray(positive_fields), axis=0)
    if len(negative_fields):
        field = np.maximum(field, -np.min(np.asarray(negative_fields), axis=0))
    return field
