"""Native convex proxies partitioned by input neighborhoods and concavity."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from .spatial_regions import connected_point_regions, cuboid_levels


def native_convex_mesh(points):
    import bmesh
    from .native_geometry import GeometryArrays
    bm = bmesh.new()
    try:
        for point in np.asarray(points, float): bm.verts.new(point)
        bm.verts.ensure_lookup_table()
        result = bmesh.ops.convex_hull(bm, input=list(bm.verts), use_existing_faces=False)
        unused = list({g for g in result.get('geom_unused', [])+result.get('geom_interior', []) if isinstance(g, bmesh.types.BMVert) and g.is_valid})
        if unused: bmesh.ops.delete(bm, geom=unused, context='VERTS')
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bm.verts.ensure_lookup_table(); bm.verts.index_update()
        return GeometryArrays.capture([tuple(v.co) for v in bm.verts], [[v.index for v in f.verts] for f in bm.faces])
    finally:
        bm.free()


def convex_proxy_programs(target, seed, maximum=8):
    from reconstruction.point_cloud import target_surface_points
    from primitives.shape_program import ShapeNode
    points, _ = target_surface_points(target, resolution=32, max_points=2048)
    components = connected_point_regions(points, minimum=8, max_regions=maximum)
    if not components: components = [points]
    partitions = []
    for component in components:
        # Concave components get spatial leaves instead of a single filled hull.
        levels = cuboid_levels(component, fine_parts=max(2, maximum//max(1, len(components))), levels=(4, 2, 1))
        partitions.extend(levels[0] if levels else [component])
    partitions = partitions[:maximum]
    alternatives = [partitions]
    # Adjacent merges are scored by occupied support and actual boundary error.
    from reconstruction.visibility import point_support
    from reconstruction.projected_metrics import projected_mesh_metrics
    from reconstruction.grouped_solids import concatenate, signed_volume
    meshes = [native_convex_mesh(p) for p in partitions]
    while len(partitions) > 2:
        choices = []
        centers = np.array([p.mean(0) for p in partitions])
        for i in range(len(partitions)):
            neighbors = np.argsort(np.linalg.norm(centers-centers[i], axis=1))[1:3]
            for j in neighbors:
                if j <= i: continue
                points_merged = np.vstack([partitions[i], partitions[j]])
                merged = native_convex_mesh(points_merged)
                probes = np.vstack([merged.vertices, merged.vertices[merged.faces].mean(1)])
                supported, observed = point_support(target, probes)
                occupied = float(np.mean(supported[observed > 0])) if np.any(observed > 0) else 0.
                rows = projected_mesh_metrics(target, merged.vertices, merged.faces)
                boundary = np.mean([r['boundary_iou'] for r in rows.values() if r.get('boundary_iou') is not None])
                extra_volume = max(0., abs(signed_volume(merged))-abs(signed_volume(meshes[i]))-abs(signed_volume(meshes[j])))
                if occupied >= .98:
                    choices.append((extra_volume+float(1.-boundary), i, int(j), points_merged, merged))
        if not choices: break
        _, a, b, points_merged, merged = min(choices, key=lambda row: row[:3])
        partitions = [p for i, p in enumerate(partitions) if i not in {a, b}]+[points_merged]
        meshes = [p for i, p in enumerate(meshes) if i not in {a, b}]+[merged]
        alternatives.append(list(partitions))
    programs = []
    for partition in alternatives:
        nodes = tuple(ShapeNode('convex_'+str(i), 'add', 'convex_hull', parameters={'points_world': p.tolist(),
            'input_point_count': len(p), 'x': 0., 'y': 0., 'z': 0.}, name='Input region convex '+str(i)) for i, p in enumerate(partition))
        programs.append(replace(seed, root_nodes=nodes, constraints=(), residual_patches=(),
            metadata={**seed.metadata, 'proposal': 'native_spatial_convex_proxy', 'output_transform': 'identity_world',
                'editable_correspondence': {n.node_id: n.parameters['input_point_count'] for n in nodes}}))
    return programs
