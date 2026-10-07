"""Input-derived section sampling and conservative silhouette occupancy hierarchy."""
from __future__ import annotations
import numpy as np


def adaptive_slices(profile, minimum=10, maximum=64, tolerance=.012):
    from geometry.profile_models import EllipticalSlice
    t = np.asarray(profile.heights_t, float)
    columns = [np.asarray(profile.rx), np.asarray(profile.ry)]
    columns += [np.zeros_like(t) if c is None else np.asarray(c) for c in (profile.cx, profile.cy)]
    values = np.column_stack(columns)
    scale = max(float(np.ptp(values, axis=0).max()), float(values[:, :2].max()), 1e-9)
    chosen = set(np.linspace(0, len(t)-1, min(len(t), max(2, minimum))).round().astype(int))
    # The residual against interpolated sections detects corners and necks as
    # well as smooth curvature. Every ring uses the same cardinal landmarks.
    while len(chosen) < min(maximum, len(t)):
        indices = sorted(chosen)
        reconstructed = np.column_stack([np.interp(t, t[indices], values[indices, j]) for j in range(4)])
        error = np.linalg.norm(values-reconstructed, axis=1)/scale
        error[indices] = -1
        worst = int(np.argmax(error))
        if error[worst] <= tolerance:
            break
        chosen.add(worst)
    return [EllipticalSlice(z=profile.z0+float(t[i])*profile.world_height,
                rx=float(values[i, 0]), ry=float(values[i, 1]),
                cx=float(values[i, 2]), cy=float(values[i, 3])) for i in sorted(chosen)]


def hierarchical_hull(target, resolution=64, *, conservative=False,storage='hierarchical',chunk_size=16):
    from reconstruction.projection_contract import pixel_cell_viewport
    from reconstruction.visibility import valid_evidence
    from reconstruction.point_cloud.bounds import volume_bounds_from_target
    from volume import DenseVolumeGrid
    n = int(resolution)
    if not 8 <= n <= 256:
        raise ValueError("adaptive hull resolution must be 8..256")
    lo, hi = np.asarray(target.bounds.to_min_max(), float)
    # Volume transforms locate cell centers, not interval endpoints. Query and
    # extracted world geometry must use exactly that same physical transform.
    coords = [lo[a]+(np.arange(n)+.5)/n*(hi[a]-lo[a]) for a in range(3)]
    records = []
    for c in target.constraints:
        mask = np.asarray(getattr(c.mask, 'mask', c.mask), bool)
        allowed = mask | ~valid_evidence(c)
        axes, viewport = pixel_cell_viewport(target, c)
        sat = np.pad(allowed.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        records.append((axes, viewport, allowed.shape, sat))
    if not records:
        raise ValueError("hull requires silhouette constraints")
    accepted_boxes=[]
    stack = [(np.zeros(3, int), np.full(3, n, int))]
    counts = dict(rejected=0, accepted=0, subdivided=0, leaves=0)
    while stack:
        start, end = stack.pop()
        all_inside = True
        reject = False
        for axes, (u0, u1, v0, v1), (h, w), sat in records:
            a, b = axes
            px = (np.array([coords[a][start[a]], coords[a][end[a]-1]])-u0)/(u1-u0)*w-.5
            py = (v1-np.array([coords[b][end[b]-1], coords[b][start[b]]]))/(v1-v0)*h-.5
            margin_x = ((hi[a]-lo[a])/n/(u1-u0)*w/2.) if conservative else 0.
            margin_y = ((hi[b]-lo[b])/n/(v1-v0)*h/2.) if conservative else 0.
            x0, x1 = int(np.floor(px[0]-margin_x+.5)), int(np.floor(px[1]+margin_x+.5))+1
            y0, y1 = int(np.floor(py[0]-margin_y+.5)), int(np.floor(py[1]+margin_y+.5))+1
            outside = x0 < 0 or x1 > w or y0 < 0 or y1 > h
            x0, x1 = np.clip([x0, x1], 0, w)
            y0, y1 = np.clip([y0, y1], 0, h)
            area = int((x1-x0)*(y1-y0))
            count = int(sat[y1, x1]-sat[y0, x1]-sat[y1, x0]+sat[y0, x0])
            # Outside the observed viewport is unknown, never an empty ray.
            if count == 0 and not outside:
                reject = True
                break
            all_inside &= count == area
        if reject:
            counts['rejected'] += 1
            continue
        if all_inside or np.all(end-start == 1):
            accepted_boxes.append(np.r_[start,end])
            counts['accepted' if all_inside else 'leaves'] += 1
            continue
        counts['subdivided'] += 1
        intervals = [[(s, e)] if e-s == 1 else [(s, (s+e)//2), ((s+e)//2, e)] for s, e in zip(start, end)]
        from itertools import product
        for box in product(*intervals):
            stack.append((np.array([x[0] for x in box]), np.array([x[1] for x in box])))
    if storage=='hierarchical':
        from volume.hierarchical import HierarchicalOccupancyGrid
        grid=HierarchicalOccupancyGrid((n,n,n),volume_bounds_from_target(target),accepted_boxes,chunk_size=chunk_size)
    elif storage=='dense':
        output=np.zeros((n,n,n),bool)
        for box in accepted_boxes:output[tuple(slice(s,e) for s,e in zip(box[:3],box[3:]))]=True
        grid=DenseVolumeGrid(output,volume_bounds_from_target(target),value_type='occupancy_bool',default_value=False,chunk_size=chunk_size)
    else:raise ValueError('unsupported hierarchy storage')
    grid.adaptive_report = {**counts, 'conservative_pixel_footprints': conservative,
        'unknown_is_empty': False, 'resolution': n,'storage':storage,'stored_boxes':len(accepted_boxes),
        'query_world_convention':'VoxelTransform cell centers'}
    return grid


def contour_section_mesh(target, slices, resolution=64):
    """Loft a top-view contour envelope through adaptive elliptical sections.

    Sampling a signed section field preserves holes and separate loops. It is
    deliberately an alternative: it cannot replace the ellipse incumbent on a
    worse measured silhouette.
    """
    from reconstruction.projection_contract import project_vertices
    from reconstruction.visibility import valid_evidence
    from reconstruction.point_cloud.bounds import volume_bounds_from_target
    from volume import DenseVolumeGrid, extract_mesh
    top = next((c for c in target.constraints if c.view == 'top'), None)
    if top is None:
        return None
    from scipy.ndimage import distance_transform_edt, map_coordinates
    lo, hi = np.asarray(target.bounds.to_min_max(), float)
    n = min(128, max(16, int(resolution)))
    x, y, z = np.meshgrid(*[lo[a]+(np.arange(n)+.5)/n*(hi[a]-lo[a]) for a in range(3)], indexing='ij')
    zs = np.array([s.z for s in slices])
    rx = np.interp(z.ravel(), zs, [s.rx for s in slices])
    ry = np.interp(z.ravel(), zs, [s.ry for s in slices])
    cx = target.bounds.center[0]+np.interp(z.ravel(), zs, [s.cx or 0. for s in slices])
    cy = target.bounds.center[1]+np.interp(z.ravel(), zs, [s.cy or 0. for s in slices])
    xx, yy = x.ravel()-cx, y.ravel()-cy
    ellipse = np.sqrt((xx/np.maximum(rx, 1e-9))**2+(yy/np.maximum(ry, 1e-9))**2)-1
    # Scale the observed contour to each section using fixed axial landmarks.
    points = np.column_stack([target.bounds.center[0]+xx/np.maximum(rx, 1e-9)*max(s.rx for s in slices),
        target.bounds.center[1]+yy/np.maximum(ry, 1e-9)*max(s.ry for s in slices), z.ravel()])
    xy = project_vertices(target, top, points)
    mask = np.asarray(top.mask, bool) | ~valid_evidence(top)
    signed = distance_transform_edt(~mask)-distance_transform_edt(mask)
    contour = map_coordinates(signed, [xy[:, 1], xy[:, 0]], order=1, mode='nearest')/max(mask.shape)
    field = np.maximum(ellipse, contour).reshape((n, n, n))
    # Cap the section envelope inside the observed z interval.
    field[:, :, 0] = np.maximum(field[:, :, 0], .01)
    field[:, :, -1] = np.maximum(field[:, :, -1], .01)
    grid = DenseVolumeGrid(field, volume_bounds_from_target(target), value_type='signed_distance', default_value=1.)
    return extract_mesh(grid, level=0.)


def refine_hull_boundary(target, geometry, resolution):
    from .projection_contract import project_vertices, pixel_cell_viewport
    from .visibility import valid_evidence
    from .native_geometry import GeometryArrays
    from .projected_metrics import projected_mesh_metrics
    from scipy.ndimage import distance_transform_edt
    vertices = np.array(geometry.vertices, copy=True)
    lo, hi = np.asarray(target.bounds.to_min_max(), float)
    limit = np.max(hi-lo)/max(1, resolution)*.6
    for _ in range(2):
        for c in target.constraints:
            mask = np.asarray(c.mask, bool) | ~valid_evidence(c)
            if not mask.any(): continue
            _, indices = distance_transform_edt(~mask, return_indices=True)
            xy = project_vertices(target, c, vertices)
            px = np.rint(xy).astype(int)
            h, w = mask.shape
            inside = (px[:, 0]>=0)&(px[:, 0]<w)&(px[:, 1]>=0)&(px[:, 1]<h)
            ids = np.flatnonzero(inside)
            x, y = px[ids].T
            ids = ids[~mask[y, x]]
            if not len(ids): continue
            x, y = px[ids].T
            nearest_y, nearest_x = indices[:, y, x]
            axes, (u0, u1, v0, v1) = pixel_cell_viewport(target, c)
            destinations = np.column_stack([u0+(nearest_x+.5)/w*(u1-u0), v1-(nearest_y+.5)/h*(v1-v0)])
            delta = destinations-vertices[np.ix_(ids, axes)]
            delta *= np.minimum(1., limit/np.maximum(np.linalg.norm(delta, axis=1), 1e-12))[:, None]
            vertices[np.ix_(ids, axes)] += delta
    proposal = GeometryArrays.capture(vertices, geometry.faces)
    def score(data):
        rows = projected_mesh_metrics(target, data.vertices, data.faces).values()
        values = [r['area_iou'] for r in rows if r.get('area_iou') is not None]
        return min(values, default=0.), np.mean(values) if values else 0.
    before, after = score(geometry), score(proposal)
    # Reject folded/degenerate proposals as well as silhouette regression.
    from .grouped_solids import solid_guard
    accepted = after[0] >= before[0]-.002 and after[1] > before[1]+1e-6 and solid_guard(proposal)['valid_solid']
    return (proposal if accepted else geometry), {'accepted': accepted, 'before': before, 'after': after,
        'source': 'known_foreground_boundary', 'maximum_vertex_step_world': limit}
