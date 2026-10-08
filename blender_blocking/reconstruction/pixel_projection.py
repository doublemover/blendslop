"""Explicit pixel-center and pixel-area tracks, separate from legacy PIL masks.

The camera mapping returns integer pixel-center coordinates. Canonical cell i
therefore spans [i-.5,i+.5]. Area prediction corresponds to ideal opaque box
coverage; real renderer sampling/filtering still needs independent acceptance.
"""
import numpy as np

CENTER_METRIC = 'opaque_triangle_union_pixel_centers_v1'
AREA_METRIC = 'opaque_triangle_union_pixel_cell_area_hard_half_v1'
LEGACY_METRIC = 'legacy_pil_polygon_endpoint_rounding_v1'


def triangle_pixel_centers(projected, faces, shape):
    height, width = shape
    mask = np.zeros(shape, bool)
    for face in faces:
        points = np.asarray(projected)[face]
        first, second = points[1]-points[0], points[2]-points[0]
        area = first[0]*second[1]-first[1]*second[0]
        if area == 0:
            continue
        x0, x1 = max(0, int(np.ceil(points[:, 0].min()))), min(width-1, int(np.floor(points[:, 0].max())))
        y0, y1 = max(0, int(np.ceil(points[:, 1].min()))), min(height-1, int(np.floor(points[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue
        xx, yy = np.meshgrid(np.arange(x0, x1+1), np.arange(y0, y1+1))
        inside = np.ones(xx.shape, bool)
        for start, end in zip(points, np.roll(points, -1, axis=0)):
            cross = (end[0]-start[0])*(yy-start[1])-(end[1]-start[1])*(xx-start[0])
            inside &= cross >= 0 if area > 0 else cross <= 0
        mask[y0:y1+1, x0:x1+1] |= inside
    return mask


def triangle_pixel_areas(projected, faces, shape, *, maximum_faces=2048):
    from shapely import Polygon, union_all, box, intersection, area
    projected, faces = np.asarray(projected, float), np.asarray(faces, np.int64)
    if len(faces) > maximum_faces:
        raise ValueError('bounded canonical pixel-area triangle allowance exceeded')
    triangles = []
    for face in faces:
        points = projected[face]
        a, b = points[1]-points[0], points[2]-points[0]
        if a[0]*b[1]-a[1]*b[0] != 0:
            triangles.append(Polygon(points))
    if not triangles:
        return np.zeros(shape, float)
    union = union_all(triangles)
    if union.is_empty or not union.is_valid:
        raise ValueError('canonical opaque projected union is invalid')
    yy, xx = np.indices(shape)
    cells = box(xx-.5, yy-.5, xx+.5, yy+.5)
    return np.clip(area(intersection(union, cells)), 0., 1.)


def canonical_mesh_projections(target, vertices, faces, *, mode='pixel_area_half'):
    from .projection_contract import project_vertices
    if mode not in ('pixel_centers', 'pixel_area_half'):
        raise ValueError('unsupported canonical opaque pixel operator')
    result = {}
    for constraint in target.constraints:
        shape = np.asarray(getattr(constraint.mask, 'mask', constraint.mask)).shape
        projected = project_vertices(target, constraint, vertices)
        if mode == 'pixel_centers':
            result[constraint.view] = triangle_pixel_centers(projected, faces, shape).astype(float)
        else:
            result[constraint.view] = triangle_pixel_areas(projected, faces, shape)
    return result


def canonical_mesh_metrics(target, vertices, faces, *, mode='pixel_area_half'):
    from .visibility import evaluate_visible_pair
    from .pixel_evidence import observed_pixel_evidence
    from blender_blocking.evaluation.silhouette_eval import SilhouetteGateConfig
    predictions = canonical_mesh_projections(target, vertices, faces, mode=mode)
    rows = {}
    for constraint in target.constraints:
        reference = np.asarray(getattr(constraint.mask, 'mask', constraint.mask), bool)
        prediction = predictions[constraint.view]
        row = evaluate_visible_pair(reference, prediction >= .5, constraint, view=constraint.view,
                                    config=SilhouetteGateConfig(min_area_iou=.7), required=True)
        evidence = observed_pixel_evidence(constraint)
        weight = float(evidence.weights.sum())
        row.update(candidate_projection_source=AREA_METRIC if mode == 'pixel_area_half' else CENTER_METRIC,
                   observed_probability_weighted_l2=float(np.sum((prediction-evidence.foreground)**2*evidence.weights)/weight) if weight else None,
                   probability_comparison_scope='ideal opaque box coverage versus supplied spatial probability; grayscale likelihood is not a measured alpha coverage')
        rows[constraint.view] = row
    return rows
