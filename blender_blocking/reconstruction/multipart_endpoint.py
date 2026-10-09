"""One frozen multipart endpoint from a newly acquired silhouette only."""
from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from .frozen_family import _camera, coverage_contour_points, validate_family_budget
from .multipart_family import projected_box_polygon, retained_multipart_program


def boxes_from_program(program):
    boxes = []
    for node in program.root_nodes:
        p = node.parameters
        center = np.array([p[key] for key in ('x', 'y', 'z')], float)
        size = np.array([p[key] for key in ('width_world', 'depth_world', 'height_world')], float)
        if not np.isfinite(center).all() or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError('saved endpoint proposal requires finite positive box dimensions')
        boxes.append((center - size / 2, center + size / 2))
    return boxes


def projected_union_signed_distance(points, boxes, camera):
    """Exact distance to the projected union boundary, excluding internal seams."""
    from shapely import Polygon, contains_xy, union_all
    shape = union_all([Polygon(projected_box_polygon(a, b, camera)) for a, b in boxes])
    if shape.is_empty or not shape.is_valid:
        raise ValueError('projected box union must be a valid nonempty polygon')
    polygons = [shape] if shape.geom_type == 'Polygon' else list(shape.geoms)
    rings = [np.asarray(ring.coords, float) for polygon in polygons
             for ring in (polygon.exterior, *polygon.interiors)]
    starts = np.vstack([ring[:-1] for ring in rings])
    ends = np.vstack([ring[1:] for ring in rings])
    edges = ends - starts
    points = np.asarray(points, float)
    delta = points[:, None, :] - starts
    fraction = np.clip(np.einsum('nej,ej->ne', delta, edges) / (edges * edges).sum(axis=1), 0., 1.)
    nearest = delta - fraction[..., None] * edges
    distance = np.sqrt((nearest * nearest).sum(axis=2).min(axis=1))
    inside = contains_xy(shape, points[:, 0], points[:, 1])
    return np.where(inside, -distance, distance)


def refine_short_far_endpoint(wire, coverage, camera, *, view='complementary_minus15_20',
                              max_evaluations=64, max_elapsed_s=1.):
    """Change only short-arm far Y, using one explicit new source observation.

    No reference arrays, authored dimensions, old masks, or holdout inputs enter
    this interface. The original interval remains an explicit frozen bound.
    """
    from scipy.optimize import minimize_scalar
    validate_family_budget(max_evaluations, max_elapsed_s)
    if not 8 <= max_evaluations <= 64 or max_elapsed_s > 1.:
        raise ValueError('endpoint update requires8..64 calls and at most one second')
    if view == 'oblique_145_40':
        raise ValueError('original oblique145 is reserved for independent admission')
    started = time.perf_counter()
    program = retained_multipart_program(wire)
    boxes = boxes_from_program(program)
    image = np.asarray(coverage, float)
    contour = coverage_contour_points(image)  # Uncensored new measurement only.
    interval = np.asarray(program.metadata['identifiability']['original_short_far_interval'], float)
    near = float(boxes[2][0][1])
    original_far = float(boxes[2][1][1])
    if (interval.shape != (2,) or not np.isfinite(interval).all() or interval[1] <= interval[0]
            or not interval[0] <= original_far <= interval[1]):
        raise ValueError('saved endpoint must retain its finite original interval')
    lower = max(float(interval[0]), near + np.finfo(float).eps)
    upper = float(interval[1])
    if lower >= upper:
        raise ValueError('original endpoint interval contains no positive short arm')
    matrix, scale = _camera(camera)
    if (camera.get('shift_x', 0.) != 0. or camera.get('shift_y', 0.) != 0.
            or list(camera.get('pixel_aspect', [1., 1.])) != [1., 1.]):
        raise ValueError('endpoint stage requires its frozen zero-shift square-pixel acquisition')
    h, w = image.shape
    if h != w:
        raise ValueError('bounded endpoint stage uses square orthographic acquisition')
    contour = contour[np.linspace(0, len(contour) - 1, min(384, len(contour)), dtype=int)]
    uv = np.column_stack(((contour[:, 0] / w - .5) * scale, (.5 - contour[:, 1] / h) * scale))
    yy, xx = np.mgrid[:20, :20]
    x, y = (xx.ravel() + .5) * w / 20, (yy.ravel() + .5) * h / 20
    known = image[np.minimum(y.astype(int), h - 1), np.minimum(x.astype(int), w - 1)]
    keep = (known < .05) | (known > .95)
    grid = np.column_stack(((x / w - .5) * scale, (.5 - y / h) * scale))[keep]
    inside = known[keep] > .5
    points = np.vstack((uv, grid))
    calls, best, score = 0, original_far, float('inf')
    termination = 'solver_completed'

    def vector(endpoint):
        parts = [(a.copy(), b.copy()) for a, b in boxes]
        parts[2][1][1] = endpoint
        values = projected_union_signed_distance(points, parts, camera)
        boundary = values[:len(uv)] / (scale * np.sqrt(len(uv)))
        tail = values[len(uv):]
        violation = np.where(inside, np.maximum(tail, 0.), np.minimum(tail, 0.))
        return np.r_[boundary, violation * .25 / (scale * np.sqrt(len(grid)))]

    class AllowanceEnded(Exception):
        pass

    def residual(values):
        nonlocal calls, best, score, termination
        if calls >= max_evaluations - 4 or (calls and time.perf_counter() - started >= max_elapsed_s):
            termination = 'evaluation_allowance' if calls >= max_evaluations - 4 else 'elapsed_allowance'
            raise AllowanceEnded
        calls += 1
        result = vector(float(values[0]))
        current = float(result @ result)
        if current < score:
            best, score = float(values[0]), current
        return result

    try:
        # One scalar can cross nearest-edge branches. Nine fixed interval probes
        # avoid a local derivative stopping on the wrong visible edge. Every
        # probe consumes the same fixed residual-call allowance.
        residual([original_far])
        probes = np.linspace(lower, upper, min(9, max_evaluations - 5))
        for probe in probes:
            residual([probe])
        index = int(np.argmin(np.abs(probes - best)))
        bracket = (probes[max(0, index - 1)], probes[min(len(probes) - 1, index + 1)])
        def objective(endpoint):
            result = residual([endpoint])
            return float(result @ result)
        if bracket[1] > bracket[0]:
            minimize_scalar(objective, bounds=bracket, method='bounded',
                            options={'maxiter': max_evaluations - calls - 4, 'xatol': scale / w * 1e-4})
    except AllowanceEnded:
        pass
    sensitivity = None
    if calls + 2 <= max_evaluations and time.perf_counter() - started < max_elapsed_s:
        delta = scale / w * .1
        a, b = max(lower, best - delta), min(upper, best + delta)
        sensitivity = float(np.linalg.norm((vector(b) - vector(a)) / (b - a)))
        calls += 2
    active = min(best - lower, upper - best) <= scale / w * .25
    identified = sensitivity is not None and sensitivity > 1e-8 and not active
    node = program.root_nodes[2]
    parameters = dict(node.parameters)
    parameters['y'] = (near + best) / 2
    parameters['depth_world'] = best - near
    roots = (*program.root_nodes[:2], replace(node, parameters=parameters))
    metadata = {**program.metadata, 'endpoint_refinement': {
        'fit_view': view, 'input_scope': 'only newly acquired float32 coverage and actual camera',
        'objective': 'exact projected union boundary; no interior box edges',
        'free_parameter': 'short_arm_far_y', 'all_other_recipe_parameters_fixed': True,
        'original_far': original_far, 'refined_far': best, 'fixed_near': near,
        'original_interval': interval.tolist(), 'effective_positive_interval': [lower, upper],
        'residual_calls': calls, 'elapsed_seconds': time.perf_counter() - started,
        'squared_residual': score if np.isfinite(score) else None, 'termination': termination,
        'local_sensitivity_norm': sensitivity, 'interval_bound_active': active,
        'identifiability': 'locally_identified' if identified else 'underconstrained',
        'scope': 'local one-scalar observation; no global hidden-surface uniqueness claim',
        'heldout_oblique145_fit_used': False, 'raw_surface_limits': None,
    }}
    return replace(program, root_nodes=roots, metadata=metadata)
