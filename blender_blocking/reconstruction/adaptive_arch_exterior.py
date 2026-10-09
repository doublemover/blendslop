"""Separate three-control arch exterior stage using the complete extrusion.

Opening width and notch height measured from the bottom stay fixed. Symmetric
outer-height changes move the bottom and absolute notch roof together. This is
not the existing inner-only cavity roof control. No source dimensions enter
this model; fixed pose and the ordinary polygon triangulation remain unchanged.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import math
import hashlib
import time
import numbers
import numpy as np
from .adaptive_family import _validated_program, _arch_outline, _model_camera, _json_hash
from .frozen_family import coverage_contour_points

CONTROLS = ('outer_width_world', 'outer_height_world', 'extrusion_depth_world')
CONVENTION = {
    'stage': 'arch_exterior_dimension_preserving_v1',
    'fixed_notch_controls': ['opening_width_world', 'notch_height_from_bottom_world'],
    'outer_height_coupling': 'outer midpoint fixed; bottom attachments and absolute notch roof move together',
    'distinct_from_inner_control': 'inner stage varied bottom-relative cavity_roof_height_world at fixed exterior; this stage fixes that dimension while varying exterior',
    'pose_and_triangulation_generator_fixed': True,
}


def _geometry(wire):
    program = _validated_program(wire, 'concave_arch')
    p = program.root_nodes[0].parameters
    outline, scale = _arch_outline(p)
    left, right = outline[0, 0], outline[2, 0]
    bottom, top, roof = outline[0, 1], outline[1, 1], outline[5, 1]
    notch_left, notch_right = outline[7, 0], outline[4, 0]
    expected = np.array([[left, bottom], [left, top], [right, top], [right, bottom],
                         [notch_right, bottom], [notch_right, roof],
                         [notch_left, roof], [notch_left, bottom]])
    if (not np.array_equal(outline, expected) or not left < notch_left < notch_right < right
            or not bottom < roof < top):
        raise ValueError('exterior stage requires the exact retained rectangular eight-corner arch')
    depth = p.get('height_world')
    if (isinstance(depth, bool) or not isinstance(depth, numbers.Real)
            or not math.isfinite(depth) or depth <= 0.):
        raise ValueError('arch physical extrusion depth must be finite and positive')
    return program, outline, scale


def arch_exterior_controls(wire):
    program, outline, _ = _geometry(wire)
    return dict(zip(CONTROLS, (float(np.ptp(outline[:, 0])), float(np.ptp(outline[:, 1])),
                              float(program.root_nodes[0].parameters['height_world']))))


def arch_notch_dimensions(wire):
    _, outline, _ = _geometry(wire)
    return {'opening_width_world': float(outline[4, 0]-outline[7, 0]),
            'notch_height_from_bottom_world': float(outline[5, 1]-outline[4, 1])}


def arch_exterior_update(wire, values):
    program, outline, scale = _geometry(wire)
    previous = arch_exterior_controls(wire)
    if not values or not set(values).issubset(previous):
        raise ValueError('arch exterior update contains an undeclared control')
    controls = {**previous, **dict(values)}
    if any(isinstance(v, bool) or not isinstance(v, numbers.Real)
           or not math.isfinite(v) or v <= 0. for v in controls.values()):
        raise ValueError('arch exterior dimensions must be finite positive numbers')
    controls = {k: float(v) for k, v in controls.items()}
    if controls == previous:
        return program
    notch = arch_notch_dimensions(wire)
    width, height, depth = (controls[k] for k in CONTROLS)
    cx, cy = .5*(outline.min(axis=0)+outline.max(axis=0))
    left, right, bottom, top = cx-width/2, cx+width/2, cy-height/2, cy+height/2
    if not (left < outline[7, 0] < outline[4, 0] < right
            and height > notch['notch_height_from_bottom_world']):
        raise ValueError('exterior update would cross the fixed notch dimensions')
    outline[[0, 1], 0] = left; outline[[2, 3], 0] = right
    outline[[0, 3, 4, 7], 1] = bottom; outline[[1, 2], 1] = top
    outline[[5, 6], 1] = bottom+notch['notch_height_from_bottom_world']
    # Keep the existing outline coordinate scale convention; coupled aliases
    # make the compiler evaluate exactly this physical outline.
    p = deepcopy(dict(program.root_nodes[0].parameters))
    p['outer'] = (outline/scale).tolist()
    p['width_world'], p['depth_world'], p['height_world'] = width, height, depth
    updated = replace(program, root_nodes=(replace(program.root_nodes[0], parameters=p),))
    _geometry(updated.to_dict())
    return updated


def arch_exterior_mesh(wire):
    from primitives.polygon_extrusion import PolygonExtrusionPrimitive
    program, outline, _ = _geometry(wire)
    p = program.root_nodes[0].parameters
    return PolygonExtrusionPrimitive(outline, height=p['height_world'],
        rotation=p.get('rotation'), center=[p.get(k, 0.) for k in ('x', 'y', 'z')]).to_mesh_data()


def arch_exterior_boundary(wire, camera, *, return_metadata=False):
    """Full concave projected triangle union, never a filled convex hull."""
    from primitives.contour_silhouette import projected_union
    from primitives.primitive_protocol import MeshData
    from shapely.affinity import affine_transform
    matrix, scale = _model_camera(camera)
    mesh = arch_exterior_mesh(wire)
    local = MeshData((mesh.vertices-matrix[:3, 3])@matrix[:3, :3], mesh.faces)
    class CameraLocalMesh:
        def to_mesh_data(self, resolution):
            return local
    size = camera['resolution'][0]
    pixel_camera = SimpleNamespace(image_size=(size, size), axes=(0, 1),
                                    world_bounds=(-scale/2, scale/2, -scale/2, scale/2))
    shape, _, metadata = projected_union(CameraLocalMesh(), pixel_camera, 16)
    step = scale/size
    shape = affine_transform(shape, [step, 0., 0., -step,
                                     -scale/2+.5*step, scale/2-.5*step])
    return (shape, metadata) if return_metadata else shape


def arch_exterior_signed_distance(points, wire, camera):
    from shapely import contains_xy, distance, points as geometry_points
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('arch contour points must be finite Nx2')
    shape = arch_exterior_boundary(wire, camera)
    result = distance(geometry_points(points[:, 0], points[:, 1]), shape.boundary)
    return np.where(contains_xy(shape, points[:, 0], points[:, 1]), -result, result)


def arch_exterior_candidate_sensitivity(wire, cameras, *, heldout_views=(), points_per_view=384):
    """Visible internal candidate boundary only; no source observation claim."""
    from shapely import get_coordinates, line_interpolate_point
    if (not isinstance(cameras, dict) or not cameras or set(cameras)&set(heldout_views)
            or isinstance(points_per_view, bool) or not isinstance(points_per_view, numbers.Integral)
            or not 32 <= points_per_view <= 384):
        raise ValueError('candidate diagnostic excludes held-outs and allows 32..384 samples')
    controls = arch_exterior_controls(wire)
    records, evidence = {}, []
    for view, camera in sorted(cameras.items()):
        _, scale = _model_camera(camera)
        shape, metadata = arch_exterior_boundary(wire, camera, return_metadata=True)
        samples = get_coordinates(line_interpolate_point(shape.boundary,
            np.linspace(0., shape.boundary.length, int(points_per_view), endpoint=False)))
        visible = np.all((samples >= -scale/2)&(samples <= scale/2), axis=1)
        points = samples[visible]
        if len(points) < 4: raise ValueError('too few visible internal boundary samples')
        b = shape.bounds
        records[view] = {'candidate_frame_censored': bool(min(b[:2]) < -scale/2 or max(b[2:]) > scale/2),
                         'internal_candidate_samples': len(points), 'bounds_world_camera': list(b),
                         'frame_border_closure_added': False, 'source_complete_contour_claim': False,
                         'projection_metadata': metadata}
        evidence.append((points, camera, scale, math.sqrt(len(points))))
    columns = []
    for name, value in controls.items():
        span = .2*value
        lower = arch_exterior_update(wire, {name: value-span*.001}).to_dict()
        upper = arch_exterior_update(wire, {name: value+span*.001}).to_dict()
        columns.append(np.concatenate([(arch_exterior_signed_distance(points, upper, camera)
            -arch_exterior_signed_distance(points, lower, camera))/.002/scale/weight
            for points, camera, scale, weight in evidence]))
    singular = np.linalg.svd(np.column_stack(columns), compute_uv=False)
    rank = int(np.count_nonzero(singular > max(1e-8, singular[0]*1e-5)))
    return {'protocol': 'candidate_only_arch_exterior_sensitivity_v1', 'local_rank': rank,
            'singular_values': singular.tolist(), 'observations': records, 'source_observations': 0,
            'heldout_fit_or_view_choice_used': False, 'global_uniqueness_established': False,
            'source_identifiability': 'unqualified until genuine observations', 'parameter_convention': CONVENTION}


def refine_arch_exterior(wire, observations, *, parameter_bounds,
                         heldout_views, prior_view_exposure, max_evaluations=96,
                         max_elapsed_s=1., baseline_geometry_hash=None,
                         baseline_program_sha256=None):
    """Bounded subset update from internal crossings of controlled observations.

    Intervals must be frozen from candidate/engineering policy before acquiring
    selected detail truth. Bounds or rank failure retain an unqualified local
    state, independently of future silhouettes and raw surface observations.
    """
    from scipy.optimize import least_squares
    if (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, numbers.Integral)
            or not 8 <= max_evaluations <= 192 or isinstance(max_elapsed_s, bool)
            or not isinstance(max_elapsed_s, numbers.Real) or not math.isfinite(max_elapsed_s)
            or not 0 < max_elapsed_s <= 3.):
        raise ValueError('detail update exceeds bounded 8..192 calls / >0..3 seconds')
    original = arch_exterior_controls(wire)
    for name, value in (('baseline_geometry_hash', baseline_geometry_hash),
                        ('baseline_program_sha256', baseline_program_sha256)):
        if value is not None and (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError(name+' must be a frozen lowercase SHA256 identity')
    retained_detail = wire.get('metadata', {}).get('adaptive_detail')
    if retained_detail is not None and (baseline_geometry_hash is None or baseline_program_sha256 is None):
        raise ValueError('retained inner observations require their original baseline geometry and recipe binding')
    if (not isinstance(parameter_bounds, dict) or not parameter_bounds
            or not set(parameter_bounds).issubset(original) or len(parameter_bounds) > 3):
        raise ValueError('detail update needs one to three declared bounded physical controls')
    names = tuple(parameter_bounds)
    intervals = np.asarray([parameter_bounds[name] for name in names], float)
    initial = np.asarray([original[name] for name in names])
    if (intervals.shape != (len(names), 2) or not np.isfinite(intervals).all()
            or (intervals[:, 0] <= 0.).any() or (intervals[:, 1] <= intervals[:, 0]).any()
            or (initial < intervals[:, 0]).any() or (initial > intervals[:, 1]).any()):
        raise ValueError('each frozen positive interval must include the retained control')
    if not isinstance(prior_view_exposure, dict) or not all(v in prior_view_exposure for v in heldout_views):
        raise ValueError('held-out history must disclose prior view exposure explicitly')
    if not observations or set(observations) & set(heldout_views):
        raise ValueError('reserved held-out observations cannot enter fitting')
    evidence, records, precision = [], {}, []
    for view, record in sorted(observations.items()):
        coverage = np.asarray(record['coverage'], float)
        matrix, scale = _model_camera(record['camera'], coverage.shape)
        # Only observed neighbour crossings; no padded frame closure.
        pixels = coverage_contour_points(coverage, allow_clipped=True)
        count = len(pixels)
        pixels = pixels[np.linspace(0, count-1, min(count, 384), dtype=int)]
        height, width = coverage.shape
        xy = np.column_stack(((pixels[:, 0]/width-.5)*scale,
                              (.5-pixels[:, 1]/height)*scale))
        arch_exterior_boundary(wire, record['camera'])
        evidence.append((xy, record['camera'], scale, math.sqrt(len(xy))))
        touched = bool((coverage[0] >= .5).any() or (coverage[-1] >= .5).any()
                       or (coverage[:, 0] >= .5).any() or (coverage[:, -1] >= .5).any())
        records[view] = {'camera_sha256': _json_hash(record['camera']),
                         'coverage_sha256': hashlib.sha256(coverage.astype('<f8').tobytes()).hexdigest(),
                         'all_observed_crossings': count, 'fitted_crossings': len(xy),
                         'frame_censored': touched, 'complete_contour_claim': not touched}
        precision.append(scale/width)
    spans = intervals[:, 1]-intervals[:, 0]
    x0 = (initial-intervals[:, 0])/spans
    started = time.perf_counter()
    calls, best, score, termination = 0, x0.copy(), float('inf'), 'solver_completed'
    def decode(x):
        return dict(zip(names, intervals[:, 0]+x*spans))
    class AllowanceEnded(Exception):
        pass
    def residual(x, *, retain=True):
        nonlocal calls, best, score, termination
        if calls >= max_evaluations or (calls and time.perf_counter()-started >= max_elapsed_s):
            termination = 'evaluation_allowance' if calls >= max_evaluations else 'elapsed_allowance'
            raise AllowanceEnded
        calls += 1
        updated = arch_exterior_update(wire, decode(x)).to_dict()
        result = np.concatenate([arch_exterior_signed_distance(xy, updated, camera)
                                 / scale / weight for xy, camera, scale, weight in evidence])
        current = float(result @ result)
        if retain and current < score:
            best, score = x.copy(), current
        return result
    residual(x0)
    try:
        least_squares(residual, x0, bounds=(np.zeros(len(names)), np.ones(len(names))),
                      diff_step=1e-4, max_nfev=max_evaluations, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    except AllowanceEnded:
        pass
    selected = best.copy()
    columns = []
    try:
        for axis in range(len(names)):
            lower, upper = selected.copy(), selected.copy()
            lower[axis] = max(0., lower[axis]-1e-3)
            upper[axis] = min(1., upper[axis]+1e-3)
            columns.append((residual(upper, retain=False)-residual(lower, retain=False))
                           /(upper[axis]-lower[axis]))
    except AllowanceEnded:
        columns = []
    singular = np.linalg.svd(np.column_stack(columns), compute_uv=False) if columns else np.array([])
    rank = int(np.count_nonzero(singular > max(1e-8, singular[0]*1e-5))) if len(singular) else None
    controls = decode(selected)
    bound_active = any(min(controls[name]-intervals[i, 0], intervals[i, 1]-controls[name])
                       <= max(1e-6*spans[i], min(precision)*.25) for i, name in enumerate(names))
    updated = arch_exterior_update(wire, controls)
    detail = {'protocol': 'bounded_arch_exterior_detail_v1', 'family': 'concave_arch',
              'free_controls': list(names), 'original_controls': original,
              'selected_controls': {name: float(value) for name, value in controls.items()},
              'frozen_intervals': intervals.tolist(), 'fixed_pose': True, 'tessellation': 'existing fixed polygon extrusion triangulation',
              'residual_calls': calls, 'elapsed_seconds': time.perf_counter()-started,
              'termination': termination, 'squared_residual': score, 'observations': records,
              'local_sensitivity_singular_values': singular.tolist(), 'local_rank': rank,
              'interval_bound_active': bool(bound_active),
              'identifiability': 'locally_identified' if rank == len(names) and not bound_active else 'underconstrained',
              'global_uniqueness_established': False, 'heldout_views': list(heldout_views),
              'heldout_fit_or_roi_used': False, 'prior_view_exposure': deepcopy(prior_view_exposure),
              'artist_surface_limits': None, 'full_native_silhouette_admission': 'unrun',
              'model_scope': 'discrete projected boundary; actual native geometry and measurement remain independent'}
    detail['parameter_convention'] = deepcopy(CONVENTION)
    detail['fixed_notch_dimensions'] = arch_notch_dimensions(wire)
    detail['baseline_stage_provenance'] = {
        'geometry_hash': baseline_geometry_hash, 'program_file_sha256': baseline_program_sha256,
        'program_content_sha256': _json_hash(wire),
        'prior_observations_apply_to_current_geometry': False,
        'inner_dimension_values_remain_fixed': True}
    metadata = deepcopy(dict(updated.metadata))
    if retained_detail is not None:
        # Preserve the old record exactly, including its selected inner controls
        # and observations. Its geometry is the verified baseline, never this
        # new exterior result, even though the fixed notch values remain valid.
        history = list(metadata.get('historical_adaptive_stages', []))
        history.append({'source_metadata_key': 'adaptive_detail',
                        'scope': 'historical inner stage at fixed exterior',
                        'baseline_geometry_hash': baseline_geometry_hash,
                        'baseline_program_file_sha256': baseline_program_sha256,
                        'observations_apply_to_current_geometry': False,
                        'record': deepcopy(metadata.pop('adaptive_detail'))})
        metadata['historical_adaptive_stages'] = history
    metadata['adaptive_arch_exterior'] = detail
    return replace(updated, metadata=metadata)
