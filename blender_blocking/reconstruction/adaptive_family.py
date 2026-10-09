"""Bounded local family updates from genuine controlled detail acquisitions.

Only declared fitting observations enter the objective. Partial contours remain
partial; frame borders are never closed. Fixed pose and discrete tessellation
are part of the model, and local sensitivity never implies global uniqueness.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
import numbers
import time

import numpy as np

from .frozen_family import _camera, coverage_contour_points, retained_family_program
from .adaptive_axis_family import (
    AXIS_CONTROLS, AXIS_PRIMITIVES, axis_family_control_values,
    axis_family_program_update, axis_projected_family_boundary,
)

FAMILY_CONTROLS = {
    'capsule': ('radius_world', 'segment_height_world'),
    'tapered_frustum': ('radius_bottom', 'radius_top', 'height_world'),
    'torus': ('major_radius', 'minor_radius'),
    'concave_arch': ('opening_width_world', 'cavity_roof_height_world'),
    'rounded_triangle_dot': ('scale_x', 'scale_y', 'corner_radius_world'),
}
PRIMITIVES = {'capsule': 'capsule', 'tapered_frustum': 'frustum',
              'torus': 'torus', 'concave_arch': 'polygon_extrusion',
              'rounded_triangle_dot': 'rounded_triangle'}
FAMILY_CONTROLS.update(AXIS_CONTROLS)
PRIMITIVES.update(AXIS_PRIMITIVES)


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def _validated_program(wire, family):
    if family not in FAMILY_CONTROLS:
        raise ValueError('adaptive family model is unsupported')
    program = retained_family_program(wire)
    if (len(program.root_nodes) != 1 or program.constraints or program.residual_patches
            or program.root_nodes[0].primitive_type != PRIMITIVES[family]):
        raise ValueError('detail update requires one connected matching family recipe')
    node = program.root_nodes[0]
    if node.operation != 'add' or node.children:
        raise ValueError('detail update cannot alter assembly or Boolean operations')
    params = node.parameters
    if any(key in params for key in ('rotation_euler', 'rotation_row_major', 'offset_x_normalized')):
        raise ValueError('detail update requires the explicit fixed rotation/position recipe contract')
    pose = np.asarray(params.get('rotation', np.eye(3)), float)
    position = np.array([params.get(k, 0.) for k in ('x', 'y', 'z')], float)
    if (pose.shape != (3, 3) or not np.isfinite(pose).all()
            or not np.allclose(pose.T @ pose, np.eye(3), atol=1e-5)
            or np.linalg.det(pose) < .99999 or not np.isfinite(position).all()):
        raise ValueError('adaptive recipe needs a fixed finite proper pose')
    return program


def _arch_outline(params):
    outer = np.asarray(params.get('outer'), float)
    if (outer.shape != (8, 2) or not np.isfinite(outer).all() or params.get('holes')
            or not np.allclose(outer[4:6, 0], outer[4, 0])
            or not np.allclose(outer[6:8, 0], outer[6, 0])
            or not np.isclose(outer[5, 1], outer[6, 1])
            or not np.isclose(outer[4, 1], outer[7, 1])):
        raise ValueError('arch detail model requires the retained eight-corner open notch')
    spans = np.ptp(outer, axis=0)
    scale = np.array([params['width_world'], params['depth_world']], float) / spans
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('arch outline scale must be positive')
    return outer * scale, scale


def family_control_values(wire, family):
    """Physical controls in local world units, independent of source parameters."""
    if family in AXIS_CONTROLS:
        return axis_family_control_values(wire, family)
    p = _validated_program(wire, family).root_nodes[0].parameters
    if family == 'rounded_triangle_dot':
        if 'width_world' in p or 'depth_world' in p:
            raise ValueError('triangle detail controls require scale_xy without absolute-dimension overrides')
        scales = np.asarray(p.get('scale_xy', (1., 1.)), float)
        if scales.shape != (2,):
            raise ValueError('triangle needs two declared scale values')
        values = {'scale_x': float(scales[0]), 'scale_y': float(scales[1]),
                  'corner_radius_world': float(p.get('corner_radius_world', p.get('corner_radius', .16)))}
    elif family == 'concave_arch':
        outline, _ = _arch_outline(p)
        values = {'opening_width_world': float(outline[4, 0] - outline[7, 0]),
                  'cavity_roof_height_world': float(outline[5, 1] - outline[4, 1])}
        if not (0 < values['opening_width_world'] < np.ptp(outline[:, 0])
                and 0 < values['cavity_roof_height_world'] < np.ptp(outline[:, 1])):
            raise ValueError('arch opening must remain inside its fixed exterior')
    else:
        values = {key: float(p[key]) for key in FAMILY_CONTROLS[family]}
    if not all(math.isfinite(value) and value > 0. for value in values.values()):
        raise ValueError('family physical controls must be finite and positive')
    if family == 'torus' and values['minor_radius'] >= values['major_radius']:
        raise ValueError('ring detail model must preserve its open hole')
    return values


def family_program_update(wire, family, values):
    """Change only declared controls and their necessary dimension aliases."""
    if family in AXIS_CONTROLS:
        return axis_family_program_update(wire, family, values)
    program = _validated_program(wire, family)
    previous = family_control_values(wire, family)
    if not values or not set(values).issubset(previous):
        raise ValueError('update contains an undeclared family control')
    controls = {**previous, **dict(values)}
    if any(isinstance(value, bool) or not isinstance(value, numbers.Real)
           or not math.isfinite(value) or value <= 0. for value in controls.values()):
        raise ValueError('updated controls must be finite positive physical dimensions')
    controls = {key: float(value) for key, value in controls.items()}
    if all(controls[key] == previous[key] for key in controls):
        return program
    node = program.root_nodes[0]
    params = deepcopy(dict(node.parameters))
    if family == 'rounded_triangle_dot':
        params['scale_xy'] = [controls['scale_x'], controls['scale_y']]
        params['corner_radius_world'] = controls['corner_radius_world']
    elif family == 'concave_arch':
        outline, scale = _arch_outline(params)
        width, roof = controls['opening_width_world'], controls['cavity_roof_height_world']
        midpoint = .5 * (outline[4, 0] + outline[7, 0])
        if not (outline[:, 0].min() < midpoint-width/2 < midpoint+width/2 < outline[:, 0].max()
                and 0 < roof < np.ptp(outline[:, 1])):
            raise ValueError('updated arch controls would close or cross the fixed exterior')
        outline[4:6, 0] = midpoint + width/2
        outline[6:8, 0] = midpoint - width/2
        outline[5:7, 1] += roof - previous['cavity_roof_height_world']
        params['outer'] = (outline / scale).tolist()
    else:
        params.update(controls)
        if family == 'capsule':
            params['width_world'] = params['depth_world'] = 2*controls['radius_world']
            params['height_world'] = 2*controls['radius_world']+controls['segment_height_world']
        elif family == 'tapered_frustum':
            params['width_world'] = params['depth_world'] = 2*max(controls['radius_bottom'], controls['radius_top'])
        elif controls['minor_radius'] >= controls['major_radius']:
            raise ValueError('updated torus must preserve its open hole')
    return replace(program, root_nodes=(replace(node, parameters=params),))


def _model_camera(record, shape=None):
    matrix, scale = _camera(record)
    if (record.get('projection') != 'ORTHO' or record.get('shift_x', 0.) != 0.
            or record.get('shift_y', 0.) != 0. or record.get('pixel_aspect', [1., 1.]) != [1., 1.]):
        raise ValueError('detail model requires unshifted square-pixel orthographic acquisition')
    resolution = record.get('resolution')
    if (not isinstance(resolution, (list, tuple)) or len(resolution) != 2
            or any(isinstance(n, bool) or not isinstance(n, numbers.Integral) or not 32 <= n <= 2048
                   for n in resolution) or resolution[0] != resolution[1]
            or (shape is not None and tuple(resolution) != (shape[1], shape[0]))):
        raise ValueError('detail acquisition resolution must be matched square 32..2048')
    return matrix, scale


def _hull_polygon(vertices, matrix):
    from scipy.spatial import ConvexHull
    from shapely import Polygon
    projected = (vertices - matrix[:3, 3]) @ matrix[:3, :2]
    unique = np.unique(projected, axis=0)
    return Polygon(unique[ConvexHull(unique).vertices])


@lru_cache(maxsize=8)
def _triangle_vertex_basis(serialized_fixed_params):
    """Existing discrete template is affine in radius before its XY scaling.

    The fixed recipe's template, depth, front fraction and corner/dome counts
    generate both basis meshes through the ordinary primitive implementation.
    This proposal cache retains all vertices; it does not lower tessellation.
    """
    from primitives.rounded_triangle import RoundedTrianglePrimitive
    params=json.loads(serialized_fixed_params)
    meshes=[]
    for radius in (1.,2.):
        meshes.append(RoundedTrianglePrimitive.from_program_parameters(
            {**params,'scale_xy':[1.,1.],'corner_radius_world':radius},world=False).to_mesh_data())
    if meshes[0].faces!=meshes[1].faces:
        raise ValueError('triangle radius changes the declared template connectivity')
    radial=meshes[1].vertices-meshes[0].vertices
    base=meshes[0].vertices-radial
    base.flags.writeable=False;radial.flags.writeable=False
    return base,radial


def _triangle_local_vertices(serialized_params):
    params=json.loads(serialized_params)
    radius=float(params.pop('corner_radius_world',params.get('corner_radius',.16)))
    params.pop('corner_radius',None)
    scale=np.asarray(params.pop('scale_xy',[1.,1.]),float)
    base,radial=_triangle_vertex_basis(json.dumps(params,sort_keys=True,allow_nan=False))
    vertices=base+radius*radial
    vertices[:,:2]*=scale
    return vertices


def projected_family_boundary(wire, family, camera, *, segments=96):
    """Projected discrete model boundary; no renderer or reference geometry.

    Capsule/frustum are convex connected meshes. The ring adapter requires the
    camera to look along its local axis; the arch requires the extrusion axis.
    A cropped camera may keep either supported basis and move its origin.
    """
    if family in AXIS_CONTROLS:
        return axis_projected_family_boundary(wire, family, camera, segments=segments)
    from shapely import Polygon
    if isinstance(segments, bool) or not isinstance(segments, numbers.Integral) or not 12 <= segments <= 128:
        raise ValueError('discrete family segments must be an integer 12..128')
    program = _validated_program(wire, family)
    family_control_values(wire, family)
    params = program.root_nodes[0].parameters
    rotation = np.asarray(params.get('rotation', np.eye(3)), float)
    center = np.array([params.get(k, 0.) for k in ('x', 'y', 'z')], float)
    matrix, _ = _model_camera(camera)
    if family in ('capsule', 'rounded_triangle_dot'):
        if family == 'capsule':
            from primitives.capsule import CapsulePrimitive
            local = CapsulePrimitive.from_program_parameters(params, world=False).to_mesh_data(int(segments)).vertices
        else:
            local = _triangle_local_vertices(json.dumps(dict(params), sort_keys=True, allow_nan=False))
        vertices = local @ rotation.T + center
        polygon = _hull_polygon(vertices, matrix)
    elif family == 'tapered_frustum':
        angle = np.arange(segments) * 2*np.pi/segments
        circle = np.column_stack((np.cos(angle), np.sin(angle)))
        local = np.concatenate([np.column_stack((circle*params[key], np.full(segments, sign*params['height_world']/2)))
                                for key, sign in (('radius_bottom', -1), ('radius_top', 1))])
        polygon = _hull_polygon(local @ rotation.T + center, matrix)
    else:
        if not np.isclose(abs(matrix[:3, 2] @ rotation[:, 2]), 1., atol=1e-6):
            raise ValueError('ring/arch detail view must look along its fixed local axis')
        if family == 'torus':
            angle = np.arange(segments) * 2*np.pi/segments
            circle = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(segments)))
            radii = [params['major_radius']+params['minor_radius'], params['major_radius']-params['minor_radius']]
            rings = [((circle*radius) @ rotation.T + center - matrix[:3, 3])
                     @ matrix[:3, :2] for radius in radii]
            polygon = Polygon(rings[0], [rings[1]])
        else:
            outline, _ = _arch_outline(params)
            points = np.column_stack((outline, np.zeros(8)))
            xy = (points @ rotation.T + center - matrix[:3, 3]) @ matrix[:3, :2]
            polygon = Polygon(xy)
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0.:
        raise ValueError('projected family boundary is invalid')
    return polygon


def projected_family_signed_distance(points, wire, family, camera, *, segments=96):
    """Distance to the external/hole boundary, excluding any internal mesh seams."""
    from shapely import contains_xy, distance, points as geometry_points
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('projected observation points must be finite Nx2')
    polygon = projected_family_boundary(wire, family, camera, segments=segments)
    result = distance(geometry_points(points[:, 0], points[:, 1]), polygon.boundary)
    return np.where(contains_xy(polygon, points[:, 0], points[:, 1]), -result, result)


def adaptive_crop_request(reference, candidate, camera, *, fit_view, heldout_views,
                          prior_view_exposure, pixel_span=160, source_geometry_available=False):
    """A real acquisition request selected exclusively from an allowed fit view."""
    from evaluation.adaptive_measurement import residual_crop_request
    if fit_view in heldout_views:
        raise ValueError('reserved held-out view cannot select a detail crop')
    if not isinstance(prior_view_exposure, dict) or not all(v in prior_view_exposure for v in heldout_views):
        raise ValueError('held-out history must disclose prior view exposure explicitly')
    if not isinstance(source_geometry_available, bool):
        raise ValueError('source availability must be explicitly boolean')
    if not source_geometry_available:
        return {'status': 'unsupported', 'reason': 'fixed images cannot supply newly observed crop detail',
                'source_geometry_available_required': True, 'heldout_fit_or_roi_used': False}
    result = residual_crop_request(reference, candidate, camera, pixel_span=pixel_span)
    return {**result, 'fit_view': fit_view, 'heldout_views': list(heldout_views),
            'prior_view_exposure': deepcopy(prior_view_exposure),
            'heldout_fit_or_roi_used': False}


def refine_family_detail(wire, family, observations, *, parameter_bounds,
                         heldout_views, prior_view_exposure, max_evaluations=96,
                         max_elapsed_s=1., segments=96):
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
    original = family_control_values(wire, family)
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
        projected_family_boundary(wire, family, record['camera'], segments=segments)
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
        updated = family_program_update(wire, family, decode(x)).to_dict()
        result = np.concatenate([projected_family_signed_distance(xy, updated, family, camera, segments=segments)
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
    updated = family_program_update(wire, family, controls)
    detail = {'protocol': 'bounded_discrete_family_detail_v1', 'family': family,
              'free_controls': list(names), 'original_controls': original,
              'selected_controls': {name: float(value) for name, value in controls.items()},
              'frozen_intervals': intervals.tolist(), 'fixed_pose': True, 'segments': None if family == 'rounded_triangle_dot' else int(segments),
              'residual_calls': calls, 'elapsed_seconds': time.perf_counter()-started,
              'termination': termination, 'squared_residual': score, 'observations': records,
              'local_sensitivity_singular_values': singular.tolist(), 'local_rank': rank,
              'interval_bound_active': bool(bound_active),
              'identifiability': 'locally_identified' if rank == len(names) and not bound_active else 'underconstrained',
              'global_uniqueness_established': False, 'heldout_views': list(heldout_views),
              'heldout_fit_or_roi_used': False, 'prior_view_exposure': deepcopy(prior_view_exposure),
              'artist_surface_limits': None, 'full_native_silhouette_admission': 'unrun',
              'model_scope': 'discrete projected boundary; actual native geometry and measurement remain independent'}
    if family == 'rounded_triangle_dot':
        from primitives.rounded_triangle import RoundedTrianglePrimitive
        part = RoundedTrianglePrimitive.from_program_parameters(updated.root_nodes[0].parameters, world=False)
        detail['tessellation'] = {'corner_segments': part.corner_segments, 'dome_segments': part.dome_segments,
                                  'fixed_from_recipe': True}
        detail['fixed_triangle_controls'] = ['height_world', 'front_fraction', 'vertices_xy',
                                            'position', 'rotation', 'corner_segments', 'dome_segments']
    if family in AXIS_CONTROLS:
        if family in ('sphere', 'anisotropic_ellipsoid'):
            detail['tessellation'] = {'radial_segments': int(segments),
                                      'ring_count': max(6, int(segments)//2),
                                      'fixed_for_update': True,
                                      'generator': 'existing EllipsoidPrimitive UV parameter lattice'}
        elif family == 'cylinder':
            detail['tessellation'] = {'radial_segments': int(segments), 'native_end_rings': 2,
                                      'fixed_for_update': True,
                                      'generator': 'existing SuperFrustum convex support at equal cap radii'}
        else:
            detail['segments'] = None
            detail['tessellation'] = {'primitive': 'box', 'corners': 8, 'fixed_for_update': True}
        detail['fixed_axis_contract'] = {'coupled_sphere_axis_ratios': family == 'sphere',
                                         'coupled_cylinder_cap_radii': family == 'cylinder',
                                         'pose_and_tessellation_fixed': True,
                                         'native_geometry_identity': 'independent exact guard required'}
    return replace(updated, metadata={**updated.metadata, 'adaptive_detail': detail})
