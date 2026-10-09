"""Held-out-safe discrete axis-family adapters, pending adaptive integration.

These helpers consume saved candidate recipes only. View diagnostics predict
local model sensitivity; they do not observe source truth or qualify geometry.
"""
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import itertools
import math
import numbers

import numpy as np

from .frozen_family import retained_family_program

AXIS_CONTROLS = {
    'sphere': ('radius_world',),
    'anisotropic_ellipsoid': ('width_world', 'depth_world', 'height_world'),
    'cylinder': ('radius_world', 'height_world'),
    'thin_plate': ('width_world', 'depth_world', 'height_world'),
}
AXIS_PRIMITIVES = {'sphere':'ellipsoid', 'anisotropic_ellipsoid':'ellipsoid',
                   'cylinder':'cylinder', 'thin_plate':'box'}
DIMENSIONS = ('width_world', 'depth_world', 'height_world')


def _axis_program(wire, family):
    if family not in AXIS_CONTROLS:
        raise ValueError('axis family model is unsupported')
    program = retained_family_program(wire)
    if (len(program.root_nodes)!=1 or program.constraints or program.residual_patches
            or program.root_nodes[0].primitive_type!=AXIS_PRIMITIVES[family]
            or program.root_nodes[0].operation!='add' or program.root_nodes[0].children):
        raise ValueError('axis update requires one matching connected additive recipe')
    p = program.root_nodes[0].parameters
    if any(key in p for key in ('rotation_euler', 'rotation_row_major', 'offset_x_normalized')):
        raise ValueError('axis update needs explicit fixed rotation/position')
    rotation = np.asarray(p.get('rotation', np.eye(3)), float)
    center = np.array([p.get(key, 0.) for key in ('x', 'y', 'z')], float)
    if (rotation.shape!=(3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5)
            or np.linalg.det(rotation)<.99999 or not np.isfinite(center).all()):
        raise ValueError('axis recipe needs a finite proper fixed pose')
    for key in DIMENSIONS:
        value = p.get(key)
        if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not math.isfinite(value) or value<=0.):
            raise ValueError('axis recipe dimensions must be finite positive physical values')
    if family in ('sphere', 'anisotropic_ellipsoid') and any(p[key]<2e-6 for key in DIMENSIONS):
        raise ValueError('UV axis dimension is below the existing compiler scale floor')
    if family=='cylinder':
        bottom, top = p.get('radius_bottom'), p.get('radius_top')
        if (any(isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not math.isfinite(value) or value<1e-6 for value in (bottom, top)) or top!=bottom
                or p['width_world']!=2*bottom or p['depth_world']!=2*bottom):
            raise ValueError('cylinder needs exact equal cap radii and coupled diameter aliases')
    return program


def axis_family_control_values(wire, family):
    p = _axis_program(wire, family).root_nodes[0].parameters
    if family=='sphere':
        # This is the existing compiler's construction radius. Retained slight
        # anisotropy remains fixed; an update cannot replace it with GT equality.
        return {'radius_world': float(max(p[key] for key in DIMENSIONS)/2)}
    if family=='cylinder':
        return {'radius_world':float(p['radius_bottom']), 'height_world':float(p['height_world'])}
    return {key:float(p[key]) for key in DIMENSIONS}


def axis_family_program_update(wire, family, values):
    program = _axis_program(wire, family)
    previous = axis_family_control_values(wire, family)
    if not values or not set(values).issubset(previous):
        raise ValueError('axis update contains an undeclared control')
    controls = {**previous, **dict(values)}
    if any(isinstance(value, bool) or not isinstance(value, numbers.Real)
           or not math.isfinite(value) or value<=0. for value in controls.values()):
        raise ValueError('updated axis controls must be finite positive physical values')
    controls = {key:float(value) for key, value in controls.items()}
    if all(controls[key]==previous[key] for key in controls):
        return program
    node = program.root_nodes[0]
    p = deepcopy(dict(node.parameters))
    if family=='sphere':
        ratio = controls['radius_world']/previous['radius_world']
        for key in DIMENSIONS:
            p[key] = float(p[key]*ratio)
    elif family=='cylinder':
        p['radius_bottom'] = p['radius_top'] = controls['radius_world']
        p['width_world'] = p['depth_world'] = 2*controls['radius_world']
        p['height_world'] = controls['height_world']
    else:
        p.update(controls)
    updated = replace(program, root_nodes=(replace(node, parameters=p),))
    _axis_program(updated.to_dict(), family)
    return updated


@lru_cache(maxsize=8)
def _unit_uv_vertices(segments):
    from primitives.analytic_primitives import EllipsoidPrimitive
    vertices = EllipsoidPrimitive(radii=(1., 1., 1.)).to_mesh_data(segments).vertices
    vertices.flags.writeable = False
    return vertices


@lru_cache(maxsize=8)
def _unit_cylinder_vertices(segments):
    from primitives.superfrustum import SuperFrustum
    vertices = SuperFrustum(radius_bottom=1., radius_top=1., height=1.).to_mesh_data(segments).vertices
    vertices.flags.writeable = False
    return vertices


def axis_family_local_vertices(wire, family, *, segments=96):
    if (isinstance(segments, bool) or not isinstance(segments, numbers.Integral)
            or not 12<=segments<=128):
        raise ValueError('axis discrete segments must be integer12..128')
    p = _axis_program(wire, family).root_nodes[0].parameters
    dimensions = np.array([p[key] for key in DIMENSIONS], float)
    if family in ('sphere', 'anisotropic_ellipsoid'):
        return _unit_uv_vertices(int(segments))*(dimensions/2)
    if family=='cylinder':
        return _unit_cylinder_vertices(int(segments))*[p['radius_bottom'], p['radius_bottom'], p['height_world']]
    return np.array(list(itertools.product((-.5, .5), repeat=3)))*dimensions


def axis_projected_family_boundary(wire, family, camera, *, segments=96):
    # Runtime imports avoid a cycle when the frozen core later delegates here.
    from .adaptive_family import _hull_polygon, _model_camera
    p = _axis_program(wire, family).root_nodes[0].parameters
    rotation = np.asarray(p.get('rotation', np.eye(3)), float)
    center = np.array([p.get(key, 0.) for key in ('x', 'y', 'z')], float)
    matrix, _ = _model_camera(camera)
    local = axis_family_local_vertices(wire, family, segments=segments)
    polygon = _hull_polygon(local@rotation.T+center, matrix)
    if polygon.is_empty or not polygon.is_valid or polygon.area<=0.:
        raise ValueError('axis projected boundary is invalid')
    return polygon


def axis_projected_signed_distance(points, wire, family, camera, *, segments=96):
    from shapely import contains_xy, distance, points as geometry_points
    points = np.asarray(points, float)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
        raise ValueError('axis contour points must be finiteNx2')
    polygon = axis_projected_family_boundary(wire, family, camera, segments=segments)
    result = distance(geometry_points(points[:, 0], points[:, 1]), polygon.boundary)
    return np.where(contains_xy(polygon, points[:, 0], points[:, 1]), -result, result)


def axis_candidate_view_sensitivity(wire, family, cameras, *, heldout_views=(),
                                    points_per_view=192, segments=96):
    """Candidate-only local rank using internal visible boundary samples.

    This prepares acquisition choices without reference/candidate comparisons.
    It never closes a clipped contour or calls a source observation complete.
    """
    from shapely import get_coordinates, line_interpolate_point
    from .adaptive_family import _model_camera
    controls = axis_family_control_values(wire, family)
    if (not isinstance(cameras, dict) or not cameras or set(cameras)&set(heldout_views)
            or isinstance(points_per_view, bool) or not isinstance(points_per_view, numbers.Integral)
            or not 32<=points_per_view<=384):
        raise ValueError('candidate view diagnostic excludes held-outs and allows32..384 points')
    records, evidence = {}, []
    for view, camera in sorted(cameras.items()):
        _, scale = _model_camera(camera)
        polygon = axis_projected_family_boundary(wire, family, camera, segments=segments)
        samples = get_coordinates(line_interpolate_point(polygon.exterior,
            np.linspace(0., polygon.length, int(points_per_view), endpoint=False)))
        visible = np.all((samples>=-scale/2)&(samples<=scale/2), axis=1)
        points = samples[visible]
        if len(points)<4:
            raise ValueError('candidate view lacks enough observed internal boundary samples')
        b = polygon.bounds
        censored = min(b[:2]) < -scale/2 or max(b[2:]) > scale/2
        records[view] = {'candidate_frame_censored':bool(censored),
                         'internal_candidate_samples':len(points), 'frame_border_closure_added':False,
                         'source_complete_contour_claim':False}
        evidence.append((points, camera, scale, math.sqrt(len(points))))
    columns, pixel_movements = [], {}
    for name, original in controls.items():
        # Match the existing normalized +/-10% interval and1e-3 local probe.
        span = .2*original
        lower = axis_family_program_update(wire, family, {name:original-span*.001}).to_dict()
        upper = axis_family_program_update(wire, family, {name:original+span*.001}).to_dict()
        column, movement = [], []
        for points, camera, scale, weight in evidence:
            delta = (axis_projected_signed_distance(points, upper, family, camera, segments=segments)
                     -axis_projected_signed_distance(points, lower, family, camera, segments=segments))/.002
            column.append(delta/scale/weight)
            movement.append(float(np.max(np.abs(delta))*camera['resolution'][0]/scale))
        columns.append(np.concatenate(column));pixel_movements[name] = max(movement)
    singular = np.linalg.svd(np.column_stack(columns), compute_uv=False)
    rank = int(np.count_nonzero(singular>max(1e-8, singular[0]*1e-5)))
    condition = float(singular[0]/singular[-1]) if singular[-1]>0. else None
    return {'protocol':'candidate_only_axis_view_sensitivity_v1', 'family':family,
            'control_names':list(controls), 'local_rank':rank, 'singular_values':singular.tolist(),
            'condition_number':condition, 'normalized_interval_pixel_sensitivity':pixel_movements,
            'observations':records, 'source_observations':0, 'heldout_fit_or_view_choice_used':False,
            'global_uniqueness_established':False, 'source_identifiability':'unqualified until genuine observations',
            'artist_surface_limits':None, 'native_candidate_acceptance':'not supplied'}
