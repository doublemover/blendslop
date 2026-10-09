"""Three-control triangle dome follow-up at fixed template and orientation.

Only world Y/Z position and front depth fraction vary. Outline, total thickness,
X position, rigid orientation and the actual corner/dome tessellation stay fixed.
The projected convex boundary uses the ordinary generated finite mesh.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import json
import math
import numbers

import numpy as np

from .adaptive_family import _validated_program, family_control_values, _model_camera
from .arch_tail_refinement import _bounded_tail_update

CONTROLS = ('center_y_world', 'center_z_world', 'front_fraction')


def triangle_pose_controls(wire):
    program = _validated_program(wire, 'rounded_triangle_dot')
    family_control_values(wire, 'rounded_triangle_dot')
    params = program.root_nodes[0].parameters
    fraction = params.get('front_fraction', .5)
    if (isinstance(fraction, bool) or not isinstance(fraction, numbers.Real)
            or not math.isfinite(fraction) or not 0 < fraction < 1):
        raise ValueError('triangle front depth fraction must be finite inside (0, 1)')
    return {'center_y_world': float(params.get('y', 0.)),
            'center_z_world': float(params.get('z', 0.)), 'front_fraction': float(fraction)}


def triangle_pose_update(wire, values):
    program = _validated_program(wire, 'rounded_triangle_dot')
    previous = triangle_pose_controls(wire)
    if not values or not set(values).issubset(previous):
        raise ValueError('triangle follow-up contains an undeclared position/depth control')
    controls = {**previous, **dict(values)}
    if (any(isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v)
            for v in controls.values()) or not 0 < controls['front_fraction'] < 1):
        raise ValueError('triangle positions must be finite and depth fraction inside (0, 1)')
    if controls == previous: return program
    node = program.root_nodes[0]; params = deepcopy(dict(node.parameters))
    params.update(y=float(controls['center_y_world']), z=float(controls['center_z_world']),
                  front_fraction=float(controls['front_fraction']))
    return replace(program, root_nodes=(replace(node, parameters=params),))


@lru_cache(maxsize=8)
def _symmetric_vertices(fixed_params):
    from primitives.rounded_triangle import RoundedTrianglePrimitive
    mesh = RoundedTrianglePrimitive.from_program_parameters(
        json.loads(fixed_params), world=False).to_mesh_data()
    mesh.vertices.flags.writeable = False
    return mesh.vertices


def triangle_pose_boundary(wire, camera):
    program = _validated_program(wire, 'rounded_triangle_dot')
    controls = triangle_pose_controls(wire)
    params = dict(program.root_nodes[0].parameters)
    fixed = deepcopy(params)
    for key in ('x', 'y', 'z', 'rotation'): fixed.pop(key, None)
    fixed['front_fraction'] = .5
    vertices = _symmetric_vertices(json.dumps(fixed, sort_keys=True, allow_nan=False)).copy()
    fraction = controls['front_fraction']
    vertices[:, 2] *= np.where(vertices[:, 2] >= 0, 2*fraction, 2*(1-fraction))
    rotation = np.asarray(params.get('rotation', np.eye(3)), float)
    center = np.array([params.get(k, 0.) for k in ('x', 'y', 'z')], float)
    matrix, _ = _model_camera(camera)
    from scipy.spatial import ConvexHull
    from shapely import Polygon
    # Qhull accepts duplicates; every actual generated vertex is still supplied.
    # Avoid sorting all 6239 points with np.unique for each finite difference.
    projected = (vertices@rotation.T+center-matrix[:3, 3])@matrix[:3, :2]
    return Polygon(projected[ConvexHull(projected).vertices])


def triangle_pose_signed_distance(points, wire, camera):
    from shapely import contains_xy, distance, points as geometry_points
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('triangle contour points must be finite Nx2')
    shape = triangle_pose_boundary(wire, camera)
    result = distance(geometry_points(points[:, 0], points[:, 1]), shape.boundary)
    return np.where(contains_xy(shape, points[:, 0], points[:, 1]), -result, result)


def refine_triangle_pose(wire, observations, *, parameter_bounds, heldout_views,
                         prior_view_exposure, baseline_geometry_hash,
                         baseline_program_sha256, max_evaluations=96, max_elapsed_s=1.):
    return _bounded_tail_update(wire, observations, controls=triangle_pose_controls(wire),
        update=triangle_pose_update, distance=triangle_pose_signed_distance,
        parameter_bounds=parameter_bounds, heldout_views=heldout_views,
        prior_view_exposure=prior_view_exposure, baseline_geometry_hash=baseline_geometry_hash,
        baseline_program_sha256=baseline_program_sha256, max_evaluations=max_evaluations,
        max_elapsed_s=max_elapsed_s, protocol='bounded_triangle_pose_tail_v1',
        metadata_key='triangle_pose_tail')
