"""Continuous, source-only certificate for the frozen authored triangular lens.

The exact binary64 author points define the analytic offset outline. Neither an
ideal equilateral replacement nor correctly rounded producer libm is assumed.
Actual vertex construction shifts are included in the continuous bound.
"""
from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json

import numpy as np

from .torus_reference import (
    _add, _alternating_atan, _cyclic, _float_upper, _mul, _neg,
    _pi_interval, _sqrt_upper, _trig_turns,
)

PROTOCOL = 'authored_rounded_triangle_parameter_cover_v1'
FROZEN_SOURCE_HASH = '2681272b491af3370176a3751e408ef36496ba592d230bce7c733b8c41cadc64'
CORNER_SEGMENTS = 32
DOME_SEGMENTS = 64
OUTLINE_POINTS = 3 * (CORNER_SEGMENTS + 1)
_GRID = 1 << 100


def _outward(interval):
    """Exact outward rounding keeps the exhaustive rational calculation bounded."""
    low, high = interval
    return F((low * _GRID).__floor__(), _GRID), F((high * _GRID).__ceil__(), _GRID)


def _scale(interval, value):
    return _outward(_mul(interval, (value, value)))


def _frozen_recipe():
    from blender_blocking.synthetic.quality_contracts import quality_workload
    row = next(x for x in quality_workload()['cases'] if x['name'] == 'rounded_triangle_dot')
    expected = {
        'builder': 'rounded_triangle',
        'vertices_xy': [[0, .9], [-.7794228634059948, -.45], [.7794228634059948, -.45]],
        'corner_radius': .16, 'thickness': .48,
        'dome': 'z=+-thickness/2*cos(phi), outline_scale=sin(phi)',
        'corner_segments': CORNER_SEGMENTS, 'dome_segments': DOME_SEGMENTS,
    }
    if row['parameters'] != expected:
        raise ValueError('frozen authored rounded triangle declaration changed')
    points = tuple(tuple(F.from_float(float(x)) for x in p) for p in expected['vertices_xy'])
    center = tuple(sum(p[k] for p in points) / 3 for k in range(2))
    # These are properties of the exact supplied binary64 numbers, not an
    # equilateral idealization. The unequal edge lengths remain unequal.
    if (points[0][0] != 0 or points[1][0] != -points[2][0]
            or points[1][1] != points[2][1] or center != (0, 0)
            or points[0][1] <= points[1][1]):
        raise ValueError('frozen exact isosceles coordinate interpretation changed')
    return expected, points, center, F.from_float(.16), F.from_float(.48) / 2


@lru_cache(maxsize=1)
def _beta_interval():
    _, points, _, _, _ = _frozen_recipe()
    ratio = points[2][0] / (points[0][1] - points[1][1])
    t = (ratio - 1) / (ratio + 1)
    if not -F(1, 3) < t < 0:
        raise ValueError('frozen beta alternating-series convergence guard failed')
    # atan(ratio) = pi/4 + atan(t), with exact rational alternating remainder.
    return _add(_scale(_pi_interval(), F(1, 4)), _alternating_atan(1 / t, 40))


@lru_cache(maxsize=99)
def _outline_angle(corner, sample):
    if corner not in range(3) or sample not in range(CORNER_SEGMENTS + 1):
        raise ValueError('outline parameter index outside frozen cover')
    t = F(sample, CORNER_SEGMENTS)
    coefficients = ((t, 1 - 2*t), (1 + t/2, -1 + t), (-F(1, 2) + t/2, t))
    a, b = coefficients[corner]
    return _outward(_add(_scale(_pi_interval(), a), _scale(_beta_interval(), b)))


@lru_cache(maxsize=110)
def _angle_trig(angle):
    """True interval-angle sin/cos, via directed turns and a Lipschitz extension."""
    pi = _pi_interval()
    turns = (angle[0] + angle[1]) / (2 * (pi[0] + pi[1]))
    anchor = _mul((2*turns, 2*turns), pi)
    error = max(abs(a-b) for a in angle for b in anchor)
    sine, cosine = _trig_turns(turns)
    return tuple(_outward((max(F(-1), x[0]-error), min(F(1), x[1]+error)))
                 for x in (sine, cosine))


@lru_cache(maxsize=1)
def _analytic_data():
    _, points, center, radius, height = _frozen_recipe()
    outline, normals, supports = [], [], []
    for corner, point in enumerate(points):
        for sample in range(CORNER_SEGMENTS + 1):
            sine, cosine = _angle_trig(_outline_angle(corner, sample))
            normal = cosine, sine
            outline.append(tuple(_outward(_add((point[k],)*2, _scale(normal[k], radius)))
                                 for k in range(2)))
            normals.append(normal)
            support = (radius, radius)
            for k in range(2):
                support = _outward(_add(support, _scale(normal[k], point[k]-center[k])))
            supports.append(support)
    phi = tuple(tuple(_outward(x) for x in _trig_turns(F(k, 2*DOME_SEGMENTS)))
                for k in range(DOME_SEGMENTS + 1))
    # Exact poles, rather than uncertain near-zero trig terms.
    phi = (((F(0), F(0)), (F(1), F(1))), *phi[1:-1],
           ((F(0), F(0)), (F(-1), F(-1))))
    expected = [(tuple((x, x) for x in center) + ((height, height),))]
    for k in range(1, DOME_SEGMENTS):
        sp, cp = phi[k]
        for q in outline:
            expected.append(tuple(_outward(_add((center[axis],)*2,
                _mul(sp, _add(q[axis], (-center[axis],)*2)))) for axis in range(2))
                + (_scale(cp, height),))
    expected.append(tuple((x, x) for x in center) + ((-height, -height),))
    endpoint_normals = []
    for k in range(DOME_SEGMENTS + 1):
        sp, cp = phi[k]
        endpoint_normals.append(tuple(
            (_outward(_mul(_scale(sp, height), n[0])),
             _outward(_mul(_scale(sp, height), n[1])),
             _outward(_mul(cp, support)))
            for n, support in zip(normals, supports)))
    return tuple(expected), tuple(endpoint_normals)


def _inventory(faces):
    """One oriented triangle per pole cell, two per other cell, including seams."""
    count, m = OUTLINE_POINTS, DOME_SEGMENTS
    expected = {}
    def retain(triangle, edge, row):
        expected[_cyclic(list(triangle))] = (edge, row)
    for i in range(count):
        retain((0, 1+i, 1+(i+1) % count), i, 0)
    for row in range(m-2):
        a, b = 1+row*count, 1+(row+1)*count
        for i in range(count):
            j = (i+1) % count
            retain((a+i, b+i, b+j), i, row+1)
            retain((a+i, b+j, a+j), i, row+1)
    a, bottom = 1+(m-2)*count, 1+(m-1)*count
    for i in range(count):
        retain((a+i, bottom, a+(i+1) % count), i, m-1)
    if faces.shape != (len(expected), 3):
        raise ValueError('source lacks the complete frozen pole-fan and quad inventory')
    visited, cells = set(), []
    for face in faces:
        key = _cyclic(list(map(int, face)))
        if key not in expected:
            raise ValueError('source facet differs from frozen parameter cover/winding/diagonal')
        if key in visited:
            raise ValueError('source parameter cover has duplicate facets and a hole')
        visited.add(key)
        cells.append(expected[key])
    if len(visited) != len(expected):
        raise ValueError('source lacks complete once-only parameter coverage')
    return cells


def _cross(a, b):
    return tuple(a[(k+1) % 3]*b[(k+2) % 3]-a[(k+2) % 3]*b[(k+1) % 3]
                 for k in range(3))


def _facet_tangent_squared(cross, normal):
    dot = (F(0), F(0))
    for x, interval in zip(cross, normal):
        dot = _add(dot, _mul((x, x), interval))
    if dot[0] <= 0:
        raise ValueError('actual facet is degenerate or not outward throughout its endpoint cone')
    numerator = F(0)
    for k in range(3):
        interval = _add(_mul((cross[(k+1) % 3],)*2, normal[(k+2) % 3]),
                        _neg(_mul((cross[(k+2) % 3],)*2, normal[(k+1) % 3])))
        numerator += max(abs(x) for x in interval)**2
    return numerator / dot[0]**2


def _independent_bounds():
    _, points, center, radius, height = _frozen_recipe()
    pi, beta = _pi_interval(), _beta_interval()
    arcs = (_add(pi, _neg(_scale(beta, 2))), _add(_scale(pi, F(1, 2)), beta))
    maximum_span = max(x[1] for x in arcs) / CORNER_SEGMENTS
    if any(x[0] <= 0 or x[1] >= pi[0] for x in arcs):
        raise ValueError('outline arcs must be positive convex turns smaller than pi')
    radial = F.from_float(_sqrt_upper(max(sum((p[k]-center[k])**2 for k in range(2))
                                        for p in points))) + radius
    meridian = max(radial, height) * (pi[1]/DOME_SEGMENTS)**2 / 8
    outline = radius * maximum_span**2 / 8
    # The centroid-to-edge distances bound every outward support of the
    # convex triangle from below, independently of sampled outline values.
    distances = []
    for i, p in enumerate(points):
        q = points[(i+1) % 3]
        edge = tuple(q[k]-p[k] for k in range(2))
        area = edge[0]*(center[1]-p[1])-edge[1]*(center[0]-p[0])
        if area <= 0:
            raise ValueError('exact authored centroid is not inside the CCW triangle')
        length_upper = F.from_float(_sqrt_upper(sum(x*x for x in edge)))
        distances.append(area / length_upper)
    minimum_normal_length = min(height, min(distances)+radius)
    cosine = _angle_trig((maximum_span/2, maximum_span/2))[1]
    if cosine[0] <= 0:
        raise ValueError('circular endpoint normal cone spans a half turn')
    correction = radius * (1/cosine[0]-1)
    if correction >= minimum_normal_length:
        raise ValueError('circular support correction cannot prove an oriented normal cone')
    correction_angle = correction / (minimum_normal_length-correction)
    return meridian, outline, correction, correction_angle, minimum_normal_length


def _certificate(reference):
    parameters, _, _, _, _ = _frozen_recipe()
    vertices = np.asarray(reference.vertices, float)
    faces = np.asarray(reference.faces)
    if (vertices.shape != (2+(DOME_SEGMENTS-1)*OUTLINE_POINTS, 3)
            or not np.isfinite(vertices).all() or not np.issubdtype(faces.dtype, np.integer)):
        raise ValueError('source must be the complete frozen indexed triangular lens')
    from blender_blocking.reconstruction.native_geometry import GeometryArrays
    if GeometryArrays.capture(vertices, faces).content_hash != getattr(reference, 'content_hash', None):
        raise ValueError('source array identity differs from its declared content hash')
    cells = _inventory(faces)
    actual = [tuple(F.from_float(float(x)) for x in vertex) for vertex in vertices]
    expected, normals = _analytic_data()
    max_shift_squared = max(sum(max(abs(x-a), abs(x-b))**2
                                for x, (a, b) in zip(vertex, ideal))
                            for vertex, ideal in zip(actual, expected))
    tangent_squared = F(0)
    straight_tangent_squared = F(0)
    circular_tangent_squared = F(0)
    for face, (edge, row) in zip(faces, cells):
        a, b, c = [actual[int(i)] for i in face]
        cross = _cross(tuple(b[k]-a[k] for k in range(3)), tuple(c[k]-a[k] for k in range(3)))
        straight = edge % (CORNER_SEGMENTS+1) == CORNER_SEGMENTS
        # For a straight outline edge, its two endpoint normals are equal.
        # Circular cells use all four corners, including pole endpoints.
        ids = (edge,) if straight else (edge, (edge+1) % OUTLINE_POINTS)
        maximum = max(_facet_tangent_squared(cross, normals[k][i])
                      for k in (row, row+1) for i in ids)
        tangent_squared = max(tangent_squared, maximum)
        if straight:
            straight_tangent_squared = max(straight_tangent_squared, maximum)
        else:
            circular_tangent_squared = max(circular_tangent_squared, maximum)
    meridian, outline, correction, correction_angle, minimum_length = _independent_bounds()
    shift = _sqrt_upper(max_shift_squared)
    circular_angle = F.from_float(_sqrt_upper(circular_tangent_squared)) + correction_angle
    straight_angle = F.from_float(_sqrt_upper(straight_tangent_squared))
    angle = max(circular_angle, straight_angle)
    pi_low = _pi_interval()[0]
    if angle >= pi_low/2:
        raise ValueError('normal cone cannot prove outward orientation over every parameter cell')
    return {
        'protocol': PROTOCOL, 'status': 'certified', 'family': 'rounded_triangle_dot',
        'reference_geometry_hash': reference.content_hash,
        'frozen_parameters_sha256': hashlib.sha256(json.dumps(parameters, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'authored_parameter_cover': {
            'outline_points': OUTLINE_POINTS, 'corner_segments': CORNER_SEGMENTS,
            'dome_segments': DOME_SEGMENTS, 'vertices': len(vertices), 'triangles': len(faces),
            'circular_cells': 3*CORNER_SEGMENTS*DOME_SEGMENTS,
            'straight_cells': 3*DOME_SEGMENTS, 'collapsed_pole_fans': 2,
            'pole_fan_triangles': 2*OUTLINE_POINTS, 'oriented_once_only': True,
            'outline_seams_verified': True, 'diagonal': '00-to-11',
        },
        'maximum_source_facet_distance_world': _float_upper(meridian+outline+F.from_float(shift)),
        'meridian_interpolation_distance_bound_world': _float_upper(meridian),
        'circular_outline_interpolation_distance_bound_world': _float_upper(outline),
        'straight_outline_interpolation_distance_bound_world': 0.,
        'maximum_vertex_construction_shift_world': shift,
        'maximum_normal_angle_degrees': _float_upper(angle*180/pi_low),
        'facet_to_endpoint_cone_tangent_upper_bound': _sqrt_upper(tangent_squared),
        'circular_support_correction_world_upper_bound': _float_upper(correction),
        'circular_support_correction_angle_radians_upper_bound': _float_upper(correction_angle),
        'analytic_normal_length_lower_bound': -_float_upper(-minimum_length),
        'construction_scope': 'actual exact binary64 source coordinates versus analytic interval vertices; no libm rounding assumption or construction acceptance tolerance',
        'distance_correspondence': 'bidirectional continuous analytic-cell to ideal planar chord-quad to actual-facet cover; poles collapse to complete triangular fans',
        'normal_correspondence': 'each actual facet to every outward analytic normal in its corresponding parameter cell; edge nearest-normal uniqueness not asserted',
        'coverage_proof': 'exact once-only oriented triangles for every cell and both collapsed pole fans, including cyclic outline seams',
        'distance_proof': 'sequential meridian and circular-outline interpolation remainder <= maximum second derivative times span squared/8; all ideal chord quads planar; barycentric actual construction displacement',
        'normal_proof': 'exact facet cross and interval dot/cross endpoint cones; positive phi and theta endpoint coefficients; circular vertical support correction r*(sec(theta_span/2)-1); atan(t)<=t and angle triangle inequality',
        'arithmetic': 'exact rational binary64 author/source data and outward 100-bit interval grid; rational Machin pi and alternating atan; existing directed Decimal50 Taylor trig; rationally checked upper float endpoints and square roots',
        'artist_surface_limits': None, 'candidate_boundary_qualification': 'not supplied',
    }


def rounded_triangle_reference_certificate(reference):
    """Certify only the exact frozen authored source, otherwise give its blocker."""
    try:
        if getattr(reference, 'content_hash', None) != FROZEN_SOURCE_HASH:
            raise ValueError('source changed from the frozen original authored triangle identity')
        return _certificate(reference)
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return {'protocol': PROTOCOL, 'status': 'unsupported', 'family': 'rounded_triangle_dot',
                'reference_geometry_hash': getattr(reference, 'content_hash', None),
                'reason': type(exc).__name__ + ': ' + str(exc),
                'artist_surface_limits': None, 'candidate_boundary_qualification': 'not supplied'}
