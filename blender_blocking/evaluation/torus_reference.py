"""Continuous source-only certificates for the frozen authored 128x24 torus.

Coverage is an exact oriented parameter-triangle inventory, including seams.
Distance and normal bounds use analytic derivatives, not surface samples.
"""
from __future__ import annotations

from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
import math

import numpy as np

MAJOR_SEGMENTS = 128
MINOR_SEGMENTS = 24
PROTOCOL = 'authored_torus_parameter_cover_v1'
FROZEN_SOURCE_HASH = '3d1743e165e096fc7e413237523e2402a809a7e71c6209f95d1de9f705b1738a'


def _alternating_atan(inverse, terms):
    x = F(1, inverse)
    total = sum(((-1)**k * x**(2*k+1) / (2*k+1) for k in range(terms)), F(0))
    endpoint = total + (-1)**terms * x**(2*terms+1) / (2*terms+1)
    return min(total, endpoint), max(total, endpoint)


@lru_cache(maxsize=1)
def _pi_interval():
    # Machin's identity plus the alternating-series remainder is rational.
    a, b = _alternating_atan(5, 40), _alternating_atan(239, 12)
    return 16*a[0]-4*b[1], 16*a[1]-4*b[0]


def _add(a, b):
    return a[0]+b[0], a[1]+b[1]


def _neg(a):
    return tuple(x.copy_negate() if isinstance(x, Decimal) else -x for x in (a[1], a[0]))


def _mul(a, b):
    values = [x*y for x in a for y in b]
    return min(values), max(values)


def _decimal(fraction, rounding):
    with localcontext() as context:
        context.prec = 50
        context.rounding = rounding
        return Decimal(fraction.numerator) / Decimal(fraction.denominator)


def _decimal_op(a, b, operation):
    values = []
    for rounding in (ROUND_FLOOR, ROUND_CEILING):
        with localcontext() as context:
            context.prec = 50
            context.rounding = rounding
            values.append([operation(x, y) for x in a for y in b])
    return min(values[0]), max(values[1])


@lru_cache(maxsize=304)
def _trig_turns(turns):
    """Outward rational sin/cos intervals, with no libm trig assumptions."""
    turns = turns % 1
    quadrant = int(4*turns)
    residual = turns-F(quadrant, 4)
    low, high = _pi_interval()
    x = (_decimal(2*residual*low, ROUND_FLOOR),
         _decimal(2*residual*high, ROUND_CEILING))
    square = _decimal_op(x, x, lambda a, b: a*b)
    sine_term, cosine_term = x, (Decimal(1), Decimal(1))
    sine, cosine = sine_term, cosine_term
    for k in range(1, 24):
        sine_term = _decimal_op(_neg(sine_term), square, lambda a, b: a*b)
        cosine_term = _decimal_op(_neg(cosine_term), square, lambda a, b: a*b)
        sine_term = _decimal_op(sine_term, (Decimal(2*k*(2*k+1)),)*2, lambda a, b: a/b)
        cosine_term = _decimal_op(cosine_term, (Decimal((2*k-1)*2*k),)*2, lambda a, b: a/b)
        sine = _decimal_op(sine, sine_term, lambda a, b: a+b)
        cosine = _decimal_op(cosine, cosine_term, lambda a, b: a+b)
    # x<2: the first omitted sine/cosine terms are both <1e-45.
    assert F(2)**48/F(math.factorial(48)) < F(1, 10**45)
    remainder = (Decimal('-1e-45'), Decimal('1e-45'))
    sine = tuple(F(v) for v in _decimal_op(sine, remainder, lambda a, b: a+b))
    cosine = tuple(F(v) for v in _decimal_op(cosine, remainder, lambda a, b: a+b))
    return ((sine, cosine), (cosine, _neg(sine)),
            (_neg(sine), _neg(cosine)), (_neg(cosine), sine))[quadrant]


def _sqrt_upper(value):
    result = math.sqrt(float(value))
    while F.from_float(result)**2 < value:
        result = math.nextafter(result, math.inf)
    return result


def _float_upper(value):
    result = float(value)
    while F.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _cyclic(face):
    return min(tuple(face[i:]+face[:i]) for i in range(3))


def _frozen_recipe():
    from blender_blocking.synthetic.quality_contracts import quality_workload
    row = next(x for x in quality_workload()['cases'] if x['name']=='torus')
    p = row['parameters']
    if p != {'primitive':'torus', 'major_radius':.7, 'minor_radius':.22}:
        raise ValueError('frozen authored torus declaration changed')
    return p, F.from_float(p['major_radius']), F.from_float(p['minor_radius'])


def _certificate(reference):
    parameters, major, minor = _frozen_recipe()
    vertices = np.asarray(reference.vertices, float)
    faces = np.asarray(reference.faces)
    n, m = MAJOR_SEGMENTS, MINOR_SEGMENTS
    if (vertices.shape!=(n*m, 3) or not np.isfinite(vertices).all()
            or faces.shape!=(2*n*m, 3) or not np.issubdtype(faces.dtype, np.integer)
            or faces.min()<0 or faces.max()>=n*m):
        raise ValueError('source must be the complete frozen 128x24 indexed triangle lattice')
    from blender_blocking.reconstruction.native_geometry import GeometryArrays
    if GeometryArrays.capture(vertices, faces).content_hash != getattr(reference, 'content_hash', None):
        raise ValueError('source array identity differs from its declared content hash')
    actual = [tuple(F.from_float(float(x)) for x in vertex) for vertex in vertices]
    max_shift_squared = F(0)
    for i in range(n):
        st, ct = _trig_turns(F(i, n))
        for j in range(m):
            sp, cp = _trig_turns(F(j, m))
            radial = _add((major, major), _mul((minor, minor), cp))
            expected = (_mul(radial, ct), _mul(radial, st), _mul((minor, minor), sp))
            shift_squared = sum(max(abs(value-a), abs(value-b))**2
                                for value, (a, b) in zip(actual[i*m+j], expected))
            max_shift_squared = max(max_shift_squared, shift_squared)
    inventory = {}
    for i in range(n):
        for j in range(m):
            a, b = i*m+j, ((i+1)%n)*m+j
            c, d = ((i+1)%n)*m+(j+1)%m, i*m+(j+1)%m
            for triangle in ((a, b, c), (a, c, d)):
                inventory[_cyclic(list(triangle))] = (i, j)
    maximum_tangent_squared = F(0)
    visited = set()
    for face in faces:
        ids = list(map(int, face))
        key = _cyclic(ids)
        if key not in inventory:
            raise ValueError('source facet differs from frozen oriented parameter cover/winding')
        if key in visited:
            raise ValueError('source parameter cover has duplicate facets and a hole')
        visited.add(key)
        i, j = inventory[key]
        st, ct = _trig_turns(F(2*i+1, 2*n))
        sp, cp = _trig_turns(F(2*j+1, 2*m))
        normal = (_mul(ct, cp), _mul(st, cp), sp)
        a = [actual[ids[1]][k]-actual[ids[0]][k] for k in range(3)]
        b = [actual[ids[2]][k]-actual[ids[0]][k] for k in range(3)]
        cross = [a[(k+1)%3]*b[(k+2)%3]-a[(k+2)%3]*b[(k+1)%3] for k in range(3)]
        dot = (F(0), F(0))
        for component, interval in zip(cross, normal):
            dot = _add(dot, _mul((component, component), interval))
        if dot[0] <= 0:
            raise ValueError('actual source facet is degenerate or not outward at its analytic cell center')
        tangent_numerator_squared = F(0)
        for k in range(3):
            term = _add(_mul((cross[(k+1)%3],)*2, normal[(k+2)%3]),
                        _neg(_mul((cross[(k+2)%3],)*2, normal[(k+1)%3])))
            tangent_numerator_squared += max(abs(term[0]), abs(term[1]))**2
        maximum_tangent_squared = max(maximum_tangent_squared,
                                      tangent_numerator_squared/dot[0]**2)
    if len(visited)!=len(inventory):
        raise ValueError('source lacks complete parameter boundary coverage')
    pi_low, pi_high = _pi_interval()
    # Taylor expectation remainder: Hessian norm bounds (R+r), r, r.
    interpolation = pi_high**2*((major+minor)/n**2+2*minor/(n*m)+minor/m**2)/2
    shift = _sqrt_upper(max_shift_squared)
    distance = _float_upper(interpolation+F.from_float(shift))
    center_tangent = _sqrt_upper(maximum_tangent_squared)
    cell_normal_radius = _sqrt_upper(pi_high**2*(F(1, n*n)+F(1, m*m)))
    normal_radians = F.from_float(center_tangent)+F.from_float(cell_normal_radius)
    if normal_radians >= pi_low/2:
        raise ValueError('normal cone cannot prove outward orientation throughout every parameter cell')
    normal_degrees = _float_upper(normal_radians*180/pi_low)
    return {
        'protocol':PROTOCOL, 'status':'certified', 'family':'torus',
        'reference_geometry_hash':reference.content_hash,
        'frozen_parameters_sha256':hashlib.sha256(json.dumps(parameters, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'authored_parameter_lattice':{'major_segments':n, 'minor_segments':m, 'triangles':len(faces),
                                    'diagonal':'00-to-11', 'periodic_seams_verified':True},
        'maximum_source_facet_distance_world':distance,
        'analytic_interpolation_distance_bound_world':_float_upper(interpolation),
        'maximum_vertex_construction_shift_world':shift,
        'construction_scope':'actual vertex displacement, no construction acceptance tolerance; public API requires exact original authored source identity',
        'maximum_normal_angle_degrees':normal_degrees,
        'face_to_cell_center_tangent_upper_bound':center_tangent,
        'cell_analytic_normal_radius_radians_upper_bound':cell_normal_radius,
        'normal_correspondence':'each actual facet to every analytic normal in its corresponding parameter cell; edge nearest-normal uniqueness not asserted',
        'distance_correspondence':'bidirectional continuous surface/facet cover via identical parameter barycentric coordinates',
        'coverage_proof':'exact once-only inventory of both outward parameter triangles in every frozen periodic cell',
        'distance_proof':'Hessian expectation bound with each parameter variance <= span squared/4, plus actual vertex construction shifts',
        'normal_proof':'atan(tangent)<=tangent at cell center; analytic normal parameter metric bounded by identity; spherical triangle inequality over half-cell diagonal',
        'arithmetic':'exact rational source crosses and bounds; Machin alternating-series pi enclosure; directed Decimal50 Taylor trig enclosures; float square-root/endpoints rationally checked',
        'artist_surface_limits':None, 'candidate_boundary_qualification':'not supplied',
    }


def torus_reference_certificate(reference):
    """Certify only the fixed authored source, or return an explicit reason."""
    try:
        if getattr(reference, 'content_hash', None) != FROZEN_SOURCE_HASH:
            raise ValueError('source changed from the frozen original authored torus identity')
        return _certificate(reference)
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return {'protocol':PROTOCOL, 'status':'unsupported', 'family':'torus',
                'reference_geometry_hash':getattr(reference, 'content_hash', None),
                'reason':type(exc).__name__+': '+str(exc), 'artist_surface_limits':None,
                'candidate_boundary_qualification':'not supplied'}
