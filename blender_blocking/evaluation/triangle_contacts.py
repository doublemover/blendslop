"""Exact predicates for decoded finite binary64 triangle coordinates.

Separating axes plus exact plane-cut intervals; no tolerance removes a contact.
Potential axes: https://www.geometrictools.com/GTE/Mathematics/IntrTriangle3Triangle3.h
Implementation uses Python integers/Fractions, independently of Open3D.
"""
from fractions import Fraction
import math


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def integer_vertices(vertices):
    """Scale all finite IEEE coordinates to a common exact power-of-two grid."""
    ratios = []
    exponent = 0
    for vertex in vertices:
        row = []
        for coordinate in vertex:
            value = float(coordinate)
            if not math.isfinite(value):
                raise ValueError('triangle predicates require finite coordinates')
            numerator, denominator = value.as_integer_ratio()
            shift = denominator.bit_length()-1
            exponent = max(exponent, shift)
            row.append((numerator, shift))
        if len(row) != 3:
            raise ValueError('triangle predicates require xyz vertices')
        ratios.append(row)
    return [tuple(value << (exponent-shift) for value, shift in row) for row in ratios]


def _interval(triangle, axis):
    values = [_dot(vertex, axis) for vertex in triangle]
    return min(values), max(values)


def _separated(a, b, axis):
    lo_a, hi_a = _interval(a, axis)
    lo_b, hi_b = _interval(b, axis)
    return hi_a < lo_b or hi_b < lo_a, hi_a == lo_b or hi_b == lo_a


def _plane_cut(triangle, distances, axis):
    values = [Fraction(vertex[axis]) for vertex, distance in zip(triangle, distances) if distance == 0]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        if distances[i]*distances[j] < 0:
            values.append(Fraction(triangle[i][axis]*distances[j]-triangle[j][axis]*distances[i],
                                   distances[j]-distances[i]))
    return min(values), max(values)


def integer_triangle_relation(triangle_a, triangle_b):
    """Return disjoint/crossing/coplanar/contact/degenerate without rounding."""
    origin = triangle_a[0]
    a, b = [tuple(_sub(v, origin)) for v in triangle_a], [tuple(_sub(v, origin)) for v in triangle_b]
    edges_a = [_sub(a[(i+1)%3], a[i]) for i in range(3)]
    edges_b = [_sub(b[(i+1)%3], b[i]) for i in range(3)]
    normal_a, normal_b = _cross(edges_a[0], edges_a[1]), _cross(edges_b[0], edges_b[1])
    if not any(normal_a) or not any(normal_b):
        return {'relation': 'degenerate', 'predicate': 'exact_integer', 'certified': False}
    da = [_dot(normal_b, _sub(v, b[0])) for v in a]
    db = [_dot(normal_a, v) for v in b]
    for name, axis in (('normal_a', normal_a), ('normal_b', normal_b)):
        if _separated(a, b, axis)[0]:
            return {'relation': 'disjoint', 'predicate': 'exact_integer', 'separating_axis': name, 'certified': True}
    direction = _cross(normal_a, normal_b)
    if not any(direction):
        if set(a) == set(b):
            return {'relation': 'duplicate_triangle', 'predicate': 'exact_integer', 'certified': True}
        contact_axes = []
        for normal, edges in ((normal_a, edges_a), (normal_b, edges_b)):
            for edge in edges:
                separate, touches = _separated(a, b, _cross(normal, edge))
                if separate:
                    return {'relation': 'disjoint', 'predicate': 'exact_integer', 'separating_axis': 'coplanar_edge', 'certified': True}
                if touches:
                    contact_axes.append(_cross(normal, edge))
        dimension = (0 if any(any(_cross(a, b)) for a in contact_axes for b in contact_axes) else 1) if contact_axes else 2
        return {'relation': 'boundary_contact' if contact_axes else 'coplanar_area_overlap',
                'predicate': 'exact_integer', 'certified': True, 'intersection_dimension': dimension}
    for edge_a in edges_a:
        for edge_b in edges_b:
            if _separated(a, b, _cross(edge_a, edge_b))[0]:
                return {'relation': 'disjoint', 'predicate': 'exact_integer', 'separating_axis': 'edge_cross', 'certified': True}
    # Independent exact interval construction also distinguishes contact from crossing.
    axis = max(range(3), key=lambda i: abs(direction[i]))
    ia, ib = _plane_cut(a, da, axis), _plane_cut(b, db, axis)
    low, high = max(ia[0], ib[0]), min(ia[1], ib[1])
    if low > high:
        raise ArithmeticError('exact SAT and plane-cut interval predicates disagree')
    interior_a, interior_b = min(da) < 0 < max(da), min(db) < 0 < max(db)
    relation = 'proper_crossing' if low < high and interior_a and interior_b else 'boundary_contact'
    return {'relation': relation, 'predicate': 'exact_integer_and_rational_plane_cut', 'certified': True,
            'intersection_dimension': 1 if low < high else 0}


def triangle_relation(triangle_a, triangle_b):
    vertices = integer_vertices([*triangle_a, *triangle_b])
    return integer_triangle_relation(vertices[:3], vertices[3:])


def _indexed_adjacency_contact(a, b, shared):
    """Certify ordinary adjacency without evaluating every separating axis."""
    if len(shared) == 2:
        left, right = list(shared)
        edge = _sub(a[left], a[right])
        normal_a = _cross(edge, _sub(a[next(i for i in a if i not in shared)], a[right]))
        normal_b = _cross(edge, _sub(b[next(i for i in b if i not in shared)], b[right]))
        if not any(normal_a) or not any(normal_b):
            return False
        # Distinct planes meet only along the shared edge. Coplanar triangles
        # on opposite sides also meet only on that edge.
        return bool(any(_cross(normal_a, normal_b)) or _dot(normal_a, normal_b) < 0)
    if len(shared) == 1:
        common = next(iter(shared))
        aa = [_sub(v, a[common]) for i,v in a.items() if i != common]
        bb = [_sub(v, b[common]) for i,v in b.items() if i != common]
        normal_a, normal_b = _cross(*aa), _cross(*bb)
        if not any(normal_a) or not any(normal_b):
            return False
        da, db = [_dot(normal_b,v) for v in aa], [_dot(normal_a,v) for v in bb]
        # One triangle's cut against the other plane is only the shared vertex.
        return da[0]*da[1] > 0 or db[0]*db[1] > 0
    return False


def face_components(vertex_count, faces):
    parents = list(range(vertex_count))
    def component(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for face in faces:
        for index in face[1:]:
            parents[component(int(index))] = component(int(face[0]))
    return [component(int(face[0])) for face in faces]


def verify_reported_pairs(vertices, faces, pairs, *, max_pairs=1000, progress=None):
    """Verify native candidate pairs; an unverified remainder cannot qualify."""
    from collections import Counter
    import numpy as np
    vertices, faces, pairs = np.asarray(vertices, float), np.asarray(faces, int), np.asarray(pairs, int).reshape(-1, 2)
    if max_pairs is not None and max_pairs < 0:
        raise ValueError('max_pairs must be nonnegative or None')
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= len(faces)):
        raise ValueError('reported triangle index out of range')
    integers = integer_vertices(vertices)
    components = face_components(len(vertices), faces)
    limit = len(pairs) if max_pairs is None else min(max_pairs, len(pairs))
    counts, relations, samples = Counter(), Counter(), []
    for index, (a, b) in enumerate(pairs[:limit]):
        relation = integer_triangle_relation([integers[i] for i in faces[a]], [integers[i] for i in faces[b]])
        between = components[a] != components[b]
        kind = ('numerical_false_positive' if relation['relation'] == 'disjoint' else
                'duplicate_or_coincident_surface' if relation['relation'] in {'duplicate_triangle', 'coplanar_area_overlap'} else
                ('multipart_overlap' if between else 'true_self_intersection') if relation['relation'] == 'proper_crossing' else
                'boundary_contact' if relation['relation'] == 'boundary_contact' else 'indeterminate_degenerate')
        counts[kind] += 1
        relations[relation['relation']] += 1
        if index < 20 or sum(s['classification'] == kind for s in samples) < 3:
            samples.append({'faces': [int(a), int(b)], 'classification': kind,
                'left_component': components[a], 'right_component': components[b],
                'between_components': between, **relation})
        if progress is not None and (index+1) % 1000 == 0:
            progress(index+1, len(pairs))
    unresolved = len(pairs)-limit
    non_disjoint = limit-counts['numerical_false_positive']
    complete = unresolved == 0 and not counts['indeterminate_degenerate']
    return {'predicate_version': 'exact_binary64_triangle_contacts_v1', 'reported_pairs': len(pairs),
        'verified_pairs': limit, 'unverified_pairs': unresolved, 'complete': complete,
        'non_disjoint_pairs': non_disjoint, 'counts': dict(counts), 'relations': dict(relations),
        'samples': samples, 'component_count': len(set(components)),
        'qualification_policy': 'only exactly disjoint native reports are removed; contact/degenerate/unverified reports remain blocking'}


def within_part_boundary_guard(vertices, faces, *, timeout_s=None, progress=None):
    """Check all within-component AABB candidates; permit indexed adjacency.

    This guards deformation of individual parts, not the union of an assembly.
    Shared-index boundary contacts are expected adjacency; area overlap and
    proper crossing are rejected even when indices are shared.
    """
    import numpy as np
    import time
    started = time.monotonic()
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, int)
    labels = np.asarray(face_components(len(vertices), faces))
    exact = integer_vertices(vertices)
    # Reuse each face's exact coordinates and indexed vertex set. The same
    # immutable face participates in many pairs; rebuilding dictionaries and
    # sets per pair spends the helper allowance without adding evidence.
    face_indices = [tuple(int(index) for index in face) for face in faces]
    face_vertex_sets = [frozenset(face) for face in face_indices]
    exact_faces = [{index: exact[index] for index in face} for face in face_indices]
    exact_triangles = [tuple(exact[index] for index in face) for face in face_indices]
    triangles = vertices[faces]
    lower, upper = triangles.min(axis=1), triangles.max(axis=1)
    # Sweep along the axis with the smallest average box extent relative to
    # the mesh span. Dense lofts have tiny height slabs but broad x/y boxes;
    # choosing x unconditionally made broad-phase filtering nearly quadratic.
    # Every closed AABB overlap remains included; exact predicates are unchanged.
    span = upper.max(axis=0) - lower.min(axis=0)
    ratio = np.divide((upper - lower).mean(axis=0), span,
                      out=np.full(3, np.inf), where=span > 0.)
    sweep_axis = int(np.argmin(ratio))
    other_axes = [axis for axis in range(3) if axis != sweep_axis]
    order = np.argsort(lower[:, sweep_axis], kind='stable')
    min_axis = lower[order, sweep_axis]
    tested, allowed = 0, 0
    def result(status, passed, **extra):
        return {'status': status, 'passed': passed, 'tested_pairs': tested,
            'indexed_adjacency_contacts': allowed, 'component_count': len(set(labels)),
            'elapsed_s': time.monotonic()-started, 'geometry_scope': 'within indexed parts only; between-part overlap permitted',
            'predicate_version': 'exact_binary64_triangle_contacts_v1', 'sweep_axis': sweep_axis, **extra}
    for position, a in enumerate(order):
        if timeout_s is not None and time.monotonic()-started >= max(0., timeout_s):
            return result('unavailable', False, reason='remaining_candidate_allowance_exhausted')
        end = np.searchsorted(min_axis, upper[a, sweep_axis], side='right')
        candidates = order[position+1:end]
        candidates = candidates[(labels[candidates] == labels[a]) &
            np.all(lower[candidates][:, other_axes] <= upper[a, other_axes], axis=1) &
            np.all(upper[candidates][:, other_axes] >= lower[a, other_axes], axis=1)]
        for b in candidates:
            if timeout_s is not None and time.monotonic()-started >= max(0., timeout_s):
                return result('unavailable', False, reason='remaining_candidate_allowance_exhausted')
            tested += 1
            shared = face_vertex_sets[a] & face_vertex_sets[b]
            if shared and _indexed_adjacency_contact(exact_faces[a], exact_faces[b], shared):
                allowed += 1
                continue
            relation = integer_triangle_relation(exact_triangles[a], exact_triangles[b])
            kind = relation['relation']
            if kind == 'disjoint':
                continue
            if kind == 'boundary_contact' and (len(shared) >= 2 or (shared and relation.get('intersection_dimension') == 0)):
                allowed += 1
                continue
            return result('rejected', False, reason='within_part_boundary_defect',
                first_blocking_pair={'faces': [int(a), int(b)], **relation})
        if progress is not None:
            progress(position+1, len(order))
    return result('checked', True, reason=None,
        limitation='surface guard; closed topology/volume and external single-solid qualification are separate')
