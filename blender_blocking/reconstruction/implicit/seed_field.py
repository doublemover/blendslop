"""Bounded signed-distance samples of a closed, screened source mesh."""
import time

import numpy as np


def triangle_distance_and_winding(points, vertices, faces):
    """Euclidean triangle/edge distance and oriented solid-angle winding.

    This numerical field sampler is not an exact contact or native-solid check.
    Source qualification must remain separate from floating-point field values.
    """
    points = np.asarray(points, float)
    vertices = np.asarray(vertices, float)
    raw_faces = np.asarray(faces)
    if (raw_faces.dtype.kind not in 'iuf' or not np.isfinite(raw_faces).all()
            or np.any(raw_faces != np.floor(raw_faces))):
        raise ValueError('seed triangles must contain finite integer indices')
    faces = raw_faces.astype(np.int64)
    if (points.ndim != 2 or points.shape[1] != 3 or vertices.ndim != 2
            or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
            or not np.isfinite(points).all() or not np.isfinite(vertices).all()
            or not len(faces) or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError('seed distance requires finite points and valid triangles')
    triangles = vertices[faces]
    a, b, c = (triangles[:, index] for index in range(3))
    ab, ac = b-a, c-a
    normals = np.cross(ab, ac)
    normal_squared = np.sum(normals*normals, axis=1)
    if np.any(normal_squared <= 0):
        raise ValueError('seed distance cannot use collapsed triangles')
    pa = points[:, None, :]-a
    plane_numerator = np.einsum('pfi,fi->pf', pa, normals)
    projected = pa-plane_numerator[:, :, None]/normal_squared[None, :, None]*normals
    d00, d01, d11 = np.sum(ab*ab, axis=1), np.sum(ab*ac, axis=1), np.sum(ac*ac, axis=1)
    d20 = np.einsum('pfi,fi->pf', projected, ab)
    d21 = np.einsum('pfi,fi->pf', projected, ac)
    denominator = d00*d11-d01*d01
    if np.any(denominator <= 0):
        raise ValueError('seed triangles have numerically unresolved barycentric coordinates')
    v = (d11*d20-d01*d21)/denominator
    w = (d00*d21-d01*d20)/denominator
    inside = (v >= 0) & (w >= 0) & (v+w <= 1)
    squared = np.where(inside, plane_numerator**2/normal_squared, np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end-start
        length = np.sum(edge*edge, axis=1)
        if np.any(length <= 0):
            raise ValueError('seed contains collapsed edges')
        delta = points[:, None, :]-start
        fraction = np.clip(np.einsum('pfi,fi->pf', delta, edge)/length, 0, 1)
        residual = delta-fraction[:, :, None]*edge
        squared = np.minimum(squared, np.einsum('pfi,pfi->pf', residual, residual))
    va, vb, vc = (corner[None, :, :]-points[:, None, :] for corner in (a, b, c))
    la, lb, lc = (np.linalg.norm(value, axis=2) for value in (va, vb, vc))
    determinant = np.einsum('pfi,pfi->pf', va, np.cross(vb, vc))
    angle_denominator = (la*lb*lc + np.einsum('pfi,pfi->pf', va, vb)*lc
                         + np.einsum('pfi,pfi->pf', vb, vc)*la
                         + np.einsum('pfi,pfi->pf', vc, va)*lb)
    winding = (2*np.arctan2(determinant, angle_denominator)).sum(axis=1)/(4*np.pi)
    return np.sqrt(squared.min(axis=1)), winding


def signed_seed_field(data, bounds_min, bounds_max, *, resolution=16, timeout_s=8.):
    """Sample cell centers of bounded, strictly separated indexed solids.

    Scope: <=512 triangles, <=8 strictly AABB-separated parts, 16/32 grids.
    Touching, overlap, nested cavities and larger grids remain unsupported.
    """
    from blender_blocking.reconstruction.grouped_solids import solid_guard
    from blender_blocking.evaluation.triangle_contacts import within_part_boundary_guard, face_components
    from ..native_geometry import GeometryArrays

    started = time.perf_counter()
    lo, hi = np.asarray(bounds_min, float), np.asarray(bounds_max, float)
    if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite(np.r_[lo, hi]).all()
            or np.any(hi <= lo) or resolution not in (16, 32)
            or not np.isfinite(timeout_s) or timeout_s <= 0 or len(data.faces) > 512):
        raise ValueError('invalid bounded seed-field domain or triangle allowance')
    guard = solid_guard(data)
    if not guard['valid_solid'] or guard.get('loose_vertices', 0):
        raise ValueError('implicit seed requires closed, oriented, noncollapsed indexed parts')
    labels = np.asarray(face_components(len(data.vertices), data.faces))
    components = []
    for label in sorted(set(labels)):
        faces = data.faces[labels == label]
        indices = np.unique(faces)
        part = GeometryArrays.capture(data.vertices[indices], np.searchsorted(indices, faces))
        screen = solid_guard(part)
        if not screen['valid_solid'] or screen['connected_components'] != 1:
            raise ValueError('each implicit source part requires independent positive closed orientation/volume')
        components.append((part.vertices.min(axis=0), part.vertices.max(axis=0), screen))
    if not 1 <= len(components) <= 8 or len(components) != guard['connected_components']:
        raise ValueError('implicit seed component allowance or indexed component identity failed')
    separation = []
    for a, (alo, ahi, _) in enumerate(components):
        for b in range(a+1, len(components)):
            blo, bhi, _ = components[b]
            axes = np.flatnonzero((ahi < blo) | (bhi < alo))
            if not len(axes):
                raise ValueError('implicit multipart seed requires strictly separated AABBs; touching, overlap and cavities unsupported')
            separation.append({'parts': [a, b], 'strictly_separated_axes': axes.tolist()})
    guard = {**guard, 'component_screens': [row[2] for row in components],
             'component_separation': separation,
             'component_scope': 'strictly AABB-separated closed positive components; no touching unions or nested cavities',
             'between_part_disjointness': 'strict endpoint comparisons of immutable binary64 vertex AABBs; no tolerance or inflation'}
    last_progress = started
    def progress(count, total):
        nonlocal last_progress
        now = time.perf_counter()
        if count == total or now-last_progress >= 2:
            print(f'implicit seed boundary faces={count}/{total} elapsed={now-started:.2f}s', flush=True)
            last_progress = now
    boundary = within_part_boundary_guard(data.vertices, data.faces, timeout_s=timeout_s,
                                          progress=progress)
    if not boundary['passed']:
        raise ValueError('implicit seed boundary screen unavailable or rejected: '+str(boundary))
    spacing = (hi-lo)/resolution
    if (np.any(data.vertices.min(axis=0) <= lo+spacing/2)
            or np.any(data.vertices.max(axis=0) >= hi-spacing/2)):
        raise ValueError('implicit seed must be strictly contained inside the fixed sampled domain')
    axes = [lo[axis]+(np.arange(resolution)+.5)*spacing[axis] for axis in range(3)]
    if any(np.any(np.diff(axis) <= 0) for axis in axes):
        raise ValueError('world coordinate precision cannot resolve the seed-field cells')
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    field = np.empty(len(points), np.float32)
    maximum_winding_error = 0.
    for offset in range(0, len(points), 128):
        if time.perf_counter()-started >= timeout_s:
            raise TimeoutError('seed-field construction allowance exhausted')
        distance, winding = triangle_distance_and_winding(points[offset:offset+128], data.vertices, data.faces)
        magnitude = np.abs(winding)
        away = distance > np.finfo(float).eps*max(1., float(np.abs(points).max()))*128
        error = np.abs(magnitude-np.rint(magnitude))
        maximum_winding_error = max(maximum_winding_error, float(error[away].max(initial=0)))
        if np.any((error > 1e-6) & away) or np.any((magnitude > 1+1e-6) & away):
            raise ValueError('source winding is unresolved or overlapping in the sampled domain')
        signed = np.where(magnitude > .5, -distance, distance)
        if not np.isfinite(signed).all():
            raise ValueError('source field contains nonfinite distances')
        field[offset:offset+len(signed)] = signed
        now = time.perf_counter()
        if offset == 0 or now-last_progress >= 2 or offset+128 >= len(points):
            print(f'implicit seed cells={min(offset+128,len(points))}/{len(points)} '
                  f'elapsed={now-started:.2f}s', flush=True)
            last_progress = now
    return {'field_xyz': field.reshape((resolution,)*3), 'origin_xyz': lo+spacing/2,
            'voxel_size_xyz': spacing, 'bounds_min': lo, 'bounds_max': hi,
            'source_geometry_hash': data.content_hash,
            'seed_screen': guard, 'exact_within_part_boundary': boundary,
            'maximum_sampled_winding_error': maximum_winding_error,
            'sign_convention': 'negative material; positive exterior; zero boundary',
            'field_scope': 'numerical cell-center samples; no native single-solid qualification',
            'elapsed_s': time.perf_counter()-started}
