"""Spatially grouped, balanced native CSG with retained editable inputs."""
from __future__ import annotations
import numpy as np
from .native_geometry import GeometryArrays, GeometryCache


def concatenate(parts):
    vertices, faces, offset = [], [], 0
    for p in parts:
        vertices.append(p.vertices); faces.append(p.faces+offset); offset += len(p.vertices)
    if not vertices:
        raise ValueError('empty assembly')
    return GeometryArrays.capture(np.concatenate(vertices), np.concatenate(faces))


def signed_volume(data):
    a, b, c = (data.vertices[data.faces[:, i]] for i in range(3))
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum()/6.)


def solid_guard(data, *, minimum_volume=1e-12):
    report = GeometryCache().topology_report(data)
    a, b, c = (data.vertices[data.faces[:, i]] for i in range(3))
    area = np.linalg.norm(np.cross(b-a, c-a), axis=1)
    volume = signed_volume(data)
    directed = np.concatenate([data.faces[:, [0, 1]], data.faces[:, [1, 2]], data.faces[:, [2, 0]]])
    edges = np.sort(directed, axis=1)
    _, inv = np.unique(edges, axis=0, return_inverse=True)
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    orientation_errors = int(np.count_nonzero(np.bincount(inv, weights=signs)))
    valid = bool(report['watertight'] and np.isfinite(volume) and volume > minimum_volume
                 and np.all(area > 1e-14) and orientation_errors == 0)
    return {**report, 'signed_volume': volume, 'orientation_errors': orientation_errors,
            'geometric_degenerate_faces': int(np.count_nonzero(area <= 1e-14)), 'valid_solid': valid,
             'topology_and_volume_valid': valid, 'single_solid_qualified': False,
             'valid_solid_scope': 'legacy topology/volume screen; boundary intersections untested',
            'self_intersection_qualification': 'not_established_by_index_topology'}


def oriented_generated_mesh(mesh):
    """Orient closed generated primitive polygons without changing coordinates.

    Fan triangulation is limited to the convex primitive faces produced here.
    This is an experimental-boundary repair, not a change to legacy exporters.
    It does not qualify geometric self intersections.
    """
    from .differentiable.mesh_projection import _triangulated_mesh_faces
    faces = _triangulated_mesh_faces(mesh.faces).astype(np.int64)
    vertices = np.asarray(mesh.vertices, float)
    adjacency = [[] for _ in faces]
    edges = {}
    for index, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append((index, 1 if a < b else -1))
    for incidences in edges.values():
        if len(incidences) != 2:
            raise ValueError('generated mesh orientation requires closed manifold edges')
        (a, sa), (b, sb) = incidences
        adjacency[a].append((b, -sa*sb))
        adjacency[b].append((a, -sa*sb))
    signs = np.zeros(len(faces), int)
    components = []
    for index in range(len(faces)):
        if signs[index]:
            continue
        signs[index] = 1
        stack, component = [index], []
        while stack:
            current = stack.pop()
            component.append(current)
            for other, relative in adjacency[current]:
                expected = signs[current]*relative
                if signs[other] and signs[other] != expected:
                    raise ValueError('generated mesh has inconsistent orientability')
                if not signs[other]:
                    signs[other] = expected
                    stack.append(other)
        components.append(component)
    faces[signs < 0] = faces[signs < 0][:, [0, 2, 1]]
    for component in components:
        ids = np.asarray(component)
        a, b, c = (vertices[faces[ids, axis]] for axis in range(3))
        if np.einsum('ij,ij->i', a, np.cross(b, c)).sum() < 0:
            faces[ids] = faces[ids][:, [0, 2, 1]]
    return GeometryArrays.capture(vertices, faces)


def overlap_groups(parts):
    boxes = [(p.vertices.min(0), p.vertices.max(0)) for p in parts]
    parent = list(range(len(parts)))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i, (lo, hi) in enumerate(boxes):
        for j in range(i):
            other_lo, other_hi = boxes[j]
            # Contact is a Boolean candidate, never permission to bridge a gap.
            # Bound only binary64 roundoff at the actual coordinate scale.
            scale = max(float(np.max(np.abs(np.concatenate((lo, hi, other_lo, other_hi))))),
                        float(np.max(hi - lo)), float(np.max(other_hi - other_lo)))
            tolerance = 8.0 * np.spacing(scale)
            if np.all(np.minimum(hi, other_hi) + tolerance >= np.maximum(lo, other_lo)):
                parent[root(i)] = root(j)
    groups = {}
    for i, part in enumerate(parts):
        groups.setdefault(root(i), []).append(part)
    return list(groups.values())


def execute_union_pair(payload):
    from .native_csg import boolean_mesh, sdf_grid_mesh
    left, right, options = payload
    solver = options.get('solver', 'EXACT')
    qualification = options.get('qualification')
    qualifications = []
    reason = None
    if solver == 'MANIFOLD':
        python = options.get('qualification_python')
        try:
            if not python:
                raise ValueError('no existing qualification interpreter configured')
            from .native_qualification import qualify_geometry, toolchain_identity
            identity = toolchain_identity(python)
            qualification = {'_toolchain_identity': identity, '_qualification_python': python}
            for operand in (left, right):
                receipt = qualify_geometry(operand, python=python, timeout_s=options.get('qualification_timeout_s', 15.))
                qualifications.append(receipt)
                qualification[operand.content_hash] = receipt
            if not all(r.get('manifold_validated') and r.get('self_intersections') == 0 for r in qualifications):
                raise ValueError('actual operand qualification did not pass')
        except Exception as exc:
            solver = 'EXACT'
            reason = str(exc)
    proposal = None
    report = {'algorithm': 'native_boolean', 'operation': 'UNION', 'solver': solver}
    try:
        proposal, report = boolean_mesh(left, right, operation='UNION', solver=solver, qualification=qualification)
    except Exception as exc:
        reason = str(exc)
    if solver == 'MANIFOLD' and (proposal is None or not solid_guard(proposal)['valid_solid']):
        solver = 'EXACT'
        try:
            proposal, report = boolean_mesh(left, right, operation='UNION', solver=solver)
        except Exception as exc:
            proposal = None
            reason = 'Exact execution failed: '+str(exc)
    report.update(requested_solver=options.get('solver', 'EXACT'), qualification=qualifications,
                  exact_fallback_reason=reason, sdf_requested=bool(options.get('sdf_fallback')))
    if not options.get('feature_thickness'):
        report['sdf_admission'] = 'unavailable: no measured input feature-thickness lower bound'

    valid = solid_guard(proposal) if proposal is not None else {'valid_solid': False, 'signed_volume': 0.}
    lower = max(abs(signed_volume(left)), abs(signed_volume(right)))
    upper = abs(signed_volume(left))+abs(signed_volume(right))
    volume_ok = lower*.98 <= valid['signed_volume'] <= upper*1.02
    if not (valid['valid_solid'] and volume_ok):
        # SDF requires at least four cells across the measured thinnest feature.
        thickness = options.get('feature_thickness')
        span = float(np.ptp(np.vstack([left.vertices, right.vertices]), axis=0).max())
        resolution = next((r for r in (128, 256) if thickness and span/r <= float(thickness)/4.), None)
        if options.get('sdf_fallback') and resolution:
            proposal, sdf_report = sdf_grid_mesh(left, right, operation='UNION', resolution=resolution)
            report.update(sdf_report, sdf_admission='four-cell measured-feature coverage',
                          feature_thickness=float(thickness))
            valid = solid_guard(proposal)
            volume_ok = lower*.97 <= valid['signed_volume'] <= upper*1.03
        if not (valid['valid_solid'] and volume_ok):
            raise ValueError('native union rejected by topology/orientation/volume guard')
    return proposal, {**report, 'solid_guard': valid, 'volume_guard': volume_ok}


def balanced_union(parts, *, executor=None, timeout_s=None, **options):
    groups = overlap_groups(list(parts))
    reports = []
    outputs = []
    for group in groups:
        level = group
        while len(level) > 1:
            payloads = [(level[i], level[i+1], options) for i in range(0, len(level)-1, 2)]
            if executor is None:
                results = [execute_union_pair(p) for p in payloads]
            else:
                outcomes = executor.map([('native_union', p, timeout_s) for p in payloads], timeout_s=timeout_s)
                results = []
                for outcome in outcomes:
                    if outcome.status != 'success':
                        raise RuntimeError(outcome.error)
                    results.append(outcome.value)
            reports.extend(r for _, r in results)
            level = [p for p, _ in results]+(level[-1:] if len(level)%2 else [])
        outputs.extend(level)
    return concatenate(outputs), {'overlap_groups': len(groups), 'balanced_unions': reports,
        'editable_input_hashes': [p.content_hash for p in parts], 'disconnected_groups_preserved': True}


def production_union(parts, config, *, executor=None, timeout_s=None, feature_thickness=None):
    """Consume the workflow's explicit native options in real assembly callers."""
    from .process_executor import current_worker_client
    queue = executor or current_worker_client()
    enabled = bool(config.get('native_union_execution', False))
    if enabled and queue is None:
        from .process_executor import PersistentProcessExecutor
        with PersistentProcessExecutor(2) as owned:
            return production_union(parts, config, executor=owned, timeout_s=timeout_s,
                                    feature_thickness=feature_thickness)
    result, report = balanced_union(parts, executor=queue if enabled else None, timeout_s=timeout_s,
        solver=config.get('native_union_solver', 'EXACT'),
        qualification_python=config.get('native_qualification_python'),
        qualification_timeout_s=min(float(config.get('native_qualification_timeout_s', 15.)), timeout_s or 15.),
        sdf_fallback=bool(config.get('native_sdf_fallback', False)),
        feature_thickness=feature_thickness or config.get('native_feature_thickness'))
    report['native_queue_used'] = bool(enabled and queue is not None)
    report['requested_options'] = {key: config.get(key) for key in ('native_union_execution',
        'native_union_solver', 'native_sdf_fallback', 'native_qualification_python')}
    return result, report
