"""Explicit, bounded same-evidence residual proposals and authoritative replay.

A signed-field surrogate never supplies opaque admission. Final native geometry
qualification is content scoped and remains unavailable outside its supported
Blender/native environment. Unqualified proposals preserve the actual seed.
"""
import hashlib
from pathlib import Path
import shutil

import numpy as np

from .field_model import NarrowBandField
from .seed_field import signed_seed_field


def prepare_implicit_job(target, seed, config):
    from ..differentiable.ray_evidence import prepare_ray_targets
    from ..evidence_identity import target_evidence_hash
    resolution = config.get('resolution', 16)
    if isinstance(resolution, bool) or resolution not in (16, 32):
        raise ValueError('implicit grid must be 16 or 32')
    views = [constraint.view for constraint in target.constraints]
    if not views or len(set(views)) != len(views) or set(views)-{'front', 'side', 'top'}:
        raise ValueError('implicit proposal requires unique canonical front/side/top evidence')
    # Fixed padding contains both source and target, with a full boundary cell
    # exterior to the seed. The original camera viewport remains authoritative.
    lo, hi = np.asarray(target.bounds.to_min_max(), float)
    center = (lo+hi)*.5
    scale = max(float(np.max(hi-lo)/1.6), float(np.max(np.abs(seed.vertices-center))/.8))
    sampled = signed_seed_field(seed, center-scale, center+scale,
                                resolution=resolution, timeout_s=float(config.get('seed_timeout_s', 8.)))
    rays = prepare_ray_targets(target, center, scale, resolution)
    empty = np.zeros((resolution,)*3, bool)
    for view, row in rays.items():
        # Only wholly observed, wholly reliable empty pixel-cell footprints
        # become hard constraints. Partial/unknown/soft foreground is evidence
        # for the objective, never a declaration of an empty 3D column.
        certain = ((row['valid'] >= 1.-1e-12) & (row['weights'] >= 1.-1e-12)
                   & (row['foreground'] == 0.))
        axis = {'front': 1, 'side': 2, 'top': 0}[view]
        empty |= np.broadcast_to(np.expand_dims(certain, axis), empty.shape)
    spacing = sampled['voxel_size_xyz']
    band = float(config.get('band_width_world', spacing.min()*2))
    displacement = float(config.get('maximum_displacement_world', spacing.min()*.9))
    job = {'execution_approved': config.get('execution_approved') is True,
           'seed_field_zyx': sampled['field_xyz'].transpose(2, 1, 0).copy(),
           'voxel_size_xyz': spacing, 'origin_xyz': sampled['origin_xyz'],
           'known_empty_zyx': empty, 'band_width': band, 'maximum_displacement': displacement,
           'ray_targets': rays, 'steps': config.get('steps', 2),
           'timeout_s': float(config.get('timeout_s', 8.)),
           'input_evidence_hash': target_evidence_hash(target)}
    job['retain_scored_checkpoints'] = config.get('checkpoint_selection', True)
    for key in ('learning_rate', 'temperature_world', 'eikonal_weight', 'displacement_weight', 'known_empty_feature_weight'):
        if key in config:
            job[key] = config[key]
    model = NarrowBandField.create(job['seed_field_zyx'], spacing, origin_xyz=job['origin_xyz'],
                                  band_width=band, maximum_displacement=displacement,
                                  known_empty_zyx=empty)
    job['objective'] = config.get('objective', 'minimum_distance_rays')
    if job['objective'] == 'original_pixel_extracted_mesh':
        job['original_pixel_targets'] = prepare_original_pixel_targets(target, model)
    elif job['objective'] != 'minimum_distance_rays':
        raise ValueError('unsupported implicit objective')
    report = {'source_geometry_hash': seed.content_hash, 'input_evidence_hash': job['input_evidence_hash'],
              'source_screen': sampled['seed_screen'],
              'source_component_scope': sampled['seed_screen']['component_scope'],
              'exact_source_boundary': sampled['exact_within_part_boundary'],
              'fixed_transform': model.report(), 'confirmed_empty_cells': int(empty.sum()),
              'empty_constraint_rule': 'complete observed reliable pixel footprint with zero foreground',
              'camera_contract': 'original fixed camera pixel cells; no viewport rebounding',
              'ray_operator': 'minimum cell-center signed distance sigmoid; proposal only',
              'native_qualification': False}
    return job, report


def prepare_original_pixel_targets(target, model):
    from ..pixel_evidence import observed_pixel_evidence
    from ..projection_contract import pixel_cell_viewport
    from ..feature_evidence import known_hole_regions
    lo = model.origin_xyz-model.voxel_size_xyz*.5
    hi = lo+np.asarray(model.seed_zyx.shape[::-1])*model.voxel_size_xyz
    targets = {}
    for constraint in target.constraints:
        (a, b), (u0, u1, v0, v1) = pixel_cell_viewport(target, constraint)
        evidence = observed_pixel_evidence(constraint)
        height, width = evidence.foreground.shape
        du, dv = (u1-u0)/width, (v1-v0)/height
        left = max(0, int(np.ceil((u0-lo[a])/du)))+1
        right = max(0, int(np.ceil((hi[a]-u1)/du)))+1
        bottom = max(0, int(np.ceil((v0-lo[b])/dv)))+1
        top = max(0, int(np.ceil((hi[b]-v1)/dv)))+1
        n = max(width+left+right, height+top+bottom, 8)
        if n > 192:
            raise ValueError('original camera support exceeds bounded implicit pixel allowance')
        right += n-width-left-right
        bottom += n-height-top-bottom
        pad = ((top, bottom), (left, right))
        regions = known_hole_regions(constraint)
        feature = np.logical_or.reduce(regions) if regions else np.zeros(evidence.valid.shape, bool)
        # DVX rows increase in world y; image rows increase down the screen.
        targets[constraint.view] = {
            'foreground': np.pad(evidence.foreground, pad)[::-1].copy(),
            'valid': np.pad(evidence.valid.astype(float), pad)[::-1].copy(),
            'weights': np.pad(evidence.weights, pad)[::-1].copy(),
            'empty_feature_weights': np.pad(evidence.weights*feature, pad)[::-1].copy(),
            'empty_feature_regions': len(regions),
            'empty_feature_semantics': 'same observed enclosed-empty evidence reweighted; not an independent observation',
            'square_resolution': n, 'original_shape': [height, width],
            'padded_world_bounds': [u0-left*du, u1+right*du, v0-bottom*dv, v1+top*dv],
            'original_pixel_size_world': [du, dv],
            'padding_screen_tblr': [top, bottom, left, right],
            'padding_semantics': 'unknown zero weight; original pixel size and camera orientation unchanged'}
    return targets


def validate_fitted_field(job, fitted):
    """Verify helper identity and actual scored field before extraction."""
    model = NarrowBandField.create(job['seed_field_zyx'], job['voxel_size_xyz'],
                                  origin_xyz=job['origin_xyz'], band_width=job['band_width'],
                                  maximum_displacement=job['maximum_displacement'],
                                  known_empty_zyx=job['known_empty_zyx'])
    if (fitted.get('input_evidence_hash') != job['input_evidence_hash']
            or fitted.get('model', {}).get('constraint_hash') != model.report()['constraint_hash']):
        raise ValueError('implicit helper response changed fixed evidence or field constraints')
    if fitted.get('objective', 'minimum_distance_rays') != job.get('objective', 'minimum_distance_rays'):
        raise ValueError('implicit helper response changed the declared objective operator')
    field = np.asarray(fitted['field_zyx'])
    if field.dtype.kind != 'f' or field.shape != model.seed_zyx.shape or not np.isfinite(field).all():
        raise ValueError('implicit helper response has an invalid retained field')
    evaluation = fitted.get('best_evaluation')
    count = fitted.get('objective_evaluations')
    total = np.asarray(fitted.get('best_total', np.nan))
    history = np.asarray(fitted.get('history', []))
    if (isinstance(evaluation, bool) or not isinstance(evaluation, int)
            or isinstance(count, bool) or not isinstance(count, int) or not 1 <= evaluation <= count <= 17
            or total.shape != () or total.dtype.kind not in 'fi' or not np.isfinite(total)
            or history.ndim != 1 or history.dtype.kind not in 'fi' or not evaluation <= len(history) <= count
            or not np.isfinite(history).all() or total != history[evaluation-1]):
        raise ValueError('implicit helper field lacks a finite evaluated checkpoint')
    identity = hashlib.sha256(model.report()['constraint_hash'].encode()+field.tobytes()).hexdigest()
    if identity != fitted.get('retained_field_hash'):
        raise ValueError('implicit retained field identity mismatch')
    tolerance = np.finfo(field.dtype).eps*max(float(np.abs(model.seed_zyx).max()),
                                             model.maximum_displacement, model.minimum_empty_distance)*16
    decoded = model.decode(fitted['parameters'])
    if not np.allclose(decoded, field, atol=tolerance, rtol=0.):
        raise ValueError('implicit evaluated field does not match its finite residual coordinates')
    if (np.max(np.abs(field-model.seed_zyx)) > model.maximum_displacement+tolerance
            or np.any(field[model.protected_empty_zyx] <= 0)
            or np.any(field[model.protected_empty_zyx] < model.minimum_empty_distance-tolerance)):
        raise ValueError('implicit helper field violates world residual or protected-empty bounds')
    inactive = np.ones(field.size, bool)
    inactive[model.active_flat] = False
    # Feasibility conditioning can change protected cells, even outside the
    # active band. Exact comparison uses the corresponding zero-coordinate base.
    if not np.array_equal(field.ravel()[inactive], model.decode(np.zeros(len(model.active_flat))).astype(field.dtype).ravel()[inactive]):
        raise ValueError('implicit helper changed fixed inactive field values')
    return model, field


def extract_implicit_geometry(field, model):
    from blender_blocking.volume import Bounds3D, DenseVolumeGrid, extract_mesh
    from ..native_geometry import GeometryArrays
    lo = model.origin_xyz-model.voxel_size_xyz*.5
    hi = lo+np.asarray(field.shape[::-1])*model.voxel_size_xyz
    grid = DenseVolumeGrid(field.transpose(2, 1, 0), Bounds3D.from_min_max(lo, hi),
                           value_type='signed_distance', default_value=1.)
    mesh = extract_mesh(grid)
    return GeometryArrays.capture(mesh.vertices, mesh.faces), mesh.topology


def select_output_checkpoint(job, fitted, target, seed, *, projection_mode='legacy_pil', timeout_s=8.):
    """Rank only evaluated residual fields on the actual original-camera metric.

    This numerical ordering is not boundary/native admission. The original
    source remains available to the backend regardless of checkpoint ordering.
    A shared elapsed allowance bounds all extraction/projection work.
    """
    import time
    from ..projected_metrics import projected_mesh_metrics
    from ..pixel_projection import canonical_mesh_metrics
    from ..types import CandidateMetrics
    if projection_mode not in ('legacy_pil', 'pixel_area_half'):
        raise ValueError('unsupported output checkpoint metric')
    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('output checkpoint selection requires a finite positive allowance')
    metric = projected_mesh_metrics if projection_mode == 'legacy_pil' else canonical_mesh_metrics
    model, _ = validate_fitted_field(job, fitted)
    checkpoints = fitted.get('scored_checkpoints', [])
    if not isinstance(checkpoints, (tuple, list)) or len(checkpoints) > 17:
        raise ValueError('implicit scored checkpoint allowance or container failed')
    started = time.perf_counter()
    baseline = CandidateMetrics(per_view=metric(target, seed.vertices, seed.faces))
    from ..projected_metrics import projected_mesh_masks
    from ..pixel_projection import canonical_mesh_projections
    from ..pixel_evidence import observed_pixel_evidence
    from ..feature_evidence import known_empty_feature_guard
    from ..grouped_solids import solid_guard
    masks = (lambda data: projected_mesh_masks(target, data.vertices, data.faces)) if projection_mode == 'legacy_pil' else (
        lambda data: {view: values >= .5 for view, values in canonical_mesh_projections(target, data.vertices, data.faces).items()})
    source_masks = masks(seed)
    report = {'metric_identity': ('legacy_pil_polygon_endpoint_rounding_v1' if projection_mode == 'legacy_pil'
                                  else 'opaque_triangle_union_pixel_cell_area_hard_half_v1'),
              'source_always_available': True, 'source_per_view': baseline.per_view,
              'surrogate_best_evaluation': fitted['best_evaluation'], 'surrogate_best_total': fitted['best_total'],
              'checkpoints': [], 'stop_reason': 'all_evaluated_checkpoints',
              'admission_scope': 'ordering only; exact boundary, known-empty and native guards remain mandatory'}
    if not checkpoints:
        proposed, topology = extract_implicit_geometry(fitted['field_zyx'], model)
        report.update(selected_evaluation=fitted['best_evaluation'], selection_scope='single helper-scored field; checkpoint pool unavailable')
        return fitted, proposed, topology, report
    seen = set(); best = None
    checkpoint_keys = {'field_zyx', 'parameters', 'best_evaluation', 'best_total', 'retained_field_hash'}
    history = fitted.get('history', [])
    for row in checkpoints:
        if not isinstance(row, dict) or set(row) != checkpoint_keys:
            raise ValueError('implicit scored checkpoint must contain only the evaluated field record')
        evaluation = row.get('best_evaluation')
        if (isinstance(evaluation, bool) or not isinstance(evaluation, int)
                or not 1 <= evaluation <= min(17, fitted.get('objective_evaluations', 0)) or evaluation in seen):
            raise ValueError('implicit checkpoint evaluation identity failed')
        seen.add(evaluation)
        if len(history) < evaluation or row['best_total'] != history[evaluation-1]:
            raise ValueError('implicit checkpoint loss does not match its evaluated history')
        current = {**fitted, **row}
        _, field = validate_fitted_field(job, current)
        if time.perf_counter()-started >= timeout_s:
            report['stop_reason'] = 'shared_extraction_projection_allowance'
            break
        try:
            proposed, topology = extract_implicit_geometry(field, model)
            metrics = CandidateMetrics(per_view=metric(target, proposed.vertices, proposed.faces))
        except (ValueError, RuntimeError) as exc:
            report['checkpoints'].append({'evaluation': evaluation, 'status': 'extraction_or_projection_unavailable', 'reason': str(exc)})
            continue
        proposal_masks = masks(proposed)
        empty_passed = True; added_empty = {}
        for constraint in target.constraints:
            evidence = observed_pixel_evidence(constraint)
            certain = evidence.valid & (evidence.weights == 1.) & (evidence.foreground == 0.) & ~evidence.hard_mask
            added_empty[constraint.view] = int(np.count_nonzero(certain & proposal_masks[constraint.view] & ~source_masks[constraint.view]))
            empty_passed &= known_empty_feature_guard(constraint, proposal_masks[constraint.view])['passed']
        eligible = bool(solid_guard(proposed)['valid_solid'] and empty_passed and not any(added_empty.values())
            and all(row.get('passed') for row in metrics.per_view.values() if row.get('required', True))
            and metrics.area_iou_min >= baseline.area_iou_min-.002
            and metrics.area_iou_mean > baseline.area_iou_mean+.002
            and metrics.boundary_iou_mean >= baseline.boundary_iou_mean-.002)
        boundary = {'status': 'not_required_for_ineligible_output', 'passed': False}
        boundary_allowance_exhausted = False
        if eligible:
            from blender_blocking.evaluation.triangle_contacts import within_part_boundary_guard
            remaining = timeout_s-(time.perf_counter()-started)
            boundary = within_part_boundary_guard(proposed.vertices, proposed.faces,
                                                 timeout_s=max(0., remaining))
            eligible = bool(boundary['passed'])
            boundary_allowance_exhausted = boundary['status'] == 'unavailable'
        key = (int(eligible), metrics.area_iou_min, metrics.area_iou_mean, metrics.boundary_iou_mean, -len(proposed.faces), -evaluation)
        if not np.isfinite(key).all():
            raise ValueError('implicit checkpoint output metric is nonfinite')
        report['checkpoints'].append({'evaluation': evaluation, 'surrogate_total': row['best_total'],
            'field_hash': row['retained_field_hash'],
            'geometry_hash': proposed.content_hash, 'per_view': metrics.per_view, 'rank_key': list(key),
            'preliminary_output_eligible': eligible, 'known_empty_features_passed': bool(empty_passed),
            'exact_boundary_guard': boundary,
            'added_known_empty_pixels': added_empty,
            'status': 'original_camera_opaque_scored'})
        if best is None or key > best[0]:
            best = (key, current, proposed, topology)
        if boundary_allowance_exhausted:
            report['stop_reason'] = 'shared_extraction_projection_boundary_allowance'
            break
    if best is None:
        raise ValueError('implicit checkpoint pool has no extracted original-camera scored field within allowance')
    _, current, proposed, topology = best
    report.update(selected_evaluation=current['best_evaluation'], selection_scope='bounded finite evaluated fields ranked on unchanged output metric')
    return current, proposed, topology, report


def write_implicit_artifacts(root, seed, proposed, retained, job, fitted, report, source_artist=None):
    """Keep source/artist, topology-changing field, and selected output distinct."""
    from ..artifacts import write_json, hash_file
    from ..mesh_io import write_obj
    root = Path(root)
    state = root/'h'/'implicit-state.npz'
    state.parent.mkdir(parents=True, exist_ok=True)
    selection = report.get('output_checkpoint_selection') or {}
    scored_ids = {row['evaluation'] for row in selection.get('checkpoints', [])
                  if row.get('status') == 'original_camera_opaque_scored'}
    scored = [row for row in fitted.get('scored_checkpoints', []) if row['best_evaluation'] in scored_ids]
    checkpoint_fields = (np.stack([row['field_zyx'] for row in scored]) if scored
                         else np.empty((0,)+fitted['field_zyx'].shape, dtype=fitted['field_zyx'].dtype))
    checkpoint_parameters = (np.stack([row['parameters'] for row in scored]) if scored
                             else np.empty((0, len(fitted['parameters'])), dtype=fitted['parameters'].dtype))
    np.savez_compressed(state, seed_field_zyx=job['seed_field_zyx'], field_zyx=fitted['field_zyx'],
                        parameters=fitted['parameters'], known_empty_zyx=job['known_empty_zyx'],
                        origin_xyz=job['origin_xyz'], voxel_size_xyz=job['voxel_size_xyz'],
                        source_vertices=seed.vertices, source_faces=seed.faces,
                        retained_vertices=retained.vertices, retained_faces=retained.faces,
                        opaque_scored_checkpoint_fields_zyx=checkpoint_fields,
                        opaque_scored_checkpoint_parameters=checkpoint_parameters,
                        opaque_scored_checkpoint_evaluations=np.asarray([row['best_evaluation'] for row in scored],int),
                        opaque_scored_checkpoint_surrogate_totals=np.asarray([row['best_total'] for row in scored],float))
    paths = {'field_state': state, 'seed_mesh_obj': write_obj(root/'seed'/'seed.obj', seed),
             'proposal_mesh_obj': write_obj(root/'proposal'/'implicit.obj', proposed),
             'mesh_obj': write_obj(root/'m'/'retained.obj', retained)}
    metadata = {**report, 'contract': 'bounded_implicit_residual_with_separate_artist_source_v1',
                'output_checkpoint_archive': 'finite_scored_field_and_residual_coordinates_v1',
                'field_state_sha256': hash_file(state),
                'retained_field_hash': fitted['retained_field_hash'],
                'source_geometry_hash': seed.content_hash, 'proposal_geometry_hash': proposed.content_hash,
                'retained_geometry_hash': retained.content_hash,
                'source_parameters_reproduce_final_field': False,
                'field_replay_authoritative': True}
    if source_artist and Path(source_artist).is_file():
        artist = root/'seed'/'artist-source.json'
        shutil.copyfile(source_artist, artist)
        paths['artist_source_json'] = artist
        metadata['artist_source_sha256'] = hash_file(artist)
    metadata['mesh_file_sha256'] = {name: hash_file(path) for name, path in paths.items() if name.endswith('_obj')}
    metadata['artifacts'] = {name: str(path.resolve()) for name, path in paths.items()}
    paths['implicit_contract'] = write_json(root/'implicit-contract.json', metadata)
    return paths, metadata


def replay_implicit_artifacts(metadata, *, include_output_checkpoints=False):
    from ..artifacts import hash_file
    from ..native_geometry import GeometryArrays
    from blender_blocking.evaluation.comparable_geometry import read_obj
    if metadata.get('contract') != 'bounded_implicit_residual_with_separate_artist_source_v1':
        raise ValueError('unsupported implicit artifact contract')
    paths = metadata['artifacts']
    if hash_file(paths['field_state']) != metadata['field_state_sha256']:
        raise ValueError('implicit field artifact bytes changed')
    with np.load(paths['field_state'], allow_pickle=False) as packet:
        transform = metadata['fixed_transform']
        model = NarrowBandField.create(packet['seed_field_zyx'], packet['voxel_size_xyz'],
                                      origin_xyz=packet['origin_xyz'], band_width=transform['band_width_world'],
                                      maximum_displacement=transform['maximum_displacement_world'],
                                      known_empty_zyx=packet['known_empty_zyx'],
                                      minimum_empty_distance=transform['minimum_empty_distance_world'])
        field = packet['field_zyx']
        identity = hashlib.sha256(model.report()['constraint_hash'].encode()+field.tobytes()).hexdigest()
        if (model.report()['constraint_hash'] != transform['constraint_hash']
                or identity != metadata['retained_field_hash']):
            raise ValueError('implicit field constraint or content identity changed')
        proposed, topology = extract_implicit_geometry(field, model)
        source = GeometryArrays.capture(packet['source_vertices'], packet['source_faces'])
        retained = GeometryArrays.capture(packet['retained_vertices'], packet['retained_faces'])
        if (source.content_hash != metadata['source_geometry_hash']
                or retained.content_hash != metadata['retained_geometry_hash']):
            raise ValueError('implicit source or retained authoritative geometry changed')
        checkpoints = _replay_scored_checkpoint_pool(packet, model, metadata) if include_output_checkpoints else []
    if proposed.content_hash != metadata['proposal_geometry_hash']:
        raise ValueError('implicit field no longer reproduces proposed mesh')
    for key, data in (('seed_mesh_obj', source), ('mesh_obj', retained), ('proposal_mesh_obj', proposed)):
        # OBJ serialization is rounded; its file digest is checked separately,
        # while authoritative field replay uses exact geometry hashes.
        vertices, faces = read_obj(Path(paths[key]))
        if (vertices.shape != data.vertices.shape or not np.array_equal(faces, data.faces)
                or np.max(np.abs(vertices-data.vertices)) > 1e-8*max(1., float(np.abs(data.vertices).max()))):
            raise ValueError('implicit saved OBJ disagrees with authoritative geometry: '+key)
        if metadata.get('mesh_file_sha256', {}).get(key) != hash_file(paths[key]):
            raise ValueError('implicit saved mesh artifact bytes changed: '+key)
    if 'artist_source_json' in paths and hash_file(paths['artist_source_json']) != metadata.get('artist_source_sha256'):
        raise ValueError('implicit artist-source artifact bytes changed')
    return {'proposal': proposed, 'source': source, 'retained': retained, 'topology': topology, 'native_qualification': False,
            'output_checkpoints': checkpoints,
            'retained_output_is_proposal': metadata['retained_geometry_hash'] == proposed.content_hash}


def _replay_scored_checkpoint_pool(packet, model, metadata):
    selection = metadata.get('output_checkpoint_selection') or {}
    rows = [row for row in selection.get('checkpoints', [])
            if row.get('status') == 'original_camera_opaque_scored']
    if len(rows) > 17:
        raise ValueError('implicit artifact checkpoint allowance failed')
    if not rows:
        return []
    if metadata.get('output_checkpoint_archive') != 'finite_scored_field_and_residual_coordinates_v1':
        raise ValueError('full checkpoint-coordinate replay is unavailable for this older artifact contract')
    fields = packet['opaque_scored_checkpoint_fields_zyx']
    parameters = packet['opaque_scored_checkpoint_parameters']
    evaluations = packet['opaque_scored_checkpoint_evaluations']
    totals = packet['opaque_scored_checkpoint_surrogate_totals']
    if (fields.shape != (len(rows),)+model.seed_zyx.shape
            or parameters.shape != (len(rows), len(model.active_flat))
            or evaluations.shape != (len(rows),) or evaluations.dtype.kind not in 'iu'
            or totals.shape != (len(rows),) or not np.isfinite(totals).all()):
        raise ValueError('implicit artifact checkpoint arrays failed')
    job = {'seed_field_zyx': packet['seed_field_zyx'], 'voxel_size_xyz': packet['voxel_size_xyz'],
           'origin_xyz': packet['origin_xyz'], 'known_empty_zyx': packet['known_empty_zyx'],
           'band_width': model.band_width, 'maximum_displacement': model.maximum_displacement,
           'input_evidence_hash': metadata['input_evidence_hash'],
           'objective': metadata['proposal_objective']}
    seen = set(); checkpoints = []
    for index, row in enumerate(rows):
        evaluation = row['evaluation']
        if (isinstance(evaluation, bool) or not isinstance(evaluation, int)
                or not 1 <= evaluation <= 17 or evaluation in seen or evaluations[index] != evaluation
                or totals[index] != row['surrogate_total']):
            raise ValueError('implicit artifact checkpoint evaluation identity changed')
        seen.add(evaluation)
        current = {'field_zyx': fields[index], 'parameters': parameters[index],
                   'model': model.report(), 'input_evidence_hash': job['input_evidence_hash'],
                   'objective': job['objective'], 'best_evaluation': evaluation,
                   'best_total': totals[index], 'retained_field_hash': row['field_hash'],
                   'objective_evaluations': metadata['optimization']['objective_evaluations'],
                   'history': metadata['optimization']['history']}
        _, scored_field = validate_fitted_field(job, current)
        geometry, _ = extract_implicit_geometry(scored_field, model)
        if geometry.content_hash != row['geometry_hash']:
            raise ValueError('implicit artifact checkpoint no longer reproduces scored mesh')
        checkpoints.append({'evaluation': evaluation, 'surrogate_total': float(totals[index]),
                            'field_hash': row['field_hash'], 'geometry': geometry})
    return checkpoints
