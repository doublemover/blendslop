"""Optional pinned-CPU signed-field optimization, separate from the DVX operator."""
import time
import hashlib

import numpy as np

from .field_model import NarrowBandField

DEPTH_AXIS_ZYX = {'front': 1, 'side': 2, 'top': 0}


def field_ray_predictions(field, temperature):
    """Voxel-center minimum-distance ray surrogate, not exact pixel coverage."""
    import torch
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('implicit ray temperature must be finite positive world distance')
    return {view: torch.sigmoid(-torch.amin(field, dim=axis)/temperature)
            for view, axis in DEPTH_AXIS_ZYX.items()}


def field_regularizers(field, model):
    import torch
    residual = field-torch.tensor(model.seed_zyx.copy(), dtype=field.dtype)
    indices = torch.tensor(model.active_flat.copy(), dtype=torch.int64)
    displacement = (residual.reshape(-1)[indices]/model.maximum_displacement).square().mean()
    spacing = model.voxel_size_xyz[::-1]
    gz = (field[2:, 1:-1, 1:-1]-field[:-2, 1:-1, 1:-1])/(2*spacing[0])
    gy = (field[1:-1, 2:, 1:-1]-field[1:-1, :-2, 1:-1])/(2*spacing[1])
    gx = (field[1:-1, 1:-1, 2:]-field[1:-1, 1:-1, :-2])/(2*spacing[2])
    norm = torch.sqrt(gz.square()+gy.square()+gx.square()+1e-12)
    band = torch.tensor((np.abs(model.seed_zyx[1:-1, 1:-1, 1:-1]) <= model.band_width).copy())
    eikonal = (norm[band]-1).square().mean() if bool(band.any()) else norm.sum()*0
    return displacement, eikonal


def fit_implicit_field_job(payload):
    """Numerical foundation only; no automatic output admission or native calls."""
    if not payload.get('execution_approved', False):
        raise PermissionError('implicit field numerical execution has not been approved')
    import torch
    if torch.__version__ != '2.14.1+cpu' or torch.version.cuda is not None or torch.version.hip is not None:
        raise RuntimeError('implicit field fixtures currently require pinned Torch 2.14.1+cpu')
    torch.set_num_threads(1)
    model = NarrowBandField.create(
        payload['seed_field_zyx'], payload['voxel_size_xyz'], origin_xyz=payload['origin_xyz'],
        band_width=float(payload['band_width']), maximum_displacement=float(payload['maximum_displacement']),
        known_empty_zyx=payload.get('known_empty_zyx'),
        minimum_empty_distance=payload.get('minimum_empty_distance'))
    steps = payload.get('steps', 2)
    timeout = float(payload.get('timeout_s', 5.))
    learning_rate = float(payload.get('learning_rate', .03))
    temperature = float(payload.get('temperature_world', model.voxel_size_xyz.min()*.25))
    displacement_weight = float(payload.get('displacement_weight', .02))
    eikonal_weight = float(payload.get('eikonal_weight', .0001))
    feature_weight = float(payload.get('known_empty_feature_weight', 0.))
    retain_checkpoints = payload.get('retain_scored_checkpoints', False)
    if not isinstance(retain_checkpoints, bool):
        raise ValueError('scored checkpoint retention must be an explicit Boolean')
    if (isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= 16
            or not np.isfinite([timeout, learning_rate, temperature, displacement_weight, eikonal_weight, feature_weight]).all()
            or timeout <= 0 or not 0 < learning_rate <= 1 or temperature <= 0
            or displacement_weight < 0 or eikonal_weight < 0 or feature_weight < 0):
        raise ValueError('invalid bounded implicit field optimization settings')
    objective = payload.get('objective', 'minimum_distance_rays')
    if objective not in ('minimum_distance_rays', 'original_pixel_extracted_mesh'):
        raise ValueError('unsupported implicit field objective')
    if feature_weight > 0 and objective != 'original_pixel_extracted_mesh':
        raise ValueError('known-empty feature objective requires actual original-pixel extracted support')
    projected_reports = {}
    feature_rays = {}
    feature_denominator = 0.
    rays = {}
    denominator = 0.
    n = model.seed_zyx.shape[0]
    target_records = payload.get('original_pixel_targets' if objective == 'original_pixel_extracted_mesh' else 'ray_targets', {})
    for view, row in dict(target_records).items():
        if view not in DEPTH_AXIS_ZYX:
            raise ValueError('implicit observed rays require canonical front/side/top cameras')
        shape = (int(row['square_resolution']),)*2 if objective == 'original_pixel_extracted_mesh' else (n, n)
        foreground = np.asarray(row['foreground'], float)
        valid = np.asarray(row.get('valid', np.ones(shape)), float)
        weights = np.asarray(row.get('weights', valid), float)
        if (foreground.shape != shape or valid.shape != shape or weights.shape != shape
                or not np.isfinite(valid).all() or not np.isfinite(weights).all()
                or np.any(valid < 0) or np.any(valid > 1) or np.any(weights < 0)
                or np.any(weights > valid+1e-8)
                or not np.isfinite(foreground[weights > 0]).all()
                or np.any(foreground[weights > 0] < 0) or np.any(foreground[weights > 0] > 1)):
            raise ValueError('implicit ray evidence requires finite observed fractions and reliability weights')
        foreground = np.where(weights > 0, foreground, 0.)
        rays[view] = (torch.tensor(foreground, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32))
        denominator += float(weights.sum())
        if objective == 'original_pixel_extracted_mesh':
            feature = np.asarray(row.get('empty_feature_weights', np.zeros(shape)), float)
            if (feature.shape != shape or not np.isfinite(feature).all()
                    or np.any(feature < 0) or np.any(feature > weights+1e-8)):
                raise ValueError('implicit empty-feature evidence weights must be a subset of observed reliability')
            feature_rays[view] = torch.tensor(feature, dtype=torch.float32)
            feature_denominator += float(feature.sum())
    if denominator <= 0:
        raise ValueError('implicit refinement has no positive observed-ray evidence')
    parameters = torch.zeros(len(model.active_flat), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([parameters], lr=learning_rate)
    started = time.perf_counter()
    history, term_history = [], []
    evaluations = updates = 0
    best_value = float('inf')
    best_field = None
    best_parameters = None
    best_evaluation = None
    best_projected_reports = {}
    operator_failure = None
    stop_reason = 'requested_steps'
    final_update_evaluated = False
    checkpoints = []

    def snapshot():
        field_hash = hashlib.sha256(model.report()['constraint_hash'].encode()+best_field.tobytes()).hexdigest()
        return {'field_zyx': best_field, 'parameters': best_parameters,
                'retained_field_hash': field_hash, 'input_evidence_hash': payload.get('input_evidence_hash'),
                'seed_field_zyx': model.seed_zyx.copy(), 'active_flat': model.active_flat.copy(),
                'protected_empty_zyx': model.protected_empty_zyx.copy(), 'model': model.report(),
                'history': history, 'term_history': term_history,
                'objective_evaluations': evaluations, 'optimizer_updates': updates,
                'best_evaluation': best_evaluation, 'best_total': best_value,
                'final_update_evaluated': final_update_evaluated, 'stop_reason': stop_reason,
                'optimization_wall_s': time.perf_counter()-started, 'device': 'cpu',
                'runtime': {'torch': str(torch.__version__), 'operator': ('Torch field-to-extracted-mesh pullback and DVX2D closed-form pixel support' if objective == 'original_pixel_extracted_mesh'
                            else 'Torch signed-field residual; DVX is not used')},
                'objective': objective, 'objective_weights': {'global_observed_coverage': 1., 'known_empty_feature': feature_weight,
                    'displacement': displacement_weight, 'eikonal': eikonal_weight},
                'known_empty_feature_weight_sum': feature_denominator,
                'feature_evidence_scope': 'same original observed enclosed-empty evidence; no additional observation or clearance certificate',
                'projected_reports': best_projected_reports, 'operator_failure': operator_failure,
                'ray_operator': ('extracted zero-set opaque triangle union with original-pixel box support' if objective == 'original_pixel_extracted_mesh'
                                 else 'voxel-center minimum signed distance and sigmoid; not exact pixel-area mesh projection'),
                'native_qualification': False, 'automatic_admission': False,
                'scored_checkpoints': checkpoints,
                'checkpoint_scope': 'at most 17 finite evaluated fields; optimizer proposals without an evaluation are excluded',
                'inference_scope': 'evidence-conditioned hypothesis; unobserved cavities are not recovered truth'}

    def evaluate():
        nonlocal evaluations, best_value, best_field, best_parameters, best_evaluation, projected_reports, best_projected_reports
        field = model.torch_decode(parameters)
        if objective == 'original_pixel_extracted_mesh':
            from .mesh_objective import original_pixel_mesh_predictions
            predictions, projected_reports = original_pixel_mesh_predictions(field, model, target_records)
        else:
            predictions = field_ray_predictions(field, temperature)
        numerator = sum(((predictions[view]-target).square()*weights).sum()
                        for view, (target, weights) in rays.items())
        evidence = numerator/denominator
        feature_loss = (sum(((predictions[view]-target).square()*feature_rays[view]).sum()
                            for view, (target, _) in rays.items())/feature_denominator
                        if feature_denominator > 0 else evidence*0.)
        displacement, eikonal = field_regularizers(field, model)
        loss = evidence+feature_weight*feature_loss+displacement_weight*displacement+eikonal_weight*eikonal
        evaluations += 1
        if bool(torch.isfinite(loss)):
            value = float(loss.detach())
            history.append(value)
            term_history.append({'evidence': float(evidence.detach()),
                                 'displacement': float(displacement.detach()), 'eikonal': float(eikonal.detach()), 'known_empty_feature': float(feature_loss.detach())})
            if retain_checkpoints:
                scored_field = field.detach().numpy().copy()
                checkpoints.append({'field_zyx': scored_field,
                    'parameters': parameters.detach().numpy().copy(),
                    'best_evaluation': evaluations, 'best_total': value,
                    'retained_field_hash': hashlib.sha256(model.report()['constraint_hash'].encode()+scored_field.tobytes()).hexdigest()})
            if value < best_value:
                best_value, best_evaluation = value, evaluations
                best_field = field.detach().numpy().copy()
                best_parameters = parameters.detach().numpy().copy()
                from copy import deepcopy
                best_projected_reports = deepcopy(projected_reports)
                if payload.get('progress_paths'):
                    from pathlib import Path
                    import os, pickle
                    for output in payload['progress_paths']:
                        path = Path(output)
                        temporary = path.with_name(path.name+'.tmp')
                        with temporary.open('wb') as stream:
                            pickle.dump({'value': snapshot()}, stream, protocol=pickle.HIGHEST_PROTOCOL)
                        os.replace(temporary, path)
        print(f'implicit field evaluations={evaluations} updates={updates} active={len(model.active_flat)} '
              f'elapsed={time.perf_counter()-started:.2f}s', flush=True)
        return loss

    for _ in range(steps):
        if time.perf_counter()-started >= timeout:
            stop_reason = 'elapsed_time_budget'
            break
        optimizer.zero_grad(set_to_none=True)
        try:
            loss = evaluate()
        except (ValueError, RuntimeError) as exc:
            if best_field is None or objective != 'original_pixel_extracted_mesh':
                raise
            stop_reason, operator_failure = 'extracted_operator_unavailable_after_scored_checkpoint', str(exc)
            break
        if not bool(torch.isfinite(loss)):
            stop_reason = 'nonfinite_loss'
            break
        loss.backward()
        if parameters.grad is None or not bool(torch.isfinite(parameters.grad).all()):
            stop_reason = 'nonfinite_or_missing_gradient'
            break
        optimizer.step()
        updates += 1
    if stop_reason == 'requested_steps' and time.perf_counter()-started < timeout:
        try:
            with torch.no_grad():
                final_loss = evaluate()
            final_update_evaluated = bool(torch.isfinite(final_loss))
        except (ValueError, RuntimeError) as exc:
            if best_field is None or objective != 'original_pixel_extracted_mesh':
                raise
            stop_reason, operator_failure = 'extracted_operator_unavailable_after_scored_checkpoint', str(exc)
    elif stop_reason == 'requested_steps':
        stop_reason = 'elapsed_time_budget'
    if best_field is None:
        raise RuntimeError('implicit field job has no finite evaluated checkpoint')
    return snapshot()
