"""Opt-in topology-changing residual proposal backend, preserving qualified seeds."""
from dataclasses import replace
import time

import numpy as np

from ..backend import BackendCapabilities, BaseBackend
from ..types import CandidateMetrics, CandidateResult


class ImplicitResidualBackend(BaseBackend):
    def __init__(self):
        super().__init__(name='implicit_residual', version='1', capabilities=BackendCapabilities(
            supports_multi_view=True, supports_top_view=True, supports_uncertainty=True,
            supports_constraints=True, outputs_mesh=True, outputs_volume=True,
            supports_gradients=True, editability_score=.15,
            optional_dependencies=('torch', 'scikit-image', 'dvx-python', 'shapely')))

    def validate_config(self, config):
        errors = []
        if not isinstance(config.get('checkpoint_selection', True), bool):
            errors.append('checkpoint selection must be an explicit Boolean')
        if isinstance(config.get('resolution', 16), bool) or config.get('resolution', 16) not in (16, 32):
            errors.append('implicit resolution must be 16 or 32')
        if config.get('objective', 'minimum_distance_rays') not in ('minimum_distance_rays', 'original_pixel_extracted_mesh'):
            errors.append('unsupported implicit objective')
        if config.get('projection_metric', 'legacy_pil') not in ('legacy_pil', 'pixel_area_half'):
            errors.append('unsupported implicit output projection metric')
        try:
            feature_weight = float(config.get('known_empty_feature_weight', 0.))
            if not np.isfinite(feature_weight) or feature_weight < 0:
                raise ValueError()
            if feature_weight > 0 and config.get('objective') != 'original_pixel_extracted_mesh':
                errors.append('known-empty feature weighting requires original-pixel extracted support')
        except (TypeError, ValueError):
            errors.append('known-empty feature weight must be finite nonnegative')
        steps = config.get('steps', 2)
        if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= 16:
            errors.append('implicit steps must be an integer from 0 through 16')
        for key in ('timeout_s', 'seed_timeout_s', 'boundary_timeout_s', 'checkpoint_selection_timeout_s'):
            try:
                value = float(config.get(key, 8.))
                if not np.isfinite(value) or value <= 0:
                    raise ValueError()
            except (TypeError, ValueError):
                errors.append(key+' must be a finite positive allowance')
        return errors

    def reconstruct(self, request):
        started = time.perf_counter()
        config = dict(request.config)
        deadline = None if request.budget.timeout_s is None else started+float(request.budget.timeout_s)
        def allowance(key, default=8.):
            value = float(config.get(key, default))
            if deadline is not None:
                value = min(value, deadline-time.perf_counter())
            if not np.isfinite(value) or value <= 0:
                raise TimeoutError('implicit candidate allowance exhausted before '+key)
            return value
        errors = self.validate_config(config)
        if errors:
            return CandidateResult(request.candidate_id, self.name, 'failed', errors=tuple(errors))
        if config.get('execution_approved') is not True:
            return self.unavailable(request, 'implicit numerical execution requires explicit approval')
        from ..differentiable.seed_selection import select_existing_seed
        seed, selection = select_existing_seed(request.target, config.get('seed_results'),
                                               maximum_vertices=4096, maximum_faces=512)
        if seed is None:
            return CandidateResult(request.candidate_id, self.name, 'skipped',
                                   warnings=('no bounded same-evidence validated source seed',),
                                   metric_result=CandidateMetrics(extras={'seed_selection': selection}))
        source_result = dict(config['seed_results'])[selection['source']]
        def retain_source(reason):
            report = {'seed_selection': selection, 'source_geometry_hash': seed.content_hash,
                      'retained_geometry_hash': seed.content_hash, 'full_geometry_admitted': False,
                      'proposal_unavailable_reason': reason,
                      'opaque_seed_per_view': source_result.metric_result.per_view,
                      'public_selection_metric': 'existing_same_evidence_source_metric',
                      'native_blender_render_acceptance': 'not_run',
                      'admission_scope': 'original validated source retained; no new implicit or native qualification'}
            metrics = replace(source_result.metric_result, elapsed_s=time.perf_counter()-started,
                              extras={**source_result.metric_result.extras,
                                      'input_evidence_hash': selection['input_evidence_hash'],
                                      'metrics_refer_to_output_hash': seed.content_hash,
                                      'implicit_residual': report})
            return CandidateResult(request.candidate_id, self.name, 'degraded', degraded=True,
                                   geometry=seed, mesh_path=source_result.mesh_path,
                                   primitive_path=source_result.primitive_path,
                                   artifacts=source_result.artifacts,
                                   warnings=('implicit proposal unavailable; actual same-evidence source retained: '+reason,),
                                   metric_result=metrics)
        try:
            from ..implicit.pipeline import (prepare_implicit_job, validate_fitted_field,
                                             extract_implicit_geometry, write_implicit_artifacts, select_output_checkpoint)
            config['seed_timeout_s'] = allowance('seed_timeout_s')
            job, preparation = prepare_implicit_job(request.target, seed, config)
            job['timeout_s'] = allowance('timeout_s')
            if config.get('helper_python'):
                from ..implicit.helper import implicit_session
                owner = implicit_session(config['helper_python'])
                state = owner.call(timeout_s=job['timeout_s'])
                if not state.get('available'):
                    return retain_source('selected helper lacks pinned Torch2.14.1+cpu')
                if job['objective'] == 'original_pixel_extracted_mesh' and not state.get('original_pixel_mesh_available'):
                    return retain_source('original-pixel field mesh requires pinned DVX0.1.1, scikit-image0.26.0 and Shapely2.1.2 in the selected helper')
                job['timeout_s'] = allowance('timeout_s')
                fitted = owner.call(job, timeout_s=job['timeout_s'])
            else:
                from ..implicit.numeric_solver import fit_implicit_field_job
                fitted = fit_implicit_field_job(job)
            model, field = validate_fitted_field(job, fitted)
            checkpoint_report = None
            if config.get('checkpoint_selection', True):
                fitted, proposed, topology, checkpoint_report = select_output_checkpoint(job, fitted, request.target, seed,
                    projection_mode=config.get('projection_metric', 'legacy_pil'),
                    timeout_s=allowance('checkpoint_selection_timeout_s'))
            else:
                proposed, topology = extract_implicit_geometry(field, model)
            from ..grouped_solids import solid_guard
            from blender_blocking.evaluation.triangle_contacts import within_part_boundary_guard
            guard = solid_guard(proposed)
            boundary = within_part_boundary_guard(proposed.vertices, proposed.faces,
                                                 timeout_s=allowance('boundary_timeout_s'))
            from ..projected_metrics import projected_mesh_metrics
            legacy_initial = CandidateMetrics(per_view=projected_mesh_metrics(request.target, seed.vertices, seed.faces))
            legacy_final = CandidateMetrics(per_view=projected_mesh_metrics(request.target, proposed.vertices, proposed.faces))
            projection_mode = config.get('projection_metric', 'legacy_pil')
            initial, final = legacy_initial, legacy_final
            if projection_mode == 'pixel_area_half':
                from ..pixel_projection import canonical_mesh_metrics
                initial = CandidateMetrics(per_view=canonical_mesh_metrics(request.target, seed.vertices, seed.faces))
                final = CandidateMetrics(per_view=canonical_mesh_metrics(request.target, proposed.vertices, proposed.faces))
            elif projection_mode != 'legacy_pil':
                raise ValueError('unsupported implicit output projection metric')
            from ..output_qualification import qualify_retained_output
            native_options = dict(config)
            if config.get('native_qualification_python'):
                native_options['native_qualification_timeout_s'] = allowance('native_qualification_timeout_s', 15.)
            qualification = qualify_retained_output(proposed, native_options) if guard['valid_solid'] and boundary['passed'] else {
                'status': 'rejected', 'boundary_qualified': False, 'single_solid_qualified': False,
                'geometry_content_hash': proposed.content_hash, 'reason': 'output_topology_or_exact_boundary_failed'}
            from ..projected_metrics import projected_mesh_masks
            from ..pixel_evidence import observed_pixel_evidence
            seed_masks = projected_mesh_masks(request.target, seed.vertices, seed.faces)
            final_masks = projected_mesh_masks(request.target, proposed.vertices, proposed.faces)
            if projection_mode == 'pixel_area_half':
                from ..pixel_projection import canonical_mesh_projections
                seed_masks = {view: value >= .5 for view, value in canonical_mesh_projections(request.target, seed.vertices, seed.faces).items()}
                final_masks = {view: value >= .5 for view, value in canonical_mesh_projections(request.target, proposed.vertices, proposed.faces).items()}
            added_empty_violations = {}
            for constraint in request.target.constraints:
                evidence = observed_pixel_evidence(constraint)
                certain_empty = (evidence.valid & (evidence.weights == 1.)
                                 & (evidence.foreground == 0.) & ~evidence.hard_mask)
                added_empty_violations[constraint.view] = int(np.count_nonzero(
                    certain_empty & final_masks[constraint.view] & ~seed_masks[constraint.view]))
            from ..feature_evidence import known_empty_feature_guard
            empty_rows = {c.view: known_empty_feature_guard(c, final_masks[c.view]) for c in request.target.constraints}
            empty_features = {'passed': all(row['passed'] for row in empty_rows.values()), 'per_view': empty_rows}
            passed_views = all(row.get('passed') for row in final.per_view.values() if row.get('required', True))
            geometric_improvement = (final.area_iou_min >= initial.area_iou_min-.002
                                     and final.area_iou_mean > initial.area_iou_mean+.002
                                     and final.boundary_iou_mean >= initial.boundary_iou_mean-.002)
            admitted = bool(guard['valid_solid'] and boundary['passed'] and passed_views
                            and geometric_improvement and empty_features['passed']
                            and not any(added_empty_violations.values())
                            and qualification.get('boundary_qualified')
                            and qualification.get('geometry_content_hash') == proposed.content_hash)
            retained = proposed if admitted else seed
            # Shared candidate ranking keeps the existing legacy score. The
            # explicit canonical experiment is a separate identified track;
            # mixing two raster contracts would silently change ensemble rank.
            metrics = legacy_final if admitted else legacy_initial
            report = {**preparation, 'seed_selection': selection, 'fixed_transform': model.report(),
                      'output_checkpoint_selection': checkpoint_report,
                      'proposal_topology': topology, 'proposal_solid_guard': guard,
                      'proposal_exact_boundary': boundary, 'proposal_native_qualification': qualification,
                      'opaque_metric_contract': ('opaque_triangle_union_pixel_cell_area_hard_half_v1' if projection_mode == 'pixel_area_half'
                                                 else 'legacy_pil_polygon_endpoint_rounding_v1'),
                      'legacy_opaque_seed_per_view': legacy_initial.per_view,
                      'legacy_opaque_proposal_per_view': legacy_final.per_view,
                      'opaque_proposal_per_view': final.per_view,
                      'new_original_pixel_empty_violations': added_empty_violations,
                      'known_empty_feature_guard': empty_features,
                      'opaque_seed_per_view': initial.per_view,
                      'full_geometry_admitted': admitted,
                      'public_selection_metric': 'legacy_pil_polygon_endpoint_rounding_v1',
                      'admission_metric_retained_per_view': (final if admitted else initial).per_view, 'native_blender_render_acceptance': 'not_run',
                      'proposal_objective': fitted.get('objective'), 'proposal_ray_operator': fitted.get('ray_operator'),
                      'projection_support': {view: {key: row[key] for key in ('original_shape', 'padded_world_bounds',
                          'original_pixel_size_world', 'padding_screen_tblr', 'padding_semantics',
                          'empty_feature_regions', 'empty_feature_semantics')}
                          for view, row in job.get('original_pixel_targets', {}).items()},
                      'optimization': {key: fitted.get(key) for key in ('best_total', 'best_evaluation',
                          'history', 'term_history', 'objective_evaluations', 'optimizer_updates',
                          'stop_reason', 'final_update_evaluated', 'partial', 'helper_session', 'projected_reports',
                          'objective_weights', 'known_empty_feature_weight_sum', 'feature_evidence_scope')},
                      'admission_scope': 'exact boundary, native content qualification, original-size polygon projection; native render acceptance remains separate',
                      'inference_scope': 'evidence-conditioned residual hypothesis; invisible cavities are not recovered truth'}
            paths = {}
            root = request.candidate_artifact_root()
            if root is not None:
                paths, report = write_implicit_artifacts(root, seed, proposed, retained, job, fitted, report,
                                                        selection.get('source_primitive_path'))
            return CandidateResult(request.candidate_id, self.name, 'success' if admitted else 'degraded',
                degraded=not admitted, geometry=retained, mesh_path=paths.get('mesh_obj'),
                volume_path=paths.get('field_state'), artifacts=paths,
                warnings=() if admitted else ('implicit proposal not admitted; actual same-evidence source retained',),
                metric_result=replace(metrics, editability_score=.15 if admitted else .65,
                    elapsed_s=time.perf_counter()-started, extras={'input_evidence_hash': job['input_evidence_hash'],
                        'metrics_refer_to_output_hash': retained.content_hash, 'implicit_residual': report}))
        except (ValueError, TypeError, KeyError, RuntimeError, TimeoutError, ImportError, PermissionError, OSError) as exc:
            return retain_source(str(exc))
