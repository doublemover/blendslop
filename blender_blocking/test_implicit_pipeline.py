"""Bounded source/evidence preparation, retained-output safety and field replay."""
from dataclasses import replace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from reconstruction.backends.implicit_residual import ImplicitResidualBackend
from reconstruction.implicit.pipeline import (prepare_implicit_job, validate_fitted_field,
                                              extract_implicit_geometry, write_implicit_artifacts,
                                              replay_implicit_artifacts, select_output_checkpoint)
from reconstruction.implicit.field_model import NarrowBandField
from reconstruction.types import CandidateRequest, CandidateResult, CandidateMetrics, CandidateBudget
from reconstruction.evidence_identity import target_evidence_hash
from reconstruction.differentiable.seed_selection import select_existing_seed
from test_quality_geometry import target_for_masks
from test_view_suggestions import cube


def fixture():
    return target_for_masks({'front': np.ones((24, 32), bool)}), cube([.1, -.1, .05], radius=.4)


def hole_fixture():
    from reconstruction.native_geometry import GeometryArrays
    from reconstruction.projected_metrics import projected_mesh_masks
    _, source = fixture()
    vertices = np.array(source.vertices)
    vertices[:, 2] = .05+(vertices[:, 2]-.05)*.18
    source = GeometryArrays.capture(vertices, source.faces)
    target = target_for_masks({'top': np.ones((64, 64), bool)})
    mask = projected_mesh_masks(target, source.vertices, source.faces)['top'].copy()
    mask[29:35, 29:35] = False
    return target_for_masks({'top': mask}), source


def separated_fixture():
    from reconstruction.grouped_solids import concatenate
    from reconstruction.projected_metrics import projected_mesh_masks
    seed = concatenate([cube([-.5, 0., 0.], radius=.2), cube([.5, 0., 0.], radius=.2)])
    target = target_for_masks({view: np.zeros((32, 32), bool) for view in ('front','side','top')})
    return target_for_masks(projected_mesh_masks(target, seed.vertices, seed.faces)), seed


def fake_fit(job):
    model = NarrowBandField.create(job['seed_field_zyx'], job['voxel_size_xyz'],
                                  origin_xyz=job['origin_xyz'], band_width=job['band_width'],
                                  maximum_displacement=job['maximum_displacement'],
                                  known_empty_zyx=job['known_empty_zyx'])
    parameters = np.zeros(len(model.active_flat), np.float32)
    field = model.decode(parameters).astype(np.float32)
    return {'field_zyx': field, 'parameters': parameters, 'model': model.report(),
            'input_evidence_hash': job['input_evidence_hash'], 'objective': job.get('objective', 'minimum_distance_rays'), 'best_evaluation': 1, 'best_total': .1,
            'retained_field_hash': hashlib.sha256(model.report()['constraint_hash'].encode()+field.tobytes()).hexdigest(),
            'history': [.1], 'optimizer_updates': 0, 'objective_evaluations': 1}


def seed_result(target, seed):
    return CandidateResult('source', 'fixture', 'success', geometry=seed,
        metric_result=CandidateMetrics(per_view={c.view: {'area_iou': .9, 'passed': True} for c in target.constraints},
            extras={'input_evidence_hash': target_evidence_hash(target)}))


def ranked_fake_fit(job):
    fitted=fake_fit(job);model=validate_fitted_field(job,fitted)[0]
    first={key:fitted[key] for key in ('field_zyx','parameters','best_evaluation','best_total','retained_field_hash')}
    first={**first,'best_total':.2}
    parameters=np.ones(len(model.active_flat),np.float32)
    field=model.decode(parameters).astype(np.float32)
    second={'field_zyx':field,'parameters':parameters,'best_evaluation':2,'best_total':.1,
        'retained_field_hash':hashlib.sha256(model.report()['constraint_hash'].encode()+field.tobytes()).hexdigest()}
    return {**fitted,**second,'objective_evaluations':2,'history':[.2,.1],'scored_checkpoints':[first,second]}


class ImplicitPipelineTests(unittest.TestCase):
    def test_checkpoint_ordering_keeps_safe_earlier_output_when_higher_score_is_folded(self):
        from reconstruction.native_geometry import GeometryArrays
        from reconstruction.grouped_solids import solid_guard
        from blender_blocking.evaluation.triangle_contacts import within_part_boundary_guard
        target,seed=fixture();safe=cube([0.,0.,0.],radius=.5)
        vertices=safe.vertices.copy();vertices[0]=[.7,.2,.1]
        folded=GeometryArrays.capture(vertices,safe.faces)
        self.assertTrue(solid_guard(folded)['valid_solid'])
        self.assertFalse(within_part_boundary_guard(folded.vertices,folded.faces,timeout_s=2.)['passed'])
        job,_=prepare_implicit_job(target,seed,{})
        score=lambda value:{'front':{'area_iou':value,'boundary_iou':value,'passed':True}}
        with patch('reconstruction.projected_metrics.projected_mesh_metrics',side_effect=[score(.7),score(.8),score(.9)]), \
             patch('reconstruction.implicit.pipeline.extract_implicit_geometry',side_effect=[(safe,{}),(folded,{})]):
            selected,proposal,_,report=select_output_checkpoint(job,ranked_fake_fit(job),target,seed)
        self.assertEqual(selected['best_evaluation'],1)
        self.assertEqual(proposal.content_hash,safe.content_hash)
        self.assertTrue(report['checkpoints'][0]['preliminary_output_eligible'])
        self.assertFalse(report['checkpoints'][1]['preliminary_output_eligible'])
        self.assertEqual(report['checkpoints'][1]['exact_boundary_guard']['first_blocking_pair']['relation'],'proper_crossing')
        # A later unverified boundary never displaces the earlier passed one.
        with patch('reconstruction.projected_metrics.projected_mesh_metrics',side_effect=[score(.7),score(.8),score(.9)]), \
             patch('reconstruction.implicit.pipeline.extract_implicit_geometry',side_effect=[(safe,{}),(folded,{})]), \
             patch('blender_blocking.evaluation.triangle_contacts.within_part_boundary_guard',
                   side_effect=[report['checkpoints'][0]['exact_boundary_guard'],{'status':'unavailable','passed':False}]) as guard, \
             patch('time.perf_counter',side_effect=[0.,.1,.2,.3,.4]):
            selected,_,_,partial=select_output_checkpoint(job,ranked_fake_fit(job),target,seed,timeout_s=1.)
        self.assertEqual(selected['best_evaluation'],1)
        self.assertEqual(partial['stop_reason'],'shared_extraction_projection_boundary_allowance')
        self.assertAlmostEqual(guard.call_args_list[0].kwargs['timeout_s'],.8)
        self.assertAlmostEqual(guard.call_args_list[1].kwargs['timeout_s'],.6)

    def test_opaque_checkpoint_ranking_can_reject_the_lower_surrogate_field(self):
        target,seed=fixture();job,_=prepare_implicit_job(target,seed,{})
        fitted=ranked_fake_fit(job)
        selected,proposal,_,report=select_output_checkpoint(job,fitted,target,seed)
        self.assertEqual(fitted['best_evaluation'],2)
        self.assertEqual(selected['best_evaluation'],1)
        self.assertEqual(report['surrogate_best_evaluation'],2)
        self.assertEqual(report['metric_identity'],'legacy_pil_polygon_endpoint_rounding_v1')
        self.assertTrue(report['source_always_available'])
        self.assertGreater(report['checkpoints'][0]['per_view']['front']['area_iou'],
                           report['checkpoints'][1]['per_view']['front']['area_iou'])
        self.assertFalse(any(row['preliminary_output_eligible'] for row in report['checkpoints']))

    def test_checkpoint_pool_validates_each_scored_field_and_evaluation_identity(self):
        target,seed=fixture();job,_=prepare_implicit_job(target,seed,{})
        original=ranked_fake_fit(job)
        wrong=deepcopy(original);wrong['scored_checkpoints'][0]['field_zyx'].flat[0]+=.001
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            select_output_checkpoint(job,wrong,target,seed)
        duplicate=deepcopy(original);duplicate['scored_checkpoints'][1]['best_evaluation']=1
        with self.assertRaisesRegex(ValueError,'evaluation identity'):
            select_output_checkpoint(job,duplicate,target,seed)
        with self.assertRaisesRegex(ValueError,'allowance'):
            select_output_checkpoint(job,{**original,'scored_checkpoints':original['scored_checkpoints']*9},target,seed)
        mislabeled=deepcopy(original);mislabeled['scored_checkpoints'][0]['best_total']=.15
        with self.assertRaisesRegex(ValueError,'evaluated history'):
            select_output_checkpoint(job,mislabeled,target,seed)
        expanded=deepcopy(original);expanded['scored_checkpoints'][0]['model']={}
        with self.assertRaisesRegex(ValueError,'only the evaluated field record'):
            select_output_checkpoint(job,expanded,target,seed)
        wrong_coordinates=deepcopy(original);wrong_coordinates['scored_checkpoints'][0]['parameters']+=.1
        with self.assertRaisesRegex(ValueError,'residual coordinates'):
            select_output_checkpoint(job,wrong_coordinates,target,seed)

    def test_checkpoint_selection_uses_one_allowance_and_preserves_a_scored_prefix(self):
        target,seed=fixture();job,_=prepare_implicit_job(target,seed,{})
        fitted=ranked_fake_fit(job)
        with patch('time.perf_counter',side_effect=[0.,.1,1.]):
            selected,_,_,report=select_output_checkpoint(job,fitted,target,seed,timeout_s=.5)
        self.assertEqual(selected['best_evaluation'],1)
        self.assertEqual(len(report['checkpoints']),1)
        self.assertEqual(report['stop_reason'],'shared_extraction_projection_allowance')
        self.assertTrue(ImplicitResidualBackend().validate_config({'checkpoint_selection_timeout_s':float('nan')}))

    def test_backend_replays_the_opaque_selected_field_and_keeps_the_seed(self):
        target,seed=fixture();backend=ImplicitResidualBackend()
        with tempfile.TemporaryDirectory() as root:
            request=CandidateRequest('checkpoint-ranked',backend.name,target,
                config={'execution_approved':True,'seed_results':{'source':seed_result(target,seed)}},artifact_root=Path(root))
            with patch('reconstruction.implicit.numeric_solver.fit_implicit_field_job',side_effect=ranked_fake_fit):
                result=backend.reconstruct(request)
            self.assertEqual(result.status,'degraded',result.errors)
            report=result.metric_result.extras['implicit_residual']
            self.assertEqual(report['output_checkpoint_selection']['selected_evaluation'],1)
            self.assertEqual(report['output_checkpoint_selection']['surrogate_best_evaluation'],2)
            self.assertEqual(result.geometry.content_hash,seed.content_hash)
            self.assertEqual(replay_implicit_artifacts(report)['proposal'].content_hash,report['proposal_geometry_hash'])

    def test_full_output_checkpoint_replay_preserves_every_scored_mesh_and_rejects_misassociation(self):
        from reconstruction.artifacts import hash_file
        target,seed=fixture();backend=ImplicitResidualBackend()
        with tempfile.TemporaryDirectory() as root:
            request=CandidateRequest('pool-replay',backend.name,target,
                config={'execution_approved':True,'seed_results':{'source':seed_result(target,seed)}},artifact_root=Path(root))
            with patch('reconstruction.implicit.numeric_solver.fit_implicit_field_job',side_effect=ranked_fake_fit):
                result=backend.reconstruct(request)
            report=result.metric_result.extras['implicit_residual']
            replay=replay_implicit_artifacts(report,include_output_checkpoints=True)
            rows=report['output_checkpoint_selection']['checkpoints']
            self.assertEqual([row['evaluation'] for row in replay['output_checkpoints']],[1,2])
            self.assertEqual([row['geometry'].content_hash for row in replay['output_checkpoints']],
                             [row['geometry_hash'] for row in rows])
            older={key:value for key,value in report.items() if key != 'output_checkpoint_archive'}
            self.assertEqual(replay_implicit_artifacts(older)['proposal'].content_hash,report['proposal_geometry_hash'])
            with self.assertRaisesRegex(ValueError,'older artifact contract'):
                replay_implicit_artifacts(older,include_output_checkpoints=True)
            wrong=deepcopy(report);wrong['output_checkpoint_selection']['checkpoints'][0]['geometry_hash']='wrong'
            with self.assertRaisesRegex(ValueError,'scored mesh'):
                replay_implicit_artifacts(wrong,include_output_checkpoints=True)
            state=Path(report['artifacts']['field_state'])
            with np.load(state,allow_pickle=False) as packet:
                arrays={key:packet[key] for key in packet.files}
            arrays['opaque_scored_checkpoint_parameters'][0]+=.1
            np.savez_compressed(state,**arrays)
            wrong={**report,'field_state_sha256':hash_file(state)}
            with self.assertRaisesRegex(ValueError,'residual coordinates'):
                replay_implicit_artifacts(wrong,include_output_checkpoints=True)

    def test_unknown_pixels_cannot_change_ray_preparation_or_final_checkpoint_ordering(self):
        target,seed=fixture();c=target.constraints[0]
        valid=np.ones_like(c.mask);valid[:,:12]=False
        one=replace(target,constraints=(replace(c,valid_mask=valid),))
        changed=c.mask.copy();changed[~valid]=~changed[~valid]
        two=replace(target,constraints=(replace(c,mask=changed,valid_mask=valid),))
        first,_=prepare_implicit_job(one,seed,{});second,_=prepare_implicit_job(two,seed,{})
        np.testing.assert_array_equal(first['known_empty_zyx'],second['known_empty_zyx'])
        for key in ('foreground','valid','weights'):
            np.testing.assert_array_equal(first['ray_targets']['front'][key],second['ray_targets']['front'][key])
        for mode in ('legacy_pil','pixel_area_half'):
            a=select_output_checkpoint(first,ranked_fake_fit(first),one,seed,projection_mode=mode)[3]
            b=select_output_checkpoint(second,ranked_fake_fit(second),two,seed,projection_mode=mode)[3]
            self.assertEqual(a['selected_evaluation'],b['selected_evaluation'])
            self.assertEqual(a['source_per_view'],b['source_per_view'])
            self.assertEqual(a['checkpoints'],b['checkpoints'])

    def test_separated_components_field_matches_analytic_box_union_at_16_and_32(self):
        from reconstruction.implicit.seed_field import signed_seed_field
        _, seed=separated_fixture();original=seed.content_hash
        lo,hi=np.array([-1.,-.6,-.5]),np.array([1.,.6,.5])
        for n in (16,32):
            sampled=signed_seed_field(seed,lo,hi,resolution=n)
            axes=[sampled['origin_xyz'][i]+np.arange(n)*sampled['voxel_size_xyz'][i] for i in range(3)]
            points=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1)
            def box(center):
                q=np.abs(points-np.array(center))-.2
                return np.linalg.norm(np.maximum(q,0.),axis=-1)+np.minimum(q.max(axis=-1),0.)
            expected=np.minimum(box([-.5,0,0]),box([.5,0,0]))
            np.testing.assert_allclose(sampled['field_xyz'],expected,atol=3e-8,rtol=1e-7)
            screen=sampled['seed_screen']
            self.assertEqual(len(screen['component_screens']),2)
            self.assertEqual(len(screen['component_separation']),1)
            self.assertTrue(sampled['exact_within_part_boundary']['passed'])
        self.assertEqual(original,seed.content_hash)

    def test_touching_overlap_cavity_and_inward_component_seeds_fail_closed(self):
        from reconstruction.implicit.seed_field import signed_seed_field
        from reconstruction.grouped_solids import concatenate
        from reconstruction.native_geometry import GeometryArrays
        for second in (cube([.4,0,0],radius=.2),cube([.3,0,0],radius=.2),cube([0,0,0],radius=.1)):
            seed=concatenate([cube([0,0,0],radius=.2),second])
            with self.assertRaisesRegex(ValueError,'strictly separated AABBs'):
                signed_seed_field(seed,[-1,-1,-1],[1,1,1])
        a,b=cube([-.5,0,0],radius=.25),cube([.5,0,0],radius=.15)
        inward=GeometryArrays.capture(b.vertices,b.faces[:,::-1])
        with self.assertRaisesRegex(ValueError,'independent positive'):
            signed_seed_field(concatenate([a,inward]),[-1,-1,-1],[1,1,1])

    def test_multipart_backend_scores_actual_extraction_and_retains_original_assembly(self):
        from reconstruction.projected_metrics import projected_mesh_metrics
        target,seed=separated_fixture();backend=ImplicitResidualBackend()
        source=replace(seed_result(target,seed),metric_result=CandidateMetrics(
            per_view=projected_mesh_metrics(target,seed.vertices,seed.faces),
            extras=seed_result(target,seed).metric_result.extras))
        with tempfile.TemporaryDirectory() as root:
            request=CandidateRequest('multipart',backend.name,target,
                config={'execution_approved':True,'seed_results':{'source':source}},artifact_root=Path(root))
            with patch('reconstruction.implicit.numeric_solver.fit_implicit_field_job',side_effect=fake_fit):
                result=backend.reconstruct(request)
            self.assertEqual(result.status,'degraded',result.errors)
            self.assertEqual(result.geometry.content_hash,seed.content_hash)
            report=result.metric_result.extras['implicit_residual']
            self.assertEqual(report['source_screen']['connected_components'],2)
            self.assertFalse(report['full_geometry_admitted'])
            self.assertEqual(set(report['opaque_proposal_per_view']),{'front','side','top'})
            replay=replay_implicit_artifacts(report)
            self.assertEqual(replay['retained'].content_hash,seed.content_hash)

    def test_full_and_empty_camera_distance_metrics_are_finite_and_symmetric(self):
        from metrics.silhouette import signed_distance_silhouette_loss
        full, empty = np.ones((24, 32), bool), np.zeros((24, 32), bool)
        with np.errstate(over='raise', invalid='raise'):
            self.assertEqual(signed_distance_silhouette_loss(full, full), 0.)
            self.assertEqual(signed_distance_silhouette_loss(empty, empty), 0.)
            self.assertAlmostEqual(signed_distance_silhouette_loss(full, empty), 2.5)
            self.assertEqual(signed_distance_silhouette_loss(full, empty),
                             signed_distance_silhouette_loss(empty, full))

    def test_original_non_square_camera_and_fixed_world_field_are_not_rebounded(self):
        target, seed = fixture()
        job, report = prepare_implicit_job(target, seed, {'execution_approved': True})
        model = validate_fitted_field(job, fake_fit(job))[0]
        self.assertEqual(job['ray_targets']['front']['foreground'].shape, (16, 16))
        self.assertTrue((job['ray_targets']['front']['valid'] < 1).any())
        self.assertEqual(report['source_geometry_hash'], seed.content_hash)
        self.assertEqual(report['input_evidence_hash'], target_evidence_hash(target))
        self.assertTrue(report['exact_source_boundary']['passed'])
        self.assertFalse(model.report()['native_qualification'])

    def test_soft_or_partly_unknown_background_is_never_a_hard_empty_column(self):
        target, seed = fixture()
        mask = np.zeros((24, 32), bool)
        valid = np.zeros_like(mask)
        valid[:2] = True
        uncertainty = SimpleNamespace(foreground_prob=np.full(mask.shape, .2),
                                      confidence=np.ones(mask.shape), boundary_uncertainty=np.zeros(mask.shape))
        target = replace(target, constraints=(replace(target.constraints[0], mask=mask,
                         valid_mask=valid, uncertainty=uncertainty),))
        job, _ = prepare_implicit_job(target, seed, {})
        self.assertFalse(job['known_empty_zyx'].any())
        self.assertTrue(np.all(job['ray_targets']['front']['foreground'][job['ray_targets']['front']['valid'] > 0] > 0))

    def test_complete_reliable_empty_ray_has_correct_depth_axis(self):
        target, seed = fixture()
        mask = np.ones((24, 32), bool)
        mask[:, :5] = False
        target = replace(target, constraints=(replace(target.constraints[0], mask=mask),))
        job, _ = prepare_implicit_job(target, seed, {})
        empty = job['known_empty_zyx']
        self.assertTrue(empty.any())
        np.testing.assert_array_equal(empty[:, 0, :], empty[:, -1, :])
        self.assertFalse(empty[8, 8, 8])

    def test_anisotropic_16_and_32_fields_extract_at_world_cell_centers_without_half_cell_shift(self):
        for n in (16, 32):
            spacing = np.array([1.6, 1.12, .8])/n
            lo = np.array([1., -3., 2.])
            origin = lo+spacing*.5
            center = lo+spacing*n*.5
            axes = [origin[a]+np.arange(n)*spacing[a] for a in range(3)]
            points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
            half = np.array([.31, .39, .18])
            q = np.abs(points-center)-half
            phi = (np.linalg.norm(np.maximum(q, 0), axis=-1)+np.minimum(q.max(axis=-1), 0)).astype(np.float32)
            model = NarrowBandField.create(phi.transpose(2, 1, 0), spacing, origin_xyz=origin,
                                          band_width=.1, maximum_displacement=.05)
            geometry, _ = extract_implicit_geometry(model.decode(np.zeros(len(model.active_flat))), model)
            np.testing.assert_allclose(geometry.vertices.min(axis=0), center-half, atol=1e-7)
            np.testing.assert_allclose(geometry.vertices.max(axis=0), center+half, atol=1e-7)

    def test_duplicate_or_unsupported_views_fail(self):
        target, seed = fixture()
        with self.assertRaisesRegex(ValueError, 'unique canonical'):
            prepare_implicit_job(replace(target, constraints=target.constraints*2), seed, {})
        with self.assertRaisesRegex(ValueError, 'unique canonical'):
            prepare_implicit_job(replace(target, constraints=(replace(target.constraints[0], view='oblique'),)), seed, {})

    def test_helper_response_identity_and_scored_checkpoint_are_required(self):
        target, seed = fixture()
        job, _ = prepare_implicit_job(target, seed, {})
        fitted = fake_fit(job)
        for edit in ({'input_evidence_hash': 'wrong'}, {'retained_field_hash': 'wrong'}, {'best_evaluation': None},
                     {'best_evaluation':True}, {'best_evaluation':'1'}, {'best_evaluation':2},
                     {'objective_evaluations':True}, {'objective_evaluations':18},
                     {'best_total':.2}, {'history':[float('nan')]}, {'history':['.1']}):
            with self.assertRaises(ValueError):
                validate_fitted_field(job, {**fitted, **edit})
        wrong = fitted['field_zyx'].copy()
        wrong[0, 0, 0] = -1
        with self.assertRaises(ValueError):
            validate_fitted_field(job, {**fitted, 'field_zyx': wrong})

    def test_artifact_replay_preserves_distinct_source_field_and_retained_output(self):
        target, seed = fixture()
        job, report = prepare_implicit_job(target, seed, {})
        fitted = fake_fit(job)
        model, field = validate_fitted_field(job, fitted)
        proposed, _ = extract_implicit_geometry(field, model)
        with tempfile.TemporaryDirectory() as root:
            artist = Path(root)/'original.json'
            artist.write_text('{"artist": "original"}')
            paths, contract = write_implicit_artifacts(Path(root)/'artifacts', seed, proposed, seed,
                                                       job, fitted, report, artist)
            replay = replay_implicit_artifacts(contract)
            np.testing.assert_array_equal(replay['proposal'].vertices, proposed.vertices)
            np.testing.assert_array_equal(replay['proposal'].faces, proposed.faces)
            self.assertEqual(replay['retained'].content_hash, seed.content_hash)
            self.assertFalse(replay['retained_output_is_proposal'])
            self.assertFalse(replay['native_qualification'])
            artist_copy = paths['artist_source_json']
            artist_copy.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'artist-source'):
                replay_implicit_artifacts(contract)

    def test_modified_field_archive_or_obj_fails_replay(self):
        target, seed = fixture()
        job, report = prepare_implicit_job(target, seed, {})
        fitted = fake_fit(job)
        model, field = validate_fitted_field(job, fitted)
        proposed, _ = extract_implicit_geometry(field, model)
        with tempfile.TemporaryDirectory() as root:
            paths, contract = write_implicit_artifacts(root, seed, proposed, seed, job, fitted, report)
            paths['mesh_obj'].write_text('changed')
            with self.assertRaises(ValueError):
                replay_implicit_artifacts(contract)

    def test_seed_face_allowance_is_applied_before_ranking(self):
        target, seed = fixture()
        other = cube([0, 0, 0], radius=.5)
        data, selection = select_existing_seed(target, {'small': seed_result(target, seed),
                                                       'other': seed_result(target, other)}, maximum_faces=11)
        self.assertIsNone(data)
        self.assertEqual(selection['reasons'][0]['reason'].split(';')[0], 'seed_triangle_allowance')

    def test_topology_changing_hole_is_scored_opaquely_and_worse_geometry_keeps_source(self):
        target, source = hole_fixture()
        request = CandidateRequest('hole', 'implicit_residual', target,
            config={'execution_approved': True, 'resolution': 32,
                    'seed_results': {'source': seed_result(target, source)}})
        with patch('reconstruction.implicit.numeric_solver.fit_implicit_field_job', side_effect=fake_fit), \
             patch('reconstruction.output_qualification.qualify_retained_output',
                   side_effect=lambda data, config: {'boundary_qualified': True,
                       'geometry_content_hash': data.content_hash, 'status': 'fixture_only'}):
            result = ImplicitResidualBackend().reconstruct(request)
        self.assertEqual(result.status, 'degraded', result.errors)
        report = result.metric_result.extras['implicit_residual']
        self.assertEqual(report['proposal_topology']['euler_characteristic'], 0)
        self.assertTrue(report['proposal_exact_boundary']['passed'])
        self.assertLess(report['opaque_proposal_per_view']['top']['area_iou'],
                        report['opaque_seed_per_view']['top']['area_iou'])
        self.assertFalse(report['full_geometry_admitted'])
        self.assertEqual(result.geometry.content_hash, source.content_hash)
        # Native evidence here is explicitly a mock. The actual opaque score,
        # exact contact guard and topology are executed numerical fixtures.

    def test_canonical_admission_track_does_not_silently_replace_public_legacy_score(self):
        target, source = fixture()
        request = CandidateRequest('track', 'implicit_residual', target,
            config={'execution_approved': True, 'projection_metric': 'pixel_area_half',
                    'seed_results': {'source': seed_result(target, source)}})
        with patch('reconstruction.implicit.numeric_solver.fit_implicit_field_job', side_effect=fake_fit):
            result = ImplicitResidualBackend().reconstruct(request)
        self.assertEqual(result.status, 'degraded', result.errors)
        report = result.metric_result.extras['implicit_residual']
        self.assertEqual(report['opaque_metric_contract'], 'opaque_triangle_union_pixel_cell_area_hard_half_v1')
        self.assertEqual(result.metric_result.per_view, report['legacy_opaque_seed_per_view'])
        self.assertEqual(report['public_selection_metric'], 'legacy_pil_polygon_endpoint_rounding_v1')
        self.assertEqual(result.geometry.content_hash, source.content_hash)

    def test_unknown_pixels_stay_excluded_from_final_candidate_metrics(self):
        target, source = fixture()
        valid = np.ones((24, 32), bool)
        valid[:, :8] = False
        base = replace(target.constraints[0], valid_mask=valid)
        left = replace(target, constraints=(base,))
        changed = np.array(base.mask)
        changed[~valid] = False
        right = replace(target, constraints=(replace(base, mask=changed),))
        records = []
        with patch('reconstruction.implicit.numeric_solver.fit_implicit_field_job', side_effect=fake_fit):
            for evidence in (left, right):
                request = CandidateRequest('unknown', 'implicit_residual', evidence,
                    config={'execution_approved': True, 'seed_results': {'source': seed_result(evidence, source)}})
                result = ImplicitResidualBackend().reconstruct(request)
                self.assertEqual(result.status, 'degraded', result.errors)
                records.append(result.metric_result)
        self.assertEqual(records[0].per_view, records[1].per_view)
        self.assertEqual(records[0].extras['implicit_residual']['opaque_proposal_per_view'],
                         records[1].extras['implicit_residual']['opaque_proposal_per_view'])

    def test_expired_candidate_budget_starts_no_numeric_work(self):
        target, source = fixture()
        request = CandidateRequest('expired', 'implicit_residual', target,
            config={'execution_approved': True, 'seed_results': {'source': seed_result(target, source)}},
            budget=CandidateBudget(timeout_s=0.))
        with patch('reconstruction.implicit.numeric_solver.fit_implicit_field_job', side_effect=AssertionError('no work')):
            result = ImplicitResidualBackend().reconstruct(request)
        self.assertEqual(result.status, 'degraded')
        self.assertEqual(result.geometry.content_hash,source.content_hash)
        self.assertIn('allowance exhausted', result.warnings[0])

    def test_optional_failures_retain_source_geometry_artifacts_and_its_existing_metrics(self):
        target,seed=fixture();backend=ImplicitResidualBackend()
        source=replace(seed_result(target,seed),mesh_path=Path('existing-source.obj'),
                       primitive_path=Path('existing-artist.json'),artifacts={'mesh_obj':Path('existing-source.obj')})
        request=CandidateRequest('unavailable',backend.name,target,
            config={'execution_approved':True,'seed_results':{'source':source}})
        for error in (RuntimeError('worker unavailable'),ImportError('optional dependency unavailable')):
            with patch('reconstruction.implicit.numeric_solver.fit_implicit_field_job',side_effect=error):
                result=backend.reconstruct(request)
            self.assertEqual(result.status,'degraded')
            self.assertEqual(result.geometry.content_hash,seed.content_hash)
            self.assertEqual(result.mesh_path,source.mesh_path)
            self.assertEqual(result.primitive_path,source.primitive_path)
            self.assertEqual(result.artifacts,source.artifacts)
            self.assertEqual(result.metric_result.per_view,source.metric_result.per_view)
            self.assertEqual(result.metric_result.extras['metrics_refer_to_output_hash'],seed.content_hash)
            self.assertFalse(result.metric_result.extras['implicit_residual']['full_geometry_admitted'])
        corrupt=lambda job:{**fake_fit(job),'retained_field_hash':'wrong'}
        with patch('reconstruction.implicit.numeric_solver.fit_implicit_field_job',side_effect=corrupt):
            result=backend.reconstruct(request)
        self.assertEqual(result.geometry.content_hash,seed.content_hash)
        self.assertIn('identity mismatch',result.warnings[0])

    def test_unavailable_pinned_helper_does_not_drop_a_valid_source(self):
        target,seed=fixture();backend=ImplicitResidualBackend()
        request=CandidateRequest('missing-helper',backend.name,target,
            config={'execution_approved':True,'seed_results':{'source':seed_result(target,seed)},'helper_python':'missing'})
        owner=SimpleNamespace(call=lambda **kwargs:{'available':False})
        with patch('reconstruction.implicit.helper.implicit_session',return_value=owner):
            result=backend.reconstruct(request)
        self.assertEqual(result.status,'degraded')
        self.assertEqual(result.geometry.content_hash,seed.content_hash)
        self.assertIn('lacks pinned',result.warnings[0])

    def test_missing_observed_seed_view_is_declined_before_field_construction(self):
        target, source = fixture()
        target = replace(target, constraints=target.constraints+(replace(target.constraints[0], view='top'),))
        result = seed_result(target, source)
        result = replace(result, metric_result=replace(result.metric_result, per_view={'front': {'passed': True}}))
        data, report = select_existing_seed(target, {'source': result})
        self.assertIsNone(data)
        self.assertEqual(report['reasons'][0]['reason'], 'required_observed_views_not_validated')

    def test_backend_requires_approval_without_loading_torch_or_starting_helpers(self):
        target, seed = fixture()
        backend = ImplicitResidualBackend()
        request = CandidateRequest('residual', backend.name, target)
        with patch('reconstruction.implicit.helper.implicit_session', side_effect=AssertionError('must stay lazy')):
            self.assertEqual(backend.reconstruct(request).status, 'skipped')
        for config in ({'resolution': True}, {'steps': True}, {'timeout_s': float('nan')}):
            self.assertTrue(backend.validate_config(config))

    def test_unqualified_proposal_retains_actual_seed_and_reports_output_metrics(self):
        target, seed = fixture()
        backend = ImplicitResidualBackend()
        with tempfile.TemporaryDirectory() as root:
            request = CandidateRequest('residual', backend.name, target,
                config={'execution_approved': True, 'seed_results': {'source': seed_result(target, seed)}},
                artifact_root=Path(root))
            with patch('reconstruction.implicit.numeric_solver.fit_implicit_field_job', side_effect=fake_fit):
                result = backend.reconstruct(request)
            self.assertEqual(result.status, 'degraded', result.errors)
            self.assertEqual(result.geometry.content_hash, seed.content_hash)
            self.assertEqual(result.metric_result.extras['metrics_refer_to_output_hash'], seed.content_hash)
            report = result.metric_result.extras['implicit_residual']
            self.assertFalse(report['full_geometry_admitted'])
            self.assertFalse(report['proposal_native_qualification']['boundary_qualified'])
            self.assertEqual(report['native_blender_render_acceptance'], 'not_run')
            replay_implicit_artifacts(report)


if __name__ == '__main__':
    unittest.main()
