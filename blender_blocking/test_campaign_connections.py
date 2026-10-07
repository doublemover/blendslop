"""Cheap contracts for declared experiments and observation transport."""
import sys, unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from test_quality_geometry import target_for_masks


class CampaignConnectionTests(unittest.TestCase):
    def test_quality_reaches_actual_multistart_fitter(self):
        from config import BlockingConfig
        from main_integration import BlockingWorkflow
        from placement.resfit import backend_adapter as adapter
        from reconstruction.types import CandidateRequest
        from blender_blocking.reconstruction.process_executor import JobOutcome, execute_job
        from blender_blocking.reconstruction.option_receipts import ConsumedOptions
        class InlineExecutor:
            def map(self, jobs, **kwargs):
                return [JobOutcome('success', execute_job(kind, payload)) for kind, payload, _ in jobs]
        points = np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
        target = target_for_masks({view:np.ones((8,8),bool) for view in ('front','side','top')})
        for preset, expected in [('default',2),('quality',4)]:
            cfg = BlockingConfig(); cfg.reconstruction.quality_preset = preset
            options = BlockingWorkflow(config=cfg)._config_for_backend('primitive_fit_refine')
            options.update(primitive_families=['ellipsoid'],optimization_steps=0,residual_rounds=0,primitive_count=1)
            options = ConsumedOptions(options)
            request = CandidateRequest('contract','primitive_fit_refine',target,options)
            with patch('reconstruction.point_cloud.target_surface_points',return_value=(points,{})), \
                 patch('reconstruction.point_cloud.target_occupied_points',return_value=(points,{})), \
                 patch('blender_blocking.reconstruction.process_executor.current_worker_client',return_value=InlineExecutor()), \
                 patch.object(adapter,'fit_residual_primitives_multistart',wraps=adapter.fit_residual_primitives_multistart) as fitter:
                adapter.run_primitive_fit_pipeline(request)
            self.assertTrue(fitter.called)
            self.assertEqual(fitter.call_args.kwargs['max_attempts'],expected)
            self.assertIn('max_multistart_attempts',options.reads)

    def test_performance_worker_transports_variant_and_overlay(self):
        from argparse import Namespace
        from run_native_performance_phase import candidate_worker_arguments
        args = Namespace(case='box', seed=77, variant='cpu_dvx', config_overlay=Path('cpu.json'))
        candidate = candidate_worker_arguments(args, Path('cold'))
        self.assertEqual(candidate.experiment, 'cpu_dvx')
        self.assertEqual(candidate.config_overlay, args.config_overlay)
        self.assertEqual(candidate.mode, 'ensemble')
        args.config_overlay = None
        self.assertIsNone(candidate_worker_arguments(args, Path('cold')).experiment)

    def test_overlay_rejects_ignored_setting(self):
        from config import BlockingConfig
        from run_improvement_pass import apply_config_overlay
        with self.assertRaises(ValueError):
            apply_config_overlay(BlockingConfig(),{'primitive_fit':{'max_attempts':4}})

    def test_selected_ensemble_assembly_uses_its_own_parameters(self):
        import json,tempfile
        from verify_candidate_exports import selected_backend,primitive_parameter_sources
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'result.json'
            source.write_text(json.dumps({'backend_result':{'selected':{
                'primitive_path':str(root/'selected-parts.json'),'backend_name':'primitive_fit_refine'}}}))
            row = {'requested_mode':'ensemble','selected_backend':'primitive_fit_refine','result_path':str(source)}
            self.assertEqual(selected_backend(row),'primitive_fit_refine')
            self.assertEqual(primitive_parameter_sources(row),[root/'selected-parts.json'])

    def test_dvx_overlay_reaches_workflow(self):
        from config import BlockingConfig
        from main_integration import BlockingWorkflow
        from run_improvement_pass import apply_config_overlay
        cfg = apply_config_overlay(BlockingConfig(),{'differentiable_render':{'backend':'dvx',
            'dvx_execution_approved':True,'dvx_helper_python':'cpu-helper','dvx_steps':12}})
        payload = BlockingWorkflow(config=cfg)._config_for_backend('differentiable_refine')
        self.assertEqual(payload['backend'],'dvx')
        self.assertTrue(payload['dvx_execution_approved'])
        self.assertEqual(payload['dvx_steps'],12)

    def test_main_workflow_crops_pixels_validity_and_camera_together(self):
        from config import BlockingConfig
        from main_integration import BlockingWorkflow
        cfg = BlockingConfig(); cfg.reconstruction.view_crops = {'front':[2,1,6,7]}
        cfg.reconstruction.view_calibration = {'front':{'world_bounds':[-2,2,-2,2],'axes':[0,2]}}
        valid = np.ones((8,8),bool); valid[:,4:] = False
        workflow = BlockingWorkflow(config=cfg,valid_evidence_masks={'front':valid})
        workflow.views = {'front':np.full((8,8,4),255,np.uint8)}
        workflow._prepare_evidence_inputs()
        self.assertEqual(workflow.views['front'].shape,(6,4,4))
        np.testing.assert_array_equal(workflow.valid_evidence_masks['front'],valid[1:7,2:6])
        np.testing.assert_allclose(cfg.reconstruction.view_calibration['front']['world_bounds'],[-1,1,-1.5,1.5])
        workflow._prepare_evidence_inputs()
        self.assertEqual(workflow.views['front'].shape,(6,4,4))

    def test_main_target_receives_validity(self):
        from config import BlockingConfig
        from main_integration import BlockingWorkflow
        cfg = BlockingConfig(); cfg.reconstruction.view_calibration = {
            view:{'world_bounds':[-1,1,-1,1],'axes':axes} for view,axes in [('front',[0,2]),('side',[1,2]),('top',[0,1])]}
        images = np.zeros((16,16,4),np.uint8); images[3:13,3:13] = 255
        valid = np.ones((16,16),bool); valid[:,10:] = False
        workflow = BlockingWorkflow(config=cfg,valid_evidence_masks={'front':valid})
        workflow.views = {view:images.copy() for view in ('front','side','top')}
        target = workflow.build_reconstruction_target().target
        constraint = next(c for c in target.constraints if c.view == 'front')
        np.testing.assert_array_equal(constraint.valid_mask,valid)
        self.assertFalse(np.asarray(constraint.mask)[:,10:].any())

    def test_unknown_region_cannot_become_observed_hole(self):
        from reconstruction.projection_contract import observed_holes
        mask = np.ones((8,8),bool); mask[2:6,2:6] = False
        target = target_for_masks({'front':mask})
        self.assertEqual(observed_holes(target)['front'],16)
        valid = np.ones_like(mask); valid[3,3] = False
        target = replace(target,constraints=(replace(target.constraints[0],valid_mask=valid),))
        self.assertEqual(observed_holes(target)['front'],0)

    def test_external_evidence_accepts_unobserved_view_without_fake_average(self):
        from blender_blocking.e2e.evidence import attach_evaluation_evidence
        payload = {'validation_mode':'render-iou','views':{
            'front':{'area_iou':.8,'boundary_iou':.6,'signed_distance_loss':.01,'required':True,'passed':True},
            'top':{'area_iou':None,'boundary_iou':None,'signed_distance_loss':None,'required':False,'passed':True}},
            'average_iou':.8,'min_view_iou':.8}
        result = attach_evaluation_evidence(payload)
        self.assertIsNone(result['views']['top']['area_iou'])

    def test_unknown_contour_score_excluded(self):
        from reconstruction.visibility import observed_area_score
        self.assertEqual(observed_area_score([{'area_iou':None},{'area_iou':.8}]),.8)
        self.assertEqual(observed_area_score([{'area_iou':None}]),0.)

    def test_consumption_distinguishes_unused_switch(self):
        from blender_blocking.reconstruction.option_receipts import reconstruct_with_receipt
        from reconstruction.types import CandidateRequest,CandidateResult
        class Consumer:
            def reconstruct(self,request):
                request.config.get('used')
                return CandidateResult('r','contract','success')
        request = CandidateRequest('r','contract',target_for_masks({'front':np.ones((4,4),bool)}),{'used':True,'ignored':True})
        receipt = reconstruct_with_receipt(Consumer(),request).metric_result.extras['option_consumption']
        self.assertEqual(receipt['consumed_keys'],['used'])
        self.assertEqual(receipt['unconsumed_keys'],['ignored'])

    def test_native_assets_activate_selected_metric(self):
        import evaluate_protocol_campaign as campaign
        config = {'reference':'native.obj','source_sha256':{'reference':'verified'},
            'reference_metadata':{'upstream_preparation_receipt':{'target_sha256':'verified'}}}
        with patch.object(campaign,'file_hash',return_value='verified'), \
             patch.object(campaign,'evaluate_mesh_profile',return_value={'status':'available'}) as evaluator:
            result = campaign.evaluate_native_profile('target.obj','prediction.obj','superfit_native_v1',config,77,Path('unused'))
        self.assertEqual(result['status'],'available')
        self.assertEqual(evaluator.call_args.kwargs['profile'],'superfit_native_v1')
        self.assertEqual(evaluator.call_args.args[0].name,'native.obj')

    def test_missing_dtu_affects_its_track_only(self):
        import evaluate_protocol_campaign as campaign
        result = campaign.evaluate_dtu_track({'case':'example'},None,'dp_gs',Path('unused'))
        self.assertEqual(result['status'],'unavailable')
        self.assertIn('DTU',result['reason'])


if __name__ == '__main__':
    unittest.main()
