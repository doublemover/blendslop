"""Correctness regressions for failed receipts and initialization/deformation contracts."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class CorrectiveRecoveryTests(unittest.TestCase):
    def test_missing_renders_are_unavailable_not_zero_or_perfect(self):
        from scripts.run_improvement_pass import measure_saved_views
        payload = {'failure_code': 'missing_renderable_mesh', 'error': 'worker deadline exceeded'}
        result = measure_saved_views(payload, {'views': {}})
        self.assertFalse(result['passed'])
        self.assertIsNone(result['average_iou'])
        self.assertIsNone(result['min_view_iou'])
        self.assertEqual(set(result['unavailable_views']), {'front', 'side', 'top'})
        self.assertEqual(payload['error'], 'worker deadline exceeded')

    def test_missing_optional_view_does_not_fail_known_views(self):
        from PIL import Image
        from scripts.run_improvement_pass import measure_saved_views
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'mask.png'
            pixels = np.zeros((24, 24, 4), dtype=np.uint8)
            pixels[6:18, 6:18, 3] = 255
            Image.fromarray(pixels).save(path)
            result = measure_saved_views({'rendered_paths': {'front': str(path)}},
                {'views': {'front': str(path)}, 'required_views': ['front']})
        self.assertTrue(result['passed'])
        self.assertEqual(result['average_iou'], 1.)

    def test_guard_failure_is_distinct_from_missing_toolchain(self):
        from blender_blocking.reconstruction.native_geometry import GeometryArrays
        from blender_blocking.reconstruction.native_qualification import qualify_geometry
        data = GeometryArrays.capture([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
        with patch('blender_blocking.reconstruction.native_qualification.toolchain_identity') as identity:
            result = qualify_geometry(data, python='unused')
            identity.assert_not_called()
        self.assertEqual(result['reason'], 'topology_or_volume_guard_failed')
        self.assertFalse(result['single_solid_qualified'])
        self.assertEqual(result['triangle_count'], 1)

    def test_partial_fit_counts_recorded_work_separately_from_reserved_allowance(self):
        from dataclasses import replace
        from blender_blocking.placement.resfit.optimizer import fit_residual_primitives, fit_residual_primitives_multistart
        from blender_blocking.placement.resfit.config import ResFitPipelineConfig
        from blender_blocking.reconstruction.process_executor import JobOutcome
        config = ResFitPipelineConfig()
        config = replace(config, optimizer=replace(config.optimizer, iterations=0, max_objective_evaluations=8))
        points = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
        scored = fit_residual_primitives(points, config)

        class Executor:
            def map(self, jobs, timeout_s=None):
                return [JobOutcome('timeout', scored, 'worker deadline exceeded',
                    partial=True, recorded_evaluations=scored.objective_evaluations,
                    stop_reason='worker deadline exceeded')]

        retained = fit_residual_primitives_multistart(points, config, max_attempts=1, executor=Executor())
        self.assertEqual(retained.objective_evaluations, scored.objective_evaluations)
        self.assertGreater(retained.search_budget['reserved_evaluation_charge'], retained.objective_evaluations)
        self.assertTrue(retained.search_budget['unreported_work_possible'])
        self.assertEqual(retained.final_loss.total, scored.final_loss.total)

    def test_dvx_seed_and_final_state_are_replayed_separately(self):
        from blender_blocking.reconstruction.differentiable.dvx_adapter import run_candidate
        from blender_blocking.reconstruction.differentiable.dvx_artifacts import replay_dvx_artifacts
        from blender_blocking.reconstruction.process_executor import JobOutcome
        from blender_blocking.reconstruction.types import CandidateRequest, CandidateBudget, Bounds3D, ReconstructionTarget
        from blender_blocking.primitives.analytic_primitives import EllipsoidPrimitive
        import json

        class Executor:
            def map(self, jobs, timeout_s=None):
                payload = jobs[0][1]
                return [JobOutcome('success', {'vertices': payload['vertices'] + [.001, 0, 0],
                    'faces': payload['faces'], 'history': [1., .5],
                    'fixed_transform': {'center': payload['center'].tolist(), 'scale': payload['scale']},
                    'requested_steps': 1, 'optimizer_updates': 1, 'objective_evaluations': 2,
                    'best_evaluation': 2, 'stop_reason': 'requested_steps',
                    'final_update_evaluated': True, 'partial': False})]

        points = np.random.default_rng(123).uniform(-.4, .4, (32, 3))
        target = ReconstructionTarget(bounds=Bounds3D.from_min_max((-1.,)*3, (1.,)*3))
        score = lambda i: {'front': {'area_iou': i, 'boundary_iou': i, 'passed': True}}
        with tempfile.TemporaryDirectory() as root:
            request = CandidateRequest('dvx-replay', 'differentiable_refine', target,
                artifact_root=Path(root), budget=CandidateBudget(timeout_s=10.),
                config={'dvx_execution_approved': True, 'dvx_resolution': 16},
                context=SimpleNamespace(process_executor=Executor()))
            with patch('blender_blocking.reconstruction.differentiable.dvx_adapter.dependency_state', return_value={'available': True}), \
                 patch('reconstruction.point_cloud.target_surface_points', return_value=(points, {})), \
                 patch('reconstruction.visibility.point_support', side_effect=lambda target, p: (np.ones(len(p)), np.ones(len(p)))), \
                 patch('reconstruction.projected_metrics.projected_mesh_metrics', side_effect=[score(.5), score(.6)]):
                result = run_candidate(request)
            self.assertTrue(result.succeeded, result.errors)
            self.assertTrue(result.primitive_path.is_file())
            payload = json.loads(result.primitive_path.read_text())
            primitives = [EllipsoidPrimitive.from_dict(p) for p in payload['primitives']]
            replay = replay_dvx_artifacts(payload, primitives)
            self.assertFalse(replay['receipt']['seed_parameters_reproduce_final_deformation'])
            self.assertNotEqual(replay['seed'].content_hash, replay['retained'].content_hash)
            self.assertEqual(replay['retained'].content_hash, result.geometry.content_hash)
            self.assertFalse(result.metric_result.extras['single_solid_qualified'])
            metadata = payload['metadata']
            self.assertEqual(sum(p['face_count'] for p in metadata['part_ranges']), len(result.geometry.faces))
            self.assertEqual(sum(p['vertex_count'] for p in metadata['part_ranges']), len(result.geometry.vertices))
            with np.load(metadata['deformation_state_path'], allow_pickle=False) as values:
                seed, retained, faces = values['seed_vertices'], values['retained_vertices'], values['faces']
            np.savez(metadata['deformation_state_path'], seed_vertices=seed,
                     retained_vertices=retained + .01, faces=faces)
            with self.assertRaisesRegex(ValueError, 'output identity'):
                replay_dvx_artifacts(payload, primitives)


if __name__ == '__main__':
    unittest.main()
