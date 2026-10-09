"""Standalone fallback ownership admission; no child, native kernel or fit launches."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from blender_blocking.utils import owned_process_supervisor as supervisor
import numpy as np

from blender_blocking.reconstruction import process_executor as executor
from blender_blocking.reconstruction.grouped_solids import production_union
from blender_blocking.reconstruction.native_geometry import GeometryArrays
from blender_blocking.reconstruction.types import (Bounds3D, CandidateMetrics, CandidateRequest,
    CandidateResult, ReconstructionTarget)
from blender_blocking.placement.resfit.config import ResFitPipelineConfig, ResFitPipelineResult
from blender_blocking.placement.resfit_objective import ResFitObjectiveResult


class MapPool:
    def __init__(self, workers=2, *, process_budget=None, outcome=None):
        self.max_workers = workers
        self.process_budget = process_budget
        self.root = Path('unused-fallback-budget-fixture')
        self.outcome = outcome or executor.JobOutcome('failed', error='fixture: no native execution')
        self.calls = []
        self.enters = self.closes = 0
        self._budget_deadline = 777.

    def __enter__(self):
        self.enters += 1
        return self

    def __exit__(self, *args):
        self.closes += 1

    def map(self, jobs, **kwargs):
        self.calls.append((jobs, kwargs))
        return [self.outcome for job in jobs]


class TestFallbackProcessBudget(unittest.TestCase):
    def setUp(self):
        # These fixtures never launch workers; model only the supported
        # capability gate while keeping real numeric resource validation.
        capability = patch.object(supervisor, 'os',
            SimpleNamespace(name='nt', PathLike=os.PathLike))
        capability.start()
        self.addCleanup(capability.stop)
        self.budget = executor.WorkerProcessBudget(25., 1073741824,
            max_rss_bytes=536870912, join_timeout_s=2., max_restarts=1)
        self.context = SimpleNamespace(worker_process_budget=self.budget)
        worker = patch.object(executor, '_WORKER_CLIENT', None)
        worker.start();self.addCleanup(worker.stop)
        self.points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.data = GeometryArrays.capture(self.points, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])

    def factory(self, outcome=None):
        pools = []
        def create(*args, **kwargs):
            pool = MapPool(*args, outcome=outcome, **kwargs);pools.append(pool);return pool
        return pools, patch.object(executor, 'PersistentProcessExecutor', side_effect=create)

    def request(self, backend, config=None, context=None):
        return CandidateRequest('fallback', backend, ReconstructionTarget(
            bounds=Bounds3D(-1., 1., -1., 1., -1., 1.)), config=config or {},
            context=self.context if context is None else context)

    def fit_result(self):
        return ResFitPipelineResult((), ResFitObjectiveResult(2., {}),
            ResFitObjectiveResult(1., {}), (), (), objective_evaluations=1)

    def fit_config(self):
        config = ResFitPipelineConfig()
        return replace(config, optimizer=replace(config.optimizer, iterations=0,
            max_objective_evaluations=8))

    def test_native_union_explicit_and_context_budget_create_only_one_pool(self):
        for options in ({'process_budget': self.budget}, {'context': self.context}):
            with self.subTest(options=options):
                pools, factory = self.factory()
                with factory as construction, patch('blender_blocking.reconstruction.grouped_solids.balanced_union',
                        return_value=('fixture-result', {})) as union:
                    result, report = production_union(['fixture'], {'native_union_execution': True},
                        timeout_s=4., feature_thickness=.1, **options)
                construction.assert_called_once_with(2, process_budget=self.budget)
                self.assertIs(union.call_args.kwargs['executor'], pools[0])
                self.assertEqual(union.call_args.kwargs['timeout_s'], 4.)
                self.assertEqual(union.call_args.kwargs['feature_thickness'], .1)
                self.assertTrue(report['native_queue_used'])
                self.assertEqual(result, 'fixture-result')
                self.assertEqual((pools[0].enters, pools[0].closes), (1, 1))

    def test_union_disabled_and_shared_defaults_do_not_create_or_adopt_pools(self):
        pool = MapPool(process_budget=self.budget)
        with patch.object(executor, 'PersistentProcessExecutor') as factory, \
             patch('blender_blocking.reconstruction.grouped_solids.balanced_union', return_value=('fixture', {})) as union:
            _, report = production_union([], {}, process_budget=self.budget)
            self.assertFalse(report['native_queue_used'])
            self.assertIsNone(union.call_args.kwargs['executor'])
            production_union([], {'native_union_execution': True}, executor=pool, process_budget=self.budget)
            self.assertIs(union.call_args.kwargs['executor'], pool)
            with self.assertRaises(ValueError):
                production_union([], {'native_union_execution': True}, executor=MapPool(), process_budget=self.budget)
            factory.assert_not_called()
        self.assertEqual((pool.enters, pool.closes, pool._budget_deadline), (0, 0, 777.))

    def test_multistart_direct_budget_default_and_context_route_real_job_payload(self):
        from blender_blocking.placement.resfit.optimizer import fit_residual_primitives_multistart
        outcome = executor.JobOutcome('success', self.fit_result())
        for options, budget in (({}, None), ({'process_budget': self.budget}, self.budget),
                                ({'context': self.context}, self.budget)):
            with self.subTest(options=options):
                pools, factory = self.factory(outcome)
                with factory as construction:
                    result = fit_residual_primitives_multistart(self.points, self.fit_config(),
                        max_attempts=1, **options)
                construction.assert_called_once_with(2, **({'process_budget': budget} if budget else {}))
                self.assertEqual(pools[0].calls[0][0][0][0], 'fit_start')
                self.assertIs(pools[0].calls[0][0][0][1]['target_points'], self.points)
                self.assertEqual(result.final_loss.total, 1.)
                self.assertEqual((pools[0].enters, pools[0].closes), (1, 1))

    def test_multistart_owned_worker_client_stays_flat_and_mismatch_refuses(self):
        from blender_blocking.placement.resfit.optimizer import fit_residual_primitives_multistart
        client = executor.WorkerClient('unused-fallback-budget-fixture', 0)
        client.bind_process_budget(self.budget)
        with patch.object(executor, '_WORKER_CLIENT', client), patch.object(executor, 'PersistentProcessExecutor') as factory, \
             patch.object(client, 'map', return_value=[executor.JobOutcome('success', self.fit_result())]) as mapping:
            fit_residual_primitives_multistart(self.points, self.fit_config(), max_attempts=1,
                process_budget=self.budget)
            mapping.assert_called_once()
            with self.assertRaises(ValueError):
                fit_residual_primitives_multistart(self.points, self.fit_config(), max_attempts=1,
                    process_budget=replace(self.budget, wall_s=20.))
            factory.assert_not_called()
        self.assertIs(client.process_budget, self.budget)

    def test_family_search_reuses_one_budget_across_all_families(self):
        from blender_blocking.placement.resfit import backend_adapter as adapter
        pools, factory = self.factory()
        with factory as construction, patch.object(adapter, '_profile_initial_primitives', return_value=(None, '')), \
             patch.object(adapter, 'fit_residual_primitives_multistart', return_value=self.fit_result()) as fitting:
            result = adapter._fit_best_primitive_family(surface=self.points, base_config=self.fit_config(),
                primitive_families=['ellipsoid', 'superfrustum'], profile_rows=[],
                init_config=self.fit_config().initialization, occupied_points=self.points,
                silhouette_hook=None, topology_penalty_hook=None, constraint_penalty_hook=None,
                uncertainty_penalty_hook=None, max_attempts=1, share_budget_across_families=True,
                context=self.context)
        construction.assert_called_once_with(2, process_budget=self.budget)
        self.assertEqual(fitting.call_count, 2)
        self.assertTrue(all(call.kwargs['executor'] is pools[0] for call in fitting.call_args_list))
        self.assertEqual(len(result[0].family_attempts), 2)
        self.assertEqual((pools[0].enters, pools[0].closes, pools[0]._budget_deadline), (1, 1, 777.))

    def test_public_resfit_request_forwards_context_before_failed_fit(self):
        from blender_blocking.placement.resfit import backend_adapter as adapter
        from test_quality_geometry import target_for_masks
        points = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
        target = target_for_masks({v: np.ones((8, 8), bool) for v in ('front', 'side', 'top')})
        request = replace(self.request('primitive_fit_refine', {'primitive_families': ['ellipsoid'],
            'optimization_steps': 0, 'residual_rounds': 0, 'primitive_count': 1}), target=target)
        with patch('reconstruction.point_cloud.target_surface_points', return_value=(points, {})), \
             patch('reconstruction.point_cloud.target_occupied_points', return_value=(points, {})), \
             patch.object(adapter, '_fit_best_primitive_family', side_effect=RuntimeError('fixture stopped before fitting')) as fitting, \
             patch.object(executor, 'PersistentProcessExecutor') as factory:
            result = adapter.run_primitive_fit_pipeline(request)
        self.assertEqual(result.status, 'failed')
        self.assertIs(fitting.call_args.kwargs['context'], self.context)
        factory.assert_not_called()

    def test_program_search_admits_context_budget_without_compilation(self):
        from blender_blocking.reconstruction.backends.shape_program.geometry_search import geometric_program_search
        from blender_blocking.primitives.shape_program import ShapeNode, ShapeProgram, validate_compilable_program
        seed = ShapeProgram('shape-program-v1', 'budget-fixture', (ShapeNode('budget-box', 'add', 'box',
            parameters={'width_world': 1., 'depth_world': 1., 'height_world': 1.}),))
        self.assertFalse(validate_compilable_program(seed))
        request = self.request('shape_program', {'routed_program': seed, 'program_workers': 2})
        pools, factory = self.factory()
        with factory as construction, patch('blender_blocking.primitives.program_search.search_shape_program_candidates',
                return_value=SimpleNamespace(candidates=())):
            program, best, receipt = geometric_program_search(request, seed)
        construction.assert_called_once_with(2, process_budget=self.budget)
        self.assertIsNone(best)
        self.assertFalse(receipt['geometry_scored'])
        self.assertEqual(pools[0].calls[0][0][0][0], 'program_geometry')
        self.assertIsNone(pools[0].calls[0][0][0][1][0].context)

    def test_dvx_fallback_budget_and_warm_helper_remain_distinct(self):
        from contextlib import ExitStack
        from blender_blocking.reconstruction.differentiable import dvx_adapter as adapter
        request = self.request('differentiable_refine', {'dvx_execution_approved': True, 'dvx_resolution': 16})
        pools, factory = self.factory()
        with ExitStack() as patches:
            construction = patches.enter_context(factory)
            patches.enter_context(patch.object(adapter, 'dependency_state', return_value={'available': True}))
            patches.enter_context(patch('blender_blocking.reconstruction.differentiable.seed_selection.select_existing_seed',
                return_value=(self.data, {})))
            patches.enter_context(patch('reconstruction.visibility.point_support', return_value=(None, np.zeros(4))))
            patches.enter_context(patch('blender_blocking.reconstruction.differentiable.ray_evidence.prepare_coverage_target',
                return_value={'target_zyx': np.zeros((1, 1, 1)), 'target_valid_zyx': np.ones((1, 1, 1))}))
            result = adapter.run_candidate(request)
        construction.assert_called_once_with(1, process_budget=self.budget)
        self.assertEqual(result.status, 'failed')
        self.assertEqual(pools[0].calls[0][0][0][0], 'dvx_fit')
        warm = replace(request, config={**request.config, 'dvx_helper_python': 'unused-fixture', 'dvx_warm_helper': True})
        with patch.object(executor, 'PersistentProcessExecutor') as construction, \
             patch.object(adapter, 'helper_call', side_effect=RuntimeError('fixture unavailable warm helper')):
            skipped = adapter.run_candidate(warm)
        self.assertEqual(skipped.status, 'skipped')
        construction.assert_not_called()

    def test_hybrid_context_budget_only_opens_native_enabled_residual_scope(self):
        from blender_blocking.reconstruction.backends.hybrid_loft_hull import HybridLoftHullBackend
        seeds = {name: CandidateResult(name, name, 'success', geometry=self.data)
                 for name in ('profile_loft', 'visual_hull_voxel')}
        metrics = {'front': {'area_iou': .98, 'boundary_iou': .94, 'passed': True}}
        class NativeFixture:
            def __init__(self, data, name):
                self.data = data
                self.obj = MagicMock()
            def attach(self):
                return self.obj
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                request = self.request('hybrid_loft_hull', {'seed_results': seeds,
                    'hybrid_residual_parts': 0, 'native_union_execution': enabled})
                pools, factory = self.factory()
                with factory as construction, patch.dict(sys.modules, {'mathutils': SimpleNamespace(Vector=tuple)}), \
                     patch('blender_blocking.reconstruction.native_csg.boolean_mesh', return_value=(self.data, {})), \
                     patch('blender_blocking.reconstruction.grouped_solids.solid_guard', return_value={'valid_solid': True}), \
                     patch('blender_blocking.reconstruction.projected_metrics.projected_mesh_metrics', return_value=metrics), \
                     patch('blender_blocking.reconstruction.feature_evidence.mesh_empty_features', return_value={'passed': True}), \
                     patch('blender_blocking.reconstruction.native_geometry.NativeOwnedGeometry', NativeFixture):
                    result = HybridLoftHullBackend().reconstruct(request)
                self.assertEqual(result.status, 'success')
                if enabled:
                    construction.assert_called_once_with(2, process_budget=self.budget)
                    self.assertEqual((pools[0].enters, pools[0].closes), (1, 1))
                else:
                    construction.assert_not_called()


if __name__ == '__main__':
    unittest.main()
