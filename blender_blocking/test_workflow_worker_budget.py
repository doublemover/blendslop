"""Real workflow resource opt-in without worker/native allocation."""
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch
from blender_blocking.utils import owned_process_supervisor as supervisor

from config import BlockingConfig, EnsembleConfig
from blender_blocking.reconstruction.process_executor import WorkerProcessBudget


class WorkflowWorkerBudgetTests(unittest.TestCase):
    def setUp(self):
        # These fixtures never launch workers; model only the supported
        # capability gate while keeping real numeric resource validation.
        capability = patch.object(supervisor, 'os',
            SimpleNamespace(name='nt', PathLike=os.PathLike))
        capability.start()
        self.addCleanup(capability.stop)

    @staticmethod
    def declaration():
        return {'wall_s': 25., 'max_memory_bytes': 1073741824,
                'max_rss_bytes': 536870912, 'join_timeout_s': 2., 'max_restarts': 1}

    def test_none_preserves_default_workflow_and_config_serialization(self):
        cfg = BlockingConfig()
        cfg.validate()
        self.assertIsNone(cfg.ensemble.make_worker_process_budget())
        self.assertIsNone(cfg.to_dict()['ensemble']['worker_process_budget'])

    def test_explicit_mapping_roundtrip_has_no_alias_or_inferred_caps(self):
        cfg = EnsembleConfig(worker_process_budget=self.declaration())
        cfg.validate()
        self.assertEqual(cfg.make_worker_process_budget(), WorkerProcessBudget(**self.declaration()))
        serialized = json.loads(json.dumps(cfg.to_dict()))
        other = EnsembleConfig(worker_process_budget=serialized['worker_process_budget'])
        self.assertEqual(other.make_worker_process_budget(), cfg.make_worker_process_budget())
        serialized['worker_process_budget']['wall_s'] = 20.
        self.assertEqual(cfg.worker_process_budget['wall_s'], 25.)

    def test_invalid_or_incomplete_declarations_fail_without_pool_allocation(self):
        invalid = [False, 'automatic', {}, {'wall_s': 25.},
            {**self.declaration(), 'wall_s': True}, {**self.declaration(), 'wall_s': math.inf},
            {**self.declaration(), 'max_memory_bytes': 1},
            {**self.declaration(), 'max_restarts': True},
            {**self.declaration(), 'guess_caps': True}]
        with patch('blender_blocking.reconstruction.process_executor.PersistentProcessExecutor') as pool:
            for declaration in invalid:
                with self.subTest(declaration=declaration), self.assertRaises(ValueError):
                    EnsembleConfig(worker_process_budget=declaration).validate()
            pool.assert_not_called()

    def test_main_workflow_forwards_declared_budget_to_real_request_context(self):
        import blender_blocking.main_integration as integration
        from blender_blocking.reconstruction.ensemble import EnsembleRunResult
        from blender_blocking.reconstruction.types import ReconstructionTarget
        cfg = BlockingConfig()
        cfg.ensemble.worker_process_budget = self.declaration()
        workflow = object.__new__(integration.BlockingWorkflow)
        workflow.config = cfg
        workflow.context = None
        workflow.views = {}
        result = EnsembleRunResult(None, (), (), 'fixture')
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(integration, 'register_builtin_backends'), \
             patch.object(integration, 'EnsembleRunner') as runner, \
             patch.object(workflow, '_backend_name_for_mode', return_value='fixture'), \
             patch.object(workflow, 'build_reconstruction_target', return_value=SimpleNamespace(target=ReconstructionTarget())), \
             patch.object(workflow, '_artifact_root', return_value=Path(folder)), \
             patch.object(workflow, '_configured_ensemble_candidates', return_value=()), \
             patch.object(workflow, '_record_backend_manifest'):
            runner.return_value.run.return_value = result
            self.assertIsNone(workflow.run_backend_reconstruction(mode='ensemble'))
            context = runner.return_value.run.call_args.kwargs['context']
            self.assertEqual(context.worker_process_budget, WorkerProcessBudget(**self.declaration()))
            self.assertIs(workflow.reconstruction_result, result)


if __name__ == '__main__':
    unittest.main()
