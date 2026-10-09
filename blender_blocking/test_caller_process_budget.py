"""Caller pool-budget guards using real routing/transport and no child launches."""
from dataclasses import asdict, replace
from pathlib import Path
import pickle
from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch
from blender_blocking.utils import owned_process_supervisor as supervisor

from blender_blocking.reconstruction import process_executor as module
from blender_blocking.reconstruction.ensemble import CandidateConfig, EnsembleRunner, EnsembleRunResult
from blender_blocking.reconstruction.measured_selection import routing_run
from blender_blocking.reconstruction.types import CandidateMetrics, CandidateRequest, CandidateResult, ReconstructionTarget


class RecordingPool:
    def __init__(self, workers=2, *, process_budget=None):
        self.max_workers = workers
        self.process_budget = process_budget
        self.root = Path('unused-budget-fixture')
        self.payloads = []
        self.enters = self.closes = 0
        self._budget_deadline = 12345.
        self._launch_count = 3

    def __enter__(self):
        self.enters += 1
        return self

    def __exit__(self, *args):
        self.closes += 1

    def submit(self, kind, payload, **options):
        self.payloads.append((kind, payload, options))
        return len(self.payloads) - 1

    def result(self, job):
        request = self.payloads[job][1][0]
        result = CandidateResult(request.candidate_id, request.backend_name, 'success',
            metric_result=CandidateMetrics(per_view={'front': {
                'area_iou': .98, 'boundary_iou': .94, 'passed': True}},
                extras={'selection_evidence': 'fresh_blender_render'}))
        return module.JobOutcome('success', result)


class TestCallerProcessBudget(unittest.TestCase):
    def setUp(self):
        # These fixtures never launch workers; model only the supported
        # capability gate while keeping real numeric resource validation.
        capability = patch.object(supervisor, 'os',
            SimpleNamespace(name='nt', PathLike=os.PathLike))
        capability.start()
        self.addCleanup(capability.stop)
        self.budget = module.WorkerProcessBudget(25., 1073741824,
            max_rss_bytes=536870912, join_timeout_s=2., max_restarts=1)
        self.worker = patch.object(module, '_WORKER_CLIENT', None)
        self.worker.start()
        self.addCleanup(self.worker.stop)

    def request(self, context=None):
        return CandidateRequest('fixture', 'fixture', ReconstructionTarget(), context=context)

    @staticmethod
    def run_result(results, requests, **options):
        return EnsembleRunResult(results[0] if results else None, tuple(results), (), 'fixture')

    def test_default_and_context_budget_reach_constructor_without_inferred_caps(self):
        with patch.object(module, 'PersistentProcessExecutor', side_effect=RecordingPool) as factory:
            default = module.executor_scope(3)
            factory.assert_called_once_with(3)
            self.assertIsNone(default.process_budget)
            contextual = module.executor_scope(2, context=SimpleNamespace(worker_process_budget=self.budget))
            self.assertIs(contextual.process_budget, self.budget)
            self.assertEqual(factory.call_args.kwargs, {'process_budget': self.budget})

    def test_shared_budget_identity_does_not_enter_close_or_reset_allowance(self):
        pool = RecordingPool(process_budget=self.budget)
        state = (pool._budget_deadline, pool._launch_count)
        with patch.object(module, 'PersistentProcessExecutor') as factory:
            with module.executor_scope(context=SimpleNamespace(process_executor=pool,
                    worker_process_budget=self.budget), require_submit_result=True) as shared:
                self.assertIs(shared, pool)
            factory.assert_not_called()
        self.assertEqual((pool.enters, pool.closes), (0, 0))
        self.assertEqual(state, (pool._budget_deadline, pool._launch_count))

    def test_invalid_conflicting_and_unowned_shared_budgets_refuse_before_allocation(self):
        different = replace(self.budget, wall_s=20.)
        with patch.object(module, 'PersistentProcessExecutor') as factory:
            for kwargs in ({'process_budget': True}, {'process_budget': module.WorkerProcessBudget(1., 0)},
                    {'process_budget': self.budget, 'context': SimpleNamespace(worker_process_budget=different)},
                    {'process_budget': self.budget, 'executor': RecordingPool()},
                    {'process_budget': self.budget, 'executor': RecordingPool(process_budget=different)}):
                with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                    module.executor_scope(**kwargs)
            factory.assert_not_called()

    def test_all_candidate_contexts_must_declare_one_allowance_even_with_no_launch(self):
        first = self.request(SimpleNamespace(worker_process_budget=self.budget))
        other = replace(first, candidate_id='other', context=SimpleNamespace(
            worker_process_budget=replace(self.budget, max_restarts=2)))
        with patch.object(module, 'PersistentProcessExecutor') as factory:
            with self.assertRaises(ValueError):
                EnsembleRunner().run_requests([first, other])
            with self.assertRaises(TypeError):
                EnsembleRunner().run_requests([], process_budget=False)
            factory.assert_not_called()

    def test_ensemble_and_evidence_routing_share_the_same_explicit_pool(self):
        for measured in (False, True):
            with self.subTest(measured=measured):
                pools = []
                def create(*args, **kwargs):
                    pool = RecordingPool(*args, **kwargs);pools.append(pool);return pool
                runner = EnsembleRunner(evidence_routing=measured)
                with patch.object(module, 'PersistentProcessExecutor', side_effect=create) as factory, \
                     patch.object(runner, '_result_from_candidates', side_effect=self.run_result):
                    run = runner.run_requests([self.request()], process_budget=self.budget,
                        max_parallel_candidates=2, total_timeout_s=7.)
                factory.assert_called_once_with(2, process_budget=self.budget)
                pool = pools[0]
                self.assertEqual((pool.enters, pool.closes), (1, 1))
                self.assertTrue(run.selected.succeeded)
                settings = pool.payloads[0][1][1]
                self.assertEqual(settings['worker_process_budget'], asdict(self.budget))
                self.assertEqual(pool._budget_deadline, 12345.)
                self.assertEqual(pool._launch_count, 3)

    def test_run_and_parallel_convenience_forward_exact_budget(self):
        runner = EnsembleRunner()
        with patch.object(runner, 'run_requests', return_value='fixture') as requests:
            runner.run(target=ReconstructionTarget(), candidates=[CandidateConfig('fixture')],
                       process_budget=self.budget)
            self.assertIs(requests.call_args.kwargs['process_budget'], self.budget)
            runner._run_requests_parallel([self.request()], max_parallel_candidates=2,
                                          process_budget=self.budget)
            self.assertIs(requests.call_args.kwargs['process_budget'], self.budget)

    def test_standalone_routing_accepts_context_opt_in_and_refuses_unowned_pool(self):
        request = self.request(SimpleNamespace(worker_process_budget=self.budget))
        with patch.object(module, 'PersistentProcessExecutor', side_effect=RecordingPool) as factory:
            results, best, ledger = routing_run([request])
            factory.assert_called_once_with(2, process_budget=self.budget)
            self.assertTrue(best.succeeded)
            self.assertEqual(ledger['submitted_candidates'], 1)
        with patch.object(module, 'PersistentProcessExecutor') as factory:
            with self.assertRaises(ValueError):
                routing_run([request], executor=RecordingPool())
            factory.assert_not_called()

    def test_worker_client_reuses_scoped_protocol_without_recursive_candidates(self):
        client = module.WorkerClient('unused-budget-fixture', 0)
        client.bind_process_budget(self.budget)
        client.stack = ['parent-job']
        with patch.object(module, '_WORKER_CLIENT', client), \
             patch.object(module, 'PersistentProcessExecutor') as factory:
            with module.executor_scope(process_budget=self.budget) as shared:
                self.assertIs(shared, client)
            with self.assertRaisesRegex(TypeError, 'submit/result'):
                EnsembleRunner().run_requests([self.request()], process_budget=self.budget)
            with self.assertRaises(ValueError):
                module.executor_scope(executor=RecordingPool(process_budget=self.budget))
            outcome = module.JobOutcome('success', 'native-union-fixture')
            with patch.object(module, 'write_packet') as packet, \
                 patch.object(module, 'read_packet', return_value=[outcome]), \
                 patch.object(Path, 'exists', return_value=True), patch.object(Path, 'unlink'):
                self.assertEqual(client.map([('native_union', (), 1.)], timeout_s=2.), [outcome])
                self.assertEqual(packet.call_args.args[1]['parent'], 'parent-job')
            factory.assert_not_called()
        self.assertIs(client.process_budget, self.budget)
        with self.assertRaises(ValueError):
            client.bind_process_budget(replace(self.budget, wall_s=20.))

    def test_transport_binds_owning_declaration_and_same_context_client(self):
        request = self.request(SimpleNamespace(worker_process_budget=self.budget))
        pool = RecordingPool(process_budget=self.budget)
        payload = pickle.loads(pickle.dumps(module.candidate_payload(request, pool)))
        self.assertIsNone(payload[0].context)
        self.assertEqual(payload[1]['worker_process_budget'], asdict(self.budget))
        client = module.WorkerClient(pool.root, 0)
        seen = []
        def reconstruct(request, **kwargs):
            seen.append(request.context)
            return CandidateResult(request.candidate_id, request.backend_name, 'failed')
        with patch.object(module, '_WORKER_CLIENT', client), \
             patch('blender_blocking.reconstruction.registry.register_builtin_backends'), \
             patch('blender_blocking.reconstruction.ensemble._run_candidate_request', side_effect=reconstruct):
            module.execute_job('candidate', payload)
        self.assertEqual(client.process_budget, self.budget)
        self.assertEqual(seen[0].worker_process_budget, self.budget)
        self.assertIs(seen[0].process_executor, client)
        with self.assertRaises(ValueError):
            module.candidate_payload(request, RecordingPool())


    def test_standalone_job_dispatch_carries_pool_declaration_before_execution(self):
        import tempfile
        import time
        with tempfile.TemporaryDirectory() as folder:
            pool = module.PersistentProcessExecutor(1, root=folder, process_budget=self.budget)
            try:
                with patch.object(pool, 'start'):
                    job = pool.submit('fit_multistart', {'fixture': 'noncandidate'})
                (pool.root / 'workers' / '0').mkdir()
                pool.workers['0'] = {'ready': True, 'ownership_stopping': False,
                    'stack': [], 'waiting': set(), 'scope': None,
                    'process': SimpleNamespace(poll=lambda: None), 'started': time.monotonic()}
                pool.poll()
                envelope = module.read_packet(pool.root / 'workers' / '0' / 'inbox.pkl')
                self.assertEqual(envelope['worker_process_budget'], asdict(self.budget))
                client = module.WorkerClient(pool.root, 0)
                def run(kind, payload, **kwargs):
                    self.assertEqual(kind, 'fit_multistart')
                    self.assertEqual(client.process_budget, self.budget)
                    return 'same-owned-client'
                with patch.object(module, 'execute_job', side_effect=run):
                    client.execute(envelope)
                result = module.read_packet(pool.root / 'results' / (job + '.pkl'))
                self.assertEqual((result.status, result.value), ('success', 'same-owned-client'))
            finally:
                pool.workers.clear()
                pool.jobs.clear()
                pool.close()

    def test_envelope_budget_mismatch_refuses_job_and_legacy_envelope_stays_compatible(self):
        client = module.WorkerClient('unused-envelope-fixture', 0)
        envelope = {'id': 'fixture', 'kind': 'native_union', 'payload': (), 'deadline': None}
        with patch.object(module, 'execute_job', return_value='legacy') as execute, \
             patch.object(module, 'write_packet') as packet:
            client.execute(envelope)
            self.assertEqual(packet.call_args.args[1].status, 'success')
            self.assertIsNone(client.process_budget)
            client.execute({**envelope, 'worker_process_budget': asdict(self.budget)})
            self.assertEqual(client.process_budget, self.budget)
            execute.reset_mock()
            changed = asdict(replace(self.budget, wall_s=20.))
            client.execute({**envelope, 'worker_process_budget': changed})
            execute.assert_not_called()
            self.assertEqual(packet.call_args.args[1].status, 'failed')
            self.assertIn('cannot replace', packet.call_args.args[1].error)
            self.assertEqual(client.stack, [])
            self.assertEqual(client.process_budget, self.budget)


class TestUnsupportedWorkerBudgetPlatform(unittest.TestCase):
    def test_unsupported_platform_refuses_before_pool_allocation(self):
        budget = module.WorkerProcessBudget(25., 1073741824,
            max_rss_bytes=536870912, join_timeout_s=2., max_restarts=1)
        with patch.object(supervisor, 'os',
                SimpleNamespace(name='posix', PathLike=os.PathLike)), \
             patch.object(module, 'PersistentProcessExecutor') as factory:
            with self.assertRaisesRegex(RuntimeError, 'Windows Job Objects'):
                module.executor_scope(process_budget=budget)
            with self.assertRaisesRegex(RuntimeError, 'Windows Job Objects'):
                module.WorkerClient('unused-unsupported-platform-fixture', 0).bind_process_budget(budget)
            factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
