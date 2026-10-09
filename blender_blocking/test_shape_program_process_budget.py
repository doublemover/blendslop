"""Compile-chain caller-budget propagation, stopped before any native work."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from blender_blocking.primitives import shape_program_compiler as compiler
from blender_blocking.primitives.shape_program import ShapeNode, ShapeProgram
from blender_blocking.reconstruction import grouped_solids, native_geometry, process_executor
from blender_blocking.reconstruction.backends.shape_program import backend, compiler_bridge, geometry_search
from blender_blocking.reconstruction.types import CandidateRequest, ReconstructionTarget


class StopBeforeNative(Exception):
    pass


class StopBeforeExport(BaseException):
    pass


class ScopePool:
    def __init__(self, workers=2, *, process_budget=None):
        self.workers = workers
        self.process_budget = process_budget
        self.enters = self.closes = 0
        self.deadline = 12345.

    def __enter__(self):
        self.enters += 1
        return self

    def __exit__(self, *args):
        self.closes += 1


class TestShapeProgramProcessBudget(unittest.TestCase):
    def setUp(self):
        self.program = ShapeProgram('1', 'compile-budget', (
            ShapeNode('base', 'add', 'box', {'width_world': 1., 'depth_world': .5,
                'height_world': 2.}),
            ShapeNode('arm', 'add', 'box', {'width_world': 1., 'depth_world': .75,
                'height_world': 1.}),
        ))
        self.budget = process_executor.WorkerProcessBudget(25., 1073741824,
            max_rss_bytes=536870912, join_timeout_s=2., max_restarts=1)
        worker = patch.object(process_executor, '_WORKER_CLIENT', None)
        worker.start()
        self.addCleanup(worker.stop)

    def test_actual_bridge_forwards_context_budget_and_preserves_default_options(self):
        context = SimpleNamespace(worker_process_budget=self.budget)
        config = {'lathe_segments': 64, 'bevel_modifier': False, 'weighted_normals': False,
            'native_union_execution': True, 'program_timeout_s': 7.}
        with patch.object(compiler, 'compile_shape_program', return_value='compiled') as compile_program:
            self.assertEqual(compiler_bridge._compile_program(self.program, config,
                context=context, process_budget=self.budget), 'compiled')
            compile_program.assert_called_once_with(self.program, lathe_segments=64,
                bevel_modifier=False, weighted_normals=False, csg_options=config,
                timeout_s=7., context=context, process_budget=self.budget)
            compile_program.reset_mock()
            compiler_bridge._compile_program(self.program, {})
            compile_program.assert_called_once_with(self.program, lathe_segments=48,
                bevel_modifier=True, weighted_normals=True, csg_options={},
                timeout_s=45., context=None, process_budget=None)

    def test_actual_compiler_union_scope_admits_budget_reuses_pool_and_refuses_mismatch(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        data = native_geometry.GeometryArrays.capture(points,
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        owned_pool = ScopePool(process_budget=self.budget)
        mismatched_pool = ScopePool()
        cases = (
            ('default', {}, None, False),
            ('explicit', {'process_budget': self.budget}, self.budget, False),
            ('context', {'context': SimpleNamespace(worker_process_budget=self.budget)}, self.budget, False),
            ('shared', {'context': SimpleNamespace(worker_process_budget=self.budget,
                process_executor=owned_pool)}, self.budget, False),
            ('unowned', {'context': SimpleNamespace(worker_process_budget=self.budget,
                process_executor=mismatched_pool)}, self.budget, True),
        )
        for name, options, budget, refusal in cases:
            with self.subTest(name=name):
                parts = [SimpleNamespace(type='MESH', name=node.node_id) for node in self.program.root_nodes]
                pools = []
                def create(*args, **kwargs):
                    pool = ScopePool(*args, **kwargs)
                    pools.append(pool)
                    return pool
                with patch.object(compiler, 'BLENDER_AVAILABLE', True), \
                     patch.object(compiler, '_ensure_collection'), \
                     patch.object(compiler, '_compile_node', side_effect=[(part, ()) for part in parts]), \
                     patch.object(compiler, '_link_to_collection'), patch.object(compiler, '_tag_object'), \
                     patch.object(compiler, '_make_root_empty') as live_root, \
                     patch.object(native_geometry, 'evaluated_arrays', return_value=data) as capture, \
                     patch.object(process_executor, 'PersistentProcessExecutor', side_effect=create) as factory, \
                     patch.object(grouped_solids, 'production_union', wraps=grouped_solids.production_union) as admission, \
                     patch.object(grouped_solids, 'balanced_union', side_effect=StopBeforeNative) as union:
                    with self.assertRaises(ValueError if refusal else StopBeforeNative):
                        compiler.compile_shape_program(self.program, bevel_modifier=False,
                            csg_options={'native_union_execution': True}, timeout_s=3.5, **options)
                    capture.assert_has_calls([unittest.mock.call(part) for part in parts])
                    admission.assert_called_once()
                    self.assertIs(admission.call_args.kwargs['context'], options.get('context'))
                    self.assertIs(admission.call_args.kwargs['process_budget'], options.get('process_budget'))
                    live_root.assert_not_called()
                    if refusal:
                        factory.assert_not_called()
                        union.assert_not_called()
                    else:
                        expected = owned_pool if name == 'shared' else pools[0]
                        self.assertIs(union.call_args.kwargs['executor'], expected)
                        self.assertEqual(union.call_args.kwargs['timeout_s'], 3.5)
                        self.assertEqual(union.call_args.kwargs['feature_thickness'], .5)
                        if name == 'shared':
                            factory.assert_not_called()
                        elif budget is None:
                            factory.assert_called_once_with(2)
                        else:
                            factory.assert_called_once_with(2, process_budget=budget)
                        if pools:
                            self.assertEqual((pools[0].enters, pools[0].closes), (1, 1))
        self.assertEqual((owned_pool.enters, owned_pool.closes, owned_pool.deadline), (0, 0, 12345.))

    def test_actual_backend_final_compile_forwards_request_context_before_exports(self):
        context = SimpleNamespace(blender_available=True, worker_process_budget=self.budget)
        request = CandidateRequest('request-program', 'shape_program', ReconstructionTarget(),
            config={'routed_program': self.program, 'native_union_execution': True}, context=context)
        routed = replace(self.program, program_id=request.candidate_id)
        with patch.object(geometry_search, 'geometric_program_search', return_value=(routed, None, {})), \
             patch.object(backend, '_compile_program', side_effect=StopBeforeExport) as final_compile, \
             patch.object(backend, '_run_shape_program_export_qa') as export, \
             patch.object(backend, 'write_json') as write:
            with self.assertRaises(StopBeforeExport):
                backend.ShapeProgramBackend().reconstruct(request)
            final_compile.assert_called_once_with(routed, request.config, context=context)
            export.assert_not_called()
            write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
