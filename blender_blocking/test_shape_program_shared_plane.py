"""Declared source relation dispatch and preallocation guards; no native jobs."""
from dataclasses import replace
from types import SimpleNamespace
from contextlib import ExitStack
import sys
import unittest
from unittest.mock import patch
import numpy as np
from primitives import shape_program_compiler as compiler
from primitives.shape_program import ShapeConstraint, ResidualPatch
from reconstruction import multipart_planar_join as relation
from reconstruction.multipart_family import retained_multipart_program
from test_multipart_planar_join import _Part, recipe


class TestShapeProgramSharedPlane(unittest.TestCase):
    def compile_sources(self, program, **kwargs):
        parts = [_Part(node) for node in program.root_nodes]
        by_id = {node.node_id: part for node, part in zip(program.root_nodes, parts)}
        for node, part in zip(program.root_nodes, parts):
            part.type, part.name = "MESH", node.node_id
        fake = SimpleNamespace(context=SimpleNamespace(view_layer=SimpleNamespace(update=lambda: None)))
        stack = ExitStack()
        stack.enter_context(patch.object(compiler, "BLENDER_AVAILABLE", True))
        stack.enter_context(patch.dict(sys.modules, {"bpy": fake}))
        collection = stack.enter_context(patch.object(compiler, "_ensure_collection", return_value=object()))
        stack.enter_context(patch.object(compiler, "_link_to_collection"))
        root = stack.enter_context(patch.object(compiler, "_make_root_empty", return_value=object()))
        emit = stack.enter_context(patch.object(compiler, "_compile_node",
            side_effect=lambda node, **options: (by_id[node.node_id], ())))
        embedding = stack.enter_context(patch.object(relation, "embed_shared_plane_sources",
            wraps=relation.embed_shared_plane_sources))
        return stack, parts, collection, root, emit, embedding

    def test_generic_dispatch_preserves_defaults_collection_and_applies_once(self):
        program = relation.shared_far_plane_program(recipe())
        stack, parts, collection, root, emit, embedding = self.compile_sources(program)
        with stack:
            compiled = compiler.compile_shape_program(program, collection_name="CallerCollection")
            self.assertEqual(embedding.call_count, 1)
            collection.assert_called_once_with("CallerCollection")
            self.assertTrue(all(call.kwargs['weighted_normals'] is True for call in emit.call_args_list))
            self.assertTrue(all(call.kwargs['bevel_modifier'] is True for call in emit.call_args_list))
            self.assertEqual(parts[0].geometry().vertices[:, 1].max(), parts[1].geometry().vertices[:, 1].max())
            with patch.object(relation, "_compile_live_union", return_value=(parts[0], parts)):
                output, sources, receipt = relation.compile_shared_plane_multipart(program)
            self.assertEqual(embedding.call_count, 1)
            self.assertIs(output, parts[0]); self.assertTrue(receipt['exact_common_y_frame'])
            with self.assertRaisesRegex(ValueError, 'already applied'):
                relation.embed_shared_plane_sources(program, parts)

    def test_embed_precedes_native_array_capture_and_production_union(self):
        from blender_blocking.reconstruction import native_geometry, grouped_solids
        program = relation.shared_far_plane_program(recipe())
        stack, parts, collection, root, emit, embedding = self.compile_sources(program)
        class StopBeforeUnion(Exception): pass
        def capture(source):
            self.assertEqual(embedding.call_count, 1)
            self.assertEqual(parts[0].geometry().vertices[:, 1].max(), parts[1].geometry().vertices[:, 1].max())
            return source.geometry()
        def union(positives, options, **kwargs):
            self.assertEqual(len(positives), 3); self.assertEqual(embedding.call_count, 1)
            raise StopBeforeUnion()
        with stack, patch.object(native_geometry, "evaluated_arrays", side_effect=capture) as read_arrays, \
                patch.object(grouped_solids, "production_union", side_effect=union) as run_union:
            with self.assertRaises(StopBeforeUnion):
                compiler.compile_shape_program(program, csg_options={'native_union_execution': True})
            self.assertEqual(read_arrays.call_count, 3); run_union.assert_called_once()

    def test_ordinary_historical_metadata_does_not_dispatch(self):
        program = retained_multipart_program(recipe())
        stack, parts, collection, root, emit, embedding = self.compile_sources(program)
        with stack:
            compiler.compile_shape_program(program)
            embedding.assert_not_called()
            self.assertNotEqual(parts[0].geometry().vertices[:, 1].max(), parts[1].geometry().vertices[:, 1].max())

    def test_malformed_and_unsupported_relation_fails_before_scene_allocation(self):
        valid = relation.shared_far_plane_program(recipe())
        variants = []
        for key, value in [('protocol','other'),('axis','z'),('side','near'),('base_node_id','missing'),
                           ('base_far_world',False),('preserved_arm_near_world',float('nan'))]:
            variants.append(replace(valid,metadata={**valid.metadata,'shared_far_plane':{**valid.metadata['shared_far_plane'],key:value}}))
        variants += [replace(valid,metadata={'shared_far_plane':None}),replace(valid,root_nodes=valid.root_nodes[:2]),
            replace(valid,constraints=(ShapeConstraint('align',(valid.root_nodes[0].node_id,)),)),
            replace(valid,residual_patches=(ResidualPatch('patch','surface','front'),))]
        wrong = replace(valid.root_nodes[1], parameters={**valid.root_nodes[1].parameters,'depth_world':.2})
        variants.append(replace(valid,root_nodes=(valid.root_nodes[0],wrong,valid.root_nodes[2])))
        huge = replace(valid.root_nodes[2], parameters={**valid.root_nodes[2].parameters,'width_world':1e50})
        variants.append(replace(valid,root_nodes=(valid.root_nodes[0],valid.root_nodes[1],huge)))
        for index, program in enumerate(variants):
            with self.subTest(index=index), patch.object(compiler,'BLENDER_AVAILABLE',True), \
                    patch.object(compiler,'_ensure_collection') as collection, patch.object(compiler,'_compile_node') as emit:
                with self.assertRaises(ValueError): compiler.compile_shape_program(program)
                collection.assert_not_called();emit.assert_not_called()

    def test_declared_node_normal_style_is_not_silently_disabled(self):
        valid = relation.shared_far_plane_program(recipe())
        first = replace(valid.root_nodes[0],parameters={**valid.root_nodes[0].parameters,'weighted_normals':True})
        program=replace(valid,root_nodes=(first,*valid.root_nodes[1:]))
        stack,parts,collection,root,emit,embedding=self.compile_sources(program)
        with stack:
            compiler.compile_shape_program(program,weighted_normals=False)
            self.assertTrue(emit.call_args_list[0].args[0].parameters['weighted_normals'])
            self.assertTrue(all(call.kwargs['weighted_normals'] is False for call in emit.call_args_list))
            self.assertEqual(embedding.call_count,1)


if __name__ == '__main__': unittest.main()
