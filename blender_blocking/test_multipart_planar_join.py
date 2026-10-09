"""Explicit multipart plane model and physical source edits; no Blender session."""
import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from reconstruction.multipart_planar_join import (
    PROTOCOL, compile_shared_plane_multipart, shared_depth_coordinates,
    shared_far_plane_program, shared_y_coordinates,
)
from reconstruction.native_geometry import GeometryArrays
from primitives.shape_program import ShapeNode, ShapeProgram


_CORNERS = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)], np.float32)
_FACES = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                   [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]], np.int64)


def recipe():
    # Real retained b2c controls, copied as numbers; no old files are adopted.
    rows = [('observed_base', (0., -.000017008916509658745, -.24997877377173894),
             (1.6007063620619555, .6505698684303027, .5000196344736247)),
            ('observed_tall_arm', (-.3997710684957956, .10003823656031358, .45001706757534077),
             (.4500565996476898, .450046293581264, .9999759751152597)),
            ('observed_short_arm', (.44999909143257183, -.08005412959351006, .1976298266711268),
             (.34959914271188686, .44980482428022117, .49520149330683166))]
    nodes = tuple(ShapeNode(name, 'add', 'box', parameters={
        **dict(zip(('x', 'y', 'z'), center)),
        **dict(zip(('width_world', 'depth_world', 'height_world'), size)),
        'rotation': np.eye(3).tolist()}, name=name) for name, center, size in rows)
    return ShapeProgram('1', 'retained-positive', nodes, metadata={'prior': 'untouched'}).to_dict()


class _Pose:
    def __init__(self, values): self.values = np.array(values, np.float32)
    @property
    def y(self): return float(self.values[1])
    @y.setter
    def y(self, value): self.values[1] = value


class _Vertices:
    def __init__(self): self.coordinates = _CORNERS.copy()
    def __len__(self): return len(self.coordinates)
    def foreach_get(self, name, target): target[:] = self.coordinates.ravel()
    def foreach_set(self, name, values): self.coordinates[:] = np.asarray(values).reshape(-1, 3)


class _Part(dict):
    def __init__(self, node):
        super().__init__(blendslop_shape_node_id=node.node_id)
        p = node.parameters
        self.location = _Pose([p[k] for k in ('x', 'y', 'z')])
        self.scale = _Pose([p[k] for k in ('width_world', 'depth_world', 'height_world')])
        self.data = SimpleNamespace(vertices=_Vertices(), polygons=[SimpleNamespace(use_smooth=False)] * 6, update=lambda: None)
    def geometry(self):
        vertices = self.data.vertices.coordinates.astype(float) * self.scale.values + self.location.values
        return GeometryArrays.capture(vertices, _FACES)


class MultipartPlanarJoinTests(unittest.TestCase):
    def test_explicit_model_closes_only_declared_far_plane(self):
        wire = recipe(); original = copy.deepcopy(wire)
        updated = shared_far_plane_program(wire).to_dict()
        self.assertEqual(wire, original)
        self.assertEqual(updated['root_nodes'][0], wire['root_nodes'][0])
        self.assertEqual(updated['root_nodes'][2], wire['root_nodes'][2])
        before, after = wire['root_nodes'][1]['parameters'], updated['root_nodes'][1]['parameters']
        self.assertEqual({k:v for k,v in before.items() if k not in ('y','depth_world')},
                         {k:v for k,v in after.items() if k not in ('y','depth_world')})
        base = wire['root_nodes'][0]['parameters']
        self.assertEqual(after['y'] + after['depth_world']/2, base['y'] + base['depth_world']/2)
        self.assertEqual(after['y'] - after['depth_world']/2, before['y'] - before['depth_world']/2)
        self.assertGreater(updated['metadata']['shared_far_plane']['original_gap_world'], 0.)
        self.assertEqual(updated['metadata']['prior'], 'untouched')

    def test_no_inference_or_implicit_compilation_from_proximity(self):
        from reconstruction.multipart_family import retained_multipart_program
        original = retained_multipart_program(recipe())
        with self.assertRaisesRegex(ValueError, 'explicit supported'):
            compile_shared_plane_multipart(original)
        declared = shared_far_plane_program(recipe()).to_dict()
        with self.assertRaisesRegex(ValueError, 'independent-box'):
            shared_far_plane_program(declared)

    def test_invalid_scope_refuses_before_native_compilation(self):
        from dataclasses import replace
        program = shared_far_plane_program(recipe())
        for field, value in [('protocol','other'), ('axis','x'), ('base_node_id','missing')]:
            declaration = {**program.metadata['shared_far_plane'], field:value}
            changed = replace(program, metadata={**program.metadata, 'shared_far_plane':declaration})
            with self.subTest(field=field), self.assertRaises(ValueError):
                compile_shared_plane_multipart(changed)
        params = {**program.root_nodes[1].parameters, 'depth_world': .2}
        changed = replace(program, root_nodes=(program.root_nodes[0], replace(program.root_nodes[1], parameters=params), program.root_nodes[2]))
        with self.assertRaisesRegex(ValueError, 'exact recipe'):
            compile_shared_plane_multipart(changed)

    def test_rotated_or_shading_changed_or_invalid_boxes_are_not_repaired(self):
        for key, value in [('rotation', [[0,-1,0],[1,0,0],[0,0,1]]), ('weighted_normals', True),
                           ('depth_world', False), ('depth_world', float('nan')), ('depth_world', -.1)]:
            wire = recipe(); wire['root_nodes'][1]['parameters'][key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                shared_far_plane_program(wire)
        with self.assertRaisesRegex(ValueError, 'distinct existing'):
            shared_far_plane_program(recipe(), arm_node_id='observed_base')

    def test_common_native_frame_preserves_xz_and_exact_far_plane(self):
        center, depth = float(np.float32(-.000017008916509658745)), float(np.float32(.6505698684303027))
        embedded = shared_y_coordinates(_CORNERS, near_world=-.12498491023031842,
                                        base_center_y=center, base_depth_world=depth)
        np.testing.assert_array_equal(embedded[:,(0,2)], _CORNERS[:,(0,2)])
        far = center + depth * embedded[embedded[:,1]==.5,1].astype(float)
        np.testing.assert_array_equal(far, np.full(4,center+depth*.5))
        self.assertEqual(embedded.dtype, np.float32)

    def test_depth_control_moves_near_only_and_restores_exact_source_geometry(self):
        center, frame_depth = float(np.float32(-.000017008916509658745)), float(np.float32(.6505698684303027))
        before = shared_y_coordinates(_CORNERS, near_world=-.12498491023031842,
                                     base_center_y=center, base_depth_world=frame_depth)
        original = before.copy()
        world_depth = float(np.ptp(before[:,1].astype(float))*frame_depth)
        changed = shared_depth_coordinates(before, depth_world=world_depth*1.05, frame_depth_world=frame_depth)
        self.assertAlmostEqual(float(np.ptp(changed[:,1].astype(float))*frame_depth/world_depth), 1.05, places=6)
        np.testing.assert_array_equal(changed[before[:,1]==.5], before[before[:,1]==.5])
        np.testing.assert_array_equal(changed[:,(0,2)], before[:,(0,2)])
        baseline = GeometryArrays.capture(before.astype(float), _FACES)
        edited = GeometryArrays.capture(changed.astype(float), _FACES)
        self.assertNotEqual(baseline.content_hash, edited.content_hash)
        changed[:] = original
        self.assertEqual(GeometryArrays.capture(changed.astype(float), _FACES).content_hash, baseline.content_hash)
        np.testing.assert_array_equal(before, original)

    def test_native_adapter_keeps_source_pointers_and_physical_frame(self):
        import reconstruction.multipart_planar_join as original
        program = shared_far_plane_program(recipe())
        parts = [_Part(node) for node in program.root_nodes]
        pointers = [(id(part), id(part.data)) for part in parts]
        fake_bpy = SimpleNamespace(context=SimpleNamespace(view_layer=SimpleNamespace(update=lambda: None)))
        def compiled_relation(unused):
            original.embed_shared_plane_sources(program, parts)
            return parts[0], parts
        with patch.dict(sys.modules, {'bpy':fake_bpy}), patch.object(original, '_compile_live_union', side_effect=compiled_relation):
            output, sources, receipt = compile_shared_plane_multipart(program)
        self.assertIs(output,parts[0]);self.assertIs(sources,parts)
        self.assertEqual(pointers,[(id(part),id(part.data)) for part in parts])
        self.assertEqual(parts[0].geometry().vertices[:,1].max(),parts[1].geometry().vertices[:,1].max())
        self.assertTrue(receipt['exact_common_y_frame']);self.assertTrue(receipt['flat_shading_preserved'])
        self.assertEqual(receipt['protocol'],PROTOCOL)
        declared_near = program.metadata['shared_far_plane']['preserved_arm_near_world']
        self.assertEqual(receipt['native_near_rounding_world'], parts[1].geometry().vertices[:,1].min()-declared_near)

    def test_production_compiler_needs_no_validation_scripts_and_keeps_live_unions(self):
        import builtins
        from primitives import shape_program_compiler as production
        program = shared_far_plane_program(recipe())
        parts = [_Part(node) for node in program.root_nodes]
        unions = []
        def add_modifier(name, kind):
            modifier = SimpleNamespace(name=name, type=kind)
            unions.append(modifier)
            return modifier
        parts[0].modifiers = SimpleNamespace(new=add_modifier)
        for part in parts:
            part.hide_set = lambda value: None
        fake_bpy = SimpleNamespace(context=SimpleNamespace(view_layer=SimpleNamespace(update=lambda: None)))
        native_import = builtins.__import__
        def no_scripts(name, *args, **kwargs):
            if name.startswith('run_') or name == 'scripts' or name.startswith('scripts.'):
                raise AssertionError('production compilation depended on a validation script: ' + name)
            return native_import(name, *args, **kwargs)
        def compiled_relation(*args, **kwargs):
            from reconstruction.multipart_planar_join import embed_shared_plane_sources
            embed_shared_plane_sources(program, parts)
            return SimpleNamespace(objects=list(reversed(parts)))
        with patch.dict(sys.modules, {'bpy': fake_bpy}), patch.object(production, 'compile_shape_program',
                side_effect=compiled_relation) as compile_program, \
                patch('builtins.__import__', side_effect=no_scripts):
            obj, sources, receipt = compile_shared_plane_multipart(program)
        compile_program.assert_called_once_with(program, bevel_modifier=False, weighted_normals=False)
        self.assertIs(obj, parts[0]); self.assertEqual([id(p) for p in sources], [id(p) for p in parts])
        self.assertEqual(len(unions), 2)
        for modifier, operand in zip(unions, parts[1:]):
            self.assertEqual((modifier.type, modifier.operation, modifier.solver), ('BOOLEAN', 'UNION', 'EXACT'))
            self.assertIs(modifier.object, operand)
            self.assertTrue(operand['blendslop_export_exclude']); self.assertTrue(operand.hide_render)
        self.assertTrue(receipt['exact_common_y_frame'])

    def test_checker_refuses_changed_scope_or_model_before_native_allocation(self):
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        import run_multipart_planar_join_check as checker
        wire = recipe()
        plan = {"protocol": checker.PROTOCOL, "family": checker.FAMILY,
                "baseline_hash": checker.BASELINE, "source_hash": checker.SOURCE,
                "render_frames": 12, "candidate_frames": 11, "source_frames": 1,
                "candidate_original_masks": 5, "candidate_original_neutrals": 5,
                "display_only_unclipped_neutrals": {"candidate": 1, "source": 1},
                "reused_source_gate_masks": 5, "reused_source_neutral_frames": 5,
                "fits": 0, "raw_pairs": 1, "samples_per_direction": 4096, "seed": 61007,
                "qualifier_children": 1, "helper_timeout_seconds": 15., "threads": 2,
                "work_seconds": 85., "join_seconds": 5., "rss_bytes": 8 * 1024 ** 3,
                "committed_bytes": 8 * 1024 ** 3, "resolution": [512, 512],
                "artist_limits": None, "aggregate_accepted": False,
                "engineering_limits": {"normal_angle_p95_degrees_max": 1.,
                    "symmetric_mean_distance_world_max": .0018125506404794065},
                "entry": {"candidate_geometry_hash": checker.BASELINE, "source_geometry_hash": checker.SOURCE,
                    "candidate_recipe": wire, "original_cameras": {v:{} for v in checker.CANONICAL_VIEWS},
                    "original_masks": {v:{} for v in checker.CANONICAL_VIEWS}},
                "source_cameras": {v:{} for v in checker.CANONICAL_VIEWS},
                "proposal": shared_far_plane_program(wire).to_dict(), "input_sha256": {"frozen":"0"*64}}
        self.assertIs(checker.validate_plan(plan), plan)
        for key, value in [("family", "other"), ("render_frames", 13), ("source_hash", "0"*64),
                           ("work_seconds", 86.), ("aggregate_accepted", True),
                           ("engineering_limits", {"normal_angle_p95_degrees_max": 2.5})]:
            changed = copy.deepcopy(plan); changed[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                checker.validate_plan(changed)
        changed = copy.deepcopy(plan)
        changed['proposal']['root_nodes'][1]['parameters']['width_world'] *= 1.01
        with self.assertRaisesRegex(ValueError, 'family/recipe/camera identity'):
            checker.validate_plan(changed)

    def test_invalid_native_geometry_and_depth_refuse(self):
        for coordinates in [_CORNERS[:7], _CORNERS * 2, np.full((8,3),np.nan)]:
            with self.subTest(shape=coordinates.shape), self.assertRaises(ValueError):
                shared_y_coordinates(coordinates,near_world=-.1,base_center_y=0.,base_depth_world=1.)
        for depth in [False,0.,-1.,float('nan')]:
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                shared_depth_coordinates(_CORNERS,depth_world=depth,frame_depth_world=1.)


if __name__ == '__main__':
    unittest.main()
