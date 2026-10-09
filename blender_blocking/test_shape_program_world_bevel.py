"""World-unit rounded-box radius and live descendant edit contracts."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from primitives import shape_program_compiler as compiler
from primitives.shape_program import ShapeNode, ShapeProgram


class TestShapeProgramWorldBevel(unittest.TestCase):
    def test_anisotropic_source_scale_does_not_scale_the_world_bevel_radius(self):
        class Modifiers(list):
            def new(self, name, kind):
                result = SimpleNamespace(name=name, type=kind)
                self.append(result)
                return result
        child = SimpleNamespace(scale=np.array([2., 1., 1.5]), modifiers=Modifiers(),
            coordinates=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]]),
            select_set=lambda value: None)
        active = SimpleNamespace(active=None)
        def apply(**kwargs):
            self.assertIs(active.active, child)
            self.assertEqual(kwargs, {"location": False, "rotation": False, "scale": True})
            child.coordinates *= child.scale
            child.scale = np.ones(3)
        fake = SimpleNamespace(context=SimpleNamespace(view_layer=SimpleNamespace(objects=active)),
            ops=SimpleNamespace(object=SimpleNamespace(select_all=lambda **kwargs: None, transform_apply=apply)))
        with patch.object(compiler, "bpy", fake):
            compiler._add_bevel(child, .12, segments=8)
        np.testing.assert_array_equal(np.ptp(child.coordinates, axis=0), [2.,1.,1.5])
        np.testing.assert_array_equal(child.scale, [1.,1.,1.])
        self.assertEqual(child.modifiers[0].width, .12)
        self.assertEqual(child.modifiers[0].segments, 8)

    def test_default_segments_remain_three_and_explicit_segments_are_bounded(self):
        self.assertEqual(compiler._rounded_box_bevel_segments({}), 3)
        self.assertEqual(compiler._rounded_box_bevel_segments({"bevel_segments": 8}), 8)
        for value in (False, 3.5, 0, 65, "8"):
            with self.assertRaises(ValueError):
                compiler._rounded_box_bevel_segments({"bevel_segments": value})
        node = ShapeNode("box", "add", "rounded_box", {"bevel_segments": 65})
        with patch.object(compiler, "_cube") as cube:
            with self.assertRaises(ValueError):
                compiler._compile_node(node, lathe_segments=48, bevel_modifier=True, weighted_normals=False)
            cube.assert_not_called()

    def test_explicit_weighted_normal_style_overrides_only_the_global_default(self):
        class Source(dict):
            modifiers = []
        for global_default, explicit, expected in ((False, True, True), (True, False, False),
                                                    (True, None, True), (False, None, False)):
            parameters = {} if explicit is None else {"weighted_normals": explicit}
            node = ShapeNode("box", "add", "box", parameters)
            source = Source()
            with patch.object(compiler, "_cube", return_value=source), patch.object(compiler, "_add_weighted_normals") as add:
                compiler._compile_node(node, lathe_segments=48, bevel_modifier=True, weighted_normals=global_default)
                self.assertEqual(add.called, expected)
                if expected:
                    self.assertTrue(add.call_args.kwargs["keep_sharp"])
        for invalid in (0, 1, "true", None):
            node = ShapeNode("box", "add", "box", {"weighted_normals": invalid})
            with patch.object(compiler, "_cube") as cube:
                with self.assertRaises(ValueError):
                    compiler._compile_node(node, lathe_segments=48, bevel_modifier=True, weighted_normals=False)
                cube.assert_not_called()

    def test_authored_keep_sharp_control_is_explicit_and_validated_before_geometry(self):
        class Source(dict):
            modifiers = []
        source = Source()
        node = ShapeNode("box", "add", "box", {"weighted_normals": True, "weighted_normals_keep_sharp": False})
        with patch.object(compiler, "_cube", return_value=source), patch.object(compiler, "_add_weighted_normals") as add:
            compiler._compile_node(node, lathe_segments=48, bevel_modifier=True, weighted_normals=False)
            add.assert_called_once_with(source, keep_sharp=False)
        for invalid in (0, 1, "false", None):
            node = ShapeNode("box", "add", "box", {"weighted_normals_keep_sharp": invalid})
            with patch.object(compiler, "_cube") as cube:
                with self.assertRaises(ValueError):
                    compiler._compile_node(node, lathe_segments=48, bevel_modifier=True, weighted_normals=False)
                cube.assert_not_called()

    def test_invalid_world_radius_fails_before_applying_source_scale(self):
        with patch.object(compiler, "bpy", SimpleNamespace()):
            with self.assertRaises(ValueError):
                compiler._add_bevel(SimpleNamespace(), float("nan"))


@unittest.skipUnless(compiler.BLENDER_AVAILABLE, "native Blender geometry required")
class TestNativeShapeProgramWorldBevel(unittest.TestCase):
    def setUp(self):
        self.owned = []

    def tearDown(self):
        import bpy
        for compiled in reversed(self.owned):
            for obj in (*compiled.objects, compiled.root_object):
                if obj is not None and obj.name in bpy.data.objects:
                    mesh = obj.data if obj.type == "MESH" else None
                    bpy.data.objects.remove(obj, do_unlink=True)
                    if mesh is not None and mesh.users == 0:
                        bpy.data.meshes.remove(mesh)

    def test_world_radius_is_equal_on_three_axes_and_live_corner_edit_restores(self):
        import bpy
        from reconstruction.native_geometry import evaluated_arrays
        from reconstruction.output_targets import output_mesh_targets
        params = {"width_world": 2., "depth_world": 1., "height_world": 1.5,
                  "corner_radius_world": .12, "bevel_segments": 8}
        compiled = compiler.compile_shape_program(ShapeProgram("1", "world-bevel", (
            ShapeNode("b", "add", "rounded_box", params),)), weighted_normals=False)
        self.owned.append(compiled)
        self.assertEqual(compiled.root_object.type, "EMPTY")
        children = output_mesh_targets([compiled.root_object])
        self.assertEqual(len(children), 1)
        source = children[0]
        np.testing.assert_allclose(source.scale, [1.,1.,1.], atol=0.)
        before = evaluated_arrays(compiled.root_object)
        for axis in range(3):
            face = np.isclose(before.vertices[:, axis], [1.,.5,.75][axis], atol=1e-7)
            for other in set(range(3))-{axis}:
                self.assertAlmostEqual(float(before.vertices[face, other].max()), [1.,.5,.75][other]-.12, places=6)
        bevel = next(modifier for modifier in source.modifiers if modifier.type == "BEVEL")
        pointer = source.as_pointer()
        bevel.width = .18
        bpy.context.view_layer.update()
        edited = evaluated_arrays(compiled.root_object)
        self.assertNotEqual(edited.content_hash, before.content_hash)
        face = np.isclose(edited.vertices[:,0],1.,atol=1e-7)
        self.assertAlmostEqual(float(edited.vertices[face,1].max()),.5-.18,places=6)
        bevel.width = .12
        bpy.context.view_layer.update()
        self.assertEqual(source.as_pointer(),pointer)
        self.assertEqual(evaluated_arrays(compiled.root_object).content_hash,before.content_hash)

    def test_ordinary_box_has_no_bevel_and_default_rounded_box_segments_are_compatible(self):
        for primitive in ("box", "rounded_box"):
            compiled = compiler.compile_shape_program(ShapeProgram("1", "default-"+primitive, (
                ShapeNode("b", "add", primitive, {"width_world": 2., "depth_world": 1.}),)),
                weighted_normals=False)
            self.owned.append(compiled)
            bevels = [modifier for modifier in compiled.objects[0].modifiers if modifier.type == "BEVEL"]
            self.assertEqual(len(bevels), int(primitive=="rounded_box"))
            if bevels:
                self.assertEqual(bevels[0].segments,3)


if __name__ == "__main__":
    unittest.main()
