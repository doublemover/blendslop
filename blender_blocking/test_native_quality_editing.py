"""Bounded native regeneration/parameter edits, without a render campaign."""
import json
import unittest
import numpy as np
import bpy

from geometry.profile_models import EllipticalSlice
from integration.blender_ops.profile_loft_mesh import create_loft_mesh_from_slices, rebuild_loft_mesh
from primitives.shape_program import ShapeNode, ShapeProgram
from primitives.shape_program_compiler import compile_shape_program
from reconstruction.native_geometry import evaluated_arrays


class NativeQualityEditingTests(unittest.TestCase):
    def tearDown(self):
        # These tests run in a factory-startup test scene, never the artist scene.
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

    def test_loft_regeneration_retains_object_and_saved_recipe(self):
        slices = [EllipticalSlice(z=z, rx=r, ry=r)
                  for z, r in ((0., .8), (1.3, .4), (2.6, .8))]
        obj = create_loft_mesh_from_slices(slices, radial_segments=48)
        obj.location = (2, -3, 1)
        obj["artist_note"] = "keep me"
        material = bpy.data.materials.new("ArtistMaterial")
        obj.data.materials.append(material)
        modifier = obj.modifiers.new("ArtistBevel", "BEVEL")
        modifier.width = .001
        bpy.context.view_layer.update()
        before = evaluated_arrays(obj)
        pointer, transform = obj.as_pointer(), obj.matrix_world.copy()
        self.assertIs(rebuild_loft_mesh(obj), obj)
        np.testing.assert_array_equal(evaluated_arrays(obj).vertices, before.vertices)
        edited = [EllipticalSlice(z=s.z, rx=s.rx * 1.1, ry=s.ry) for s in slices]
        rebuild_loft_mesh(obj, slices=edited, surface_mode="sharp")
        self.assertEqual(obj.as_pointer(), pointer)
        self.assertEqual(obj.matrix_world, transform)
        self.assertEqual(obj["artist_note"], "keep me")
        self.assertEqual(obj.data.materials[0], material)
        self.assertEqual(obj.modifiers[0], modifier)
        self.assertNotEqual(evaluated_arrays(obj).content_hash, before.content_hash)
        self.assertEqual(json.loads(obj["loft_recipe_json"])["options"]["surface_mode"], "sharp")
        current = evaluated_arrays(obj).content_hash
        with self.assertRaises(ValueError):
            rebuild_loft_mesh(obj, surface_mode="unknown")
        self.assertEqual(evaluated_arrays(obj).content_hash, current)

    def test_triangle_compiler_preserves_corner_depth_and_dome_edits(self):
        arrays = []
        for i, parameters in enumerate(({}, {"height_world": .6},
                                         {"corner_radius_world": .24}, {"front_fraction": .7})):
            node = ShapeNode("dot", "add", "rounded_triangle", parameters)
            compiled = compile_shape_program(ShapeProgram("1", "triangle" + str(i), (node,)),
                                             weighted_normals=False)
            arrays.append(evaluated_arrays(compiled.root_object))
            self.assertTrue(compiled.objects[0].get("blendslop_shape_node_editable"))
        self.assertEqual(len({data.content_hash for data in arrays}), 4)
        self.assertAlmostEqual(np.ptp(arrays[1].vertices[:, 2]), .6, places=6)
        self.assertGreater(np.ptp(arrays[2].vertices[:, 0]), np.ptp(arrays[0].vertices[:, 0]))
        self.assertAlmostEqual(arrays[3].vertices[:, 2].max(), .336, places=6)
        self.assertAlmostEqual(arrays[3].vertices[:, 2].min(), -.144, places=6)
