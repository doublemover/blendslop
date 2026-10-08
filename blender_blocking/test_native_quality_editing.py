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

    def test_old_loft_recipe_accepts_bounded_regularization_edits(self):
        sections = [EllipticalSlice(z=float(i), rx=1. + .04 * (-1)**i, ry=.8)
                    for i in range(9)]
        obj = create_loft_mesh_from_slices(sections, radial_segments=24)
        recipe = json.loads(obj["loft_recipe_json"])
        for key in ("regularization_window", "regularization_max_deviation_u"):
            recipe["options"].pop(key)
        obj["loft_recipe_json"] = json.dumps(recipe)
        before = evaluated_arrays(obj).content_hash
        rebuild_loft_mesh(obj, regularization_window=5, regularization_max_deviation_u=.02)
        self.assertNotEqual(evaluated_arrays(obj).content_hash, before)
        self.assertEqual(json.loads(obj["loft_recipe_json"])["options"]["regularization_window"], 5)
        current = evaluated_arrays(obj).content_hash
        with self.assertRaises(ValueError):
            rebuild_loft_mesh(obj, regularization_window=6)
        self.assertEqual(evaluated_arrays(obj).content_hash, current)

    def test_triangle_compiler_preserves_corner_depth_and_dome_edits(self):
        arrays = []
        for i, parameters in enumerate(({}, {"height_world": .6},
                                         {"corner_radius_world": .24}, {"front_fraction": .7},
                                         {"width_world": 2.1,"depth_world":1.0})):
            node = ShapeNode("dot", "add", "rounded_triangle", parameters)
            compiled = compile_shape_program(ShapeProgram("1", "triangle" + str(i), (node,)),
                                             weighted_normals=False)
            arrays.append(evaluated_arrays(compiled.root_object))
            self.assertTrue(compiled.objects[0].get("blendslop_shape_node_editable"))
        self.assertEqual(len({data.content_hash for data in arrays}), 5)
        self.assertAlmostEqual(np.ptp(arrays[1].vertices[:, 2]), .6, places=6)
        self.assertGreater(np.ptp(arrays[2].vertices[:, 0]), np.ptp(arrays[0].vertices[:, 0]))
        self.assertAlmostEqual(arrays[3].vertices[:, 2].max(), .336, places=6)
        self.assertAlmostEqual(arrays[3].vertices[:, 2].min(), -.144, places=6)
        np.testing.assert_allclose(np.ptp(arrays[4].vertices[:,:2],axis=0),[2.1,1.0],atol=.001)


    def test_compound_reference_keeps_authored_world_coordinates_and_live_edits(self):
        from synthetic.quality_contracts import quality_workload
        from synthetic.quality_references import build_quality_reference, quality_feature_verdict
        for case in quality_workload()["cases"]:
            if case["name"] not in {"concave_arch", "asymmetric_multipart_solid"}:
                continue
            reference = build_quality_reference(case)
            before = evaluated_arrays(reference.object)
            self.assertEqual(quality_feature_verdict(case["name"], before)["status"], "passed")
            part = reference.sources[1]
            location = part.location.copy()
            part.location.z += .1
            bpy.context.view_layer.update()
            self.assertNotEqual(evaluated_arrays(reference.object).content_hash, before.content_hash)
            part.location = location
            bpy.context.view_layer.update()
            self.assertEqual(evaluated_arrays(reference.object).content_hash, before.content_hash)


    def test_oblique_quality_camera_keeps_all_bound_corners_inside_frame(self):
        from mathutils import Vector
        from bpy_extras.object_utils import world_to_camera_view
        from scripts.run_surface_quality_check import _view
        camera=bpy.data.objects.new("ObliqueFit",bpy.data.cameras.new("ObliqueFit"))
        bpy.context.collection.objects.link(camera)
        camera.data.type="ORTHO"
        scene=bpy.context.scene
        scene.render.resolution_x=scene.render.resolution_y=512
        lo,hi=Vector((-.94,-.61,-.24)),Vector((.94,1.06,.24))
        for view in ("oblique_35_28","oblique_145_40"):
            _view(camera,view,lo,hi)
            bpy.context.view_layer.update()
            for x in (lo.x,hi.x):
                for y in (lo.y,hi.y):
                    for z in (lo.z,hi.z):
                        p=world_to_camera_view(scene,camera,Vector((x,y,z)))
                        self.assertGreater(p.x,0.)
                        self.assertLess(p.x,1.)
                        self.assertGreater(p.y,0.)
                        self.assertLess(p.y,1.)

    def test_capsule_compiler_and_synthetic_builder_are_closed(self):
        from synthetic.quality_contracts import quality_workload
        from synthetic.quality_references import build_quality_reference
        from reconstruction.grouped_solids import solid_guard
        case = next(r for r in quality_workload()["cases"] if r["name"] == "capsule")
        obj = build_quality_reference(case).object
        data = evaluated_arrays(obj)
        guard = solid_guard(data)
        self.assertTrue(guard["valid_solid"], guard)
        self.assertEqual(guard["geometric_degenerate_faces"], 0)
        self.assertEqual(guard["non_manifold_edges"], 0)
        self.assertAlmostEqual(np.ptp(data.vertices[:, 2]), 2., places=6)
        node = ShapeNode("capsule", "add", "capsule", {"width_world": .8, "height_world": 2.})
        compiled = compile_shape_program(ShapeProgram("1", "cap", (node,)), weighted_normals=False)
        self.assertTrue(solid_guard(evaluated_arrays(compiled.root_object))["valid_solid"])
