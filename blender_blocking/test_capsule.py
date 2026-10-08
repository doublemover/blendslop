"""Analytic capsule mesh, field, pose and authoring contracts."""
import unittest
import numpy as np
from primitives.capsule import CapsulePrimitive
from primitives.shape_program import ShapeNode, ShapeProgram, validate_compilable_program
from reconstruction.native_geometry import GeometryArrays
from reconstruction.grouped_solids import solid_guard


class CapsuleTests(unittest.TestCase):
    def test_connected_geometry_has_no_pole_or_equator_degeneracy(self):
        for height in (0., 1.2):
            part = CapsulePrimitive(radius=.4, segment_height=height)
            mesh = part.to_mesh_data(48)
            report = solid_guard(GeometryArrays.capture(mesh.vertices, mesh.faces))
            self.assertTrue(report["valid_solid"], report)
            self.assertEqual(report["euler_characteristic"], 2)
            self.assertEqual(report["geometric_degenerate_faces"], 0)
            np.testing.assert_allclose(part.sdf_batch(mesh.vertices), 0, atol=1e-14)
            expected_volume = np.pi * .4**2 * height + 4 / 3 * np.pi * .4**3
            self.assertAlmostEqual(report["signed_volume"], expected_volume, delta=expected_volume * .015)

    def test_exact_distance_and_rotated_serialization(self):
        part = CapsulePrimitive(center=(2, 3, 4), rotation=[[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        restored = CapsulePrimitive.from_dict(part.to_dict())
        np.testing.assert_allclose(restored.to_mesh_data().vertices, part.to_mesh_data().vertices, atol=1e-14)
        plain = CapsulePrimitive()
        np.testing.assert_allclose(plain.sdf_batch([[0, 0, 0], [0, 0, 1.], [.8, 0, 0]]), [-.4, 0, .4])
        self.assertEqual(plain.sdf_batch(np.empty((0, 3))).shape, (0,))

    def test_compiler_accepts_capsule_and_rejects_invalid_dimensions(self):
        for height, passed in ((2., True), (.2, False)):
            node = ShapeNode("capsule", "add", "capsule", {"width_world": .8, "height_world": height})
            errors = validate_compilable_program(ShapeProgram("1", "cap", (node,)))
            self.assertEqual(not errors, passed)
        for options in ({"radius": 0}, {"segment_height": -1}, {"radius": float("nan")}):
            with self.assertRaises(ValueError):
                CapsulePrimitive(**options)
        with self.assertRaises(ValueError):
            CapsulePrimitive().to_mesh_data(True)
