"""Geometry and artist-edit contracts for the production triangular pebble."""
import unittest
import numpy as np

from primitives.rounded_triangle import RoundedTrianglePrimitive
from primitives.shape_program import ShapeNode, ShapeProgram, validate_compilable_program
from reconstruction.proposal_screening import whole_program_geometry
from reconstruction.grouped_solids import solid_guard
from synthetic.quality_contracts import rounded_triangle_mesh
from evaluation.surface_quality import circular_profile_continuity


class RoundedTriangleTests(unittest.TestCase):
    def test_production_default_preserves_frozen_reference(self):
        vertices, faces = rounded_triangle_mesh()
        data = RoundedTrianglePrimitive().to_mesh_data()
        np.testing.assert_array_equal(data.vertices, vertices)
        np.testing.assert_array_equal(data.faces, faces)
        program = ShapeProgram("1", "triangle", (ShapeNode("dot", "add", "rounded_triangle"),))
        self.assertEqual(validate_compilable_program(program), ())
        self.assertTrue(solid_guard(whole_program_geometry(program, resolution=16))["valid_solid"])

    def test_pose_serialization_and_independent_depth_edits(self):
        angle = .4
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        part = RoundedTrianglePrimitive(front_fraction=.7, thickness=.6,
                                        center=(2, -1, 3), rotation=rotation)
        mesh = part.to_mesh_data()
        self.assertAlmostEqual(mesh.vertices[:, 2].max(), 3.42)
        self.assertAlmostEqual(mesh.vertices[:, 2].min(), 2.82)
        restored = RoundedTrianglePrimitive.from_dict(part.to_dict())
        np.testing.assert_allclose(restored.to_mesh_data().vertices, mesh.vertices, atol=1e-14)
        self.assertAlmostEqual(np.ptp(mesh.vertices[:, 2]), .6)

    def test_signed_field_handles_interior_poles_and_asymmetric_depths(self):
        part = RoundedTrianglePrimitive(front_fraction=.7, thickness=.6)
        points = [[0, 0, 0], [0, 0, .42], [0, 0, -.18], [0, 0, .5], [2, 0, 0]]
        field = part.sdf_batch(points)
        self.assertLess(field[0], 0)
        np.testing.assert_allclose(field[1:3], 0, atol=1e-14)
        self.assertGreater(field[3], 0)
        self.assertGreater(field[4], 0)
        self.assertEqual(part.sdf_batch(np.empty((0, 3))).shape, (0,))

    def test_invalid_recipe_rejected_before_compilation(self):
        for values in ({"front_fraction": 0}, {"corner_segments": 8.5},
                       {"dome_segments": True}, {"corner_radius": -1},
                       {"thickness": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RoundedTrianglePrimitive(**values)
        node = ShapeNode("dot", "add", "rounded_triangle", {"front_fraction": 2})
        self.assertTrue(validate_compilable_program(ShapeProgram("1", "bad", (node,))))

    def test_continuity_detects_shoulders_despite_correct_outer_radius(self):
        z = np.linspace(0, 2.6, 65)
        radii = .6 + .2 * np.cos(2 * np.pi * z / 2.6)
        vertices = np.column_stack((radii, np.zeros(len(z)), z))
        self.assertTrue(circular_profile_continuity(vertices)["passed"])
        stepped = np.vstack((vertices, [[radii[12] + .02, 0, z[12]]]))
        result = circular_profile_continuity(stepped)
        self.assertFalse(result["passed"])
        self.assertEqual(result["shoulder_ring_count"], 1)
        self.assertAlmostEqual(result["planar_shoulder_width_max_world"], .02)


if __name__ == "__main__":
    unittest.main()
