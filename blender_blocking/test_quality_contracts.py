"""Local numerical contracts, separate from native reconstruction acceptance."""

import unittest
import numpy as np

from evaluation.surface_quality import oriented_normal_angles
from synthetic.quality_contracts import quality_workload, rounded_triangle_mesh, triangle_preservation
from reconstruction.native_geometry import GeometryArrays
from reconstruction.grouped_solids import solid_guard


class QualityContractTests(unittest.TestCase):
    def test_normal_metric_does_not_hide_flipped_normals(self):
        values = oriented_normal_angles([[1, 0, 0], [0, 0, 1]], [[-1, 0, 0], [0, 0, 2]])
        np.testing.assert_allclose(values, [180, 0])
        with self.assertRaises(ValueError):
            oriented_normal_angles([[0, 0, 0]], [[1, 0, 0]])

    def test_workload_freezes_exactly_twelve_required_families(self):
        payload = quality_workload()
        self.assertEqual(len(payload["cases"]), 12)
        self.assertEqual(len({row["name"] for row in payload["cases"]}), 12)
        self.assertEqual(len(payload["views"]), 5)
        self.assertEqual(payload["status"], "prepared_unmeasured")
        self.assertTrue(all(row["status"] == "unmeasured" for row in payload["cases"]))
        self.assertIn("rounded_triangle_dot", payload["held_out_before_tuning"])

    def test_rounded_triangle_reference_is_deterministic_closed_oriented_solid(self):
        vertices, faces = rounded_triangle_mesh()
        other, other_faces = rounded_triangle_mesh()
        np.testing.assert_array_equal(vertices, other)
        np.testing.assert_array_equal(faces, other_faces)
        guard = solid_guard(GeometryArrays.capture(vertices, faces))
        self.assertTrue(guard["valid_solid"])
        self.assertEqual(guard["connected_components"], 1)
        self.assertTrue(triangle_preservation(vertices)["passed"])

    def test_circularized_and_flattened_controls_fail_authored_shape_gate(self):
        vertices, _ = rounded_triangle_mesh()
        theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        circle = np.column_stack((1.06 * np.cos(theta), 1.06 * np.sin(theta), .24 * np.cos(theta * 2)))
        circular = triangle_preservation(circle)
        self.assertGreater(circular["support_error_world"], .4)
        self.assertFalse(circular["passed"])
        flat = vertices.copy()
        flat[:, 2] *= .1
        self.assertFalse(triangle_preservation(flat)["passed"])

    def test_triangle_rejects_unknown_or_invalid_input_geometry(self):
        params = dict(quality_workload()["cases"][-1]["parameters"])
        params["vertices_xy"] = list(reversed(params["vertices_xy"]))
        with self.assertRaises(ValueError):
            rounded_triangle_mesh(params)
        with self.assertRaises(ValueError):
            triangle_preservation([[0, 0, float("nan")]])


if __name__ == "__main__":
    unittest.main()
