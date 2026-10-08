"""Analytic contact fixtures; no native screen or broad solid campaign required."""
import unittest
import numpy as np
from blender_blocking.evaluation.intersection_diagnostics import (
    classify_triangle_contact, intersection_pair_diagnostics,
)


class IntersectionDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.left = np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])

    def classify(self, other):
        return classify_triangle_contact(self.left, other)["classification"]

    def test_coplanar_area_overlap_and_disjoint_triangles(self):
        self.assertEqual(self.classify(self.left + [.1,.1,0]), "coplanar_area_overlap")
        self.assertEqual(self.classify(self.left + [3.,0,0]), "no_contact_at_tolerance")
        result = classify_triangle_contact(self.left, self.left)
        self.assertAlmostEqual(result["coplanar_overlap_area"], 2.)

    def test_point_and_edge_contacts_are_reported_without_qualification(self):
        self.assertEqual(self.classify([[2,0,0],[3,0,0],[2,-1,0]]), "point_contact")
        self.assertEqual(self.classify([[0,0,0],[2,0,0],[1,-1,0]]), "boundary_segment_contact")

    def test_non_coplanar_crossing_and_boundary_segment(self):
        self.assertEqual(self.classify([[.5,.5,-1],[.5,.5,1],[.5,1.5,0]]), "proper_crossing")
        self.assertEqual(self.classify([[0,0,-1],[0,0,1],[0,2,0]]), "boundary_segment_contact")

    def test_degenerate_triangles_stay_indeterminate(self):
        self.assertEqual(self.classify([[0,0,0],[1,0,0],[2,0,0]]), "indeterminate_degenerate")

    def test_pair_budget_keeps_unclassified_denominator(self):
        vertices = np.concatenate((self.left, self.left + [.1,.1,0]))
        faces = np.array([[0,1,2],[3,4,5]])
        result = intersection_pair_diagnostics(vertices, faces, np.array([[0,1],[0,1]]), max_pairs=1)
        self.assertEqual(result["status"], "bounded_partial")
        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(result["classified_count"], 1)
        self.assertIn("still prevent", result["qualification_policy"])
        with self.assertRaises(ValueError):
            intersection_pair_diagnostics(vertices, faces, np.array([[0,2]]))

    def test_native_pair_classification_does_not_relax_solid_eligibility(self):
        try:
            import open3d
        except ImportError:
            self.skipTest("existing optional Open3D is unavailable")
        from blender_blocking.evaluation.solid_validity import solid_validity_report
        right = np.array([[.5,.5,-1],[.5,.5,1],[.5,1.5,0]])
        vertices = np.concatenate((self.left, right))
        result = solid_validity_report(vertices, np.array([[0,1,2],[3,4,5]]), native_intersections=True)
        self.assertFalse(result["single_solid_eligible"])
        screen = result["self_intersection"]
        self.assertEqual(screen["intersecting_triangle_pairs"], 1)
        self.assertEqual(screen["contact_diagnostics"]["counts"], {"proper_crossing": 1})

    def test_contact_type_is_stable_under_scale_translation_and_winding(self):
        right = np.array([[.5,.5,-1],[.5,.5,1],[.5,1.5,0]])
        for scale in (.001, 1000.):
            for a, b in ((self.left, right), (self.left[::-1], right[::-1])):
                result = classify_triangle_contact(a * scale + 10, b * scale + 10)
                self.assertEqual(result["classification"], "proper_crossing")


if __name__ == "__main__":
    unittest.main()
