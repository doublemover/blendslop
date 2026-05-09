"""Pure tests for topology, surface, and budget metrics."""

from __future__ import annotations

import unittest

import numpy as np

from metrics.budgets import compare_metric_delta, evaluate_budgets
from metrics.silhouette import silhouette_metric_result
from metrics.surface import chamfer_distance, volume_overlap
from metrics.topology import (
    mesh_topology_report,
    safe_topology_repair,
    topology_penalty,
    topology_repair_plan,
)


class MetricsFoundationTests(unittest.TestCase):
    def test_topology_report_for_closed_tetrahedron(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        faces = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
        report = mesh_topology_report(vertices, faces)

        self.assertTrue(report.watertight)
        self.assertEqual(report.boundary_edges, 0)
        self.assertEqual(report.connected_components, 1)
        self.assertTrue(report.to_dict()["passed"])
        self.assertEqual(report.to_dict()["reason"], "")
        self.assertEqual(topology_penalty(report), 0.0)

    def test_safe_topology_repair_removes_invalid_loose_and_extra_components(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0, 5.0, 5.0],
                [6.0, 5.0, 5.0],
                [5.0, 6.0, 5.0],
                [9.0, 9.0, 9.0],
            ]
        )
        faces = (
            (0, 1, 2),
            (0, 1, 1),
            (0, 1, 2),
            (3, 4, 5),
            (0, 1, 99),
        )

        repair = safe_topology_repair(vertices, faces)
        payload = repair.to_dict()
        operations = {operation["operation"] for operation in payload["operations"]}

        self.assertTrue(repair.changed)
        self.assertGreaterEqual(repair.after.topology_score, repair.before.topology_score)
        self.assertEqual(len(repair.faces), 1)
        self.assertEqual(len(repair.vertices), 3)
        self.assertIn("drop_invalid_or_degenerate_faces", operations)
        self.assertIn("drop_duplicate_faces", operations)
        self.assertIn("drop_non_largest_components", operations)
        self.assertIn("drop_loose_vertices", operations)

        plan = topology_repair_plan(repair.before)
        self.assertEqual(plan["status"], "repair_recommended")
        self.assertTrue(plan["steps"])

    def test_silhouette_metric_contract_contains_required_pass_reason(self) -> None:
        reference = np.zeros((8, 8), dtype=bool)
        candidate = np.zeros((8, 8), dtype=bool)
        reference[2:6, 2:6] = True
        candidate[3:7, 3:7] = True
        probability = candidate.astype(np.float32) * 0.8
        confidence = np.full(candidate.shape, 0.75, dtype=np.float32)

        result = silhouette_metric_result(
            reference,
            candidate,
            view="front",
            min_area_iou=0.9,
            reference_probability=reference.astype(np.float32),
            candidate_probability=probability,
            candidate_confidence=confidence,
        )
        payload = result.to_dict()

        self.assertIn("area_iou", payload)
        self.assertIn("boundary_iou", payload)
        self.assertIn("soft_iou", payload)
        self.assertIn("signed_distance_loss", payload)
        self.assertTrue(payload["required"])
        self.assertFalse(payload["pass"])
        self.assertIn("area_iou", payload["reason"])

    def test_surface_and_volume_metrics(self) -> None:
        points_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        points_b = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        surface = chamfer_distance(points_a, points_b)
        self.assertGreater(surface.chamfer_l1, 0.0)

        volume_a = np.zeros((3, 3, 3), dtype=bool)
        volume_b = np.zeros((3, 3, 3), dtype=bool)
        volume_a[1, 1, 1] = True
        volume_b[1, 1, 1] = True
        volume_b[2, 2, 2] = True
        overlap = volume_overlap(volume_a, volume_b)
        self.assertAlmostEqual(overlap.iou, 0.5)
        self.assertGreater(overlap.false_positive_rate, 0.0)

    def test_budget_reports(self) -> None:
        report = evaluate_budgets(
            {"iou": {"front": 0.92}, "elapsed_ms": 12.0},
            {
                "iou.front": {"threshold": 0.9, "mode": "min"},
                "elapsed_ms": {"threshold": 20.0, "mode": "max"},
            },
        )
        self.assertTrue(report.passed)

        regression = compare_metric_delta(
            {"score": 0.95},
            {"score": 0.93},
            tolerance={"score": 0.01},
        )
        self.assertFalse(regression.passed)


if __name__ == "__main__":
    unittest.main()
