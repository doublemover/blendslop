"""Pure tests for topology, surface, and budget metrics."""

from __future__ import annotations

import unittest

import numpy as np

from metrics.budgets import compare_metric_delta, evaluate_budgets
from metrics.surface import chamfer_distance, volume_overlap
from metrics.topology import mesh_topology_report, topology_penalty


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
        self.assertEqual(topology_penalty(report), 0.0)

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
