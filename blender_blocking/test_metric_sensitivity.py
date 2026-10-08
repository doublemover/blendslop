"""Pure tests for deterministic metric sensitivity probes."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.metric_sensitivity import (
    sensitivity_metric_table,
    silhouette_sensitivity_report,
)


class MetricSensitivityTests(unittest.TestCase):
    def test_silhouette_sensitivity_report_detects_expected_metric_direction(self) -> None:
        mask = np.zeros((48, 48), dtype=bool)
        mask[14:34, 12:36] = True

        report = silhouette_sensitivity_report(mask)
        by_id = {probe.probe_id: probe for probe in report.probes}

        self.assertTrue(report.passed, report.warnings)
        self.assertEqual(report.subject, "silhouette")
        self.assertAlmostEqual(report.baseline_metrics["area_iou"], 1.0)
        self.assertAlmostEqual(report.baseline_metrics["boundary_iou"], 1.0)
        self.assertAlmostEqual(report.baseline_metrics["signed_distance_loss"], 0.0)
        self.assertLess(
            by_id["shift_5px"].metrics["area_iou"],
            by_id["shift_1px"].metrics["area_iou"],
        )
        self.assertLess(
            by_id["shift_5px"].metrics["boundary_iou"],
            by_id["shift_1px"].metrics["boundary_iou"],
        )
        self.assertGreater(
            by_id["shift_5px"].metrics["signed_distance_loss"],
            by_id["shift_1px"].metrics["signed_distance_loss"],
        )
        self.assertGreater(
            by_id["false_island_far"].metrics["signed_distance_loss"],
            0.0,
        )

    def test_sensitivity_report_serializes_to_flat_table(self) -> None:
        mask = np.zeros((24, 24), dtype=bool)
        mask[6:18, 6:18] = True

        report = silhouette_sensitivity_report(mask)
        table = sensitivity_metric_table(report)

        self.assertEqual(table[0]["probe_id"], "baseline")
        self.assertTrue(table[0]["passed"])
        self.assertGreaterEqual(len(table), 6)
        self.assertTrue(all("area_iou" in row for row in table))
        self.assertTrue(all("boundary_iou" in row for row in table))
        self.assertTrue(all("signed_distance_loss" in row for row in table))

    def test_sensitivity_rejects_non_2d_masks(self) -> None:
        with self.assertRaises(ValueError):
            silhouette_sensitivity_report(np.zeros((4, 4, 4), dtype=bool))


if __name__ == "__main__":
    unittest.main()
