"""Pure tests for JSON quality/performance budget evaluation."""

from __future__ import annotations

import unittest

from scripts.quality_budget import evaluate_budget_payloads


class QualityBudgetTests(unittest.TestCase):
    def test_threshold_mode_does_not_shadow_benchmark_record_matching(self) -> None:
        current = {
            "schema_version": "benchmark_perf_results_v2",
            "results": [
                {
                    "case": "quality-smoke",
                    "name": "canonicalize_mask",
                    "metrics": {"per_iter_ms": 0.5},
                }
            ],
        }
        budget = {
            "schema_version": "quality_perf_budget_v1",
            "name": "test",
            "thresholds": [
                {
                    "id": "canonicalize.max",
                    "artifact": "benchmark",
                    "case": "quality-smoke",
                    "name": "canonicalize_mask",
                    "metric": "metrics.per_iter_ms",
                    "mode": "max",
                    "threshold": 1.0,
                    "required": True,
                }
            ],
        }

        report = evaluate_budget_payloads(current, budget)
        self.assertTrue(report["passed"])
        self.assertEqual(report["checks"][0]["value"], 0.5)

    def test_record_mode_selector_still_matches_e2e_rows(self) -> None:
        current = {
            "matrix": [
                {
                    "suite": "smoke",
                    "shape_id": "cube",
                    "mode": "visual_hull_voxel",
                    "metrics": {"average_iou": 0.8},
                }
            ],
        }
        budget = {
            "schema_version": "quality_perf_budget_v1",
            "name": "test",
            "thresholds": [
                {
                    "id": "vh.iou",
                    "artifact": "e2e",
                    "case": "smoke",
                    "mode": "visual_hull_voxel",
                    "metric": "metrics.average_iou",
                    "threshold": 0.75,
                    "required": True,
                }
            ],
        }

        report = evaluate_budget_payloads(current, budget)
        self.assertTrue(report["passed"])
        self.assertEqual(report["checks"][0]["mode"], "min")
        self.assertEqual(report["checks"][0]["value"], 0.8)


if __name__ == "__main__":
    unittest.main()
