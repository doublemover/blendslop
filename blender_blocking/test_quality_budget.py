"""Pure tests for JSON quality/performance budget evaluation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

    def test_allow_missing_passes_absent_optional_mode_but_fails_present_rows(self) -> None:
        budget = {
            "schema_version": "quality_perf_budget_v1",
            "name": "mode-floor",
            "thresholds": [
                {
                    "id": "primitive.floor",
                    "artifact": "e2e",
                    "mode": "primitive_fit_refine",
                    "metric": "metrics.area_iou_min",
                    "threshold": 0.65,
                    "required": True,
                    "allow_missing": True,
                }
            ],
        }

        absent = {"matrix": [{"mode": "visual_hull_voxel", "metrics": {"area_iou_min": 0.9}}]}
        present_bad = {"matrix": [{"mode": "primitive_fit_refine", "metrics": {"area_iou_min": 0.2}}]}

        self.assertTrue(evaluate_budget_payloads(absent, budget)["passed"])
        self.assertFalse(evaluate_budget_payloads(present_bad, budget)["passed"])

    def test_evaluation_bundle_records_are_budgetable_by_sota_metrics(self) -> None:
        current = {
            "schema_version": "evaluation-bundle-v1",
            "suite": "smoke",
            "candidate_id": "candidate-a",
            "target_id": "cube",
            "mode": "shape_program",
            "metric_groups": [
                {
                    "name": "editability",
                    "status": "pass",
                    "metrics": [
                        {
                            "name": "editability.editable_reconstruction_index",
                            "value": 0.82,
                        }
                    ],
                },
                {
                    "name": "export_qa",
                    "status": "pass",
                    "metrics": [
                        {"name": "export.qa_score", "value": 0.95},
                    ],
                },
            ],
        }
        budget = {
            "schema_version": "quality_perf_budget_v1",
            "name": "evaluation",
            "thresholds": [
                {
                    "id": "editability.floor",
                    "artifact": "evaluation",
                    "case": "smoke",
                    "mode": "shape_program",
                    "shape_id": "cube",
                    "metric": "metrics.editability.editable_reconstruction_index",
                    "threshold": 0.8,
                    "required": True,
                },
                {
                    "id": "export.floor",
                    "artifact": "evaluation",
                    "metric": "metrics.export.qa_score",
                    "threshold": 0.9,
                    "required": True,
                },
            ],
        }

        report = evaluate_budget_payloads(current, budget)

        self.assertTrue(report["passed"])
        self.assertEqual(len(report["checks"]), 2)
        self.assertEqual(report["checks"][0]["value"], 0.82)

    def test_nested_e2e_evaluation_bundles_are_budgetable(self) -> None:
        current = {
            "validation_mode": "backend-status",
            "mode": "ensemble",
            "backend_result": {
                "selected": {"status": "success"},
                "evaluation_bundles": [
                    {
                        "schema_version": "evaluation-bundle-v1",
                        "suite": "smoke",
                        "candidate_id": "shape-program",
                        "target_id": "chair",
                        "mode": "shape_program",
                        "metric_groups": [
                            {
                                "name": "silhouette",
                                "status": "pass",
                                "metrics": [
                                    {
                                        "name": "silhouette.min_view_iou",
                                        "value": 0.76,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        }
        budget = {
            "schema_version": "quality_perf_budget_v1",
            "name": "nested",
            "thresholds": [
                {
                    "id": "min-view",
                    "artifact": "evaluation",
                    "case": "smoke",
                    "mode": "shape_program",
                    "shape_id": "chair",
                    "metric": "metrics.silhouette.min_view_iou",
                    "threshold": 0.75,
                    "required": True,
                }
            ],
        }

        report = evaluate_budget_payloads(current, budget)

        self.assertTrue(report["passed"])
        self.assertEqual(report["checks"][0]["value"], 0.76)

    def test_evaluation_bundle_recoverability_metrics_are_budgetable(self) -> None:
        current = {
            "schema_version": "evaluation-bundle-v1",
            "suite": "smoke",
            "candidate_id": "visual-hull",
            "target_id": "sphere",
            "mode": "visual_hull_voxel",
            "metric_groups": [
                {
                    "name": "geometry",
                    "status": "pass",
                    "metrics": [
                        {
                            "name": "geometry.recoverable.fscore_tau",
                            "value": 0.74,
                        },
                        {
                            "name": "geometry.ambiguity_gap_chamfer_l2",
                            "value": 0.015,
                        },
                    ],
                }
            ],
        }
        budget = {
            "schema_version": "quality_perf_budget_v1",
            "name": "recoverability",
            "thresholds": [
                {
                    "id": "recoverable.fscore",
                    "artifact": "evaluation",
                    "metric": "metrics.geometry.recoverable.fscore_tau",
                    "threshold": 0.7,
                    "required": True,
                },
                {
                    "id": "ambiguity.gap",
                    "artifact": "evaluation",
                    "metric": "metrics.geometry.ambiguity_gap_chamfer_l2",
                    "mode": "max",
                    "threshold": 0.02,
                    "required": True,
                },
            ],
        }

        report = evaluate_budget_payloads(current, budget)

        self.assertTrue(report["passed"])
        by_id = {check["id"]: check for check in report["checks"]}
        self.assertEqual(by_id["recoverable.fscore"]["value"], 0.74)
        self.assertEqual(by_id["ambiguity.gap"]["mode"], "max")

    def test_synthetic_geometry_matrix_metrics_are_budgetable(self) -> None:
        current = {
            "schema_version": "e2e_synthetic_matrix_v1",
            "matrix": [
                {
                    "artifact": "e2e",
                    "suite": "smoke",
                    "shape_id": "analytic-sphere",
                    "mode": "visual_hull_voxel",
                    "metrics": {
                        "passed": 1.0,
                        "synthetic_geometry_true_chamfer_l1": 0.08,
                    },
                }
            ],
        }
        budget = {
            "schema_version": "quality_perf_budget_v1",
            "name": "synthetic-geometry",
            "thresholds": [
                {
                    "id": "true.chamfer",
                    "artifact": "e2e",
                    "metric": "metrics.synthetic_geometry_true_chamfer_l1",
                    "mode": "max",
                    "threshold": 0.1,
                    "required": True,
                }
            ],
        }

        report = evaluate_budget_payloads(current, budget)

        self.assertTrue(report["passed"])
        self.assertEqual(report["checks"][0]["value"], 0.08)


if __name__ == "__main__":
    unittest.main()
