"""Tests for candidate autopsy classification."""

from __future__ import annotations

import unittest

from refinement_lab.candidate_autopsy import autopsy_candidate
from refinement_lab.contracts import ExperimentResult


class RefinementLabAutopsyTests(unittest.TestCase):
    def test_missing_metrics_pass_is_flagged(self) -> None:
        result = ExperimentResult(
            run_id="r",
            case_id="c",
            variant_id="v",
            mode="gaussian_ellipsoid_proxy",
            status="pass",
            exit_code=0,
            started_utc="s",
            finished_utc="f",
            elapsed_s=1.0,
            metrics={},
        )
        autopsy = autopsy_candidate(result)
        self.assertEqual(autopsy["category"], "missing_required_metrics")

    def test_visual_hull_catastrophic_is_transform_suspect_with_bounds(self) -> None:
        result = ExperimentResult(
            run_id="r",
            case_id="c",
            variant_id="v",
            mode="visual_hull_voxel",
            status="fail",
            exit_code=1,
            started_utc="s",
            finished_utc="f",
            elapsed_s=1.0,
            metrics={"average_iou": 0.1, "front_iou": 0.2, "side_iou": 0.02, "top_iou": 0.03},
            backend_result={"metric_result": {"extras": {"topology": {"topology_score": 1.0}}}},
        )
        autopsy = autopsy_candidate(result)
        categories = {finding["category"] for finding in autopsy["findings"]}
        self.assertIn("catastrophic_view_failure", categories)

    def test_hybrid_loft_one_view_failure_gets_specific_autopsy(self) -> None:
        result = ExperimentResult(
            run_id="r",
            case_id="chair",
            variant_id="hybrid",
            mode="hybrid_loft_hull",
            status="fail",
            exit_code=1,
            started_utc="s",
            finished_utc="f",
            elapsed_s=1.0,
            metrics={
                "average_iou": 0.69,
                "front_iou": 0.84,
                "side_iou": 0.91,
                "top_iou": 0.31,
            },
        )

        autopsy = autopsy_candidate(result)
        by_category = {finding["category"]: finding for finding in autopsy["findings"]}

        self.assertIn("hybrid_loft_one_view_failure", by_category)
        finding = by_category["hybrid_loft_one_view_failure"]
        self.assertEqual(finding["evidence"]["weak_views"], {"top": 0.31})
        self.assertIn("visual_hull_voxel", "\n".join(finding["recommended_next_actions"]))


if __name__ == "__main__":
    unittest.main()
