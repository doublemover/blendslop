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


if __name__ == "__main__":
    unittest.main()
