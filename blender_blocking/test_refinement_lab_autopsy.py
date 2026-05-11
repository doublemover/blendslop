"""Tests for candidate autopsy classification."""

from __future__ import annotations

import unittest

from refinement_lab.adaptive import proposals_from_result_payload
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

    def test_topology_problem_emits_repair_plan_and_adaptive_proposal(self) -> None:
        result = ExperimentResult(
            run_id="r",
            case_id="bad-mesh",
            variant_id="vh",
            mode="visual_hull_voxel",
            status="fail",
            exit_code=1,
            started_utc="s",
            finished_utc="f",
            elapsed_s=1.0,
            metrics={
                "average_iou": 0.91,
                "front_iou": 0.91,
                "side_iou": 0.92,
                "top_iou": 0.90,
                "topology": {
                    "score": 0.62,
                    "loose_vertices": 2,
                    "boundary_edges": 12,
                    "non_manifold_edges": 4,
                    "degenerate_faces": 1,
                    "zero_area_faces": 0,
                },
            },
        )

        autopsy = autopsy_candidate(result)
        actions = "\n".join(autopsy["topology_repair_plan"]["actions"])
        proposals = proposals_from_result_payload({"autopsy_pack": autopsy})
        proposal_ids = {proposal.proposal_id for proposal in proposals}

        self.assertIn("topology_repair_plan", autopsy)
        self.assertIn("remove tiny shells", actions)
        self.assertIn("fill boundary loops", actions)
        self.assertIn("weld close vertices", actions)
        self.assertIn("avoid repairs that erase silhouette detail", actions)
        self.assertTrue(
            any(
                proposal_id.startswith("autopsy-topology-repair")
                for proposal_id in proposal_ids
            )
        )


if __name__ == "__main__":
    unittest.main()
