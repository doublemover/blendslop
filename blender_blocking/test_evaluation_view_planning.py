"""Tests for active-view planning signals."""

from __future__ import annotations

import unittest

from evaluation.view_planning import (
    active_view_plan_payload,
    ensemble_disagreement_signal,
    suggest_next_views,
)


class ActiveViewPlanningTests(unittest.TestCase):
    def test_ensemble_disagreement_prioritizes_diagonal_view(self) -> None:
        candidates = (
            {
                "metric_result": {
                    "per_view": {
                        "front": {"area_iou": 0.92},
                        "side": {"area_iou": 0.38},
                        "top": {"area_iou": 0.86},
                    }
                }
            },
            {
                "metric_result": {
                    "per_view": {
                        "front": {"area_iou": 0.45},
                        "side": {"area_iou": 0.9},
                        "top": {"area_iou": 0.84},
                    }
                }
            },
        )

        signal = ensemble_disagreement_signal(candidates)

        self.assertEqual(signal.candidate_count, 2)
        self.assertGreater(signal.view_scores["front"], 0.0)
        self.assertGreater(signal.recommended_view_scores["front_side_45"], 0.0)
        self.assertGreater(signal.entropy, 0.0)

    def test_suggest_next_views_uses_ensemble_disagreement_metadata(self) -> None:
        candidates = (
            {
                "metrics": {
                    "front_iou": 0.94,
                    "side_iou": 0.35,
                    "top_iou": 0.88,
                }
            },
            {
                "metrics": {
                    "front_iou": 0.42,
                    "side_iou": 0.91,
                    "top_iou": 0.87,
                }
            },
        )
        bundle = {
            "status": "degraded",
            "metrics": {
                "silhouette.min_view_iou": 0.42,
                "silhouette.mean_boundary_iou": 0.4,
            },
            "candidate_bundles": candidates,
        }

        requests = suggest_next_views(
            bundle,
            existing_views=("front", "side", "top"),
            candidate_bundles=candidates,
            max_views=2,
        )

        self.assertEqual(requests[0].view_id, "front_side_45")
        self.assertGreater(requests[0].metadata["view_signal"], 0.0)
        self.assertIn("ensemble_disagreement", requests[0].metadata)
        self.assertIn("Ensemble disagreement", requests[0].reason)

    def test_plan_payload_includes_disagreement_summary(self) -> None:
        payload = active_view_plan_payload(
            {
                "candidate_id": "ensemble-a",
                "candidate_bundles": [
                    {"metrics": {"front_iou": 0.9, "side_iou": 0.4}},
                    {"metrics": {"front_iou": 0.45, "side_iou": 0.88}},
                ],
            },
            existing_views=("front", "side"),
            max_views=1,
        )

        self.assertEqual(payload["candidate_id"], "ensemble-a")
        self.assertIn("ensemble_disagreement", payload)
        self.assertEqual(payload["requests"][0]["view_id"], "front_side_45")


if __name__ == "__main__":
    unittest.main()
