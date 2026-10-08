"""Pure tests for first-class silhouette evaluation gates."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.silhouette_eval import (
    SilhouetteGateConfig,
    evaluate_silhouette_pair,
    missing_silhouette_view,
    summarize_silhouette_views,
)


class SilhouetteEvaluationTests(unittest.TestCase):
    def test_summary_fails_when_one_required_view_fails_despite_high_average(self) -> None:
        views = {
            "front": {
                "area_iou": 0.99,
                "iou": 0.99,
                "boundary_iou": 0.9,
                "signed_distance_loss": 0.01,
                "required": True,
                "passed": True,
            },
            "side": {
                "area_iou": 0.99,
                "iou": 0.99,
                "boundary_iou": 0.9,
                "signed_distance_loss": 0.01,
                "required": True,
                "passed": True,
            },
            "top": {
                "area_iou": 0.2,
                "iou": 0.2,
                "boundary_iou": 0.1,
                "signed_distance_loss": 0.4,
                "required": True,
                "passed": False,
                "reason": "area_iou below threshold",
            },
        }

        summary = summarize_silhouette_views(
            views,
            config=SilhouetteGateConfig(min_area_iou=0.7),
        )

        self.assertGreater(summary["average_iou"], 0.7)
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["required_views_passed"])
        self.assertEqual(summary["failed_required_views"], ["top"])

    def test_pair_payload_contains_boundary_sdf_and_shape_diagnostics(self) -> None:
        reference = np.zeros((32, 32), dtype=bool)
        candidate = np.zeros((32, 32), dtype=bool)
        reference[8:24, 8:24] = True
        candidate[10:26, 10:26] = True

        payload = evaluate_silhouette_pair(
            reference,
            candidate,
            view="front",
            config=SilhouetteGateConfig(
                min_area_iou=0.5,
                min_boundary_iou=0.1,
                max_signed_distance_loss=0.2,
            ),
        )

        self.assertIn("boundary_iou", payload)
        self.assertIn("signed_distance_loss", payload)
        self.assertIn("precision", payload)
        self.assertIn("recall", payload)
        self.assertIn("centroid_delta_px", payload)
        self.assertEqual(payload["centroid_delta_px"], [2.0, 2.0])
        self.assertTrue(payload["passed"])

    def test_missing_required_view_is_explicit_failure(self) -> None:
        missing = missing_silhouette_view(
            "side",
            reason="missing_reference_or_render",
            config=SilhouetteGateConfig(min_area_iou=0.7),
        )
        summary = summarize_silhouette_views(
            {"front": missing},
            config=SilhouetteGateConfig(min_area_iou=0.7, required_views=("front",)),
        )

        self.assertFalse(missing["passed"])
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["missing_required_view_count"], 1)


if __name__ == "__main__":
    unittest.main()
