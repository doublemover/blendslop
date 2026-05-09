"""Tests for target uncertainty reports."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from evaluation.uncertainty import uncertainty_report_from_target
from reconstruction.types import (
    CandidateMetrics,
    OrthographicCameraSpec,
    ReconstructionTarget,
    ViewConstraint,
)


class EvaluationUncertaintyTests(unittest.TestCase):
    def test_target_uncertainty_report_combines_masks_and_profile_variance(self) -> None:
        probability = np.full((4, 4), 0.5, dtype=np.float32)
        confidence = np.full((4, 4), 0.75, dtype=np.float32)
        boundary = np.zeros((4, 4), dtype=np.float32)
        boundary[1:3, 1:3] = 0.5
        target = ReconstructionTarget(
            constraints=(
                ViewConstraint(
                    view="front",
                    mask=np.ones((4, 4), dtype=bool),
                    camera=OrthographicCameraSpec("front", "z"),
                    uncertainty=SimpleNamespace(
                        foreground_prob=probability,
                        confidence=confidence,
                        boundary_uncertainty=boundary,
                    ),
                ),
            ),
            extras={
                "profile_band_distribution": {
                    "front": {
                        "width_std_mean": 2.0,
                        "center_std_mean": 1.0,
                    }
                }
            },
        )

        report = uncertainty_report_from_target(target)
        payload = report.to_dict()

        self.assertEqual(payload["view_count"], 1)
        self.assertAlmostEqual(payload["confidence_mean"], 0.75)
        self.assertAlmostEqual(payload["foreground_entropy_mean"], 1.0, places=6)
        self.assertGreater(payload["consistency_score"], 0.0)
        self.assertLess(payload["consistency_score"], 1.0)
        self.assertEqual(payload["views"][0]["profile_width_std_mean"], 2.0)  # type: ignore[index]

    def test_candidate_metrics_derives_uncertainty_consistency_from_extras(self) -> None:
        metrics = CandidateMetrics(
            extras={
                "uncertainty_report": {
                    "consistency_score": 0.42,
                    "confidence_score": 0.5,
                }
            }
        )

        self.assertEqual(metrics.uncertainty_consistency, 0.42)


if __name__ == "__main__":
    unittest.main()

