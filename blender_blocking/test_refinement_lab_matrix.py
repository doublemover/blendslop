"""Tests for refinement lab matrix generation."""

from __future__ import annotations

from pathlib import Path
import unittest

from refinement_lab.matrix import build_experiment_plan


class RefinementLabMatrixTests(unittest.TestCase):
    def test_default_vase_plan(self) -> None:
        plan = build_experiment_plan(
            suite="default-vase",
            track="visual-hull-transform",
            search="coordinate",
            objective="visual_hull_alignment",
            output_root=Path("temp/refinement-runs/test"),
            max_runs=4,
        )
        self.assertEqual(len(plan.cases), 1)
        self.assertEqual(len(plan.variants), 4)
        self.assertEqual(plan.variants[0].stage, "baseline")

    def test_random_is_seeded(self) -> None:
        first = build_experiment_plan(
            suite="default-vase",
            track="profile-loft-refinement",
            search="random",
            objective="quality_win",
            output_root=Path("temp/a"),
            seed=42,
            max_runs=6,
        )
        second = build_experiment_plan(
            suite="default-vase",
            track="profile-loft-refinement",
            search="random",
            objective="quality_win",
            output_root=Path("temp/a"),
            seed=42,
            max_runs=6,
        )
        self.assertEqual(
            [variant.variant_hash() for variant in first.variants],
            [variant.variant_hash() for variant in second.variants],
        )

    def test_synthetic_smoke_plan_has_cases(self) -> None:
        plan = build_experiment_plan(
            suite="synthetic-smoke",
            track="mask-refinement",
            search="coordinate",
            objective="min_view_iou",
            output_root=Path("temp/synthetic"),
            max_runs=2,
        )
        self.assertGreaterEqual(len(plan.cases), 1)
        self.assertEqual(plan.cases[0].source, "synthetic")


if __name__ == "__main__":
    unittest.main()
