"""Tests for refinement lab matrix generation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    from refinement_lab.contracts import ExperimentVariant
    from refinement_lab.matrix import build_experiment_plan, load_variants
except ModuleNotFoundError:  # pragma: no cover - package unittest path
    from blender_blocking.refinement_lab.contracts import ExperimentVariant
    from blender_blocking.refinement_lab.matrix import (
        build_experiment_plan,
        load_variants,
    )


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

    def test_material_appearance_suite_carries_appearance_targets(self) -> None:
        plan = build_experiment_plan(
            suite="synthetic-material-appearance",
            track="shape-program-editability",
            search="coordinate",
            objective="profile_editable",
            output_root=Path("temp/material-appearance"),
            max_runs=2,
        )

        self.assertGreaterEqual(len(plan.cases), 1)
        first_case = plan.cases[0]
        self.assertEqual(first_case.source, "synthetic")
        self.assertIn("material_", first_case.synthetic_shape_id)
        self.assertIn("appearance", first_case.expected_targets["targets"])
        self.assertEqual(
            first_case.metadata["synthetic_suite"],
            "material-appearance",
        )

    def test_external_variants_can_replace_generated_track(self) -> None:
        variant = ExperimentVariant(
            "adaptive-boundary",
            "adaptive boundary pass",
            "ensemble",
            validation_mode="backend-status",
            cli_args=("--reconstruction-mode", "ensemble"),
            stage="adaptive_refinement",
        )

        plan = build_experiment_plan(
            suite="default-vase",
            track="visual-hull-transform",
            search="coordinate",
            objective="quality_win",
            output_root=Path("temp/adaptive"),
            max_runs=2,
            external_variants=(variant,),
            external_variant_mode="replace",
        )

        self.assertEqual(
            [item.variant_id for item in plan.variants], [variant.variant_id]
        )
        self.assertEqual(plan.metadata["external_variant_mode"], "replace")
        self.assertEqual(plan.metadata["external_variant_count"], 1)

    def test_load_variants_accepts_adaptive_variant_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adaptive-variants.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "refinement_run_adaptive_variants_v1",
                        "variants": [
                            {
                                "variant_id": "adaptive-boundary",
                                "label": "Boundary pass",
                                "mode": "ensemble",
                                "validation_mode": "backend-status",
                                "parameters": {"proposal_id": "p"},
                                "cli_args": [
                                    "--reconstruction-mode",
                                    "ensemble",
                                    "--validation-mode",
                                    "backend-status",
                                ],
                                "tags": ["adaptive"],
                                "stage": "adaptive_refinement",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            variants = load_variants(path)

            self.assertEqual(len(variants), 1)
            self.assertEqual(variants[0].variant_id, "adaptive-boundary")
            self.assertEqual(variants[0].stage, "adaptive_refinement")


if __name__ == "__main__":
    unittest.main()
