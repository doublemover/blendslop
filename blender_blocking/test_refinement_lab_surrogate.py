"""Tests for refinement-lab learned surrogate prioritization."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from refinement_lab.cli import main as refinement_cli_main
from refinement_lab.contracts import (
    ExperimentCase,
    ExperimentPlan,
    ExperimentResult,
    ExperimentVariant,
)
from refinement_lab.result_index import ResultIndex
from refinement_lab.surrogate import (
    build_training_examples,
    prioritize_variants,
    surrogate_report,
    train_surrogate,
)


def _result(variant_id: str, resolution: int, score: float) -> ExperimentResult:
    return ExperimentResult(
        run_id="run",
        case_id="case",
        variant_id=variant_id,
        mode="visual_hull_voxel",
        status="pass",
        exit_code=0,
        started_utc="2026-05-09T00:00:00Z",
        finished_utc="2026-05-09T00:00:01Z",
        elapsed_s=1.0,
        backend_result={"status": "success"},
        metrics={
            "average_iou": score,
            "front_iou": score,
            "side_iou": score - 0.01,
            "top_iou": score - 0.02,
            "boundary_iou_mean": score - 0.03,
            "topology_score": 0.8,
            "editability_score": 0.25,
        },
    )


def _variant(variant_id: str, resolution: int) -> ExperimentVariant:
    return ExperimentVariant(
        variant_id=variant_id,
        label=variant_id,
        mode="visual_hull_voxel",
        validation_mode="backend-status",
        parameters={"vh_resolution": resolution},
        cli_args=("--vh-resolution", str(resolution)),
        tags=("visual-hull",),
    )


class RefinementLabSurrogateTests(unittest.TestCase):
    def test_surrogate_trains_and_prioritizes_variants(self) -> None:
        variants = (
            _variant("vh-32", 32),
            _variant("vh-64", 64),
            _variant("vh-128", 128),
        )
        results = (
            _result("vh-32", 32, 0.62),
            _result("vh-64", 64, 0.74),
            _result("vh-128", 128, 0.9),
        )

        model = train_surrogate(
            results,
            variants_by_id={variant.variant_id: variant for variant in variants},
        )
        predictions = prioritize_variants(model, variants)

        self.assertEqual(model.example_count, 3)
        self.assertGreater(len(model.feature_names), 0)
        self.assertEqual(predictions[0].variant_id, "vh-128")
        self.assertEqual(predictions[0].rank, 1)
        self.assertGreater(predictions[0].predicted_score, predictions[-1].predicted_score)

    def test_surrogate_report_round_trips_json_shape(self) -> None:
        variants = (_variant("vh-32", 32), _variant("vh-64", 64))
        results = (_result("vh-32", 32, 0.62), _result("vh-64", 64, 0.8))

        examples = build_training_examples(
            results,
            variants_by_id={variant.variant_id: variant for variant in variants},
        )
        report = surrogate_report(results, variants, top_k=1)

        self.assertEqual(len(examples), 2)
        self.assertEqual(report["schema_version"], "refinement_surrogate_report_v1")
        self.assertEqual(report["model"]["example_count"], 2)  # type: ignore[index]
        self.assertEqual(report["prediction_count"], 1)

    def test_surrogate_cli_writes_priority_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            variants = (_variant("vh-32", 32), _variant("vh-64", 64))
            plan = ExperimentPlan(
                plan_id="plan",
                suite="synthetic-smoke",
                track="visual-hull-quality",
                search="grid",
                objective="quality_win",
                output_root=root,
                run_id="run",
                cases=(
                    ExperimentCase(
                        case_id="case",
                        suite="synthetic-smoke",
                        source="synthetic",
                    ),
                ),
                variants=variants,
            )
            plan.write(root / "plan.json")
            index = ResultIndex(root)
            index.append(_result("vh-32", 32, 0.62))
            index.append(_result("vh-64", 64, 0.8))
            out = root / "surrogate.json"

            exit_code = refinement_cli_main(
                [
                    "surrogate",
                    "--run-root",
                    str(root),
                    "--out",
                    str(out),
                    "--top-k",
                    "1",
                ]
            )
            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["prediction_count"], 1)
        self.assertEqual(payload["predictions"][0]["variant_id"], "vh-64")


if __name__ == "__main__":
    unittest.main()
