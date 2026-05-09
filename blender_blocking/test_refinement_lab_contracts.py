"""Tests for refinement lab data contracts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from refinement_lab.contracts import (
    ExperimentCase,
    ExperimentPlan,
    ExperimentResult,
    ExperimentVariant,
    ParameterSpec,
)


class RefinementLabContractsTests(unittest.TestCase):
    def test_parameter_spec_bool_cli_tokens(self) -> None:
        spec = ParameterSpec("fill", cli_arg="--ref-fill-holes", value_type="bool", values=(True, False))
        self.assertEqual(spec.cli_tokens_for_value(True), ("--ref-fill-holes",))
        self.assertEqual(spec.cli_tokens_for_value(False), ("--no-ref-fill-holes",))

    def test_variant_hash_is_stable_for_mapping_order(self) -> None:
        first = ExperimentVariant("a", "A", "profile_loft", parameters={"x": 1, "y": 2})
        second = ExperimentVariant("b", "B", "profile_loft", parameters={"y": 2, "x": 1})
        self.assertEqual(first.variant_hash(), second.variant_hash())

    def test_plan_roundtrip(self) -> None:
        case = ExperimentCase(
            "case",
            "default-vase",
            "builtin_sample",
            reference_paths={
                "front": Path("front.png"),
                "side": Path("side.png"),
                "top": Path("top.png"),
            },
        )
        variant = ExperimentVariant("v", "variant", "profile_loft")
        plan = ExperimentPlan(
            plan_id="p",
            suite="default-vase",
            track="profile-loft-refinement",
            search="grid",
            objective="quality_win",
            output_root=Path("temp/run"),
            run_id="run",
            cases=(case,),
            variants=(variant,),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            plan.write(path)
            loaded = ExperimentPlan.read(path)
        self.assertEqual(loaded.plan_id, "p")
        self.assertEqual(loaded.cases[0].case_id, "case")

    def test_result_view_helpers(self) -> None:
        result = ExperimentResult(
            run_id="r",
            case_id="c",
            variant_id="v",
            mode="profile_loft",
            status="pass",
            exit_code=0,
            started_utc="s",
            finished_utc="f",
            elapsed_s=1.0,
            metrics={"average_iou": 0.9, "front_iou": 0.8, "side_iou": 0.7, "top_iou": 0.6},
        )
        self.assertAlmostEqual(result.avg_iou, 0.9)
        self.assertAlmostEqual(result.min_iou, 0.6)
        json.dumps(result.to_dict())


if __name__ == "__main__":
    unittest.main()
