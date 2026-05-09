"""Tests for refinement lab runner helpers."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

BLENDER_BLOCKING_ROOT = Path(__file__).resolve().parent
if str(BLENDER_BLOCKING_ROOT) not in sys.path:
    sys.path.insert(0, str(BLENDER_BLOCKING_ROOT))

try:
    from config import BlockingConfig
    from refinement_lab.contracts import (
        ExperimentCase,
        ExperimentPlan,
        ExperimentResult,
        ExperimentVariant,
    )
    from refinement_lab.runner import (
        InProcessBlenderRunner,
        RunOptions,
        _apply_variant_to_config,
        _metrics_from_payload,
        _variant_command,
    )
    from test_e2e_validation import (
        E2EValidator,
        _build_refinement_plan_from_args,
        _parse_args,
        _refinement_requested,
        _render_filename_prefix,
    )
except ModuleNotFoundError:  # pragma: no cover - package unittest path
    from blender_blocking.config import BlockingConfig
    from blender_blocking.refinement_lab.contracts import (
        ExperimentCase,
        ExperimentPlan,
        ExperimentResult,
        ExperimentVariant,
    )
    from blender_blocking.refinement_lab.runner import (
        InProcessBlenderRunner,
        RunOptions,
        _apply_variant_to_config,
        _metrics_from_payload,
        _variant_command,
    )
    from blender_blocking.test_e2e_validation import (
        E2EValidator,
        _build_refinement_plan_from_args,
        _parse_args,
        _refinement_requested,
        _render_filename_prefix,
    )


class RefinementLabRunnerTests(unittest.TestCase):
    def test_apply_variant_to_config_uses_parameters_and_overrides(self) -> None:
        cfg = BlockingConfig()
        variant = ExperimentVariant(
            "v",
            "variant",
            "profile_loft",
            parameters={
                "profile_samples": 120,
                "mesh_radial_segments": 48,
            },
            config_overrides={
                "render_silhouette": {"resolution": [256, 256]},
                "profile_sampling": {"smoothing_window": 5},
            },
        )

        _apply_variant_to_config(cfg, variant)

        self.assertEqual(cfg.reconstruction.reconstruction_mode, "profile_loft")
        self.assertEqual(cfg.profile_sampling.num_samples, 120)
        self.assertEqual(cfg.profile_sampling.smoothing_window, 5)
        self.assertEqual(cfg.mesh_from_profile.radial_segments, 48)
        self.assertEqual(cfg.render_silhouette.resolution, (256, 256))

    def test_apply_variant_to_config_uses_shape_program_parameters(self) -> None:
        cfg = BlockingConfig()
        variant = ExperimentVariant(
            "v",
            "variant",
            "shape_program",
            parameters={
                "shape_root_strategy": "profile_lathe",
                "shape_residual_policy": "report",
                "shape_max_nodes": 96,
                "shape_editability_bias": 0.85,
                "shape_compile_blender": False,
                "shape_lathe_segments": 64,
                "shape_bevel_modifier": False,
                "shape_weighted_normals": False,
            },
        )

        _apply_variant_to_config(cfg, variant)

        self.assertEqual(cfg.reconstruction.reconstruction_mode, "shape_program")
        self.assertEqual(cfg.shape_program.root_strategy, "profile_lathe")
        self.assertEqual(cfg.shape_program.residual_policy, "report")
        self.assertEqual(cfg.shape_program.max_nodes, 96)
        self.assertEqual(cfg.shape_program.editability_bias, 0.85)
        self.assertFalse(cfg.shape_program.compile_blender)
        self.assertEqual(cfg.shape_program.lathe_segments, 64)
        self.assertFalse(cfg.shape_program.bevel_modifier)
        self.assertFalse(cfg.shape_program.weighted_normals)

    def test_variant_command_records_reference_paths(self) -> None:
        variant = ExperimentVariant(
            "v",
            "variant",
            "profile_loft",
            cli_args=("--reconstruction-mode", "profile_loft"),
        )
        command = _variant_command(
            variant,
            result_json=Path("temp/result.json"),
            render_dir=Path("temp/renders"),
            artifact_root=Path("temp/artifacts"),
            reference_paths={
                "front": Path("front.png"),
                "side": Path("side.png"),
                "top": Path("top.png"),
            },
        )

        self.assertIn("--front", command)
        self.assertIn("--side", command)
        self.assertIn("--top", command)
        self.assertIn("--render-output-dir", command)
        self.assertIn("--artifact-output-root", command)

    def test_metrics_from_payload_reads_nested_evaluation_bundles(self) -> None:
        payload = {
            "validation_mode": "backend-status",
            "backend_result": {
                "evaluation_bundles": [
                    {
                        "metric_groups": [
                            {
                                "name": "silhouette",
                                "metrics": [
                                    {
                                        "name": "silhouette.min_view_iou",
                                        "value": 0.74,
                                    },
                                    {
                                        "name": "silhouette.mean_boundary_iou",
                                        "value": 0.62,
                                    },
                                ],
                            },
                            {
                                "name": "editability",
                                "metrics": [
                                    {
                                        "name": "editability.editable_reconstruction_index",
                                        "value": 0.81,
                                    }
                                ],
                            },
                        ]
                    }
                ]
            },
        }

        metrics = _metrics_from_payload(payload)

        self.assertEqual(metrics["area_iou_min"], 0.74)
        self.assertEqual(metrics["boundary_iou_mean"], 0.62)
        self.assertEqual(metrics["editability_score"], 0.81)
        self.assertEqual(metrics["silhouette_min_view_iou"], 0.74)

    def test_runner_resolves_relative_run_root(self) -> None:
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
            output_root=Path("temp/refinement-runs/relative-root-test"),
            run_id="run",
            cases=(case,),
            variants=(variant,),
        )

        runner = InProcessBlenderRunner(plan=plan, options=RunOptions())

        self.assertTrue(runner.run_root.is_absolute())

    def test_runner_writes_adaptive_outputs_from_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
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
            variant = ExperimentVariant("baseline", "baseline", "ensemble")
            plan = ExperimentPlan(
                plan_id="p",
                suite="default-vase",
                track="profile-loft-refinement",
                search="grid",
                objective="quality_win",
                output_root=Path(tmp),
                run_id="run",
                cases=(case,),
                variants=(variant,),
            )
            runner = InProcessBlenderRunner(plan=plan, options=RunOptions())
            result = ExperimentResult(
                run_id="run",
                case_id="case",
                variant_id="baseline",
                mode="ensemble",
                status="fail",
                exit_code=1,
                started_utc="2026-01-01T00:00:00Z",
                finished_utc="2026-01-01T00:00:01Z",
                elapsed_s=1.0,
                metrics={
                    "area_iou_min": 0.41,
                    "boundary_iou_mean": 0.19,
                    "editability_score": 0.25,
                    "topology_score": 0.4,
                },
            )

            proposal_path, variant_path = runner._write_adaptive_outputs([result])

            proposals = json.loads(proposal_path.read_text(encoding="utf-8"))
            variants = json.loads(variant_path.read_text(encoding="utf-8"))
            self.assertEqual(
                proposals["schema_version"],
                "refinement_run_adaptive_proposals_v1",
            )
            self.assertGreater(proposals["proposal_count"], 0)
            self.assertEqual(
                variants["schema_version"],
                "refinement_run_adaptive_variants_v1",
            )
            self.assertEqual(
                variants["variant_count"],
                proposals["proposal_count"],
            )

    def test_runner_writes_lineage_with_artifact_hashes_and_reproduce_script(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = root / "refs"
            refs.mkdir()
            for view in ("front", "side", "top"):
                (refs / f"{view}.png").write_bytes(f"{view}-mask".encode("utf-8"))
            case = ExperimentCase(
                "case",
                "default-vase",
                "builtin_sample",
                reference_paths={
                    "front": refs / "front.png",
                    "side": refs / "side.png",
                    "top": refs / "top.png",
                },
            )
            variant = ExperimentVariant("baseline", "baseline", "profile_loft")
            plan = ExperimentPlan(
                plan_id="p",
                suite="default-vase",
                track="profile-loft-refinement",
                search="grid",
                objective="quality_win",
                output_root=root / "run",
                run_id="run-lineage",
                cases=(case,),
                variants=(variant,),
                seed=99,
                metadata={"parent_run_id": "parent-run"},
            )
            runner = InProcessBlenderRunner(plan=plan, options=RunOptions())
            runner._prepare_run_root()
            result_json = runner.run_root / "cases" / "case" / "variants" / "baseline" / "result.json"
            result_json.parent.mkdir(parents=True, exist_ok=True)
            result_json.write_text('{"passed": true}\n', encoding="utf-8")
            result = ExperimentResult(
                run_id="run-lineage",
                case_id="case",
                variant_id="baseline",
                mode="profile_loft",
                status="pass",
                exit_code=0,
                started_utc="2026-01-01T00:00:00Z",
                finished_utc="2026-01-01T00:00:01Z",
                elapsed_s=1.0,
                command=("python", "example.py"),
                result_json=result_json,
                metrics={"average_iou": 0.9},
            )

            lineage_path, reproduce_path = runner._write_lineage_outputs([result])

            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            artifact_by_key = {
                artifact["key"]: artifact for artifact in lineage["artifacts"]
            }
            self.assertEqual(lineage["parent_run_id"], "parent-run")
            self.assertEqual(lineage["random_seeds"]["plan_seed"], 99)
            self.assertIn("case.front", lineage["input_hashes"])
            self.assertIn("case.baseline.result_json", artifact_by_key)
            self.assertTrue(artifact_by_key["case.baseline.result_json"]["sha256"])
            self.assertIn("reproduce_script", artifact_by_key)
            self.assertTrue(reproduce_path.exists())
            self.assertIn("blender_blocking.refinement_lab.cli", reproduce_path.read_text(encoding="utf-8"))

    def test_e2e_refinement_plan_accepts_variant_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            variant_file = root / "adaptive-variants.json"
            variant_file.write_text(
                json.dumps(
                    {
                        "schema_version": "refinement_run_adaptive_variants_v1",
                        "variants": [
                            {
                                "variant_id": "adaptive-e2e",
                                "label": "E2E adaptive variant",
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
            args = _parse_args(
                [
                    "--refinement-variant-file",
                    str(variant_file),
                    "--refinement-variant-file-mode",
                    "replace",
                ]
            )

            self.assertTrue(_refinement_requested(args))
            plan, _track, summary = _build_refinement_plan_from_args(
                args,
                BlockingConfig(),
            )

            self.assertEqual(
                [variant.variant_id for variant in plan.variants], ["adaptive-e2e"]
            )
            self.assertEqual(summary["external_variants"], 1)
            self.assertEqual(summary["variant_file_mode"], "replace")

    def test_e2e_validator_resolves_relative_output_paths(self) -> None:
        validator = E2EValidator(
            render_output_dir=Path("temp/e2e-relative-renders"),
            debug_output_dir=Path("temp/e2e-relative-debug"),
            artifact_root=Path("temp/e2e-relative-artifacts"),
            result_json=Path("temp/e2e-relative-result.json"),
        )

        self.assertTrue(validator.render_output_dir.is_absolute())
        self.assertTrue(validator.debug_output_dir.is_absolute())
        self.assertTrue(validator.artifact_root.is_absolute())
        self.assertTrue(validator.result_json.is_absolute())

    def test_render_filename_prefix_is_bounded_and_readable(self) -> None:
        prefix = _render_filename_prefix(
            "vase",
            "visual_hull_voxel",
            "visual_hull_voxel-ref_largest_component-00-71b2c4eb"
            "-with-a-very-long-parameter-label-that-would-break-render-output",
        )

        self.assertTrue(prefix.startswith("vase_visua-"))
        self.assertIn("_visual_hu-", prefix)
        self.assertTrue(prefix.endswith("_"))
        self.assertLessEqual(len(prefix), 10 + 1 + 14 + 1 + 18 + 1)


if __name__ == "__main__":
    unittest.main()
