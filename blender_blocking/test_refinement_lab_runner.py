"""Tests for refinement lab runner helpers."""

from __future__ import annotations

from pathlib import Path
import unittest

from config import BlockingConfig
from refinement_lab.contracts import ExperimentCase, ExperimentPlan, ExperimentVariant
from refinement_lab.runner import (
    InProcessBlenderRunner,
    RunOptions,
    _apply_variant_to_config,
    _metrics_from_payload,
    _variant_command,
)
from test_e2e_validation import E2EValidator, _render_filename_prefix


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
