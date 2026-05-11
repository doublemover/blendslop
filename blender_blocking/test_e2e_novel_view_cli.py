#!/usr/bin/env python3
"""Pure tests for E2E novel-view CLI plumbing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from integration.blender_ops.render_utils import parse_orbit_view_degrees
from config import BlockingConfig
from e2e.validator import E2EValidator
from test_e2e_validation import (
    _aggregate_novel_reports,
    _apply_cli_args,
    _backend_cost_report_from_payload,
    _cost_gate_report,
    _cost_payload,
    _matrix_cost_summary,
    _matrix_metrics,
    _novel_threshold,
    _novel_view_gate,
    _novel_view_names_from_args,
    _parse_args,
    _parse_view_reference_entries,
    _resolve_novel_view_inputs,
)


class E2ENovelViewCliTests(unittest.TestCase):
    def test_orbit_view_parser_accepts_common_names(self) -> None:
        self.assertEqual(parse_orbit_view_degrees("orbit_045"), 45.0)
        self.assertEqual(parse_orbit_view_degrees("azimuth-135"), 135.0)
        self.assertEqual(parse_orbit_view_degrees("view_405"), 45.0)
        self.assertIsNone(parse_orbit_view_degrees("front"))

    def test_view_reference_entries_are_validated_and_normalized(self) -> None:
        refs = _parse_view_reference_entries(
            ("orbit_045=temp/ref.png", "front=C:/tmp/front.png")
        )

        self.assertEqual(refs["orbit_045"], Path("temp/ref.png"))
        self.assertEqual(refs["front"], Path("C:/tmp/front.png"))
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_view_reference_entries(("diagonal=temp/ref.png",))

    def test_novel_view_names_include_angles_references_and_manual_views(self) -> None:
        args = argparse.Namespace(
            novel_view=("orbit_090",),
            novel_view_angles=("45", "135"),
            novel_view_reference=("orbit_045=temp/ref.png",),
        )

        names = _novel_view_names_from_args(args)

        self.assertEqual(names, ("orbit_090", "orbit_045", "orbit_135"))

    def test_novel_view_manifest_supplies_references_views_and_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "novel.json"
            manifest.write_text(
                json.dumps(
                    {
                        "references": {"orbit_045": "refs/holdout_045.png"},
                        "views": [
                            {"view": "orbit_090", "reference": "refs/holdout_090.png"},
                            "orbit_135",
                        ],
                        "angles": [225],
                        "metrics": {
                            "compute_ssim": False,
                            "compute_lpips": True,
                            "psnr_threshold": 24.0,
                            "ssim_threshold": 0.7,
                            "lpips_threshold": 0.35,
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = _parse_args(
                [
                    "--reconstruction-mode",
                    "legacy",
                    "--validation-mode",
                    "novel-view",
                    "--novel-view-manifest",
                    str(manifest),
                    "--novel-view-reference",
                    "orbit_315=manual.png",
                ]
            )

            (
                references,
                names,
                compute_ssim,
                compute_lpips,
                psnr_threshold,
                ssim_threshold,
                lpips_threshold,
            ) = _resolve_novel_view_inputs(args)

        self.assertEqual(
            names,
            ("orbit_045", "orbit_090", "orbit_135", "orbit_225", "orbit_315"),
        )
        self.assertEqual(references["orbit_045"], root / "refs" / "holdout_045.png")
        self.assertEqual(references["orbit_315"], Path("manual.png"))
        self.assertFalse(compute_ssim)
        self.assertTrue(compute_lpips)
        self.assertEqual(psnr_threshold, 24.0)
        self.assertEqual(ssim_threshold, 0.7)
        self.assertEqual(lpips_threshold, 0.35)

    def test_threshold_zero_disables_metric_gate(self) -> None:
        self.assertIsNone(_novel_threshold(0.0))
        self.assertIsNone(_novel_threshold(None))
        self.assertEqual(_novel_threshold(20.0), 20.0)

    def test_novel_view_gate_tracks_per_metric_failures(self) -> None:
        gate = _novel_view_gate(
            {"psnr": 18.0, "ssim": 0.7, "lpips": 0.4},
            psnr_threshold=20.0,
            ssim_threshold=0.65,
            lpips_threshold=0.35,
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(len(gate["failures"]), 2)
        self.assertIn("PSNR", gate["failures"][0])
        self.assertIn("LPIPS", gate["failures"][1])

    def test_aggregate_novel_reports_requires_all_views_and_summary_gate(self) -> None:
        summary = _aggregate_novel_reports(
            {
                "front": {
                    "psnr": 32.0,
                    "ssim": 0.91,
                    "lpips": 0.12,
                    "mse": 0.01,
                    "gate": {"passed": True},
                },
                "orbit_045": {
                    "psnr": 26.0,
                    "ssim": 0.82,
                    "lpips": 0.18,
                    "mse": 0.02,
                    "gate": {"passed": True},
                },
            },
            missing_views=(),
            psnr_threshold=20.0,
            ssim_threshold=0.65,
            lpips_threshold=0.35,
        )

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["image_count"], 2)
        self.assertAlmostEqual(summary["psnr"], 29.0)
        self.assertAlmostEqual(summary["ssim"], 0.865)

    def test_novel_reference_map_prefers_inferred_orbit_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            front = root / "front.png"
            orbit = root / "orbit_045.png"
            front.write_bytes(b"front")
            orbit.write_bytes(b"orbit")
            validator = E2EValidator(novel_view_names=("orbit_045",))

            refs = validator._novel_reference_map({"front": str(front)})

        self.assertEqual(refs, {"orbit_045": str(orbit)})

    def test_texture_material_cli_flags_apply_to_shape_program_config(self) -> None:
        args = _parse_args(
            [
                "--reconstruction-mode",
                "shape_program",
                "--validation-mode",
                "backend-status",
                "--evaluate-texture-materials",
                "--texture-reference-dir",
                "temp/texture_refs",
                "--uv-strict",
                "--material-target",
                "simple",
                "--max-texture-memory-mb",
                "64",
            ]
        )
        cfg = BlockingConfig()

        _apply_cli_args(cfg, args)

        self.assertTrue(cfg.shape_program.evaluate_texture_materials)
        self.assertEqual(cfg.shape_program.texture_reference_dir, "temp/texture_refs")
        self.assertTrue(cfg.shape_program.uv_strict)
        self.assertEqual(cfg.shape_program.material_target, "simple")
        self.assertEqual(cfg.shape_program.max_texture_memory_mb, 64.0)

    def test_e2e_cost_payload_extracts_backend_cost_and_gates_thresholds(self) -> None:
        backend = {
            "selected": {
                "metric_result": {
                    "extras": {
                        "cost_report": {
                            "total_wall_ms": 40.0,
                            "stages": [
                                {
                                    "stage": "backend_reconstruct",
                                    "status": "pass",
                                    "wall_ms": 40.0,
                                }
                            ],
                        }
                    }
                }
            }
        }
        cost = _cost_payload({"total_wall_ms": 10.0, "stages": []}, backend)
        passing = _cost_gate_report(
            cost,
            max_wall_ms=60.0,
            max_backend_wall_ms=50.0,
        )
        failing = _cost_gate_report(
            cost,
            max_wall_ms=45.0,
            max_backend_wall_ms=30.0,
        )

        self.assertEqual(_backend_cost_report_from_payload(backend)["total_wall_ms"], 40.0)
        self.assertEqual(cost["combined_total_wall_ms"], 50.0)
        self.assertTrue(passing["passed"])
        self.assertFalse(failing["passed"])
        self.assertEqual(len(failing["failures"]), 2)

    def test_matrix_metrics_and_summary_include_e2e_costs(self) -> None:
        payload = {
            "passed": False,
            "cost_report": {
                "validation": {"total_wall_ms": 10.0},
                "backend": {"total_wall_ms": 40.0},
                "combined_total_wall_ms": 50.0,
            },
            "cost_gate": {"passed": False},
        }

        metrics = _matrix_metrics(payload, passed=False)
        summary = _matrix_cost_summary(
            [
                {
                    "case": "smoke:cube",
                    "name": "cube/visual_hull_voxel",
                    "mode": "visual_hull_voxel",
                    "status": "fail",
                    "passed": False,
                    "cost_report": payload["cost_report"],
                    "cost_gate": payload["cost_gate"],
                }
            ]
        )

        self.assertEqual(metrics["cost_combined_total_wall_ms"], 50.0)
        self.assertEqual(metrics["cost_backend_total_wall_ms"], 40.0)
        self.assertEqual(metrics["cost_gate_passed"], 0.0)
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["failed_gate_count"], 1)
        self.assertEqual(summary["total_combined_wall_ms"], 50.0)

    def test_cost_cli_flags_parse_without_blender(self) -> None:
        args = _parse_args(
            [
                "--reconstruction-mode",
                "visual_hull_voxel",
                "--validation-mode",
                "backend-status",
                "--cost-report-json",
                "temp/cost.json",
                "--cost-track-memory",
                "--cost-fail-max-wall-ms",
                "1000",
                "--cost-fail-max-backend-wall-ms",
                "500",
            ]
        )

        self.assertEqual(args.cost_report_json, Path("temp/cost.json"))
        self.assertTrue(args.cost_track_memory)
        self.assertEqual(args.cost_fail_max_wall_ms, 1000.0)
        self.assertEqual(args.cost_fail_max_backend_wall_ms, 500.0)


if __name__ == "__main__":
    unittest.main()
