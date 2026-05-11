"""Tests for refinement lab CLI helpers."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    from refinement_lab import cli
    from refinement_lab.commands.common import REPO_ROOT, _run_options_from_args
    from refinement_lab.contracts import ExperimentResult
    from refinement_lab.result_index import ResultIndex
except ModuleNotFoundError:  # pragma: no cover - package unittest path
    from blender_blocking.refinement_lab import cli
    from blender_blocking.refinement_lab.commands.common import REPO_ROOT, _run_options_from_args
    from blender_blocking.refinement_lab.contracts import ExperimentResult
    from blender_blocking.refinement_lab.result_index import ResultIndex


class RefinementLabCliTests(unittest.TestCase):
    def test_list_commands(self) -> None:
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["list-suites"]), 0)
            self.assertEqual(cli.main(["list-tracks"]), 0)

    def test_plan_command_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plan.json"
            with redirect_stdout(io.StringIO()):
                code = cli.main(
                    [
                        "plan",
                        "--suite",
                        "default-vase",
                        "--track",
                        "visual-hull-transform",
                        "--search",
                        "coordinate",
                        "--max-runs",
                        "2",
                        "--out",
                        str(out),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(out.exists())

    def test_plan_command_can_replace_with_variant_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            variant_file = root / "adaptive-variants.json"
            out = root / "plan.json"
            variant_file.write_text(
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

            with redirect_stdout(io.StringIO()):
                code = cli.main(
                    [
                        "plan",
                        "--suite",
                        "default-vase",
                        "--track",
                        "visual-hull-transform",
                        "--search",
                        "coordinate",
                        "--variant-file",
                        str(variant_file),
                        "--variant-file-mode",
                        "replace",
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["variants"]), 1)
            self.assertEqual(payload["variants"][0]["variant_id"], "adaptive-boundary")

    def test_adapt_command_writes_proposals_and_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_json = root / "result.json"
            proposals_json = root / "adaptive-proposals.json"
            variants_json = root / "adaptive-variants.json"
            result_json.write_text(
                json.dumps(
                    {
                        "backend_result": {
                            "status": "degraded",
                            "evaluation_bundles": [
                                {
                                    "status": "fail",
                                    "metrics": {
                                        "editability.editable_reconstruction_index": 0.25,
                                        "silhouette.mean_boundary_iou": 0.18,
                                        "silhouette.mean_signed_distance_loss": 0.14,
                                        "silhouette.min_view_iou": 0.42,
                                        "topology.score": 0.45,
                                    },
                                    "failures": [
                                        {"code": "boundary_mismatch"},
                                        {"code": "topology_non_manifold"},
                                    ],
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                code = cli.main(
                    [
                        "adapt",
                        "--result-json",
                        str(result_json),
                        "--out",
                        str(proposals_json),
                        "--variants-out",
                        str(variants_json),
                        "--parent-variant",
                        "baseline",
                        "--max-proposals",
                        "3",
                    ]
                )

            self.assertEqual(code, 0)
            proposals = json.loads(proposals_json.read_text(encoding="utf-8"))
            variants = json.loads(variants_json.read_text(encoding="utf-8"))
            self.assertEqual(
                proposals["schema_version"],
                "refinement_adaptive_proposals_v1",
            )
            self.assertEqual(proposals["proposal_count"], 3)
            self.assertEqual(
                variants["schema_version"],
                "refinement_adaptive_variants_v1",
            )
            self.assertEqual(variants["variant_count"], 3)
            self.assertTrue(
                all(
                    variant["parent_variant_id"] == "baseline"
                    for variant in variants["variants"]
                )
            )
            for variant in variants["variants"]:
                self.assertIn("--reconstruction-mode", variant["cli_args"])
                self.assertIn("--validation-mode", variant["cli_args"])

    def test_calibrate_masks_command_writes_executable_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "ref.png"
            cand = root / "cand.png"
            out = root / "calibration.json"
            _write_square_mask(ref, x0=10, y0=12, size=14)
            _write_square_mask(cand, x0=12, y0=13, size=14)

            with redirect_stdout(io.StringIO()):
                code = cli.main(
                    [
                        "calibrate-masks",
                        "--reference",
                        f"front={ref}",
                        "--candidate",
                        f"front={cand}",
                        "--max-offset-px",
                        "3",
                        "--step-px",
                        "1",
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "improved")
            self.assertEqual(
                payload["mask_alignment"]["best_by_view"]["front"]["offset_px"],
                [-2, -1],
            )

    def test_patch_masks_command_refines_local_residuals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "ref.png"
            cand = root / "cand.png"
            out = root / "patches.json"
            refined = root / "refined.png"
            _write_square_mask(ref, x0=10, y0=10, size=24)
            _write_square_mask(cand, x0=10, y0=10, size=24, clear_box=(24, 18, 10, 12))

            with redirect_stdout(io.StringIO()):
                code = cli.main(
                    [
                        "patch-masks",
                        "--reference",
                        str(ref),
                        "--candidate",
                        str(cand),
                        "--patch-sizes",
                        "16,24",
                        "--max-patches",
                        "3",
                        "--out",
                        str(out),
                        "--refined-mask-out",
                        str(refined),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["schema_version"],
                "content_adaptive_patch_mask_result_v1",
            )
            self.assertTrue(payload["accepted"])
            self.assertGreater(payload["improvement"]["area_iou_delta"], 0.0)
            self.assertGreaterEqual(len(payload["patches"]), 1)
            self.assertTrue(refined.exists())

    def test_module_entrypoint_help_runs_from_repo_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "blender_blocking.refinement_lab.cli",
                "--help",
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("list-tracks", completed.stdout)

    def test_promote_blocks_review_required_candidate_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = ResultIndex(root)
            index.append(
                ExperimentResult(
                    run_id="r",
                    case_id="c",
                    variant_id="shape-program",
                    mode="shape_program",
                    status="pass",
                    exit_code=0,
                    started_utc="s",
                    finished_utc="f",
                    elapsed_s=1.0,
                    backend_result={"status": "research_only"},
                    metrics={
                        "average_iou": 0.95,
                        "front_iou": 0.95,
                        "side_iou": 0.95,
                        "top_iou": 0.95,
                    },
                )
            )
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                code = cli.main(
                    [
                        "promote",
                        "--run-root",
                        str(root),
                        "--variant",
                        "shape-program",
                        "--out",
                        str(root / "promoted.json"),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("research_only", stderr.getvalue())
            self.assertFalse((root / "promoted.json").exists())

    def test_promote_can_record_review_required_candidate_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "promoted.json"
            index = ResultIndex(root)
            index.append(
                ExperimentResult(
                    run_id="r",
                    case_id="c",
                    variant_id="degraded",
                    mode="visual_hull_voxel",
                    status="pass",
                    exit_code=0,
                    started_utc="s",
                    finished_utc="f",
                    elapsed_s=1.0,
                    backend_result={"status": "degraded", "degraded": True},
                    metrics={
                        "average_iou": 0.92,
                        "front_iou": 0.92,
                        "side_iou": 0.92,
                        "top_iou": 0.92,
                    },
                )
            )
            with redirect_stdout(io.StringIO()):
                code = cli.main(
                    [
                        "promote",
                        "--run-root",
                        str(root),
                        "--variant",
                        "degraded",
                        "--out",
                        str(out),
                        "--allow-review-required",
                    ]
                )
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["promotion"]["tier"], "degraded")

    def test_run_options_resolve_relative_cache_root_under_repo(self) -> None:
        args = argparse.Namespace(
            html_report=False,
            write_overlays=False,
            bounds_debug=False,
            autopsy=False,
            fail_on_all_failed=True,
            append_global_index=False,
            report_failures="none",
            adaptive_proposals=False,
            lineage=False,
            adaptive_max_proposals=1,
            blender_exe=None,
            cache_root=Path("temp/quality-refinement-cache"),
            reference_cache=True,
            candidate_cache=True,
            resume_candidates=False,
            debug_artifact_policy="none",
            batch_index_writes=True,
        )

        options = _run_options_from_args(args)

        self.assertEqual(
            options.cache_root,
            (REPO_ROOT / "temp" / "quality-refinement-cache").resolve(strict=False),
        )


def _write_square_mask(
    path: Path,
    *,
    x0: int,
    y0: int,
    size: int,
    clear_box: tuple[int, int, int, int] | None = None,
) -> None:
    from PIL import Image

    image = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
    pixels = image.load()
    for y in range(y0, y0 + size):
        for x in range(x0, x0 + size):
            pixels[x, y] = (255, 255, 255, 255)
    if clear_box is not None:
        clear_x, clear_y, clear_w, clear_h = clear_box
        for y in range(clear_y, clear_y + clear_h):
            for x in range(clear_x, clear_x + clear_w):
                pixels[x, y] = (0, 0, 0, 0)
    image.save(path)


if __name__ == "__main__":
    unittest.main()
