"""Tests for refinement lab CLI helpers."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    from refinement_lab import cli
except ModuleNotFoundError:  # pragma: no cover - package unittest path
    from blender_blocking.refinement_lab import cli


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


if __name__ == "__main__":
    unittest.main()
