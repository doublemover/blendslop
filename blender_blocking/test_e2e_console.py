"""Pure tests for E2E console formatting helpers."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from blender_blocking.e2e.backend_status import _print_backend_summary
from blender_blocking.e2e.console import (
    REPO_ROOT,
    _display_path,
    _print_section,
    _render_summary,
    _style,
)


class E2EConsoleTests(unittest.TestCase):
    def test_display_path_shortens_repo_absolute_paths(self) -> None:
        absolute = REPO_ROOT / "temp" / "quality-refinement-runs" / "run" / "front.png"

        self.assertEqual(
            _display_path(absolute),
            "temp/quality-refinement-runs/run/front.png",
        )

    def test_render_summary_condenses_view_paths(self) -> None:
        root = REPO_ROOT / "temp" / "quality-refinement-runs" / "run" / "ref" / "case"
        output = io.StringIO()

        with redirect_stdout(output):
            _render_summary(
                "render refs car_seed_1236",
                {
                    "front": root / "front.png",
                    "side": root / "side.png",
                    "top": root / "top.png",
                },
            )

        text = output.getvalue()
        self.assertIn("render refs car_seed_1236: front side top", text)
        self.assertIn("temp/quality-refinement-runs/run/ref/case/", text)
        self.assertNotIn(str(REPO_ROOT), text)
        self.assertNotIn("front.png", text)

    def test_section_keeps_banner_on_its_own_line(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            _print_section("1/4 Reconstruct")

        text = output.getvalue()
        self.assertTrue(text.startswith("\n"))
        self.assertIn("1/4 Reconstruct", text)
        self.assertTrue(text.endswith("\n"))

    def test_backend_summary_is_compact_and_shortens_artifacts(self) -> None:
        mesh = REPO_ROOT / "temp" / "quality-refinement-runs" / "run" / "mesh.obj"
        volume = REPO_ROOT / "temp" / "quality-refinement-runs" / "run" / "volume.npz"
        output = io.StringIO()

        with redirect_stdout(output):
            passed = _print_backend_summary(
                {
                    "status": "success",
                    "selected": {
                        "backend_name": "primitive_fit",
                        "candidate_id": "baseline",
                        "status": "success",
                        "mesh_path": str(mesh),
                        "volume_path": str(volume),
                        "warnings": [],
                        "errors": [],
                        "metric_result": {
                            "area_iou_min": 0.812345,
                            "area_iou_mean": 0.9,
                            "elapsed_s": 1.25,
                        },
                    },
                    "candidates": [
                        {"candidate_id": "baseline", "status": "success"},
                        {"candidate_id": "alt", "status": "rejected"},
                    ],
                }
            )

        text = output.getvalue()
        self.assertTrue(passed)
        self.assertIn("success | backend=primitive_fit | candidate=baseline", text)
        self.assertIn("candidates total=2", text)
        self.assertIn("area_min=0.8123", text)
        self.assertNotIn(str(REPO_ROOT), text)
        self.assertLessEqual(len([line for line in text.splitlines() if line.strip()]), 5)

    def test_force_color_enables_ansi(self) -> None:
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=True):
            self.assertIn("\033[31m", _style("FAIL", "red"))


if __name__ == "__main__":
    unittest.main()
