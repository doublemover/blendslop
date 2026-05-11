#!/usr/bin/env python3
"""Pure tests for the quality/refinement smoke orchestrator."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import unittest
import uuid

from scripts.run_quality_refinement_smoke import (
    DEFAULT_AMBITIOUS_MODES,
    build_phase_plan,
    format_command,
    main as smoke_main,
    parse_args,
    resolve_run_root,
)


class QualityRefinementSmokeTests(unittest.TestCase):
    def test_default_plan_is_bounded_interactive_profile(self) -> None:
        run_id = uuid.uuid4().hex
        args = parse_args(
            [
                "--dry-run",
                "--run-root",
                f"temp/quality-refinement-runs/{run_id}",
            ]
        )

        run_root, phases = build_phase_plan(args)

        self.assertTrue(str(run_root).endswith(run_id))
        self.assertTrue(any(phase.name.startswith("matrix-") for phase in phases))
        self.assertNotIn("lpips-novel-view", {phase.name for phase in phases})
        self.assertFalse(any(phase.name.startswith("refinement-") for phase in phases))
        matrix_commands = [
            phase.command for phase in phases if phase.name.startswith("matrix-")
        ]
        self.assertTrue(
            all(",".join(DEFAULT_AMBITIOUS_MODES) in command for command in matrix_commands)
        )

    def test_full_nightly_preserves_broad_workload(self) -> None:
        args = parse_args(
            [
                "--profile",
                "full-nightly",
                "--run-root",
                f"temp/quality-refinement-runs/{uuid.uuid4().hex}",
            ]
        )

        _run_root, phases = build_phase_plan(args)

        self.assertTrue(any(phase.name.startswith("matrix-") for phase in phases))
        self.assertIn("lpips-novel-view", {phase.name for phase in phases})
        self.assertTrue(any(phase.name.startswith("refinement-") for phase in phases))

    def test_run_root_must_stay_under_temp(self) -> None:
        with self.assertRaises(ValueError):
            resolve_run_root("docs/not-temp")

    def test_lpips_phase_is_separate_from_open3d_visual_hull_phase(self) -> None:
        args = parse_args(
            [
                "--profile",
                "full-nightly",
                "--run-root",
                f"temp/quality-refinement-runs/{uuid.uuid4().hex}",
                "--matrix-suites",
                "adversarial-silhouettes",
                "--refinement-targets",
                "visual-hull-quality",
            ]
        )

        _run_root, phases = build_phase_plan(args)
        lpips = next(phase for phase in phases if phase.name == "lpips-novel-view")
        visual_hull = next(
            phase for phase in phases if phase.name == "refinement-visual-hull-quality"
        )

        self.assertTrue(lpips.uses_torch_lpips_path)
        self.assertFalse(lpips.uses_open3d_path)
        self.assertTrue(visual_hull.uses_open3d_path)
        self.assertFalse(visual_hull.uses_torch_lpips_path)
        self.assertIn("--novel-compute-lpips", lpips.command)
        self.assertNotIn("--novel-compute-lpips", visual_hull.command)
        self.assertIn("scripts\\run_refinement_lab_blender.py", visual_hull.command)
        self.assertIn("--python-exit-code", visual_hull.command)
        self.assertNotIn("--blender-exe", visual_hull.command)

    def test_dry_run_writes_no_run_root_artifacts(self) -> None:
        run_root = Path("temp") / "quality-refinement-runs" / uuid.uuid4().hex
        if run_root.exists():
            shutil.rmtree(run_root)

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = smoke_main(
                [
                    "--dry-run",
                    "--run-root",
                    str(run_root),
                    "--no-refinement-loop",
                    "--no-lpips-novel",
                    "--matrix-suites",
                    "adversarial-silhouettes",
                    "--matrix-count",
                    "1",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertFalse(run_root.exists())
        self.assertIn("Workload:", output.getvalue())

    def test_named_profiles_have_expected_phase_shapes(self) -> None:
        profiles = {
            "contract": ("matrix-",),
            "visual-hull-fast": ("refinement-visual-hull-quality",),
            "primitive-fit-fast": ("refinement-primitive-fit",),
            "gaussian-diagnostic": ("refinement-gaussian-proxy",),
            "differentiable-smoke": ("refinement-differentiable-refine",),
            "lpips-only": ("lpips-novel-view",),
        }
        for profile, expected_names in profiles.items():
            with self.subTest(profile=profile):
                args = parse_args(
                    [
                        "--profile",
                        profile,
                        "--dry-run",
                        "--run-root",
                        f"temp/quality-refinement-runs/{uuid.uuid4().hex}",
                    ]
                )
                _run_root, phases = build_phase_plan(args)
                phase_names = tuple(phase.name for phase in phases)
                for expected in expected_names:
                    self.assertTrue(
                        any(name.startswith(expected) for name in phase_names),
                        phase_names,
                    )

    def test_contract_profile_keeps_quality_budget_warn_only(self) -> None:
        args = parse_args(
            [
                "--profile",
                "contract",
                "--dry-run",
                "--run-root",
                f"temp/quality-refinement-runs/{uuid.uuid4().hex}",
                "--matrix-suites",
                "smoke",
            ]
        )

        _run_root, phases = build_phase_plan(args)
        matrix = next(phase for phase in phases if phase.name == "matrix-smoke")
        budget = next(phase for phase in phases if phase.name == "quality-budget-smoke")

        self.assertIn("--synthetic-allow-failed-rows", matrix.command)
        self.assertNotIn("--quality-budget-json", matrix.command)
        self.assertIn("--warn-only", budget.command)

    def test_command_format_quotes_windows_paths(self) -> None:
        formatted = format_command(
            (
                r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
                "--background",
            )
        )

        self.assertTrue(formatted.startswith("& "))
        self.assertIn(
            '"C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe"',
            formatted,
        )


if __name__ == "__main__":
    unittest.main()
