#!/usr/bin/env python3
"""Pure tests for the quality/refinement smoke orchestrator."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid

from scripts.run_quality_refinement_smoke import (
    DEFAULT_AMBITIOUS_MODES,
    PhaseCommand,
    PhaseResult,
    RefinementPhasePreflight,
    RefinementTarget,
    build_phase_plan,
    format_command,
    main as smoke_main,
    parse_args,
    preflight_phases,
    resolve_run_root,
    run_phases,
    write_summary,
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
        ensemble = next(
            phase for phase in phases if phase.name == "refinement-ensemble-selection"
        )
        self.assertIsNotNone(ensemble.refinement_preflight)
        self.assertEqual(
            ensemble.refinement_preflight.target.suite,
            "synthetic-blender-smoke",
        )

    def test_full_nightly_estimates_resolved_refinement_case_counts(self) -> None:
        args = parse_args(
            [
                "--profile",
                "full-nightly",
                "--dry-run",
                "--run-root",
                f"temp/quality-refinement-runs/{uuid.uuid4().hex}",
            ]
        )

        _run_root, phases = build_phase_plan(args)
        primitive = next(
            phase for phase in phases if phase.name == "refinement-primitive-fit"
        )
        ensemble = next(
            phase for phase in phases if phase.name == "refinement-ensemble-selection"
        )

        self.assertEqual(primitive.expected_case_count, 6)
        self.assertEqual(primitive.expected_candidate_count, 96)
        self.assertEqual(ensemble.expected_case_count, 5)
        self.assertEqual(ensemble.expected_candidate_count, 80)

    def test_preflight_rejects_pure_mask_refinement_suite(self) -> None:
        target = RefinementTarget(
            name="bad-ensemble",
            suite="synthetic-smoke",
            track="ensemble-selection",
            search="successive_halving",
            objective="reliability_first",
        )
        phase = PhaseCommand(
            name="refinement-bad-ensemble",
            description="bad",
            command=("python", "noop.py"),
            artifacts=(),
            requires_blender=True,
            refinement_preflight=RefinementPhasePreflight(
                result_root=Path("temp/quality-refinement-runs/preflight-bad"),
                target=target,
                case_count=None,
                max_runs=8,
                top_k=4,
                seed=1234,
            ),
        )

        report = preflight_phases((phase,))

        self.assertFalse(report["passed"])
        issue = report["issues"][0]
        self.assertEqual(
            issue["code"],
            "unsupported_synthetic_family_for_refinement",
        )
        self.assertEqual(issue["family"], "adversarial_silhouette")

    def test_preflight_passes_full_nightly_plan(self) -> None:
        args = parse_args(
            [
                "--profile",
                "full-nightly",
                "--run-root",
                f"temp/quality-refinement-runs/{uuid.uuid4().hex}",
            ]
        )

        _run_root, phases = build_phase_plan(args)
        report = preflight_phases(phases)

        self.assertTrue(report["passed"], report)

    def test_preflight_only_fails_fast_on_missing_blender_executable(self) -> None:
        run_root = Path("temp") / "quality-refinement-runs" / uuid.uuid4().hex
        phase = PhaseCommand(
            name="needs-blender",
            description="missing blender probe",
            command=("definitely_missing_blender_executable_for_test",),
            artifacts=(run_root / "artifact.json",),
            requires_blender=True,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = run_phases(run_root, (phase,), preflight_only=True)

        self.assertEqual(exit_code, 1)
        self.assertFalse(run_root.exists())
        payload = json.loads(output.getvalue())
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["checked_executable_phases"], 1)
        self.assertEqual(payload["issues"][0]["code"], "missing_blender_executable")

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

    def test_full_nightly_requires_explicit_yes_to_run(self) -> None:
        run_root = Path("temp") / "quality-refinement-runs" / uuid.uuid4().hex
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            exit_code = smoke_main(
                [
                    "--profile",
                    "full-nightly",
                    "--run-root",
                    str(run_root),
                ]
            )

        self.assertEqual(exit_code, 2)
        self.assertFalse(run_root.exists())
        self.assertIn("QUALITY / REFINEMENT SMOKE PLAN", output.getvalue())
        self.assertIn("full-nightly is the broad opt-in workload", errors.getvalue())

    def test_full_nightly_allows_dry_run_without_yes(self) -> None:
        run_root = Path("temp") / "quality-refinement-runs" / uuid.uuid4().hex
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = smoke_main(
                [
                    "--profile",
                    "full-nightly",
                    "--dry-run",
                    "--run-root",
                    str(run_root),
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertFalse(run_root.exists())
        self.assertIn("Workload:", output.getvalue())

    def test_named_profiles_have_expected_phase_shapes(self) -> None:
        profiles = {
            "contract": ("matrix-",),
            "contract-canary": ("matrix-",),
            "synthetic-family-canary": ("matrix-",),
            "adaptive-canary": (
                "refinement-visual-hull-quality",
                "refinement-primitive-fit",
                "refinement-gaussian-proxy",
                "refinement-differentiable-refine",
                "refinement-ensemble-selection",
            ),
            "ensemble-canary": ("refinement-ensemble-selection",),
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
        self.assertEqual(budget.depends_on, ("matrix-smoke",))
        self.assertIsNotNone(budget.inline_quality_budget)

    def test_full_nightly_records_expected_failed_rows_without_failing_matrix_phase(self) -> None:
        args = parse_args(
            [
                "--profile",
                "full-nightly",
                "--dry-run",
                "--run-root",
                f"temp/quality-refinement-runs/{uuid.uuid4().hex}",
                "--matrix-suites",
                "primitive-fit",
                "--no-refinement-loop",
                "--no-lpips-novel",
            ]
        )

        _run_root, phases = build_phase_plan(args)
        matrix = next(phase for phase in phases if phase.name == "matrix-primitive-fit")
        budget = next(phase for phase in phases if phase.name == "quality-budget-primitive-fit")

        self.assertIn("--synthetic-allow-failed-rows", matrix.command)
        self.assertNotIn("--quality-budget-json", matrix.command)
        self.assertIn("--warn-only", budget.command)

    def test_phase_resume_skips_completed_command_with_artifact(self) -> None:
        run_root = Path("temp") / "quality-refinement-runs" / uuid.uuid4().hex
        marker = run_root / "marker.txt"
        counter = run_root / "counter.txt"
        command = (
            sys.executable,
            "-c",
            (
                "from pathlib import Path;"
                f"marker=Path({str(marker)!r});"
                f"counter=Path({str(counter)!r});"
                "marker.parent.mkdir(parents=True, exist_ok=True);"
                "value=int(counter.read_text() or '0') if counter.exists() else 0;"
                "counter.write_text(str(value + 1));"
                "marker.write_text('ok')"
            ),
        )
        phase = PhaseCommand(
            name="resume-phase",
            description="resume test",
            command=command,
            artifacts=(marker,),
        )
        try:
            self.assertEqual(run_phases(run_root, (phase,)), 0)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")

            self.assertEqual(run_phases(run_root, (phase,), resume=True), 0)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")
        finally:
            shutil.rmtree(run_root, ignore_errors=True)

    def test_warn_quality_budget_can_run_inline_after_matrix_artifact(self) -> None:
        run_root = Path("temp") / "quality-refinement-runs" / uuid.uuid4().hex
        current = run_root / "matrix.json"
        budget = run_root / "budget.json"
        report = run_root / "quality.json"
        current.parent.mkdir(parents=True)
        current.write_text(
            json.dumps(
                {
                    "matrix": [
                        {
                            "suite": "smoke",
                            "mode": "visual_hull_voxel",
                            "metrics": {"render": {"min_view_iou": 0.0}},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        budget.write_text(
            json.dumps(
                {
                    "schema_version": "quality_perf_budget_v1",
                    "name": "warn",
                    "thresholds": [
                        {
                            "id": "min",
                            "artifact": "e2e",
                            "mode": "visual_hull_voxel",
                            "metric": "metrics.render.min_view_iou",
                            "threshold": 0.5,
                            "required": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        phase = PhaseCommand(
            name="quality-budget-smoke",
            description="inline",
            command=(sys.executable, "scripts/quality_budget.py", "--warn-only"),
            artifacts=(report,),
            inline_quality_budget=(current, budget, report),
        )
        try:
            self.assertEqual(run_phases(run_root, (phase,)), 0)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["checks"][0]["category"], "quality_below_floor")
        finally:
            shutil.rmtree(run_root, ignore_errors=True)

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

    def test_summary_marks_missing_matrix_metrics_as_na(self) -> None:
        run_root = Path("temp") / "quality-refinement-runs" / uuid.uuid4().hex
        matrix_dir = run_root / "m" / "backend"
        matrix_dir.mkdir(parents=True)
        (matrix_dir / "matrix.json").write_text(
            (
                '{"schema_version":"e2e_synthetic_matrix_v1",'
                '"matrix":[{"mode":"visual_hull_voxel","passed":true,'
                '"metrics":{"backend":{"area_iou_min":0.9}}}]}'
            ),
            encoding="utf-8",
        )
        try:
            path = write_summary(run_root, (), ())
            summary = path.read_text(encoding="utf-8")
            self.assertIn("| `visual_hull_voxel` | 1 | 1 | n/a | n/a | n/a |", summary)
        finally:
            shutil.rmtree(run_root, ignore_errors=True)

    def test_summary_quality_status_fails_on_warn_only_budget_failure(self) -> None:
        run_root = Path("temp") / "quality-refinement-runs" / uuid.uuid4().hex
        quality_dir = run_root / "m" / "smoke"
        quality_dir.mkdir(parents=True)
        (quality_dir / "quality.json").write_text(
            json.dumps(
                {
                    "schema_version": "quality_perf_budget_v1",
                    "passed": False,
                    "threshold_passed": False,
                    "comparison_passed": True,
                    "checks": [
                        {
                            "id": "min",
                            "required": True,
                            "passed": False,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        report = quality_dir / "quality.json"
        phase = PhaseCommand(
            name="quality-budget-smoke",
            description="warn-only budget",
            command=(sys.executable, "scripts/quality_budget.py", "--warn-only"),
            artifacts=(report,),
            inline_quality_budget=(quality_dir / "matrix.json", quality_dir / "budget.json", report),
        )
        result = PhaseResult(phase=phase, returncode=0, elapsed_s=0.01)
        try:
            path = write_summary(run_root, (phase,), (result,))
            summary = path.read_text(encoding="utf-8")
            payload = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))

            self.assertIn("Overall quality status: **fail**", summary)
            self.assertEqual(payload["overall_quality_status"], "fail")
            self.assertEqual(payload["quality_status_reason"], "required_budget_failures")
            self.assertEqual(payload["required_budget_failure_count"], 1)
            self.assertEqual(payload["phase_results"][0]["process_status"], "pass")
            self.assertEqual(payload["phase_results"][0]["quality_status"], "warn")
        finally:
            shutil.rmtree(run_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
