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
    _run_phase_command,
    _tee_stream,
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
        self.assertIn(str(Path("scripts") / "run_refinement_lab_blender.py"), visual_hull.command)
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
        self.assertIn("Workload", output.getvalue())

    def test_yes_confirmation_flag_is_removed(self) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            with self.assertRaises(SystemExit) as raised:
                parse_args(["--yes"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: --yes", errors.getvalue())

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
        self.assertIn("Workload", output.getvalue())

    def test_phase_command_decodes_utf8_output_without_mojibake(self) -> None:
        run_root = Path("temp") / "quality-refinement-runs" / uuid.uuid4().hex
        phase = PhaseCommand(
            name="utf8-output",
            description="unicode output decoding",
            command=(
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write('✓ unicode ok\\n'.encode('utf-8'))",
            ),
            artifacts=(),
        )
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                returncode, stdout_path, _stderr_path = _run_phase_command(
                    phase,
                    run_root=run_root,
                )

            self.assertEqual(returncode, 0)
            self.assertIn("✓ unicode ok", output.getvalue())
            self.assertNotIn("âœ", output.getvalue())
            self.assertIn("✓ unicode ok", stdout_path.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(run_root, ignore_errors=True)

    def test_tee_stream_adds_console_newline_for_partial_lines(self) -> None:
        log = io.StringIO()
        console = io.StringIO()

        _tee_stream(["render     : 512x512 BLENDER_EEVEE"], log, console)

        self.assertEqual(log.getvalue(), "render     : 512x512 BLENDER_EEVEE")
        self.assertEqual(console.getvalue(), "render     : 512x512 BLENDER_EEVEE\n")

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
            "moonshot-smoke": (
                "refinement-moonshot-sidecars",
                "refinement-primitive-fit",
                "refinement-gaussian-proxy",
                "refinement-differentiable-refine",
                "refinement-ensemble-selection",
            ),
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
        run_root = resolve_run_root(
            Path("temp") / "quality-refinement-runs" / uuid.uuid4().hex
        )
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

    def test_summary_uses_backend_matrix_metrics_when_render_metrics_absent(self) -> None:
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
            self.assertIn(
                "| `visual_hull_voxel` | 1 | 1 | 0.9000 | n/a | n/a | min_iou:backend:1 |",
                summary,
            )
        finally:
            shutil.rmtree(run_root, ignore_errors=True)

    def test_summary_writes_failed_matrix_row_details_and_machine_aliases(self) -> None:
        run_root = Path("temp") / "quality-refinement-runs" / uuid.uuid4().hex
        matrix_dir = run_root / "m" / "primitive-fit"
        matrix_dir.mkdir(parents=True)
        (matrix_dir / "matrix.json").write_text(
            json.dumps(
                {
                    "schema_version": "e2e_synthetic_matrix_v1",
                    "matrix": [
                        {
                            "suite": "primitive-fit",
                            "case": "primitive-fit:capsule",
                            "shape_id": "capsule_seed_1236",
                            "mode": "primitive_fit_refine",
                            "status": "fail",
                            "passed": False,
                            "result_json": "m/primitive-fit/result.json",
                            "metrics": {
                                "backend": {"area_iou_min": 0.002},
                                "silhouette": {
                                    "min_view_iou": 0.318,
                                    "per_view": {
                                        "front": {"area_iou": 0.318},
                                        "side": {"area_iou": 0.319},
                                        "top": {"area_iou": 0.374},
                                    },
                                },
                            },
                            "evaluation_bundle": {
                                "errors": ["backend min IoU 0.002 below 0.350"]
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        try:
            path = write_summary(run_root, (), ())
            summary = path.read_text(encoding="utf-8")
            payload = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))

            self.assertIn("## Failed Matrix Rows", summary)
            self.assertIn("primitive-fit:capsule", summary)
            self.assertIn("backend min IoU 0.002 below 0.350", summary)
            self.assertEqual(payload["matrix_failed"], 1)
            self.assertEqual(payload["matrix_failed_rows"][0]["shape_id"], "capsule_seed_1236")
            self.assertEqual(payload["matrix_failed_rows"][0]["render_metric_source"], "silhouette")
            self.assertEqual(payload["required_refinement_failures"], [])
            self.assertEqual(payload["required_refinement_failure_count"], 0)
            self.assertEqual(payload["reference_cache_totals"]["hits"], 0)
            self.assertEqual(payload["candidate_cache_totals"]["hits"], 0)
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

    def test_summary_quality_status_fails_on_all_error_refinement_rows(self) -> None:
        run_root = Path("temp") / "quality-refinement-runs" / uuid.uuid4().hex
        refinement_dir = run_root / "r" / "primitive-fit"
        generation_dir = refinement_dir / "g00"
        generation_dir.mkdir(parents=True)
        (refinement_dir / "adaptive-loop-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "refinement_adaptive_loop_summary_v1",
                    "stopped_reason": "no_promotable_parents",
                    "generation_count": 1,
                    "generations": [
                        {"generation": 0, "parent_health": {"promotable_count": 0}}
                    ],
                }
            ),
            encoding="utf-8",
        )
        (generation_dir / "leaderboard.json").write_text(
            json.dumps(
                {
                    "schema_version": "refinement_leaderboard_v1",
                    "rows": [
                        {
                            "case_id": "box_seed_1234",
                            "variant_id": "baseline",
                            "status": "error",
                            "promotable": False,
                            "promotion_tier": "blocked_error",
                        },
                        {
                            "case_id": "capsule_seed_1236",
                            "variant_id": "baseline",
                            "status": "error",
                            "promotable": False,
                            "promotion_tier": "blocked_error",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        try:
            path = write_summary(run_root, (), ())
            summary = path.read_text(encoding="utf-8")
            payload = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))

            self.assertIn("Overall quality status: **fail**", summary)
            self.assertIn("Required refinement failures: 1", summary)
            self.assertIn("all_error_rows", summary)
            self.assertEqual(payload["overall_quality_status"], "fail")
            self.assertEqual(payload["quality_status_reason"], "refinement_required_failures")
            self.assertEqual(payload["refinement_error_rows"], 2)
            self.assertEqual(payload["refinement_promotable_rows"], 0)
            self.assertEqual(
                payload["refinement_required_failures"][0]["failure_reasons"],
                [
                    "all_error_rows",
                    "zero_promotable_candidates",
                    "no_promotable_parents",
                ],
            )
        finally:
            shutil.rmtree(run_root, ignore_errors=True)

    def test_summary_includes_moonshot_portfolio_actions(self) -> None:
        run_root = Path("temp") / "quality-refinement-runs" / uuid.uuid4().hex
        refinement_dir = run_root / "r" / "moonshot-sidecars"
        generation_dir = refinement_dir / "g00"
        generation_dir.mkdir(parents=True)
        (refinement_dir / "adaptive-loop-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "refinement_adaptive_loop_summary_v1",
                    "stopped_reason": "complete",
                    "generation_count": 1,
                    "generations": [
                        {"generation": 0, "parent_health": {"promotable_count": 1}}
                    ],
                }
            ),
            encoding="utf-8",
        )
        (generation_dir / "leaderboard.json").write_text(
            json.dumps(
                {
                    "schema_version": "refinement_leaderboard_v1",
                    "moonshot_summary": {
                        "status_counts": {"ran": 6},
                        "portfolio_available_actions": [
                            {
                                "source": "shape_grammar_search",
                                "kind": "compile_selected_shape_program",
                                "score": 0.2234,
                                "case_id": "box",
                                "variant_id": "moonshot",
                            }
                        ],
                        "portfolio_actions": [
                            {
                                "source": "active_view_planning",
                                "kind": "capture_next_best_view",
                                "score": 0.1234,
                                "risk": 0.08,
                                "case_id": "box",
                                "variant_id": "moonshot",
                            }
                        ],
                        "portfolio_rejected_actions": [
                            {
                                "source": "differentiable_primitives",
                                "kind": "run_finite_difference_refinement_probe",
                                "score": 0.03,
                                "rejection": "blocked_prerequisite",
                                "case_id": "box",
                                "variant_id": "moonshot",
                            }
                        ],
                        "active_view_sequence": [
                            {
                                "order": 1,
                                "view_id": "top_oblique_060",
                                "marginal_expected_metric_delta": 0.047,
                                "case_id": "box",
                                "variant_id": "moonshot",
                            }
                        ],
                        "portfolio_dependency_edges": [
                            {
                                "before_kind": "capture_next_best_view",
                                "after_kind": "compile_selected_shape_program",
                                "type": "rerank",
                                "case_id": "box",
                                "variant_id": "moonshot",
                            }
                        ],
                        "portfolio_execution_plan": [
                            {
                                "stage": "capture_and_constraints",
                                "order": 1,
                                "actions": [{"kind": "capture_next_best_view"}],
                                "case_id": "box",
                                "variant_id": "moonshot",
                            }
                        ],
                        "portfolio_risks": [
                            {
                                "kind": "view_evidence",
                                "level": "medium",
                                "risk": 0.31,
                                "case_id": "box",
                                "variant_id": "moonshot",
                            }
                        ],
                    },
                    "rows": [
                        {
                            "case_id": "box",
                            "variant_id": "moonshot",
                            "status": "pass",
                            "promotable": True,
                            "parent_selectable": True,
                            "promotion_tier": "promotable",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        try:
            path = write_summary(run_root, (), ())
            summary = path.read_text(encoding="utf-8")
            payload = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))

            self.assertIn("Available moonshot evidence:", summary)
            self.assertIn("Portfolio actions:", summary)
            self.assertIn("capture_next_best_view", summary)
            self.assertIn("Rejected portfolio actions:", summary)
            self.assertIn("blocked_prerequisite", summary)
            self.assertIn("Portfolio dependencies:", summary)
            self.assertIn("Portfolio execution stages:", summary)
            self.assertIn("Portfolio risks:", summary)
            self.assertEqual(
                payload["moonshot_summary"]["portfolio_available_actions"][0]["kind"],
                "compile_selected_shape_program",
            )
            self.assertEqual(
                payload["moonshot_summary"]["portfolio_actions"][0]["kind"],
                "capture_next_best_view",
            )
            self.assertEqual(
                payload["moonshot_summary"]["portfolio_rejected_actions"][0]["rejection"],
                "blocked_prerequisite",
            )
            self.assertEqual(
                payload["moonshot_summary"]["portfolio_dependency_edges"][0]["after_kind"],
                "compile_selected_shape_program",
            )
        finally:
            shutil.rmtree(run_root, ignore_errors=True)

    def test_summary_includes_refinement_cache_stats(self) -> None:
        run_root = Path("temp") / "quality-refinement-runs" / uuid.uuid4().hex
        cache_dir = run_root / "r" / "primitive-fit"
        cache_dir.mkdir(parents=True)
        (cache_dir / "cache-stats.json").write_text(
            json.dumps(
                {
                    "schema_version": "refinement_cache_stats_v1",
                    "reference_cache_hits": 1,
                    "reference_cache_misses": 2,
                    "reference_cache_writes": 1,
                    "candidate_cache_hits": 3,
                    "candidate_cache_misses": 2,
                    "candidate_cache_writes": 5,
                    "candidate_cache_sources": {"shared_cache": 3},
                }
            ),
            encoding="utf-8",
        )
        try:
            path = write_summary(run_root, (), ())
            summary = path.read_text(encoding="utf-8")
            payload = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))

            self.assertIn("## Cache Stats", summary)
            self.assertIn("| **total** | 1 | 2 | 1 | 3 | 2 | 5 |", summary)
            self.assertEqual(payload["cache_totals"]["reference_cache_hits"], 1)
            self.assertEqual(payload["cache_totals"]["reference_cache_misses"], 2)
            self.assertEqual(payload["cache_totals"]["reference_cache_writes"], 1)
            self.assertEqual(payload["cache_totals"]["candidate_cache_hits"], 3)
            self.assertEqual(
                payload["cache_totals"]["candidate_cache_sources"]["shared_cache"],
                3,
            )
        finally:
            shutil.rmtree(run_root, ignore_errors=True)

    def test_summary_warns_when_candidate_cache_has_duplicates_but_no_shared_hits(self) -> None:
        run_root = Path("temp") / "quality-refinement-runs" / uuid.uuid4().hex
        refinement_dir = run_root / "r" / "primitive-fit"
        generation_dir = refinement_dir / "g00"
        generation_dir.mkdir(parents=True)
        (refinement_dir / "adaptive-loop-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "refinement_adaptive_loop_summary_v1",
                    "stopped_reason": "no_child_variants",
                    "generation_count": 1,
                    "generations": [
                        {"generation": 0, "parent_health": {"promotable_count": 1}}
                    ],
                }
            ),
            encoding="utf-8",
        )
        (generation_dir / "leaderboard.json").write_text(
            json.dumps(
                {
                    "schema_version": "refinement_leaderboard_v1",
                    "duplicate_quality_duplicate_row_count": 4,
                    "rows": [
                        {"status": "pass", "promotable": True, "parent_selectable": True}
                    ],
                }
            ),
            encoding="utf-8",
        )
        (generation_dir / "cache-stats.json").write_text(
            json.dumps(
                {
                    "schema_version": "refinement_cache_stats_v1",
                    "candidate_cache_enabled": True,
                    "candidate_cache_hits": 0,
                    "candidate_cache_misses": 4,
                    "candidate_cache_writes": 4,
                    "candidate_cache_sources": {},
                }
            ),
            encoding="utf-8",
        )
        try:
            path = write_summary(run_root, (), ())
            summary = path.read_text(encoding="utf-8")
            payload = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))

            self.assertIn("## Cache Health Warnings", summary)
            self.assertIn("candidate_cache_no_shared_hits_for_duplicates", summary)
            self.assertEqual(
                payload["cache_health_warnings"][0]["code"],
                "candidate_cache_no_shared_hits_for_duplicates",
            )
        finally:
            shutil.rmtree(run_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
