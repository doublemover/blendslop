#!/usr/bin/env python3
"""Orchestrate ambitious quality/refinement smoke runs.

The script intentionally stays above the reconstruction stack: it builds the
same public e2e and refinement-lab commands a human would run, keeps all
generated outputs under repo-root temp/, and separates LPIPS/Torch work from
Open3D-heavy visual-hull refinement processes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blender_blocking.utils.path_safety import compact_path_segment

TEMP_ROOT = REPO_ROOT / "temp"
RUN_ROOT = TEMP_ROOT / "quality-refinement-runs"
DEFAULT_BLENDER_EXE = (
    r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
)
DEFAULT_AMBITIOUS_MODES = (
    "visual_hull_voxel",
    "primitive_fit_refine",
    "gaussian_ellipsoid_proxy",
    "differentiable_refine",
)
DEFAULT_MATRIX_SUITES = (
    "adversarial-silhouettes",
    "primitive-fit",
    "visual-hull",
    "smoke",
)
DEFAULT_LPIPS_MODES = ("legacy", "profile_loft")


@dataclass(frozen=True)
class RefinementTarget:
    name: str
    suite: str
    track: str
    search: str
    objective: str
    uses_open3d_path: bool = False
    uses_torch_lpips_path: bool = False


DEFAULT_REFINEMENT_TARGETS = (
    RefinementTarget(
        name="visual-hull-quality",
        suite="synthetic-visual-hull",
        track="visual-hull-quality",
        search="successive_halving",
        objective="reliability_first",
        uses_open3d_path=True,
    ),
    RefinementTarget(
        name="primitive-fit",
        suite="synthetic-primitive-fit",
        track="primitive-fit",
        search="successive_halving",
        objective="reliability_first",
    ),
    RefinementTarget(
        name="gaussian-proxy",
        suite="synthetic-primitive-fit",
        track="gaussian-proxy",
        search="successive_halving",
        objective="reliability_first",
    ),
    RefinementTarget(
        name="differentiable-refine",
        suite="synthetic-primitive-fit",
        track="differentiable-refine",
        search="successive_halving",
        objective="reliability_first",
    ),
    RefinementTarget(
        name="ensemble-selection",
        suite="synthetic-smoke",
        track="ensemble-selection",
        search="successive_halving",
        objective="reliability_first",
    ),
)


@dataclass(frozen=True)
class SmokeProfile:
    name: str
    description: str
    matrix_suites: tuple[str, ...] = ()
    modes: tuple[str, ...] = DEFAULT_AMBITIOUS_MODES
    matrix_count: int = 1
    lpips_enabled: bool = False
    lpips_suite: str = "smoke"
    lpips_modes: tuple[str, ...] = DEFAULT_LPIPS_MODES
    lpips_count: int = 1
    refinement_targets: tuple[str, ...] = ()
    refinement_case_count: int | None = 1
    refinement_generations: int = 1
    refinement_max_runs: int = 3
    refinement_top_k: int = 2
    refinement_parent_top_k: int = 1
    refinement_children_per_parent: int = 1
    matrix_budget_mode: str = "gating"
    matrix_allow_failed_rows: bool = False
    stop_on_failure: bool = False
    expected_artifacts: tuple[str, ...] = ("commands.md", "summary.md", "summary.json")


SMOKE_PROFILES: dict[str, SmokeProfile] = {
    "interactive": SmokeProfile(
        name="interactive",
        description="Bounded default profile: short backend-status contract matrix.",
        matrix_suites=("adversarial-silhouettes",),
        matrix_count=1,
        stop_on_failure=True,
    ),
    "contract": SmokeProfile(
        name="contract",
        description="Backend-status contract matrices only.",
        matrix_suites=DEFAULT_MATRIX_SUITES,
        matrix_count=1,
        matrix_budget_mode="warn",
        matrix_allow_failed_rows=True,
        stop_on_failure=True,
    ),
    "visual-hull-fast": SmokeProfile(
        name="visual-hull-fast",
        description="One visual-hull render-IoU adaptive loop.",
        refinement_targets=("visual-hull-quality",),
        refinement_max_runs=3,
        refinement_generations=1,
        stop_on_failure=True,
    ),
    "primitive-fit-fast": SmokeProfile(
        name="primitive-fit-fast",
        description="One primitive-fit render-IoU adaptive loop.",
        refinement_targets=("primitive-fit",),
        refinement_max_runs=3,
        refinement_generations=1,
        stop_on_failure=True,
    ),
    "gaussian-diagnostic": SmokeProfile(
        name="gaussian-diagnostic",
        description="One gaussian render-IoU preflight plus diagnostic output.",
        refinement_targets=("gaussian-proxy",),
        refinement_max_runs=1,
        refinement_generations=1,
        refinement_top_k=1,
        stop_on_failure=True,
    ),
    "differentiable-smoke": SmokeProfile(
        name="differentiable-smoke",
        description="Cheap differentiable refinement smoke pass.",
        refinement_targets=("differentiable-refine",),
        refinement_max_runs=1,
        refinement_generations=1,
        refinement_top_k=1,
        stop_on_failure=True,
    ),
    "lpips-only": SmokeProfile(
        name="lpips-only",
        description="Novel-view PSNR/SSIM/LPIPS validation only.",
        lpips_enabled=True,
        lpips_count=1,
        stop_on_failure=True,
    ),
    "full-nightly": SmokeProfile(
        name="full-nightly",
        description="Broad matrix, LPIPS, and adaptive quality sweep.",
        matrix_suites=DEFAULT_MATRIX_SUITES,
        matrix_count=3,
        lpips_enabled=True,
        lpips_count=2,
        refinement_targets=tuple(target.name for target in DEFAULT_REFINEMENT_TARGETS),
        refinement_case_count=None,
        refinement_max_runs=8,
        refinement_top_k=4,
        refinement_generations=2,
        refinement_parent_top_k=2,
        refinement_children_per_parent=2,
    ),
}


@dataclass(frozen=True)
class PhaseCommand:
    name: str
    description: str
    command: tuple[str, ...]
    artifacts: tuple[Path, ...]
    requires_blender: bool = False
    uses_open3d_path: bool = False
    uses_torch_lpips_path: bool = False
    expected_candidate_count: int = 0
    expected_blender_invocations: int = 0


@dataclass(frozen=True)
class PhaseResult:
    phase: PhaseCommand
    returncode: int
    elapsed_s: float

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def _parse_csv(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = []
        for item in value:
            parts.extend(str(item).split(","))
    return tuple(part.strip() for part in parts if part.strip())


def _default_run_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RUN_ROOT / stamp


def _resolve_under_repo(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw.resolve(strict=False)
    return (REPO_ROOT / raw).resolve(strict=False)


def resolve_run_root(path: str | Path | None) -> Path:
    root = _resolve_under_repo(path or _default_run_root())
    temp = TEMP_ROOT.resolve(strict=False)
    if root != temp and temp not in root.parents:
        raise ValueError(f"run root must stay under repo temp/: {root}")
    return root


def _repo_path(path: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _python_cmd() -> str:
    return sys.executable or "python"


def _blender_cmd(blender_exe: str) -> str:
    return blender_exe or os.environ.get("BLENDER_EXE") or DEFAULT_BLENDER_EXE


def build_phase_plan(args: argparse.Namespace) -> tuple[Path, tuple[PhaseCommand, ...]]:
    run_root = resolve_run_root(args.run_root)
    profile = _profile(args.profile)
    modes = _parse_csv(args.modes) or profile.modes
    matrix_suites = _parse_csv(args.matrix_suites) or profile.matrix_suites
    matrix_count = args.matrix_count if args.matrix_count is not None else profile.matrix_count
    lpips_suite = args.lpips_suite or profile.lpips_suite
    lpips_modes = _parse_csv(args.lpips_modes) or profile.lpips_modes
    lpips_count = args.lpips_count if args.lpips_count is not None else profile.lpips_count
    refinement_targets = (
        args.refinement_targets
        if args.refinement_targets is not None
        else ",".join(profile.refinement_targets)
    )
    refinement_max_runs = (
        args.refinement_max_runs
        if args.refinement_max_runs is not None
        else profile.refinement_max_runs
    )
    refinement_case_count = (
        args.refinement_case_count
        if args.refinement_case_count is not None
        else profile.refinement_case_count
    )
    refinement_top_k = (
        args.refinement_top_k
        if args.refinement_top_k is not None
        else profile.refinement_top_k
    )
    refinement_generations = (
        args.refinement_generations
        if args.refinement_generations is not None
        else profile.refinement_generations
    )
    refinement_parent_top_k = (
        args.refinement_parent_top_k
        if args.refinement_parent_top_k is not None
        else profile.refinement_parent_top_k
    )
    refinement_children_per_parent = (
        args.refinement_children_per_parent
        if args.refinement_children_per_parent is not None
        else profile.refinement_children_per_parent
    )
    blender_exe = _blender_cmd(args.blender_exe)
    phases: list[PhaseCommand] = []

    if matrix_suites and not args.no_synthetic_matrix:
        for suite in matrix_suites:
            suite_dir = run_root / "m" / compact_path_segment(
                suite,
                max_length=24,
                fallback="suite",
            )
            output_root = suite_dir / "s"
            result_json = suite_dir / "matrix.json"
            quality_report = suite_dir / "quality.json"
            cost_report = suite_dir / "cost.json"
            use_blender = suite not in {"adversarial-silhouettes", "capture-noise"}
            matrix_args = [
                "--synthetic-matrix",
                "--synthetic-suite",
                suite,
                "--synthetic-count",
                str(matrix_count),
                "--synthetic-modes",
                ",".join(modes),
                "--validation-mode",
                "backend-status",
                "--synthetic-output-root",
                _repo_path(output_root),
                "--result-json",
                _repo_path(result_json),
                "--cost-report-json",
                _repo_path(cost_report),
                "--cost-track-memory",
                "--synthetic-strict-skips",
                "--run-id",
                f"{args.run_id_prefix}_{suite}",
                "--no-progress",
            ]
            if profile.matrix_allow_failed_rows:
                matrix_args.append("--synthetic-allow-failed-rows")
            matrix_artifacts: list[Path] = [result_json, cost_report, output_root]
            if profile.matrix_budget_mode == "gating":
                matrix_args.extend(
                    [
                        "--quality-budget-json",
                        _repo_path(REPO_ROOT / args.quality_budget_json),
                        "--quality-report-json",
                        _repo_path(quality_report),
                    ]
                )
                matrix_artifacts.insert(1, quality_report)
            command = _e2e_command(
                blender_exe=blender_exe if use_blender else None,
                args=tuple(matrix_args),
            )
            phases.append(
                PhaseCommand(
                    name=f"matrix-{suite}",
                    description=(
                        "Synthetic suite x ambitious reconstruction mode "
                        "backend-status matrix"
                    ),
                    command=command,
                    artifacts=tuple(matrix_artifacts),
                    requires_blender=use_blender,
                    expected_candidate_count=matrix_count * len(modes),
                    expected_blender_invocations=1 if use_blender else 0,
                )
            )
            if profile.matrix_budget_mode == "warn":
                phases.append(
                    PhaseCommand(
                        name=f"quality-budget-{suite}",
                        description=(
                            "Warn-only quality budget report for contract matrix "
                            "evidence"
                        ),
                        command=_quality_budget_command(
                            result_json=result_json,
                            budget_json=REPO_ROOT / args.quality_budget_json,
                            report_json=quality_report,
                            warn_only=True,
                        ),
                        artifacts=(quality_report,),
                    )
                )

    if profile.lpips_enabled and not args.no_lpips_novel:
        lpips_root = run_root / "lpips"
        phases.append(
            PhaseCommand(
                name="lpips-novel-view",
                description=(
                    "Novel-view PSNR/SSIM/LPIPS smoke in its own Blender process"
                ),
                command=_e2e_command(
                    blender_exe=blender_exe,
                    args=(
                        "--synthetic-matrix",
                        "--synthetic-suite",
                        lpips_suite,
                        "--synthetic-count",
                        str(lpips_count),
                        "--synthetic-modes",
                        ",".join(lpips_modes),
                        "--validation-mode",
                        "novel-view",
                        "--novel-view-angles",
                        args.lpips_angles,
                        "--novel-compute-ssim",
                        "--novel-compute-lpips",
                        "--novel-psnr-threshold",
                        str(args.lpips_psnr_threshold),
                        "--novel-ssim-threshold",
                        str(args.lpips_ssim_threshold),
                        "--novel-lpips-threshold",
                        str(args.lpips_threshold),
                        "--synthetic-output-root",
                        _repo_path(lpips_root / "s"),
                        "--result-json",
                        _repo_path(lpips_root / "matrix.json"),
                        "--cost-report-json",
                        _repo_path(lpips_root / "cost.json"),
                        "--run-id",
                        f"{args.run_id_prefix}_lpips",
                        "--no-progress",
                    ),
                ),
                artifacts=(
                    lpips_root / "matrix.json",
                    lpips_root / "cost.json",
                    lpips_root / "s",
                ),
                requires_blender=True,
                uses_torch_lpips_path=True,
                expected_candidate_count=lpips_count * len(lpips_modes),
                expected_blender_invocations=1,
            )
        )

    if refinement_targets and not args.no_refinement_loop:
        selected = _selected_refinement_targets(refinement_targets)
        for target in selected:
            target_root = run_root / "r" / compact_path_segment(
                target.name,
                max_length=24,
                fallback="target",
            )
            phases.append(
                PhaseCommand(
                    name=f"refinement-{target.name}",
                    description=(
                        f"Closed adaptive refinement loop for {target.suite} "
                        f"on {target.track}"
                    ),
                    command=_refinement_loop_command(
                        blender_exe=blender_exe,
                        result_root=target_root,
                        target=target,
                        case_count=refinement_case_count,
                        max_runs=refinement_max_runs,
                        top_k=refinement_top_k,
                        generations=refinement_generations,
                        parent_top_k=refinement_parent_top_k,
                        children_per_parent=refinement_children_per_parent,
                        seed=args.seed,
                    ),
                    artifacts=(
                        target_root / "adaptive-loop-summary.json",
                        target_root,
                    ),
                    requires_blender=True,
                    uses_open3d_path=target.uses_open3d_path,
                    uses_torch_lpips_path=target.uses_torch_lpips_path,
                    expected_candidate_count=(
                        refinement_max_runs
                        * refinement_generations
                        * max(1, refinement_case_count or 1)
                    ),
                    expected_blender_invocations=1,
                )
            )

    return run_root, tuple(phases)


def _profile(name: str) -> SmokeProfile:
    try:
        return SMOKE_PROFILES[str(name)]
    except KeyError as exc:
        known = ", ".join(sorted(SMOKE_PROFILES))
        raise ValueError(f"unknown profile {name!r}; known: {known}") from exc


def _e2e_command(
    *,
    blender_exe: str | None,
    args: Sequence[str],
) -> tuple[str, ...]:
    script = _repo_path(REPO_ROOT / "blender_blocking" / "test_e2e_validation.py")
    if blender_exe:
        return (
            blender_exe,
            "--background",
            "--python-exit-code",
            "1",
            "--python",
            script,
            "--",
            *args,
        )
    return (_python_cmd(), script, *args)


def _refinement_loop_command(
    *,
    blender_exe: str,
    result_root: Path,
    target: RefinementTarget,
    case_count: int | None,
    max_runs: int,
    top_k: int,
    generations: int,
    parent_top_k: int,
    children_per_parent: int,
    seed: int,
) -> tuple[str, ...]:
    command = [
        blender_exe,
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        _repo_path(REPO_ROOT / "scripts" / "run_refinement_lab_blender.py"),
        "--",
        "loop",
        "--suite",
        target.suite,
        "--track",
        target.track,
        "--search",
        target.search,
        "--objective",
        target.objective,
        "--max-runs",
        str(max_runs),
        "--top-k",
        str(top_k),
        "--generations",
        str(generations),
        "--parent-top-k",
        str(parent_top_k),
        "--children-per-parent",
        str(children_per_parent),
        "--result-root",
        _repo_path(result_root),
        "--seed",
        str(seed),
        "--report-failures",
        "all",
    ]
    if case_count is not None:
        command.extend(("--case-count", str(case_count)))
    return tuple(command)


def _quality_budget_command(
    *,
    result_json: Path,
    budget_json: Path,
    report_json: Path,
    warn_only: bool,
) -> tuple[str, ...]:
    command = [
        _python_cmd(),
        _repo_path(REPO_ROOT / "scripts" / "quality_budget.py"),
        "--current",
        _repo_path(result_json),
        "--budget",
        _repo_path(budget_json),
        "--report",
        _repo_path(report_json),
    ]
    if warn_only:
        command.append("--warn-only")
    return tuple(command)


def _selected_refinement_targets(raw: str | None) -> tuple[RefinementTarget, ...]:
    names = _parse_csv(raw)
    if not names:
        return DEFAULT_REFINEMENT_TARGETS
    by_name = {target.name: target for target in DEFAULT_REFINEMENT_TARGETS}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        known = ", ".join(sorted(by_name))
        raise ValueError(f"unknown refinement target(s) {unknown}; known: {known}")
    return tuple(by_name[name] for name in names)


def format_command(command: Sequence[str]) -> str:
    parts = [_quote_arg(part) for part in command]
    if parts and parts[0].startswith('"'):
        return "& " + " ".join(parts)
    return " ".join(parts)


def _quote_arg(value: object) -> str:
    text = str(value)
    if text == "":
        return '""'
    if any(char.isspace() for char in text) or any(char in text for char in "&()[]{}"):
        return '"' + text.replace('"', r'\"') + '"'
    return text


def write_command_recipe(run_root: Path, phases: Sequence[PhaseCommand]) -> Path:
    path = run_root / "commands.md"
    lines = [
        "# Quality Refinement Smoke Commands",
        "",
        f"Run root: `{_repo_path(run_root)}`",
        "",
        "Generated artifacts under this root are local diagnostics and must not be committed.",
        "",
    ]
    for index, phase in enumerate(phases, start=1):
        lines.extend(
            [
                f"## {index}. {phase.name}",
                "",
                phase.description,
                "",
                "```powershell",
                format_command(phase.command),
                "```",
                "",
                "Artifacts:",
            ]
        )
        lines.extend(f"- `{_repo_path(path)}`" for path in phase.artifacts)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_phases(
    run_root: Path,
    phases: Sequence[PhaseCommand],
    *,
    dry_run: bool = False,
    clean_first: bool = False,
    stop_on_failure: bool = False,
) -> int:
    if dry_run:
        _print_plan(run_root, phases)
        return 0

    if clean_first and run_root.exists():
        _assert_temp_child(run_root)
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    recipe = write_command_recipe(run_root, phases)
    print(f"Saved command recipe: {_repo_path(recipe)}")

    results: list[PhaseResult] = []
    for phase in phases:
        _print_phase_header(phase)
        started = time.perf_counter()
        completed = subprocess.run(phase.command, cwd=REPO_ROOT, check=False)
        elapsed_s = time.perf_counter() - started
        result = PhaseResult(
            phase=phase,
            returncode=completed.returncode,
            elapsed_s=elapsed_s,
        )
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}: {phase.name} ({elapsed_s:.1f}s)")
        if not result.passed and stop_on_failure:
            break

    summary = write_summary(run_root, phases, results)
    print(f"Saved summary: {_repo_path(summary)}")
    return 0 if results and all(result.passed for result in results) else 1


def _assert_temp_child(path: Path) -> None:
    temp = TEMP_ROOT.resolve(strict=False)
    resolved = path.resolve(strict=False)
    if resolved == temp or temp not in resolved.parents:
        raise ValueError(f"refusing to delete outside repo temp/: {resolved}")


def _print_plan(run_root: Path, phases: Sequence[PhaseCommand]) -> None:
    candidates = sum(phase.expected_candidate_count for phase in phases)
    blender_starts = sum(phase.expected_blender_invocations for phase in phases)
    artifact_count = sum(len(phase.artifacts) for phase in phases) + 3
    print("QUALITY / REFINEMENT SMOKE PLAN")
    print(f"Run root: {_repo_path(run_root)}")
    print(
        "Workload: "
        f"{len(phases)} phases, "
        f"{candidates} expected candidates, "
        f"{blender_starts} expected Blender starts, "
        f"{artifact_count} expected artifact roots/files"
    )
    print()
    print("| # | Phase | Candidates | Blender starts | Artifacts |")
    print("| ---: | --- | ---: | ---: | ---: |")
    for index, phase in enumerate(phases, start=1):
        print(
            "| "
            f"{index} | `{phase.name}` | "
            f"{phase.expected_candidate_count} | "
            f"{phase.expected_blender_invocations} | "
            f"{len(phase.artifacts)} |"
        )
    for index, phase in enumerate(phases, start=1):
        print()
        print(f"{index}. {phase.name}")
        print(f"   {phase.description}")
        print(f"   {format_command(phase.command)}")
        for artifact in phase.artifacts:
            print(f"   artifact: {_repo_path(artifact)}")


def _print_phase_header(phase: PhaseCommand) -> None:
    print()
    print("=" * 78)
    print(phase.name)
    print("-" * 78)
    print(phase.description)
    if phase.uses_torch_lpips_path:
        print("Dependency lane: Torch/LPIPS isolated process")
    elif phase.uses_open3d_path:
        print("Dependency lane: Open3D/OpenVDB-capable process")
    print(format_command(phase.command))
    print("=" * 78)


def write_summary(
    run_root: Path,
    phases: Sequence[PhaseCommand],
    results: Sequence[PhaseResult],
) -> Path:
    path = run_root / "summary.md"
    matrix_rows = _collect_matrix_rows(run_root)
    budget_reports = _collect_budget_reports(run_root)
    lines = [
        "# Quality Refinement Smoke Summary",
        "",
        f"Run root: `{_repo_path(run_root)}`",
        "",
        "## Phase Status",
        "",
        "| Phase | Status | Seconds |",
        "| --- | --- | ---: |",
    ]
    by_name = {result.phase.name: result for result in results}
    for phase in phases:
        result = by_name.get(phase.name)
        if result is None:
            lines.append(f"| `{phase.name}` | not-run |  |")
        else:
            status = "pass" if result.passed else f"fail ({result.returncode})"
            lines.append(f"| `{phase.name}` | {status} | {result.elapsed_s:.1f} |")
    lines.extend(["", "## Matrix Rows", ""])
    if matrix_rows:
        lines.extend(_matrix_summary_lines(matrix_rows))
    else:
        lines.append("No synthetic matrix JSON rows were found.")
    lines.extend(["", "## Quality Budget Reports", ""])
    if budget_reports:
        lines.extend(_budget_summary_lines(budget_reports))
    else:
        lines.append("No quality budget report JSON files were found.")
    lines.extend(
        [
            "",
            "## Artifact Policy",
            "",
            "All files under this run root are generated diagnostics. Do not commit them.",
            "",
            "## Command Recipe",
            "",
            "See `commands.md` in this run root for exact reproduction commands.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    _write_machine_summary(run_root, results, matrix_rows, budget_reports)
    return path


def _write_machine_summary(
    run_root: Path,
    results: Sequence[PhaseResult],
    matrix_rows: Sequence[dict],
    budget_reports: Sequence[dict],
) -> None:
    payload = {
        "schema_version": "quality_refinement_smoke_summary_v1",
        "run_root": _repo_path(run_root),
        "phase_results": [
            {
                "name": result.phase.name,
                "returncode": result.returncode,
                "passed": result.passed,
                "elapsed_s": result.elapsed_s,
                "artifacts": [_repo_path(path) for path in result.phase.artifacts],
                "expected_candidate_count": result.phase.expected_candidate_count,
                "expected_blender_invocations": result.phase.expected_blender_invocations,
            }
            for result in results
        ],
        "matrix_row_count": len(matrix_rows),
        "matrix_passed": sum(1 for row in matrix_rows if row.get("passed")),
        "matrix_failed": sum(1 for row in matrix_rows if not row.get("passed")),
        "quality_budget_reports": [
            {
                "path": report["_quality_report_json"],
                "passed": report.get("passed"),
                "threshold_passed": report.get("threshold_passed"),
                "comparison_passed": report.get("comparison_passed"),
                "failed_required_checks": _failed_required_check_count(report),
            }
            for report in budget_reports
        ],
    }
    (run_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _collect_matrix_rows(run_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(run_root.glob("**/matrix.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        matrix = payload.get("matrix")
        if not isinstance(matrix, list):
            continue
        for row in matrix:
            if isinstance(row, dict):
                enriched = dict(row)
                enriched["_matrix_json"] = _repo_path(path)
                rows.append(enriched)
    return rows


def _collect_budget_reports(run_root: Path) -> list[dict]:
    reports: list[dict] = []
    for path in sorted(run_root.glob("**/quality.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict) or "threshold_passed" not in payload:
            continue
        enriched = dict(payload)
        enriched["_quality_report_json"] = _repo_path(path)
        reports.append(enriched)
    return reports


def _budget_summary_lines(reports: Sequence[dict]) -> list[str]:
    lines = [
        "| Report | Passed | Thresholds | Comparisons | Failed required checks |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for report in reports:
        lines.append(
            "| "
            f"`{report['_quality_report_json']}` | "
            f"{_pass_text(report.get('passed'))} | "
            f"{_pass_text(report.get('threshold_passed'))} | "
            f"{_pass_text(report.get('comparison_passed'))} | "
            f"{_failed_required_check_count(report)} |"
        )
    return lines


def _pass_text(value: object) -> str:
    return "pass" if value else "fail"


def _failed_required_check_count(report: dict) -> int:
    failed = 0
    for key in ("checks", "comparison_checks"):
        checks = report.get(key)
        if not isinstance(checks, list):
            continue
        for check in checks:
            if not isinstance(check, dict):
                continue
            if check.get("required", True) and not check.get("passed"):
                failed += 1
    return failed


def _matrix_summary_lines(rows: Sequence[dict]) -> list[str]:
    by_mode: dict[str, dict[str, float]] = {}
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        bucket = by_mode.setdefault(
            mode,
            {
                "total": 0.0,
                "passed": 0.0,
                "min_iou_sum": 0.0,
                "min_iou_count": 0.0,
                "boundary_sum": 0.0,
                "boundary_count": 0.0,
                "sdf_sum": 0.0,
                "sdf_count": 0.0,
            },
        )
        bucket["total"] += 1.0
        if row.get("passed"):
            bucket["passed"] += 1.0
        _accumulate_metric(bucket, metrics, "render.min_view_iou", "min_iou")
        _accumulate_metric(bucket, metrics, "render.boundary_iou_mean", "boundary")
        _accumulate_metric(bucket, metrics, "render.signed_distance_loss_mean", "sdf")

    lines = [
        "| Mode | Passed | Total | Avg min IoU | Avg Boundary IoU | Avg SDF Loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    ranked = sorted(
        by_mode.items(),
        key=lambda item: (
            item[1]["passed"] / max(item[1]["total"], 1.0),
            _average(item[1], "min_iou"),
            _average(item[1], "boundary"),
            -_average(item[1], "sdf", missing=1e9),
        ),
        reverse=True,
    )
    for mode, bucket in ranked:
        lines.append(
            "| "
            f"`{mode}` | "
            f"{int(bucket['passed'])} | "
            f"{int(bucket['total'])} | "
            f"{_fmt_metric(_average(bucket, 'min_iou'))} | "
            f"{_fmt_metric(_average(bucket, 'boundary'))} | "
            f"{_fmt_metric(_average(bucket, 'sdf'))} |"
        )
    return lines


def _accumulate_metric(
    bucket: dict[str, float],
    metrics: dict,
    metric_name: str,
    prefix: str,
) -> None:
    value = _lookup_metric(metrics, metric_name)
    if isinstance(value, (int, float)):
        bucket[f"{prefix}_sum"] += float(value)
        bucket[f"{prefix}_count"] += 1.0


def _lookup_metric(metrics: dict, metric_name: str) -> object:
    if metric_name in metrics:
        return metrics[metric_name]
    current: object = metrics
    for part in metric_name.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _average(bucket: dict[str, float], prefix: str, *, missing: float = 0.0) -> float:
    count = bucket.get(f"{prefix}_count", 0.0)
    if count <= 0:
        return missing
    return bucket.get(f"{prefix}_sum", 0.0) / count


def _fmt_metric(value: float) -> str:
    if value == 1e9:
        return ""
    return f"{value:.4f}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ambitious synthetic quality matrix, isolated LPIPS "
            "novel-view smoke, and closed-loop refinement sweeps."
        )
    )
    parser.add_argument(
        "--profile",
        default="interactive",
        choices=tuple(sorted(SMOKE_PROFILES)),
        help="Named workload profile. Use full-nightly for the old broad sweep.",
    )
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--blender-exe", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean-first", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-id-prefix", default="quality_refinement")
    parser.add_argument("--modes", default=None)
    parser.add_argument("--matrix-suites", default=None)
    parser.add_argument("--matrix-count", type=int, default=None)
    parser.add_argument(
        "--quality-budget-json",
        default="configs/quality_perf_budget-smoke.json",
    )
    parser.add_argument("--lpips-suite", default=None)
    parser.add_argument("--lpips-modes", default=None)
    parser.add_argument("--lpips-count", type=int, default=None)
    parser.add_argument("--lpips-angles", default="45,135")
    parser.add_argument("--lpips-psnr-threshold", type=float, default=0.0)
    parser.add_argument("--lpips-ssim-threshold", type=float, default=0.0)
    parser.add_argument("--lpips-threshold", type=float, default=1.0)
    parser.add_argument("--refinement-targets", default=None)
    parser.add_argument("--refinement-case-count", type=int, default=None)
    parser.add_argument("--refinement-max-runs", type=int, default=None)
    parser.add_argument("--refinement-top-k", type=int, default=None)
    parser.add_argument("--refinement-generations", type=int, default=None)
    parser.add_argument("--refinement-parent-top-k", type=int, default=None)
    parser.add_argument("--refinement-children-per-parent", type=int, default=None)
    parser.add_argument("--no-synthetic-matrix", action="store_true")
    parser.add_argument("--no-lpips-novel", action="store_true")
    parser.add_argument("--no-refinement-loop", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_root, phases = build_phase_plan(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not phases:
        print("ERROR: no phases selected", file=sys.stderr)
        return 2
    return run_phases(
        run_root,
        phases,
        dry_run=args.dry_run,
        clean_first=args.clean_first,
        stop_on_failure=args.stop_on_failure or _profile(args.profile).stop_on_failure,
    )


if __name__ == "__main__":
    raise SystemExit(main())
