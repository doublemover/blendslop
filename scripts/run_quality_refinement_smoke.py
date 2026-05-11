#!/usr/bin/env python3
"""Orchestrate ambitious quality/refinement smoke runs.

The script intentionally stays above the reconstruction stack: it builds the
same public e2e and refinement-lab commands a human would run, keeps all
generated outputs under repo-root temp/, and separates LPIPS/Torch work from
Open3D-heavy visual-hull refinement processes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blender_blocking.utils.path_safety import compact_path_segment

TEMP_ROOT = REPO_ROOT / "temp"
RUN_ROOT = TEMP_ROOT / "quality-refinement-runs"
PHASE_STATE_SCHEMA = "quality_refinement_phase_state_v1"
DEFAULT_CACHE_ROOT = TEMP_ROOT / "quality-refinement-cache"
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
        suite="synthetic-blender-smoke",
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


@dataclass(frozen=True)
class RefinementPhasePreflight:
    result_root: Path
    target: RefinementTarget
    case_count: int | None
    max_runs: int
    top_k: int
    seed: int


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
    "contract-canary": SmokeProfile(
        name="contract-canary",
        description="One-row backend-status contract canary across full-nightly matrix suites.",
        matrix_suites=DEFAULT_MATRIX_SUITES,
        matrix_count=1,
        matrix_budget_mode="warn",
        matrix_allow_failed_rows=True,
        stop_on_failure=True,
    ),
    "synthetic-family-canary": SmokeProfile(
        name="synthetic-family-canary",
        description="One-row synthetic family compatibility canary, including pure-mask suites.",
        matrix_suites=(
            "adversarial-silhouettes",
            "capture-noise",
            "smoke",
            "blender-smoke",
            "primitive-fit",
            "visual-hull",
        ),
        matrix_count=1,
        matrix_budget_mode="warn",
        matrix_allow_failed_rows=True,
        stop_on_failure=True,
    ),
    "adaptive-canary": SmokeProfile(
        name="adaptive-canary",
        description="One-case, one-variant adaptive canary for every refinement target.",
        refinement_targets=tuple(target.name for target in DEFAULT_REFINEMENT_TARGETS),
        refinement_case_count=1,
        refinement_max_runs=1,
        refinement_top_k=1,
        refinement_generations=1,
        refinement_parent_top_k=1,
        refinement_children_per_parent=1,
        stop_on_failure=True,
    ),
    "ensemble-canary": SmokeProfile(
        name="ensemble-canary",
        description="One-case ensemble adaptive canary on mesh-backed synthetic smoke.",
        refinement_targets=("ensemble-selection",),
        refinement_case_count=1,
        refinement_max_runs=1,
        refinement_top_k=1,
        refinement_generations=1,
        refinement_parent_top_k=1,
        refinement_children_per_parent=1,
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
        matrix_budget_mode="warn",
        matrix_allow_failed_rows=True,
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
    refinement_preflight: RefinementPhasePreflight | None = None
    depends_on: tuple[str, ...] = ()
    inline_quality_budget: tuple[Path, Path, Path] | None = None


@dataclass(frozen=True)
class PhaseResult:
    phase: PhaseCommand
    returncode: int
    elapsed_s: float
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    status: str = "pass"
    reused: bool = False
    original_elapsed_s: float | None = None

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


def resolve_cache_root(path: str | Path | None) -> Path:
    root = _resolve_under_repo(path or (TEMP_ROOT / "quality-refinement-cache"))
    temp = TEMP_ROOT.resolve(strict=False)
    if root != temp and temp not in root.parents:
        raise ValueError(f"cache root must stay under repo temp/: {root}")
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
    full_nightly = profile.name == "full-nightly"
    reference_cache = _bool_default(args.reference_cache, full_nightly)
    candidate_cache = _bool_default(args.candidate_cache, full_nightly)
    resume_candidates = _bool_default(args.resume_candidates, full_nightly)
    debug_artifact_policy = args.debug_artifact_policy or (
        "failures" if full_nightly else "all"
    )
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
                        depends_on=(f"matrix-{suite}",),
                        inline_quality_budget=(
                            result_json,
                            REPO_ROOT / args.quality_budget_json,
                            quality_report,
                        ),
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
                        cache_root=resolve_cache_root(args.cache_root),
                        reference_cache=reference_cache,
                        candidate_cache=candidate_cache,
                        resume_candidates=resume_candidates,
                        debug_artifact_policy=debug_artifact_policy,
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
                    refinement_preflight=RefinementPhasePreflight(
                        result_root=target_root,
                        target=target,
                        case_count=refinement_case_count,
                        max_runs=refinement_max_runs,
                        top_k=refinement_top_k,
                        seed=args.seed,
                    ),
                )
            )

    return run_root, tuple(phases)


def _bool_default(value: bool | None, default: bool) -> bool:
    return default if value is None else bool(value)


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
    cache_root: Path,
    reference_cache: bool,
    candidate_cache: bool,
    resume_candidates: bool,
    debug_artifact_policy: str,
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
        "--cache-root",
        _repo_path(cache_root),
        "--debug-artifact-policy",
        debug_artifact_policy,
    ]
    if reference_cache:
        command.append("--reference-cache")
    else:
        command.append("--no-reference-cache")
    if candidate_cache:
        command.append("--candidate-cache")
    else:
        command.append("--no-candidate-cache")
    if resume_candidates:
        command.append("--resume-candidates")
    else:
        command.append("--no-resume-candidates")
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
    preflight_only: bool = False,
    clean_first: bool = False,
    stop_on_failure: bool = False,
    resume: bool = False,
    force_phases: Sequence[str] = (),
    serial: bool = False,
    max_workers: int = 1,
    max_blender_workers: int = 1,
    max_open3d_workers: int = 1,
    max_lpips_workers: int = 1,
) -> int:
    if dry_run:
        _print_plan(run_root, phases)
        return 0

    preflight = preflight_phases(phases)
    if preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0 if preflight["passed"] else 1

    if clean_first and run_root.exists():
        _assert_temp_child(run_root)
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    recipe = write_command_recipe(run_root, phases)
    print(f"Saved command recipe: {_repo_path(recipe)}")
    preflight_path = run_root / "preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved preflight report: {_repo_path(preflight_path)}")
    if not preflight["passed"]:
        for issue in preflight["issues"]:
            print(
                "PREFLIGHT FAIL: "
                f"{issue.get('phase')} {issue.get('code')} "
                f"{issue.get('case_id', '')} {issue.get('family', '')}"
            )
        write_summary(run_root, phases, ())
        return 1

    force_set = set(force_phases)
    if serial or stop_on_failure or max_workers <= 1:
        results = _run_phases_serial(
            phases,
            run_root=run_root,
            resume=resume,
            force_phases=force_set,
            stop_on_failure=stop_on_failure,
        )
    else:
        results = _run_phases_parallel(
            phases,
            run_root=run_root,
            resume=resume,
            force_phases=force_set,
            max_workers=max_workers,
            max_blender_workers=max_blender_workers,
            max_open3d_workers=max_open3d_workers,
            max_lpips_workers=max_lpips_workers,
        )

    summary = write_summary(run_root, phases, results)
    print(f"Saved summary: {_repo_path(summary)}")
    return 0 if results and all(result.passed for result in results) else 1


def _run_phases_serial(
    phases: Sequence[PhaseCommand],
    *,
    run_root: Path,
    resume: bool,
    force_phases: set[str],
    stop_on_failure: bool,
) -> list[PhaseResult]:
    results: list[PhaseResult] = []
    by_name: dict[str, PhaseResult] = {}
    for phase in phases:
        blocked_by = [
            dep for dep in phase.depends_on if dep in by_name and not by_name[dep].passed
        ]
        if blocked_by:
            result = PhaseResult(
                phase=phase,
                returncode=1,
                elapsed_s=0.0,
                status=f"blocked:{','.join(blocked_by)}",
            )
        else:
            result = _run_one_phase_with_resume(
                phase,
                run_root=run_root,
                resume=resume,
                force=phase.name in force_phases,
            )
        results.append(result)
        by_name[phase.name] = result
        if not result.passed and stop_on_failure:
            break
    return results


def _run_phases_parallel(
    phases: Sequence[PhaseCommand],
    *,
    run_root: Path,
    resume: bool,
    force_phases: set[str],
    max_workers: int,
    max_blender_workers: int,
    max_open3d_workers: int,
    max_lpips_workers: int,
) -> list[PhaseResult]:
    pending = list(phases)
    completed: dict[str, PhaseResult] = {}
    ordered_results: list[PhaseResult] = []
    running: dict[Future[PhaseResult], PhaseCommand] = {}
    usage = {"blender": 0, "open3d": 0, "lpips": 0}

    def can_start(phase: PhaseCommand) -> bool:
        if len(running) >= max_workers:
            return False
        if phase.requires_blender and usage["blender"] >= max_blender_workers:
            return False
        if phase.uses_open3d_path and usage["open3d"] >= max_open3d_workers:
            return False
        if phase.uses_torch_lpips_path and usage["lpips"] >= max_lpips_workers:
            return False
        return True

    def add_usage(phase: PhaseCommand, delta: int) -> None:
        if phase.requires_blender:
            usage["blender"] += delta
        if phase.uses_open3d_path:
            usage["open3d"] += delta
        if phase.uses_torch_lpips_path:
            usage["lpips"] += delta

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        while pending or running:
            made_progress = False
            for phase in tuple(pending):
                deps = [completed.get(dep) for dep in phase.depends_on]
                if any(result is None for result in deps):
                    continue
                failed_deps = [
                    result.phase.name for result in deps if result is not None and not result.passed
                ]
                if failed_deps:
                    result = PhaseResult(
                        phase=phase,
                        returncode=1,
                        elapsed_s=0.0,
                        status=f"blocked:{','.join(failed_deps)}",
                    )
                    completed[phase.name] = result
                    ordered_results.append(result)
                    pending.remove(phase)
                    made_progress = True
                    continue
                if not can_start(phase):
                    continue
                _print_phase_header(phase)
                future = executor.submit(
                    _execute_phase,
                    phase,
                    run_root=run_root,
                    resume=resume,
                    force=phase.name in force_phases,
                    print_header=False,
                )
                running[future] = phase
                add_usage(phase, 1)
                pending.remove(phase)
                made_progress = True
            if not running:
                if pending and not made_progress:
                    for phase in pending:
                        result = PhaseResult(
                            phase=phase,
                            returncode=1,
                            elapsed_s=0.0,
                            status="blocked:unsatisfied_dependencies",
                        )
                        completed[phase.name] = result
                        ordered_results.append(result)
                    pending.clear()
                continue
            done, _not_done = wait(tuple(running), return_when=FIRST_COMPLETED)
            for future in done:
                phase = running.pop(future)
                add_usage(phase, -1)
                try:
                    result = future.result()
                except Exception as exc:
                    result = PhaseResult(
                        phase=phase,
                        returncode=2,
                        elapsed_s=0.0,
                        status="error",
                        stderr_path=_write_phase_exception_log(
                            run_root,
                            phase,
                            exc,
                        ),
                    )
                    print(f"ERROR: {phase.name}: {exc}", file=sys.stderr)
                completed[phase.name] = result
                ordered_results.append(result)
    by_name = {result.phase.name: result for result in ordered_results}
    return [by_name[phase.name] for phase in phases if phase.name in by_name]


def _run_one_phase_with_resume(
    phase: PhaseCommand,
    *,
    run_root: Path,
    resume: bool,
    force: bool,
) -> PhaseResult:
    _print_phase_header(phase)
    return _execute_phase(phase, run_root=run_root, resume=resume, force=force)


def _execute_phase(
    phase: PhaseCommand,
    *,
    run_root: Path,
    resume: bool,
    force: bool,
    print_header: bool = False,
) -> PhaseResult:
    if print_header:
        _print_phase_header(phase)
    if resume and not force:
        reused = _resume_phase_result(phase, run_root=run_root)
        if reused is not None:
            print(f"REUSE: {phase.name} ({reused.original_elapsed_s or 0.0:.1f}s original)")
            return reused

    started = time.perf_counter()
    if phase.inline_quality_budget is not None:
        returncode, stdout_path, stderr_path = _run_inline_quality_budget_phase(
            phase,
            run_root=run_root,
        )
    else:
        returncode, stdout_path, stderr_path = _run_phase_command(
            phase,
            run_root=run_root,
        )
    elapsed_s = time.perf_counter() - started
    result = PhaseResult(
        phase=phase,
        returncode=returncode,
        elapsed_s=elapsed_s,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        status="pass" if returncode == 0 else "fail",
    )
    if result.passed:
        _write_phase_state(run_root, result)
    status = "PASS" if result.passed else "FAIL"
    print(f"{status}: {phase.name} ({elapsed_s:.1f}s)")
    return result


def preflight_phases(phases: Sequence[PhaseCommand]) -> dict[str, object]:
    issues: list[dict[str, object]] = []
    checked_refinement = 0
    for phase in phases:
        preflight = phase.refinement_preflight
        if preflight is None:
            continue
        checked_refinement += 1
        issues.extend(_preflight_refinement_phase(phase, preflight))
    return {
        "schema_version": "quality_refinement_preflight_v1",
        "passed": not issues,
        "checked_refinement_phases": checked_refinement,
        "issue_count": len(issues),
        "issues": issues,
    }


def _preflight_refinement_phase(
    phase: PhaseCommand,
    preflight: RefinementPhasePreflight,
) -> list[dict[str, object]]:
    try:
        from blender_blocking.refinement_lab.matrix import build_experiment_plan
        from blender_blocking.synthetic.registry import get_definition
    except ImportError:  # pragma: no cover
        from refinement_lab.matrix import build_experiment_plan
        from synthetic.registry import get_definition

    plan = build_experiment_plan(
        suite=preflight.target.suite,
        track=preflight.target.track,
        search=preflight.target.search,
        objective=preflight.target.objective,
        output_root=preflight.result_root / "_preflight",
        seed=preflight.seed,
        case_count=preflight.case_count,
        max_runs=preflight.max_runs,
        top_k=preflight.top_k,
    )
    issues: list[dict[str, object]] = []
    for case in plan.cases:
        spec = case.metadata.get("spec", {}) if isinstance(case.metadata, Mapping) else {}
        family = str(spec.get("family", ""))
        definition_name = str(case.synthetic_definition or spec.get("shape_id", ""))
        blender_supported = True
        try:
            definition = get_definition(definition_name)
            blender_supported = bool(definition.blender_supported)
        except Exception:
            blender_supported = family in {
                "analytic_primitive",
                "profile_lathe",
                "furniture",
                "vehicle_mechanical",
            }
        if case.source == "synthetic" and not blender_supported:
            issues.append(
                {
                    "phase": phase.name,
                    "code": "unsupported_synthetic_family_for_refinement",
                    "suite": preflight.target.suite,
                    "track": preflight.target.track,
                    "case_id": case.case_id,
                    "synthetic_definition": definition_name,
                    "family": family,
                    "message": (
                        "Refinement runners generate references with Blender mesh "
                        "builders; this synthetic definition is pure-mask only."
                    ),
                }
            )
    return issues


def _run_phase_command(
    phase: PhaseCommand,
    *,
    run_root: Path,
) -> tuple[int, Path, Path]:
    log_dir = run_root / "phase-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = (
        log_dir / f"{compact_path_segment(phase.name, max_length=64)}.stdout.txt"
    )
    stderr_path = (
        log_dir / f"{compact_path_segment(phase.name, max_length=64)}.stderr.txt"
    )
    with stdout_path.open(
        "w",
        encoding="utf-8",
        errors="replace",
    ) as stdout_file, stderr_path.open(
        "w",
        encoding="utf-8",
        errors="replace",
    ) as stderr_file:
        process = subprocess.Popen(
            phase.command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        threads = []
        if process.stdout is not None:
            threads.append(
                threading.Thread(
                    target=_tee_stream,
                    args=(process.stdout, stdout_file, sys.stdout),
                    daemon=True,
                )
            )
        if process.stderr is not None:
            threads.append(
                threading.Thread(
                    target=_tee_stream,
                    args=(process.stderr, stderr_file, sys.stderr),
                    daemon=True,
                )
            )
        for thread in threads:
            thread.start()
        returncode = process.wait()
        for thread in threads:
            thread.join()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
    return returncode, stdout_path, stderr_path


def _run_inline_quality_budget_phase(
    phase: PhaseCommand,
    *,
    run_root: Path,
) -> tuple[int, Path, Path]:
    if phase.inline_quality_budget is None:
        raise ValueError("inline quality budget phase is missing payload paths")
    result_json, budget_json, report_json = phase.inline_quality_budget
    stdout_path, stderr_path = _phase_log_paths(run_root, phase)
    warn_only = "--warn-only" in phase.command
    try:
        from scripts.quality_budget import evaluate_budget_files, write_report

        report = evaluate_budget_files(
            current_path=result_json,
            budget_path=budget_json,
            baseline_path=None,
        )
        write_report(report_json, report)
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        stdout_path.write_text(text, encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        print(text, end="")
        returncode = 0 if warn_only or bool(report.get("passed")) else 1
    except Exception as exc:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        print(f"INLINE QUALITY BUDGET ERROR: {exc}", file=sys.stderr)
        returncode = 2
    return returncode, stdout_path, stderr_path


def _phase_log_paths(run_root: Path, phase: PhaseCommand) -> tuple[Path, Path]:
    log_dir = run_root / "phase-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = compact_path_segment(phase.name, max_length=64)
    return log_dir / f"{stem}.stdout.txt", log_dir / f"{stem}.stderr.txt"


def _write_phase_exception_log(
    run_root: Path,
    phase: PhaseCommand,
    exc: Exception,
) -> Path:
    _stdout_path, stderr_path = _phase_log_paths(run_root, phase)
    stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
    return stderr_path


def _phase_state_path(run_root: Path, phase: PhaseCommand) -> Path:
    state_dir = run_root / ".phase-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{compact_path_segment(phase.name, max_length=80)}.json"


def _phase_command_hash(phase: PhaseCommand) -> str:
    payload = {
        "name": phase.name,
        "command": list(phase.command),
        "artifacts": [_repo_path(path) for path in phase.artifacts],
        "depends_on": list(phase.depends_on),
        "inline_quality_budget": (
            [_repo_path(path) for path in phase.inline_quality_budget]
            if phase.inline_quality_budget
            else None
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resume_phase_result(
    phase: PhaseCommand,
    *,
    run_root: Path,
) -> PhaseResult | None:
    path = _phase_state_path(run_root, phase)
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if state.get("schema_version") != PHASE_STATE_SCHEMA:
        return None
    if state.get("phase") != phase.name:
        return None
    if state.get("command_hash") != _phase_command_hash(phase):
        return None
    if int(state.get("returncode", 1)) != 0:
        return None
    if not all(_artifact_ready(path) for path in phase.artifacts):
        return None
    return PhaseResult(
        phase=phase,
        returncode=0,
        elapsed_s=0.0,
        stdout_path=_state_path_or_none(state.get("stdout_path")),
        stderr_path=_state_path_or_none(state.get("stderr_path")),
        status="reused",
        reused=True,
        original_elapsed_s=float(state.get("elapsed_s") or 0.0),
    )


def _write_phase_state(run_root: Path, result: PhaseResult) -> None:
    if not result.passed:
        return
    payload = {
        "schema_version": PHASE_STATE_SCHEMA,
        "phase": result.phase.name,
        "command_hash": _phase_command_hash(result.phase),
        "returncode": result.returncode,
        "status": result.status,
        "elapsed_s": result.original_elapsed_s
        if result.reused and result.original_elapsed_s is not None
        else result.elapsed_s,
        "stdout_path": _repo_path(result.stdout_path) if result.stdout_path else None,
        "stderr_path": _repo_path(result.stderr_path) if result.stderr_path else None,
        "artifacts": [_repo_path(path) for path in result.phase.artifacts],
        "written_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    path = _phase_state_path(run_root, result.phase)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _state_path_or_none(value: object) -> Path | None:
    if not value:
        return None
    return _resolve_under_repo(str(value))


def _artifact_ready(path: Path) -> bool:
    path = Path(path)
    if path.is_dir():
        return True
    return path.exists()


def _tee_stream(source: object, log_file: object, console: object) -> None:
    for line in source:  # type: ignore[operator]
        log_file.write(line)
        log_file.flush()
        console.write(line)
        console.flush()


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
    quality = _quality_summary(results, matrix_rows, budget_reports)
    lines = [
        "# Quality Refinement Smoke Summary",
        "",
        f"Run root: `{_repo_path(run_root)}`",
        "",
        f"Overall quality status: **{quality['overall_quality_status']}**",
        f"Quality status reason: `{quality['quality_status_reason']}`",
        f"Required budget failures: {quality['required_budget_failure_count']}",
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
            status = _phase_status_text(result)
            elapsed = result.original_elapsed_s if result.reused else result.elapsed_s
            lines.append(f"| `{phase.name}` | {status} | {elapsed or 0.0:.1f} |")
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
    _write_machine_summary(run_root, results, matrix_rows, budget_reports, quality)
    return path


def _write_machine_summary(
    run_root: Path,
    results: Sequence[PhaseResult],
    matrix_rows: Sequence[dict],
    budget_reports: Sequence[dict],
    quality: Mapping[str, object],
) -> None:
    payload = {
        "schema_version": "quality_refinement_smoke_summary_v1",
        "run_root": _repo_path(run_root),
        "overall_quality_status": quality["overall_quality_status"],
        "quality_status_reason": quality["quality_status_reason"],
        "required_budget_failures": quality["required_budget_failures"],
        "required_budget_failure_count": quality["required_budget_failure_count"],
        "phase_results": [
            {
                "name": result.phase.name,
                "returncode": result.returncode,
                "passed": result.passed,
                "status": result.status,
                "process_status": _phase_process_status(result),
                "quality_status": _phase_quality_status(result),
                "reused": result.reused,
                "original_elapsed_s": result.original_elapsed_s,
                "elapsed_s": result.elapsed_s,
                "artifacts": [_repo_path(path) for path in result.phase.artifacts],
                "expected_candidate_count": result.phase.expected_candidate_count,
                "expected_blender_invocations": result.phase.expected_blender_invocations,
                "stdout_path": _repo_path(result.stdout_path) if result.stdout_path else None,
                "stderr_path": _repo_path(result.stderr_path) if result.stderr_path else None,
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


def _quality_summary(
    results: Sequence[PhaseResult],
    matrix_rows: Sequence[dict],
    budget_reports: Sequence[dict],
) -> dict[str, object]:
    required_budget_failures = _required_budget_failures(budget_reports)
    process_failures = [result.phase.name for result in results if not result.passed]
    matrix_failed = sum(1 for row in matrix_rows if not row.get("passed"))
    if process_failures:
        status = "fail"
        reason = "phase_process_failure"
    elif required_budget_failures:
        status = "fail"
        reason = "required_budget_failures"
    elif matrix_failed and not budget_reports:
        status = "fail"
        reason = "matrix_row_failures_without_budget"
    elif not results and not matrix_rows and not budget_reports:
        status = "not_run"
        reason = "no_evidence"
    else:
        status = "pass"
        reason = "all_required_quality_gates_passed"
    return {
        "overall_quality_status": status,
        "quality_status_reason": reason,
        "required_budget_failures": required_budget_failures,
        "required_budget_failure_count": len(required_budget_failures),
        "process_failures": process_failures,
        "matrix_failed": matrix_failed,
    }


def _required_budget_failures(reports: Sequence[dict]) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for report in reports:
        failed_required_checks = _failed_required_check_count(report)
        if failed_required_checks <= 0:
            continue
        failures.append(
            {
                "path": report.get("_quality_report_json", ""),
                "failed_required_checks": failed_required_checks,
                "passed": report.get("passed"),
                "threshold_passed": report.get("threshold_passed"),
                "comparison_passed": report.get("comparison_passed"),
            }
        )
    return failures


def _phase_process_status(result: PhaseResult) -> str:
    if result.reused:
        return "reused"
    if result.passed:
        return "pass"
    if result.status.startswith("blocked"):
        return result.status
    return "fail"


def _phase_quality_status(result: PhaseResult) -> str:
    if result.phase.inline_quality_budget is not None:
        return "warn" if result.passed else "fail"
    return "pass" if result.passed else "fail"


def _phase_status_text(result: PhaseResult) -> str:
    if result.reused:
        return "reused"
    if result.status.startswith("blocked"):
        return result.status
    if result.passed:
        return "pass"
    return f"{result.status} ({result.returncode})"


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
            _sort_metric(_average(item[1], "min_iou")),
            _sort_metric(_average(item[1], "boundary")),
            -_sort_metric(_average(item[1], "sdf"), missing=1e9),
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


def _average(
    bucket: dict[str, float],
    prefix: str,
    *,
    missing: float | None = None,
) -> float | None:
    count = bucket.get(f"{prefix}_count", 0.0)
    if count <= 0:
        return missing
    return bucket.get(f"{prefix}_sum", 0.0) / count


def _sort_metric(value: float | None, *, missing: float = 0.0) -> float:
    return missing if value is None else value


def _fmt_metric(value: float | None) -> str:
    if value is None or value == 1e9:
        return "n/a"
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
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--clean-first", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-blender-workers", type=int, default=None)
    parser.add_argument("--max-open3d-workers", type=int, default=None)
    parser.add_argument("--max-lpips-workers", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force-phase", action="append", default=[])
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument(
        "--reference-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--candidate-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--resume-candidates",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--debug-artifact-policy",
        choices=("all", "failures", "top", "none"),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-id-prefix", default="quality_refinement")
    parser.add_argument("--modes", default=None)
    parser.add_argument("--matrix-suites", default=None)
    parser.add_argument("--matrix-count", type=int, default=None)
    parser.add_argument(
        "--quality-budget-json",
        default="configs/quality_perf_budget-backend-status-smoke.json",
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
    profile = _profile(args.profile)
    try:
        run_root, phases = build_phase_plan(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not phases:
        print("ERROR: no phases selected", file=sys.stderr)
        return 2
    full_nightly = profile.name == "full-nightly"
    resume = _bool_default(args.resume, full_nightly)
    max_workers = args.max_workers if args.max_workers is not None else (4 if full_nightly else 1)
    max_blender_workers = (
        args.max_blender_workers if args.max_blender_workers is not None else (2 if full_nightly else 1)
    )
    max_open3d_workers = args.max_open3d_workers if args.max_open3d_workers is not None else 1
    max_lpips_workers = args.max_lpips_workers if args.max_lpips_workers is not None else 1
    return run_phases(
        run_root,
        phases,
        dry_run=args.dry_run,
        preflight_only=args.preflight_only,
        clean_first=args.clean_first,
        stop_on_failure=args.stop_on_failure or profile.stop_on_failure,
        resume=resume,
        force_phases=args.force_phase,
        serial=args.serial,
        max_workers=max_workers,
        max_blender_workers=max_blender_workers,
        max_open3d_workers=max_open3d_workers,
        max_lpips_workers=max_lpips_workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
