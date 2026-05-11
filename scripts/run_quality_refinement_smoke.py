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
from functools import lru_cache
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
from blender_blocking.metrics.namespaces import get_metric_path

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


def _configure_stdio_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


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
        name="moonshot-sidecars",
        suite="synthetic-primitive-fit",
        track="shape-program-editability",
        search="coordinate",
        objective="profile_editable",
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
    moonshot_sidecars: bool = False
    moonshot_experiments: tuple[str, ...] = ()
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
    "moonshot-smoke": SmokeProfile(
        name="moonshot-smoke",
        description="Cheap sidecar evidence generation for moonshot diagnostics.",
        refinement_targets=(
            "moonshot-sidecars",
            "primitive-fit",
            "gaussian-proxy",
            "differentiable-refine",
            "ensemble-selection",
        ),
        refinement_max_runs=1,
        refinement_generations=1,
        refinement_top_k=1,
        moonshot_sidecars=True,
        moonshot_experiments=(
            "shape_grammar_search",
            "active_view_planning",
            "implicit_sdf_proxy",
            "editable_retopology",
            "human_constraint_learning",
            "differentiable_primitives",
            "moonshot_portfolio_optimizer",
        ),
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
    expected_case_count: int = 0


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


_ANSI_COLORS = {
    "blue": "34",
    "cyan": "36",
    "green": "32",
    "magenta": "35",
    "red": "31",
    "yellow": "33",
    "dim": "2",
}

_BADGE_SYMBOLS = {
    "blocked": "⛔",
    "error": "✗",
    "fail": "✗",
    "pass": "✓",
    "reuse": "↻",
    "warn": "!",
    "wrote": "✎",
}


def _color_enabled(stream: object | None = None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def _style(
    text: str,
    color: str | None = None,
    *,
    bold: bool = False,
    stream: object | None = None,
) -> str:
    if not _color_enabled(stream):
        return text
    codes: list[str] = []
    if bold:
        codes.append("1")
    if color:
        code = _ANSI_COLORS.get(color)
        if code:
            codes.append(code)
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def _badge(label: str, color: str, *, stream: object | None = None) -> str:
    text = f"{_BADGE_SYMBOLS.get(label.lower(), '•')} {label.upper():<7}"
    return _style(text, color, bold=True, stream=stream)


def _muted(text: str) -> str:
    return _style(text, "dim")


def _print_console_table(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[tuple[str, str, str]],
) -> None:
    if not rows:
        return
    widths = {
        key: max(len(label), *(len(str(row.get(key, ""))) for row in rows))
        for key, label, _align in columns
    }
    header_parts = []
    rule_parts = []
    for key, label, align in columns:
        width = widths[key]
        header_parts.append(f"{label:>{width}}" if align == "right" else f"{label:<{width}}")
        rule_parts.append("-" * width)
    print(_style("  " + "  ".join(header_parts), "dim", bold=True))
    print(_style("  " + "  ".join(rule_parts), "dim"))
    for row in rows:
        parts = []
        for key, _label, align in columns:
            value = str(row.get(key, ""))
            width = widths[key]
            parts.append(f"{value:>{width}}" if align == "right" else f"{value:<{width}}")
        print("  " + "  ".join(parts))


def _format_count(value: int) -> str:
    return "-" if int(value) == 0 else str(int(value))


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def build_phase_plan(args: argparse.Namespace) -> tuple[Path, tuple[PhaseCommand, ...]]:
    run_root = resolve_run_root(args.run_root)
    profile = _profile(args.profile)
    full_nightly = profile.name == "full-nightly"
    reference_cache = _bool_default(args.reference_cache, full_nightly)
    candidate_cache = _bool_default(args.candidate_cache, full_nightly)
    resume_candidates = _bool_default(args.resume_candidates, full_nightly)
    moonshot_sidecars = _bool_default(args.moonshot_sidecars, profile.moonshot_sidecars)
    moonshot_experiments = (
        _parse_csv(args.moonshot_experiments)
        or profile.moonshot_experiments
    )
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
            expected_case_count = _estimate_refinement_case_count(
                target=target,
                case_count=refinement_case_count,
                seed=args.seed,
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
                        moonshot_sidecars=moonshot_sidecars,
                        moonshot_experiments=moonshot_experiments,
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
                        * max(1, expected_case_count)
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
                    expected_case_count=expected_case_count,
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
    moonshot_sidecars: bool,
    moonshot_experiments: Sequence[str],
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
        str(result_root.resolve(strict=False)),
        "--seed",
        str(seed),
        "--report-failures",
        "all",
        "--cache-root",
        str(cache_root.resolve(strict=False)),
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
    if moonshot_sidecars:
        command.append("--moonshot-sidecars")
    else:
        command.append("--no-moonshot-sidecars")
    if moonshot_experiments:
        command.extend(("--moonshot-experiments", ",".join(moonshot_experiments)))
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


def _estimate_refinement_case_count(
    *,
    target: RefinementTarget,
    case_count: int | None,
    seed: int,
) -> int:
    return _estimate_refinement_suite_case_count(target.suite, case_count, seed)


@lru_cache(maxsize=64)
def _estimate_refinement_suite_case_count(
    suite_name: str,
    case_count: int | None,
    seed: int,
) -> int:
    try:
        from blender_blocking.refinement_lab.matrix import resolve_suite_cases
        from blender_blocking.refinement_lab.preset_catalog import get_suite_preset
    except ImportError:  # pragma: no cover - direct blender_blocking/ execution
        from refinement_lab.matrix import resolve_suite_cases
        from refinement_lab.preset_catalog import get_suite_preset

    suite = get_suite_preset(suite_name)
    cases = resolve_suite_cases(
        suite,
        seed=seed or suite.default_seed,
        count=case_count if case_count is not None else suite.default_count,
    )
    return len(cases)


def format_command(command: Sequence[str]) -> str:
    parts = [_quote_arg(part) for part in command]
    if parts and parts[0].startswith('"'):
        return "& " + " ".join(parts)
    return " ".join(parts)


def format_console_command(command: Sequence[str]) -> str:
    parts = [_quote_arg(_console_command_arg(part)) for part in command]
    if parts and parts[0].startswith('"'):
        return "& " + " ".join(parts)
    return " ".join(parts)


def _console_command_arg(value: object) -> str:
    text = str(value)
    if not text:
        return text
    try:
        path = Path(text)
    except (TypeError, ValueError):
        return text
    if not path.is_absolute():
        return text
    try:
        return str(path.resolve(strict=False).relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return path.name if path.suffix else text


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
    _configure_stdio_encoding()
    if dry_run:
        _print_plan(run_root, phases)
        return 0

    preflight = preflight_phases(phases, verify_executables=True)
    if preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0 if preflight["passed"] else 1

    if clean_first and run_root.exists():
        _assert_temp_child(run_root)
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    recipe = write_command_recipe(run_root, phases)
    print(f"{_badge('wrote', 'cyan')} command recipe   {_repo_path(recipe)}")
    preflight_path = run_root / "preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{_badge('wrote', 'cyan')} preflight report {_repo_path(preflight_path)}")
    if not preflight["passed"]:
        for issue in preflight["issues"]:
            print(
                f"{_badge('blocked', 'yellow')} "
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
    print(f"{_badge('wrote', 'cyan')} summary          {_repo_path(summary)}")
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
                    print(
                        f"{_badge('error', 'red', stream=sys.stderr)} {phase.name}: {exc}",
                        file=sys.stderr,
                    )
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
            print(
                f"{_badge('reuse', 'cyan')} {phase.name} "
                f"({reused.original_elapsed_s or 0.0:.1f}s original)"
            )
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
    label = "pass" if result.passed else "fail"
    color = "green" if result.passed else "red"
    print(f"{_badge(label, color)} {phase.name} ({elapsed_s:.1f}s)")
    return result


def preflight_phases(
    phases: Sequence[PhaseCommand],
    *,
    verify_executables: bool = False,
) -> dict[str, object]:
    issues: list[dict[str, object]] = []
    checked_refinement = 0
    checked_executables = 0
    for phase in phases:
        if verify_executables and phase.requires_blender:
            checked_executables += 1
            issues.extend(_preflight_phase_executable(phase))
        preflight = phase.refinement_preflight
        if preflight is None:
            continue
        checked_refinement += 1
        issues.extend(_preflight_refinement_phase(phase, preflight))
    return {
        "schema_version": "quality_refinement_preflight_v1",
        "passed": not issues,
        "checked_refinement_phases": checked_refinement,
        "checked_executable_phases": checked_executables,
        "issue_count": len(issues),
        "issues": issues,
    }


def _preflight_phase_executable(phase: PhaseCommand) -> list[dict[str, object]]:
    if not phase.command:
        return [
            {
                "phase": phase.name,
                "code": "missing_phase_command",
                "message": "Phase has no command to execute.",
            }
        ]
    executable = str(phase.command[0])
    if _executable_available(executable):
        return []
    return [
        {
            "phase": phase.name,
            "code": "missing_blender_executable",
            "executable": executable,
            "message": (
                "This phase requires Blender, but the configured executable "
                "does not exist and was not found on PATH."
            ),
        }
    ]


def _executable_available(executable: str) -> bool:
    path = Path(executable)
    separators = [os.sep]
    if os.altsep:
        separators.append(os.altsep)
    looks_like_path = path.is_absolute() or bool(path.drive) or any(
        separator in executable for separator in separators
    )
    if looks_like_path:
        return path.exists()
    return shutil.which(executable) is not None


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
            env=_subprocess_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
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
                    args=(process.stderr, stderr_file, sys.stderr, True),
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
        failed_required = _failed_required_check_count(report)
        passed = bool(report.get("passed"))
        status = "pass" if passed else ("warn" if warn_only else "fail")
        color = "green" if passed else ("yellow" if warn_only else "red")
        print(
            f"{_badge(status, color)} quality budget "
            f"required_failures={failed_required} report={_repo_path(report_json)}"
        )
        returncode = 0 if warn_only or bool(report.get("passed")) else 1
    except Exception as exc:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        print(
            f"{_badge('error', 'red', stream=sys.stderr)} inline quality budget: {exc}",
            file=sys.stderr,
        )
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


def _tee_stream(
    source: object,
    log_file: object,
    console: object,
    suppress_known_noise: bool = False,
) -> None:
    for line in source:  # type: ignore[operator]
        log_file.write(line)
        log_file.flush()
        if suppress_known_noise and _is_known_warning_noise(line):
            continue
        console.write(line)
        if line and not line.endswith(("\n", "\r")):
            console.write("\n")
        console.flush()


def _is_known_warning_noise(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    known_fragments = (
        "The parameter 'pretrained' is deprecated",
        "Arguments other than a weight enum or `None` for 'weights' are deprecated",
        "You are using `torch.load` with `weights_only=False`",
        "torchvision\\models\\_utils.py",
        "torchvision/models/_utils.py",
        "lpips\\lpips.py",
        "lpips/lpips.py",
        "warnings.warn",
        "self.load_state_dict(torch.load",
    )
    return any(fragment in text for fragment in known_fragments)


def _assert_temp_child(path: Path) -> None:
    temp = TEMP_ROOT.resolve(strict=False)
    resolved = path.resolve(strict=False)
    if resolved == temp or temp not in resolved.parents:
        raise ValueError(f"refusing to delete outside repo temp/: {resolved}")


def _print_plan(run_root: Path, phases: Sequence[PhaseCommand]) -> None:
    candidates = sum(phase.expected_candidate_count for phase in phases)
    blender_starts = sum(phase.expected_blender_invocations for phase in phases)
    artifact_count = sum(len(phase.artifacts) for phase in phases) + 3
    print(_style("Quality / Refinement Smoke Plan", "cyan", bold=True))
    print(f"  Run root   {_repo_path(run_root)}")
    print(
        "  Workload   "
        f"{len(phases)} phases, "
        f"{candidates} candidates, "
        f"{blender_starts} Blender starts, "
        f"{artifact_count} artifacts"
    )
    print()
    rows: list[dict[str, object]] = []
    for index, phase in enumerate(phases, start=1):
        rows.append(
            {
                "#": index,
                "phase": phase.name,
                "cases": _format_count(phase.expected_case_count),
                "candidates": _format_count(phase.expected_candidate_count),
                "blender": _format_count(phase.expected_blender_invocations),
                "artifacts": _format_count(len(phase.artifacts)),
            }
        )
    _print_console_table(
        rows,
        (
            ("#", "#", "right"),
            ("phase", "Phase", "left"),
            ("cases", "Cases", "right"),
            ("candidates", "Candidates", "right"),
            ("blender", "Blender", "right"),
            ("artifacts", "Artifacts", "right"),
        ),
    )
    print()
    print(_style("Commands", "cyan", bold=True))
    for index, phase in enumerate(phases, start=1):
        print()
        print(_style(f"{index}. {phase.name}", "magenta", bold=True))
        print(f"   {phase.description}")
        print(f"   {_muted(format_console_command(phase.command))}")
        for artifact in phase.artifacts:
            print(f"   {_style('artifact', 'dim')}: {_repo_path(artifact)}")


def _print_phase_header(phase: PhaseCommand) -> None:
    print()
    print(_style(f"▶ {phase.name}", "magenta", bold=True))
    print(f"  {phase.description}")
    if phase.uses_torch_lpips_path:
        print(f"  {_style('lane', 'dim')}: Torch/LPIPS isolated process")
    elif phase.uses_open3d_path:
        print(f"  {_style('lane', 'dim')}: Open3D/OpenVDB-capable process")
    print(f"  {_style('command', 'dim')}: {format_console_command(phase.command)}")


def write_summary(
    run_root: Path,
    phases: Sequence[PhaseCommand],
    results: Sequence[PhaseResult],
) -> Path:
    path = run_root / "summary.md"
    matrix_rows = _collect_matrix_rows(run_root)
    budget_reports = _collect_budget_reports(run_root)
    cache_stats = _collect_cache_stats(run_root)
    refinement_summaries = _collect_refinement_summaries(run_root)
    quality = _quality_summary(results, matrix_rows, budget_reports, refinement_summaries)
    lines = [
        "# Quality Refinement Smoke Summary",
        "",
        f"Run root: `{_repo_path(run_root)}`",
        "",
        f"Overall quality status: **{quality['overall_quality_status']}**",
        f"Quality status reason: `{quality['quality_status_reason']}`",
        f"Required budget failures: {quality['required_budget_failure_count']}",
        f"Required refinement failures: {quality['refinement_required_failure_count']}",
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
    failed_matrix_rows = _failed_matrix_rows(matrix_rows)
    if failed_matrix_rows:
        lines.extend(["", "## Failed Matrix Rows", ""])
        lines.extend(_failed_matrix_row_lines(failed_matrix_rows))
    lines.extend(["", "## Quality Budget Reports", ""])
    if budget_reports:
        lines.extend(_budget_summary_lines(budget_reports))
    else:
        lines.append("No quality budget report JSON files were found.")
    lines.extend(["", "## Refinement Results", ""])
    if refinement_summaries:
        lines.extend(_refinement_summary_lines(refinement_summaries))
    else:
        lines.append("No refinement adaptive summaries were found.")
    lines.extend(["", "## Moonshot Sidecars", ""])
    moonshot_lines = _moonshot_summary_lines(refinement_summaries)
    if moonshot_lines:
        lines.extend(moonshot_lines)
    else:
        lines.append("No moonshot sidecar evidence was found.")
    lines.extend(["", "## Cache Stats", ""])
    if cache_stats:
        lines.extend(_cache_summary_lines(cache_stats))
    else:
        lines.append("No refinement cache stats were found.")
    cache_warnings = _cache_health_warnings(cache_stats, refinement_summaries)
    if cache_warnings:
        lines.extend(["", "## Cache Health Warnings", ""])
        for warning in cache_warnings:
            lines.append(f"- `{warning['code']}`: {warning['message']}")
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
    _write_machine_summary(
        run_root,
        results,
        matrix_rows,
        budget_reports,
        cache_stats,
        refinement_summaries,
        quality,
        cache_warnings,
    )
    return path


def _write_machine_summary(
    run_root: Path,
    results: Sequence[PhaseResult],
    matrix_rows: Sequence[dict],
    budget_reports: Sequence[dict],
    cache_stats: Sequence[dict],
    refinement_summaries: Sequence[dict],
    quality: Mapping[str, object],
    cache_warnings: Sequence[Mapping[str, object]],
) -> None:
    cache_totals = _cache_totals(cache_stats)
    failed_matrix_rows = _failed_matrix_rows(matrix_rows)
    refinement_required_failures = quality["refinement_required_failures"]
    payload = {
        "schema_version": "quality_refinement_smoke_summary_v1",
        "run_root": _repo_path(run_root),
        "overall_quality_status": quality["overall_quality_status"],
        "quality_status_reason": quality["quality_status_reason"],
        "required_budget_failures": quality["required_budget_failures"],
        "required_budget_failure_count": quality["required_budget_failure_count"],
        "required_refinement_failures": refinement_required_failures,
        "required_refinement_failure_count": quality["refinement_required_failure_count"],
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
                "expected_case_count": result.phase.expected_case_count,
                "stdout_path": _repo_path(result.stdout_path) if result.stdout_path else None,
                "stderr_path": _repo_path(result.stderr_path) if result.stderr_path else None,
            }
            for result in results
        ],
        "matrix_row_count": len(matrix_rows),
        "matrix_passed": sum(1 for row in matrix_rows if row.get("passed")),
        "matrix_failed": sum(1 for row in matrix_rows if not row.get("passed")),
        "matrix_failed_rows": failed_matrix_rows,
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
        "cache_stats": cache_stats,
        "cache_totals": cache_totals,
        "cache_health_warnings": list(cache_warnings),
        "reference_cache_totals": {
            "hits": cache_totals["reference_cache_hits"],
            "misses": cache_totals["reference_cache_misses"],
            "writes": cache_totals["reference_cache_writes"],
        },
        "candidate_cache_totals": {
            "hits": cache_totals["candidate_cache_hits"],
            "misses": cache_totals["candidate_cache_misses"],
            "writes": cache_totals["candidate_cache_writes"],
            "local_resume_writes": cache_totals["candidate_cache_local_resume_writes"],
            "shared_writes": cache_totals["candidate_cache_shared_writes"],
            "sources": cache_totals["candidate_cache_sources"],
        },
        "refinement_summaries": refinement_summaries,
        "refinement_error_rows": sum(int(row.get("error_rows", 0) or 0) for row in refinement_summaries),
        "refinement_promotable_rows": sum(int(row.get("promotable_rows", 0) or 0) for row in refinement_summaries),
        "refinement_parent_selectable_rows": sum(
            int(row.get("parent_selectable_rows", 0) or 0)
            for row in refinement_summaries
        ),
        "moonshot_summary": _moonshot_totals(refinement_summaries),
        "refinement_required_failures": refinement_required_failures,
    }
    (run_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _quality_summary(
    results: Sequence[PhaseResult],
    matrix_rows: Sequence[dict],
    budget_reports: Sequence[dict],
    refinement_summaries: Sequence[dict],
) -> dict[str, object]:
    required_budget_failures = _required_budget_failures(budget_reports)
    refinement_required_failures = _refinement_required_failures(refinement_summaries)
    process_failures = [result.phase.name for result in results if not result.passed]
    matrix_failed = sum(1 for row in matrix_rows if not row.get("passed"))
    if process_failures:
        status = "fail"
        reason = "phase_process_failure"
    elif required_budget_failures:
        status = "fail"
        reason = "required_budget_failures"
    elif refinement_required_failures:
        status = "fail"
        reason = "refinement_required_failures"
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
        "refinement_required_failures": refinement_required_failures,
        "refinement_required_failure_count": len(refinement_required_failures),
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


def _collect_cache_stats(run_root: Path) -> list[dict]:
    stats: list[dict] = []
    for path in sorted(run_root.glob("**/cache-stats.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        enriched = dict(payload)
        enriched["_cache_stats_json"] = _repo_path(path)
        stats.append(enriched)
    return stats


def _collect_refinement_summaries(run_root: Path) -> list[dict]:
    summaries: list[dict] = []
    for summary_path in sorted((run_root / "r").glob("*/adaptive-loop-summary.json")):
        try:
            adaptive_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(adaptive_payload, dict):
            continue
        loop_root = summary_path.parent
        leaderboard_paths = sorted(loop_root.glob("g*/leaderboard.json"))
        row_count = 0
        error_rows = 0
        blocked_rows = 0
        promotable_rows = 0
        parent_selectable_rows = 0
        duplicate_quality_group_count = 0
        duplicate_quality_duplicate_row_count = 0
        moonshot_status_counts: dict[str, int] = {}
        moonshot_top_deltas: list[dict[str, object]] = []
        moonshot_active_views: list[dict[str, object]] = []
        moonshot_active_sequences: list[dict[str, object]] = []
        moonshot_portfolio_available_actions: list[dict[str, object]] = []
        moonshot_portfolio_actions: list[dict[str, object]] = []
        moonshot_portfolio_rejected_actions: list[dict[str, object]] = []
        moonshot_portfolio_dependencies: list[dict[str, object]] = []
        moonshot_portfolio_execution: list[dict[str, object]] = []
        moonshot_portfolio_risks: list[dict[str, object]] = []
        leaderboard_summaries: list[dict[str, object]] = []
        for leaderboard_path in leaderboard_paths:
            try:
                leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(leaderboard, dict):
                continue
            rows = leaderboard.get("rows")
            if not isinstance(rows, list):
                rows = []
            row_count += len(rows)
            current_error_rows = sum(1 for row in rows if isinstance(row, dict) and row.get("status") == "error")
            current_promotable_rows = sum(1 for row in rows if isinstance(row, dict) and bool(row.get("promotable")))
            current_parent_selectable_rows = sum(
                1
                for row in rows
                if isinstance(row, dict)
                and bool(row.get("parent_selectable", row.get("promotable")))
            )
            current_blocked_rows = sum(
                1
                for row in rows
                if isinstance(row, dict) and str(row.get("promotion_tier", "")).startswith("blocked")
            )
            error_rows += current_error_rows
            promotable_rows += current_promotable_rows
            parent_selectable_rows += current_parent_selectable_rows
            blocked_rows += current_blocked_rows
            duplicate_quality_group_count += int(leaderboard.get("duplicate_quality_group_count", 0) or 0)
            duplicate_quality_duplicate_row_count += int(
                leaderboard.get("duplicate_quality_duplicate_row_count", 0) or 0
            )
            moonshot_summary = leaderboard.get("moonshot_summary")
            if isinstance(moonshot_summary, Mapping):
                for status, count in dict(moonshot_summary.get("status_counts", {}) or {}).items():
                    moonshot_status_counts[str(status)] = (
                        moonshot_status_counts.get(str(status), 0) + int(count or 0)
                    )
                for item in moonshot_summary.get("top_candidate_deltas", ()) or ():
                    if isinstance(item, Mapping):
                        moonshot_top_deltas.append(dict(item))
                for item in moonshot_summary.get("active_view_suggestions", ()) or ():
                    if isinstance(item, Mapping):
                        moonshot_active_views.append(dict(item))
                for item in moonshot_summary.get("active_view_sequence", ()) or ():
                    if isinstance(item, Mapping):
                        moonshot_active_sequences.append(dict(item))
                for item in moonshot_summary.get("portfolio_available_actions", ()) or ():
                    if isinstance(item, Mapping):
                        moonshot_portfolio_available_actions.append(dict(item))
                for item in moonshot_summary.get("portfolio_actions", ()) or ():
                    if isinstance(item, Mapping):
                        moonshot_portfolio_actions.append(dict(item))
                for item in moonshot_summary.get("portfolio_rejected_actions", ()) or ():
                    if isinstance(item, Mapping):
                        moonshot_portfolio_rejected_actions.append(dict(item))
                for item in moonshot_summary.get("portfolio_dependency_edges", ()) or ():
                    if isinstance(item, Mapping):
                        moonshot_portfolio_dependencies.append(dict(item))
                for item in moonshot_summary.get("portfolio_execution_plan", ()) or ():
                    if isinstance(item, Mapping):
                        moonshot_portfolio_execution.append(dict(item))
                for item in moonshot_summary.get("portfolio_risks", ()) or ():
                    if isinstance(item, Mapping):
                        moonshot_portfolio_risks.append(dict(item))
            leaderboard_summaries.append(
                {
                    "path": _repo_path(leaderboard_path),
                    "rows": len(rows),
                    "error_rows": current_error_rows,
                    "promotable_rows": current_promotable_rows,
                    "parent_selectable_rows": current_parent_selectable_rows,
                    "blocked_rows": current_blocked_rows,
                    "duplicate_quality_group_count": int(
                        leaderboard.get("duplicate_quality_group_count", 0) or 0
                    ),
                    "duplicate_quality_duplicate_row_count": int(
                        leaderboard.get("duplicate_quality_duplicate_row_count", 0) or 0
                    ),
                    "moonshot_summary": moonshot_summary if isinstance(moonshot_summary, Mapping) else {},
                }
            )
        generation_promotable_rows = _adaptive_promotable_count(adaptive_payload)
        if generation_promotable_rows > promotable_rows:
            promotable_rows = generation_promotable_rows
        generation_parent_selectable_rows = _adaptive_parent_selectable_count(adaptive_payload)
        if generation_parent_selectable_rows > parent_selectable_rows:
            parent_selectable_rows = generation_parent_selectable_rows
        stopped_reason = str(adaptive_payload.get("stopped_reason") or "")
        failure_reasons: list[str] = []
        if row_count > 0 and error_rows == row_count:
            failure_reasons.append("all_error_rows")
        if row_count > 0 and promotable_rows <= 0 and parent_selectable_rows <= 0:
            failure_reasons.append("zero_promotable_candidates")
        if stopped_reason == "no_promotable_parents" and parent_selectable_rows <= 0:
            failure_reasons.append("no_promotable_parents")
        summaries.append(
            {
                "path": _repo_path(summary_path),
                "target": loop_root.name,
                "stopped_reason": stopped_reason,
                "generation_count": int(adaptive_payload.get("generation_count", 0) or 0),
                "result_rows": row_count,
                "error_rows": error_rows,
                "blocked_rows": blocked_rows,
                "promotable_rows": promotable_rows,
                "parent_selectable_rows": parent_selectable_rows,
                "duplicate_quality_group_count": duplicate_quality_group_count,
                "duplicate_quality_duplicate_row_count": duplicate_quality_duplicate_row_count,
                "moonshot_status_counts": dict(sorted(moonshot_status_counts.items())),
                "moonshot_ran_count": int(moonshot_status_counts.get("ran", 0)),
                "moonshot_error_count": int(moonshot_status_counts.get("error", 0)),
                "moonshot_top_deltas": _top_moonshot_deltas(moonshot_top_deltas),
                "moonshot_active_views": _top_moonshot_active_views(moonshot_active_views),
                "moonshot_active_sequence": _top_moonshot_active_sequence(
                    moonshot_active_sequences
                ),
                "moonshot_portfolio_available_actions": _top_moonshot_portfolio_actions(
                    moonshot_portfolio_available_actions
                ),
                "moonshot_portfolio_actions": _top_moonshot_portfolio_actions(
                    moonshot_portfolio_actions
                ),
                "moonshot_portfolio_rejected_actions": _top_moonshot_portfolio_actions(
                    moonshot_portfolio_rejected_actions
                ),
                "moonshot_portfolio_dependencies": _top_moonshot_portfolio_dependencies(
                    moonshot_portfolio_dependencies
                ),
                "moonshot_portfolio_execution": _top_moonshot_portfolio_execution(
                    moonshot_portfolio_execution
                ),
                "moonshot_portfolio_risks": _top_moonshot_portfolio_risks(
                    moonshot_portfolio_risks
                ),
                "required": True,
                "failure_reasons": failure_reasons,
                "leaderboards": leaderboard_summaries,
            }
        )
    return summaries


def _adaptive_promotable_count(adaptive_payload: Mapping[str, object]) -> int:
    total = 0
    generations = adaptive_payload.get("generations")
    if not isinstance(generations, list):
        return total
    for generation in generations:
        if not isinstance(generation, Mapping):
            continue
        health = generation.get("parent_health")
        if not isinstance(health, Mapping):
            continue
        total += int(health.get("promotable_count", 0) or 0)
    return total


def _adaptive_parent_selectable_count(adaptive_payload: Mapping[str, object]) -> int:
    total = 0
    generations = adaptive_payload.get("generations")
    if not isinstance(generations, list):
        return total
    for generation in generations:
        if not isinstance(generation, Mapping):
            continue
        health = generation.get("parent_health")
        if not isinstance(health, Mapping):
            continue
        total += int(
            health.get(
                "parent_selectable_count",
                health.get("promotable_count", 0),
            )
            or 0
        )
    return total


def _top_moonshot_deltas(items: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = [dict(item) for item in items]
    rows.sort(
        key=lambda item: abs(float(item.get("delta", item.get("expected_metric_delta", 0.0)) or 0.0)),
        reverse=True,
    )
    return rows[:8]


def _top_moonshot_active_views(items: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = [dict(item) for item in items]
    rows.sort(
        key=lambda item: float(item.get("expected_metric_delta", item.get("score", 0.0)) or 0.0),
        reverse=True,
    )
    return rows[:8]


def _top_moonshot_active_sequence(items: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = [dict(item) for item in items]
    rows.sort(
        key=lambda item: (
            str(item.get("case_id", "")),
            str(item.get("variant_id", "")),
            int(item.get("order", 999) or 999),
        )
    )
    return rows[:8]


def _top_moonshot_portfolio_actions(items: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = [dict(item) for item in items]
    rows.sort(
        key=lambda item: float(item.get("score", 0.0) or 0.0),
        reverse=True,
    )
    return rows[:8]


def _top_moonshot_portfolio_dependencies(
    items: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    rows = [dict(item) for item in items]
    rows.sort(
        key=lambda item: (
            str(item.get("case_id", "")),
            str(item.get("variant_id", "")),
            str(item.get("before_kind", "")),
            str(item.get("after_kind", "")),
        )
    )
    return rows[:12]


def _top_moonshot_portfolio_execution(
    items: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    rows = [dict(item) for item in items]
    rows.sort(
        key=lambda item: (
            str(item.get("case_id", "")),
            str(item.get("variant_id", "")),
            int(item.get("order", 999) or 999),
            str(item.get("stage", "")),
        )
    )
    return rows[:8]


def _top_moonshot_portfolio_risks(items: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = [dict(item) for item in items]
    rows.sort(
        key=lambda item: float(item.get("risk", 0.0) or 0.0),
        reverse=True,
    )
    return rows[:8]


def _refinement_summary_lines(summaries: Sequence[dict]) -> list[str]:
    lines = [
        "| Target | Stopped | Rows | Errors | Promotable | Parent-selectable | Duplicate rows | Moonshots | Failure reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        reasons = ", ".join(str(reason) for reason in row.get("failure_reasons", ()))
        lines.append(
            "| "
            f"`{row.get('target', '')}` | "
            f"`{row.get('stopped_reason', '') or 'n/a'}` | "
            f"{int(row.get('result_rows', 0) or 0)} | "
            f"{int(row.get('error_rows', 0) or 0)} | "
            f"{int(row.get('promotable_rows', 0) or 0)} | "
            f"{int(row.get('parent_selectable_rows', 0) or 0)} | "
            f"{int(row.get('duplicate_quality_duplicate_row_count', 0) or 0)} | "
            f"{int(row.get('moonshot_ran_count', 0) or 0)} | "
            f"{reasons or 'n/a'} |"
        )
    return lines


def _moonshot_totals(summaries: Sequence[dict]) -> dict[str, object]:
    counts: dict[str, int] = {}
    deltas: list[Mapping[str, object]] = []
    active_views: list[Mapping[str, object]] = []
    active_sequence: list[Mapping[str, object]] = []
    portfolio_available_actions: list[Mapping[str, object]] = []
    portfolio_actions: list[Mapping[str, object]] = []
    portfolio_rejected_actions: list[Mapping[str, object]] = []
    portfolio_dependencies: list[Mapping[str, object]] = []
    portfolio_execution: list[Mapping[str, object]] = []
    portfolio_risks: list[Mapping[str, object]] = []
    for row in summaries:
        row_counts = row.get("moonshot_status_counts")
        if isinstance(row_counts, Mapping):
            for status, count in row_counts.items():
                counts[str(status)] = counts.get(str(status), 0) + int(count or 0)
        for item in row.get("moonshot_top_deltas", ()) or ():
            if isinstance(item, Mapping):
                deltas.append(item)
        for item in row.get("moonshot_active_views", ()) or ():
            if isinstance(item, Mapping):
                active_views.append(item)
        for item in row.get("moonshot_active_sequence", ()) or ():
            if isinstance(item, Mapping):
                active_sequence.append(item)
        for item in row.get("moonshot_portfolio_available_actions", ()) or ():
            if isinstance(item, Mapping):
                portfolio_available_actions.append(item)
        for item in row.get("moonshot_portfolio_actions", ()) or ():
            if isinstance(item, Mapping):
                portfolio_actions.append(item)
        for item in row.get("moonshot_portfolio_rejected_actions", ()) or ():
            if isinstance(item, Mapping):
                portfolio_rejected_actions.append(item)
        for item in row.get("moonshot_portfolio_dependencies", ()) or ():
            if isinstance(item, Mapping):
                portfolio_dependencies.append(item)
        for item in row.get("moonshot_portfolio_execution", ()) or ():
            if isinstance(item, Mapping):
                portfolio_execution.append(item)
        for item in row.get("moonshot_portfolio_risks", ()) or ():
            if isinstance(item, Mapping):
                portfolio_risks.append(item)
    return {
        "status_counts": dict(sorted(counts.items())),
        "ran_count": int(counts.get("ran", 0)),
        "error_count": int(counts.get("error", 0)),
        "top_candidate_deltas": _top_moonshot_deltas(deltas),
        "active_view_suggestions": _top_moonshot_active_views(active_views),
        "active_view_sequence": _top_moonshot_active_sequence(active_sequence),
        "portfolio_available_actions": _top_moonshot_portfolio_actions(
            portfolio_available_actions
        ),
        "portfolio_actions": _top_moonshot_portfolio_actions(portfolio_actions),
        "portfolio_rejected_actions": _top_moonshot_portfolio_actions(
            portfolio_rejected_actions
        ),
        "portfolio_dependency_edges": _top_moonshot_portfolio_dependencies(
            portfolio_dependencies
        ),
        "portfolio_execution_plan": _top_moonshot_portfolio_execution(
            portfolio_execution
        ),
        "portfolio_risks": _top_moonshot_portfolio_risks(portfolio_risks),
    }


def _moonshot_summary_lines(summaries: Sequence[dict]) -> list[str]:
    totals = _moonshot_totals(summaries)
    counts = totals["status_counts"]
    if not isinstance(counts, Mapping) or not counts:
        return []
    lines = [
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status, count in counts.items():
        lines.append(f"| `{status}` | {int(count or 0)} |")
    deltas = totals.get("top_candidate_deltas")
    if isinstance(deltas, Sequence) and deltas:
        lines.extend(["", "Top deltas:"])
        for item in deltas[:5]:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "- "
                f"`{item.get('experiment_id', '')}` "
                f"{item.get('metric', 'delta')}={_fmt_metric(item.get('delta'))} "
                f"for `{item.get('case_id', '')}:{item.get('variant_id', '')}`"
            )
    active_views = totals.get("active_view_suggestions")
    if isinstance(active_views, Sequence) and active_views:
        lines.extend(["", "Active-view suggestions:"])
        for item in active_views[:5]:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "- "
                f"`{item.get('view_id', '')}` "
                f"delta={_fmt_metric(item.get('expected_metric_delta'))} "
                f"for `{item.get('case_id', '')}:{item.get('variant_id', '')}`"
            )
    active_sequence = totals.get("active_view_sequence")
    if isinstance(active_sequence, Sequence) and active_sequence:
        lines.extend(["", "Active-view sequence:"])
        for item in active_sequence[:5]:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "- "
                f"{int(item.get('order', 0) or 0)}. `{item.get('view_id', '')}` "
                f"marginal={_fmt_metric(item.get('marginal_expected_metric_delta'))} "
                f"for `{item.get('case_id', '')}:{item.get('variant_id', '')}`"
            )
    portfolio_actions = totals.get("portfolio_actions")
    available_actions = totals.get("portfolio_available_actions")
    if isinstance(available_actions, Sequence) and available_actions:
        lines.extend(["", "Available moonshot evidence:"])
        for item in available_actions[:5]:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "- "
                f"`{item.get('source', '')}:{item.get('kind', '')}` "
                f"score={_fmt_metric(item.get('score'))} "
                f"for `{item.get('case_id', '')}:{item.get('variant_id', '')}`"
            )
    if isinstance(portfolio_actions, Sequence) and portfolio_actions:
        lines.extend(["", "Portfolio actions:"])
        for item in portfolio_actions[:5]:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "- "
                f"`{item.get('kind', '')}` "
                f"score={_fmt_metric(item.get('score'))} "
                f"risk={_fmt_metric(item.get('risk'))} "
                f"for `{item.get('case_id', '')}:{item.get('variant_id', '')}`"
            )
    rejected_actions = totals.get("portfolio_rejected_actions")
    if isinstance(rejected_actions, Sequence) and rejected_actions:
        lines.extend(["", "Rejected portfolio actions:"])
        for item in rejected_actions[:5]:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "- "
                f"`{item.get('source', '')}:{item.get('kind', '')}` "
                f"reason={item.get('rejection', 'lower_rank')} "
                f"for `{item.get('case_id', '')}:{item.get('variant_id', '')}`"
            )
    dependencies = totals.get("portfolio_dependency_edges")
    if isinstance(dependencies, Sequence) and dependencies:
        lines.extend(["", "Portfolio dependencies:"])
        for item in dependencies[:5]:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "- "
                f"`{item.get('before_kind', '')}` -> `{item.get('after_kind', '')}` "
                f"({item.get('type', 'dependency')}) "
                f"for `{item.get('case_id', '')}:{item.get('variant_id', '')}`"
            )
    execution = totals.get("portfolio_execution_plan")
    if isinstance(execution, Sequence) and execution:
        lines.extend(["", "Portfolio execution stages:"])
        for item in execution[:5]:
            if not isinstance(item, Mapping):
                continue
            actions = item.get("actions")
            action_count = len(actions) if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)) else 0
            lines.append(
                "- "
                f"`{item.get('stage', '')}` actions={action_count} "
                f"for `{item.get('case_id', '')}:{item.get('variant_id', '')}`"
            )
    risks = totals.get("portfolio_risks")
    if isinstance(risks, Sequence) and risks:
        lines.extend(["", "Portfolio risks:"])
        for item in risks[:5]:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "- "
                f"`{item.get('kind', '')}` level={item.get('level', 'n/a')} "
                f"risk={_fmt_metric(item.get('risk'))} "
                f"for `{item.get('case_id', '')}:{item.get('variant_id', '')}`"
            )
    return lines


def _refinement_required_failures(summaries: Sequence[dict]) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for row in summaries:
        reasons = tuple(str(reason) for reason in row.get("failure_reasons", ()) if reason)
        if not reasons or not bool(row.get("required", True)):
            continue
        failures.append(
            {
                "path": row.get("path", ""),
                "target": row.get("target", ""),
                "stopped_reason": row.get("stopped_reason", ""),
                "result_rows": int(row.get("result_rows", 0) or 0),
                "error_rows": int(row.get("error_rows", 0) or 0),
                "promotable_rows": int(row.get("promotable_rows", 0) or 0),
                "parent_selectable_rows": int(row.get("parent_selectable_rows", 0) or 0),
                "failure_reasons": list(reasons),
            }
        )
    return failures


def _cache_summary_lines(stats: Sequence[dict]) -> list[str]:
    lines = [
        "| Report | Ref hits | Ref misses | Ref writes | Candidate hits | Candidate misses | Candidate writes | Local writes | Shared writes | Sources |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in stats:
        sources = row.get("candidate_cache_sources")
        if isinstance(sources, dict):
            source_text = ", ".join(
                f"{key}:{value}" for key, value in sorted(sources.items())
            )
        else:
            source_text = ""
        no_hit_reasons = row.get("candidate_cache_no_hit_reasons")
        if isinstance(no_hit_reasons, dict) and no_hit_reasons:
            source_text = (source_text + "; " if source_text else "") + "no-hit " + ", ".join(
                f"{key}:{value}" for key, value in sorted(no_hit_reasons.items())
            )
        lines.append(
            "| "
            f"`{row.get('_cache_stats_json', '')}` | "
            f"{int(row.get('reference_cache_hits', 0) or 0)} | "
            f"{int(row.get('reference_cache_misses', 0) or 0)} | "
            f"{int(row.get('reference_cache_writes', 0) or 0)} | "
            f"{int(row.get('candidate_cache_hits', 0) or 0)} | "
            f"{int(row.get('candidate_cache_misses', 0) or 0)} | "
            f"{int(row.get('candidate_cache_writes', 0) or 0)} | "
            f"{int(row.get('candidate_cache_local_resume_writes', 0) or 0)} | "
            f"{int(row.get('candidate_cache_shared_writes', 0) or 0)} | "
            f"{source_text or 'n/a'} |"
        )
    totals = _cache_totals(stats)
    lines.append(
        "| **total** | "
        f"{totals['reference_cache_hits']} | "
        f"{totals['reference_cache_misses']} | "
        f"{totals['reference_cache_writes']} | "
        f"{totals['candidate_cache_hits']} | "
        f"{totals['candidate_cache_misses']} | "
        f"{totals['candidate_cache_writes']} | "
        f"{totals['candidate_cache_local_resume_writes']} | "
        f"{totals['candidate_cache_shared_writes']} | "
        f"{totals['candidate_cache_sources'] or 'n/a'} |"
    )
    return lines


def _cache_totals(stats: Sequence[dict]) -> dict[str, object]:
    totals = {
        "reference_cache_hits": 0,
        "reference_cache_misses": 0,
        "reference_cache_writes": 0,
        "candidate_cache_hits": 0,
        "candidate_cache_misses": 0,
        "candidate_cache_writes": 0,
        "candidate_cache_local_resume_writes": 0,
        "candidate_cache_shared_writes": 0,
    }
    sources: dict[str, int] = {}
    no_hit_reasons: dict[str, int] = {}
    for row in stats:
        for key in totals:
            totals[key] += int(row.get(key, 0) or 0)
        row_sources = row.get("candidate_cache_sources")
        if isinstance(row_sources, dict):
            for source, count in row_sources.items():
                sources[str(source)] = sources.get(str(source), 0) + int(count or 0)
        row_no_hits = row.get("candidate_cache_no_hit_reasons")
        if isinstance(row_no_hits, dict):
            for reason, count in row_no_hits.items():
                no_hit_reasons[str(reason)] = no_hit_reasons.get(str(reason), 0) + int(count or 0)
    return {
        **totals,
        "candidate_cache_sources": dict(sorted(sources.items())),
        "candidate_cache_no_hit_reasons": dict(sorted(no_hit_reasons.items())),
    }


def _cache_health_warnings(
    stats: Sequence[dict],
    refinement_summaries: Sequence[dict],
) -> list[dict[str, object]]:
    if not any(bool(row.get("candidate_cache_enabled")) for row in stats):
        return []
    duplicate_rows = sum(
        int(row.get("duplicate_quality_duplicate_row_count", 0) or 0)
        for row in refinement_summaries
    )
    if duplicate_rows <= 0:
        return []
    totals = _cache_totals(stats)
    sources = totals.get("candidate_cache_sources")
    shared_hits = (
        int(sources.get("shared_cache", 0) or 0)
        if isinstance(sources, Mapping)
        else 0
    )
    if shared_hits > 0:
        return []
    return [
        {
            "code": "candidate_cache_no_shared_hits_for_duplicates",
            "duplicate_rows": duplicate_rows,
            "candidate_cache_hits": totals["candidate_cache_hits"],
            "candidate_cache_misses": totals["candidate_cache_misses"],
            "message": (
                "candidate cache is enabled and duplicate-quality rows exist, "
                "but no shared-cache hits were recorded"
            ),
        }
    ]


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


def _failed_matrix_rows(rows: Sequence[dict]) -> list[dict[str, object]]:
    failed: list[dict[str, object]] = []
    for row in rows:
        if row.get("passed"):
            continue
        metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
        render_min, render_source = _first_numeric_metric(
            metrics,
            (
                ("render.min_view_iou", "render"),
                ("silhouette.min_view_iou", "silhouette"),
            ),
        )
        backend_min, _backend_source = _first_numeric_metric(
            metrics,
            (("backend.area_iou_min", "backend"),),
        )
        failed.append(
            {
                "suite": row.get("suite", ""),
                "case": row.get("case", row.get("case_id", row.get("shape_id", ""))),
                "shape_id": row.get("shape_id", ""),
                "mode": row.get("mode", ""),
                "status": row.get("status", ""),
                "backend_min_iou": backend_min,
                "render_min_iou": render_min,
                "render_metric_source": render_source or "n/a",
                "front_iou": _matrix_view_iou(metrics, "front"),
                "side_iou": _matrix_view_iou(metrics, "side"),
                "top_iou": _matrix_view_iou(metrics, "top"),
                "failure_reason": _matrix_failure_reason(row),
                "result_json": row.get("result_json", ""),
                "matrix_json": row.get("_matrix_json", ""),
            }
        )
    return failed


def _failed_matrix_row_lines(rows: Sequence[Mapping[str, object]]) -> list[str]:
    lines = [
        "| Suite | Case | Mode | Status | Backend min IoU | Render min IoU | Front | Side | Top | Reason | Result |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        result = str(row.get("result_json") or "")
        lines.append(
            "| "
            f"`{row.get('suite', '')}` | "
            f"`{row.get('case') or row.get('shape_id') or ''}` | "
            f"`{row.get('mode', '')}` | "
            f"`{row.get('status', '') or 'fail'}` | "
            f"{_fmt_metric(row.get('backend_min_iou'))} | "
            f"{_fmt_metric(row.get('render_min_iou'))} "
            f"({row.get('render_metric_source', 'n/a')}) | "
            f"{_fmt_metric(row.get('front_iou'))} | "
            f"{_fmt_metric(row.get('side_iou'))} | "
            f"{_fmt_metric(row.get('top_iou'))} | "
            f"{_escape_table_text(str(row.get('failure_reason') or 'n/a'))} | "
            f"{f'`{result}`' if result else 'n/a'} |"
        )
    return lines


def _matrix_view_iou(metrics: Mapping[str, object], view: str) -> float | None:
    value, _source = _first_numeric_metric(
        metrics,
        (
            (f"render.per_view.{view}.area_iou", "render"),
            (f"silhouette.per_view.{view}.area_iou", "silhouette"),
            (f"{view}_iou", "legacy"),
        ),
    )
    return value


def _matrix_failure_reason(row: Mapping[str, object]) -> str:
    for key in ("message", "details", "error"):
        value = row.get(key)
        if value:
            return str(value)
    bundle = row.get("evaluation_bundle")
    if isinstance(bundle, Mapping):
        errors = bundle.get("errors")
        if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
            return "; ".join(str(item) for item in errors if item)
        failures = bundle.get("failures")
        if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes)):
            reasons: list[str] = []
            for failure in failures:
                if not isinstance(failure, Mapping):
                    continue
                causes = failure.get("likely_causes")
                if isinstance(causes, Sequence) and not isinstance(causes, (str, bytes)):
                    reasons.extend(str(cause) for cause in causes if cause)
                elif failure.get("code"):
                    reasons.append(str(failure["code"]))
            if reasons:
                return "; ".join(reasons)
    return "matrix row failed"


def _escape_table_text(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _matrix_summary_lines(rows: Sequence[dict]) -> list[str]:
    by_mode: dict[str, dict[str, object]] = {}
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
                "sources": {},
            },
        )
        bucket["total"] = float(bucket["total"]) + 1.0
        if row.get("passed"):
            bucket["passed"] = float(bucket["passed"]) + 1.0
        _accumulate_metric(
            bucket,
            metrics,
            (
                ("render.min_view_iou", "render"),
                ("silhouette.min_view_iou", "silhouette"),
                ("backend.area_iou_min", "backend"),
            ),
            "min_iou",
        )
        _accumulate_metric(
            bucket,
            metrics,
            (
                ("render.boundary_iou_mean", "render"),
                ("silhouette.mean_boundary_iou", "silhouette"),
                ("backend.boundary_iou_mean", "backend"),
            ),
            "boundary",
        )
        _accumulate_metric(
            bucket,
            metrics,
            (
                ("render.signed_distance_loss_mean", "render"),
                ("silhouette.mean_signed_distance_loss", "silhouette"),
            ),
            "sdf",
        )

    lines = [
        "| Mode | Passed | Total | Avg min IoU | Avg Boundary IoU | Avg SDF Loss | Metric sources |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    ranked = sorted(
        by_mode.items(),
        key=lambda item: (
            float(item[1]["passed"]) / max(float(item[1]["total"]), 1.0),
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
            f"{_fmt_metric(_average(bucket, 'sdf'))} | "
            f"{_metric_sources_text(bucket)} |"
        )
    return lines


def _accumulate_metric(
    bucket: dict[str, object],
    metrics: dict,
    metric_paths: Sequence[tuple[str, str]],
    prefix: str,
) -> None:
    value, source = _first_numeric_metric(metrics, metric_paths)
    if value is None:
        return
    bucket[f"{prefix}_sum"] = float(bucket.get(f"{prefix}_sum", 0.0)) + value
    bucket[f"{prefix}_count"] = float(bucket.get(f"{prefix}_count", 0.0)) + 1.0
    sources = bucket.setdefault("sources", {})
    if isinstance(sources, dict):
        key = f"{prefix}:{source}"
        sources[key] = int(sources.get(key, 0)) + 1


def _average(
    bucket: Mapping[str, object],
    prefix: str,
    *,
    missing: float | None = None,
) -> float | None:
    count = float(bucket.get(f"{prefix}_count", 0.0) or 0.0)
    if count <= 0:
        return missing
    return float(bucket.get(f"{prefix}_sum", 0.0) or 0.0) / count


def _first_numeric_metric(
    metrics: Mapping[str, object],
    metric_paths: Sequence[tuple[str, str]],
) -> tuple[float | None, str]:
    for path, source in metric_paths:
        value = get_metric_path(metrics, path)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value), source
    return None, ""


def _metric_sources_text(bucket: Mapping[str, object]) -> str:
    sources = bucket.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        return "n/a"
    return ", ".join(f"{key}:{value}" for key, value in sorted(sources.items()))


def _sort_metric(value: float | None, *, missing: float = 0.0) -> float:
    return missing if value is None else value


def _fmt_metric(value: object) -> str:
    if value is None or value == 1e9:
        return "n/a"
    if isinstance(value, bool):
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


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
    parser.add_argument(
        "--moonshot-sidecars",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--moonshot-experiments", default=None)
    parser.add_argument("--no-synthetic-matrix", action="store_true")
    parser.add_argument("--no-lpips-novel", action="store_true")
    parser.add_argument("--no-refinement-loop", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdio_encoding()
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
