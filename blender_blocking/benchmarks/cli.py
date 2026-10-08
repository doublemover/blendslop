from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

BLENDER_BLOCKING_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BLENDER_BLOCKING_ROOT.parent
if str(BLENDER_BLOCKING_ROOT) not in sys.path:
    sys.path.insert(0, str(BLENDER_BLOCKING_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.progress import progress_bar

from .budgets import _load_budget_report
from .cases.registry import BENCHMARK_CASES
from .contracts import BenchResult
from .results import _print_result, _results_payload, _write_json
from .runner import _default_benches, _print_case_registry, _run_case, _run_one_benchmark


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Performance micro-benchmarks")
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(BENCHMARK_CASES),
        help="Named benchmark case. Can be repeated. Overrides --bench unless --bench is also provided with --case-benches.",
    )
    parser.add_argument(
        "--case-benches",
        action="store_true",
        help="Run --bench selections inside each named --case instead of the case registry benches.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print benchmark case registry as JSON and exit.",
    )
    parser.add_argument(
        "--bench",
        default="all",
        help=(
            "Comma-separated list: visual_hull,surface_voxels,vertical_profile,"
            "vertical_width_profile,profile_interpolation,combine_profiles,slice_metrics,"
            "resfit_residual,resfit_full,resfit_optimize,canonicalize,compare,extract,"
            "silhouette_pipeline,volume_surface,target_builder,geometry_metrics,"
            "shape_program_build"
        ),
    )
    parser.add_argument("--all", action="store_true", help="Run all benches")
    parser.add_argument("--repeat", type=int, default=1, help="Repeats for visual hull")
    parser.add_argument(
        "--iterations", type=int, default=200, help="Iterations for micro-benches"
    )
    parser.add_argument(
        "--resolution", type=int, default=32, help="Voxel resolution for visual hull"
    )
    parser.add_argument(
        "--num-views", type=int, default=8, help="Lateral view count for visual hull"
    )
    parser.add_argument(
        "--include-top", action="store_true", help="Include top view in visual hull"
    )
    parser.add_argument(
        "--fill-ratio",
        type=float,
        default=0.25,
        help="Fill ratio for surface voxel benchmark",
    )
    parser.add_argument(
        "--output-size", type=int, default=256, help="Canonical size for masks"
    )
    parser.add_argument(
        "--profile-size",
        type=int,
        default=128,
        help="Image size for vertical profile benchmark",
    )
    parser.add_argument(
        "--profile-samples",
        type=int,
        default=100,
        help="Samples for vertical profile and profile-combine benches",
    )
    parser.add_argument(
        "--combine-profiles",
        type=int,
        default=12,
        help="Number of profiles for combine_profiles benchmark",
    )
    parser.add_argument(
        "--combine-method",
        type=str,
        default="median",
        help="Method for combine_profiles benchmark",
    )
    parser.add_argument(
        "--slice-profiles",
        type=int,
        default=64,
        help="Profile count for slice metrics benchmark",
    )
    parser.add_argument(
        "--resfit-points",
        type=int,
        default=1000,
        help="Point count for resfit residual benchmark",
    )
    parser.add_argument(
        "--resfit-primitives",
        type=int,
        default=5,
        help="Primitive count for resfit residual benchmark",
    )
    parser.add_argument(
        "--resfit-full-steps",
        type=int,
        default=5,
        help="Optimization steps for resfit_full benchmark",
    )
    parser.add_argument(
        "--resfit-full-iterations",
        type=int,
        default=1,
        help="Iteration count for resfit_full benchmark",
    )
    parser.add_argument(
        "--resfit-opt-steps",
        type=int,
        default=5,
        help="Optimization steps for resfit_optimize benchmark",
    )
    parser.add_argument(
        "--resfit-opt-iterations",
        type=int,
        default=1,
        help="Iteration count for resfit_optimize benchmark",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Show progress bars (requires tqdm)",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable progress bars",
    )
    parser.set_defaults(progress=True)
    parser.add_argument("--json", type=str, default=None, help="Write results to JSON")
    parser.add_argument(
        "--budget-json",
        type=str,
        default=None,
        help="Quality/perf budget JSON to evaluate against the result payload.",
    )
    parser.add_argument(
        "--baseline-json",
        type=str,
        default=None,
        help="Previous benchmark JSON used for regression comparison.",
    )
    parser.add_argument(
        "--budget-report-json",
        type=str,
        default=None,
        help="Write standalone budget comparison report JSON.",
    )
    parser.add_argument(
        "--fail-on-budget",
        action="store_true",
        help="Return exit code 1 when absolute budget thresholds fail.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Return exit code 1 when baseline comparisons fail.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.list_cases:
        _print_case_registry()
        return 0

    results: List[BenchResult] = []
    if args.case:
        for case_name in args.case:
            results.extend(_run_case(case_name, args))
    else:
        for bench in _default_benches(args):
            result = _run_one_benchmark(bench, args)
            results.append(result)
            _print_result(result)

    json_path = Path(args.json) if args.json else None
    base_payload = _results_payload(results)
    budget_report = _load_budget_report(
        current_payload=base_payload,
        current_path=json_path,
        budget_path=args.budget_json,
        baseline_path=args.baseline_json,
        report_path=args.budget_report_json,
    )

    if args.json:
        _write_json(Path(args.json), results, budget_report=budget_report)
        print(f"Wrote JSON results to: {args.json}")

    if budget_report:
        passed = bool(budget_report.get("passed", False))
        comparison_passed = bool(budget_report.get("comparison_passed", passed))
        threshold_passed = bool(budget_report.get("threshold_passed", passed))
        print(
            "Budget result: "
            f"{'PASS' if passed else 'FAIL'} "
            f"(thresholds={'PASS' if threshold_passed else 'FAIL'}, "
            f"comparisons={'PASS' if comparison_passed else 'FAIL'})"
        )
        if (args.fail_on_budget and not threshold_passed) or (
            args.fail_on_regression and not comparison_passed
        ):
            return 1

    return 0
