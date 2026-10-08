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

from .cases.registry import BENCHMARK_CASES
from .contracts import BenchResult, BenchmarkCase
from .results import _print_result
from .workloads.geometry import bench_geometry_metrics
from .workloads.profiles import (
    bench_combine_profiles,
    bench_profile_interpolation,
    bench_slice_metrics,
    bench_vertical_profile,
    bench_vertical_width_profile,
)
from .workloads.resfit import bench_resfit_full, bench_resfit_optimize, bench_resfit_residual
from .workloads.shape_program import bench_shape_program_build
from .workloads.silhouettes import (
    bench_canonicalize,
    bench_compare_silhouettes,
    bench_extract_silhouette,
    bench_silhouette_pipeline,
)
from .workloads.target_builder import bench_target_builder
from .workloads.volume import bench_surface_voxels, bench_visual_hull, bench_volume_surface_new


def _default_benches(args: argparse.Namespace) -> List[str]:
    if args.all:
        return [
            "visual_hull",
            "surface_voxels",
            "vertical_profile",
            "vertical_width_profile",
            "profile_interpolation",
            "combine_profiles",
            "slice_metrics",
            "resfit_residual",
            "resfit_optimize",
            "canonicalize",
            "compare",
            "extract",
            "silhouette_pipeline",
            "volume_surface",
            "target_builder",
            "geometry_metrics",
            "shape_program_build",
        ]
    return [b.strip() for b in args.bench.split(",") if b.strip()]


def _args_for_case(args: argparse.Namespace, case: BenchmarkCase) -> argparse.Namespace:
    case_args = argparse.Namespace(**vars(args))
    for key, value in case.overrides.items():
        setattr(case_args, key, value)
    return case_args


def _run_one_benchmark(bench: str, args: argparse.Namespace) -> BenchResult:
    if bench == "visual_hull":
        return bench_visual_hull(
            resolution=args.resolution,
            num_views=args.num_views,
            include_top=args.include_top,
            repeat=args.repeat,
            progress=args.progress,
        )
    if bench == "surface_voxels":
        return bench_surface_voxels(
            iterations=args.iterations,
            resolution=args.resolution,
            fill_ratio=args.fill_ratio,
            progress=args.progress,
        )
    if bench == "vertical_profile":
        return bench_vertical_profile(
            iterations=args.iterations,
            image_size=args.profile_size,
            num_samples=args.profile_samples,
            progress=args.progress,
        )
    if bench == "vertical_width_profile":
        return bench_vertical_width_profile(
            iterations=args.iterations,
            image_size=args.profile_size,
            num_samples=args.profile_samples,
            progress=args.progress,
        )
    if bench == "profile_interpolation":
        return bench_profile_interpolation(
            iterations=args.iterations,
            num_samples=args.profile_samples,
            progress=args.progress,
        )
    if bench == "combine_profiles":
        return bench_combine_profiles(
            iterations=args.iterations,
            num_profiles=args.combine_profiles,
            num_samples=args.profile_samples,
            method=args.combine_method,
            progress=args.progress,
        )
    if bench == "slice_metrics":
        return bench_slice_metrics(
            iterations=args.iterations,
            num_profiles=args.slice_profiles,
            progress=args.progress,
        )
    if bench == "resfit_residual":
        return bench_resfit_residual(
            iterations=args.iterations,
            num_points=args.resfit_points,
            num_primitives=args.resfit_primitives,
            progress=args.progress,
        )
    if bench == "resfit_full":
        return bench_resfit_full(
            iterations=args.resfit_full_iterations,
            num_points=args.resfit_points,
            num_primitives=args.resfit_primitives,
            steps=args.resfit_full_steps,
            progress=args.progress,
        )
    if bench == "resfit_optimize":
        return bench_resfit_optimize(
            iterations=args.resfit_opt_iterations,
            num_points=args.resfit_points,
            num_primitives=args.resfit_primitives,
            steps=args.resfit_opt_steps,
            progress=args.progress,
        )
    if bench == "canonicalize":
        return bench_canonicalize(
            iterations=args.iterations,
            output_size=args.output_size,
            progress=args.progress,
        )
    if bench == "compare":
        return bench_compare_silhouettes(
            iterations=args.iterations,
            output_size=args.output_size,
            progress=args.progress,
        )
    if bench == "extract":
        return bench_extract_silhouette(
            iterations=args.iterations,
            progress=args.progress,
        )
    if bench == "silhouette_pipeline":
        return bench_silhouette_pipeline(
            iterations=args.iterations,
            progress=args.progress,
        )
    if bench == "volume_surface":
        return bench_volume_surface_new(
            iterations=args.iterations,
            resolution=args.resolution,
            fill_ratio=args.fill_ratio,
            progress=args.progress,
        )
    if bench == "target_builder":
        return bench_target_builder(
            iterations=args.iterations,
            progress=args.progress,
        )
    if bench == "geometry_metrics":
        return bench_geometry_metrics(
            iterations=args.iterations,
            resolution=args.resolution,
            progress=args.progress,
        )
    if bench == "shape_program_build":
        return bench_shape_program_build(
            iterations=args.iterations,
            progress=args.progress,
        )
    return BenchResult(
        name=bench,
        iterations=0,
        elapsed_s=0.0,
        per_iter_ms=0.0,
        status="skip",
        skip_reason="Unknown benchmark",
    )


def _run_case(case_name: str, args: argparse.Namespace) -> List[BenchResult]:
    case = BENCHMARK_CASES[case_name]
    case_args = _args_for_case(args, case)
    benches = _default_benches(args) if args.case_benches else list(case.benches)
    results = []
    print(f"\n=== Benchmark case: {case.name} ===")
    print(case.description)
    for bench in benches:
        result = _run_one_benchmark(bench, case_args)
        result.case = case.name
        results.append(result)
        _print_result(result)
    return results


def _print_case_registry() -> None:
    payload = {
        name: {
            "description": case.description,
            "benches": list(case.benches),
            "overrides": dict(case.overrides),
        }
        for name, case in sorted(BENCHMARK_CASES.items())
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
