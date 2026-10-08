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

from ..contracts import BenchResult
from .common import (
    _make_circle_silhouette,
    _make_rect_silhouette,
    _now,
    _sample_cone_points,
    _sample_cylinder_points,
    _sample_rotated_cube_points,
    _sample_sphere_points,
)


def bench_resfit_residual(
    iterations: int,
    num_points: int,
    num_primitives: int,
    progress: bool = True,
) -> BenchResult:
    try:
        from placement.resfitting import ResidualFitter
        from primitives.superfrustum import SuperFrustum
    except Exception as exc:
        return BenchResult(
            name="resfit_residual",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")

    rng = np.random.default_rng(4242)
    target_points = rng.normal(size=(num_points, 3))
    primitives = []
    for _ in range(num_primitives):
        position = tuple(rng.normal(size=3))
        orientation = (float(rng.uniform(0.0, 3.14)), float(rng.uniform(0.0, 3.14)))
        radius_bottom = float(rng.uniform(0.5, 2.0))
        radius_top = float(rng.uniform(0.2, 1.5))
        height = float(rng.uniform(0.5, 3.0))
        primitives.append(
            SuperFrustum(
                position=position,
                orientation=orientation,
                radius_bottom=radius_bottom,
                radius_top=radius_top,
                height=height,
            )
        )

    fitter = ResidualFitter()
    progress_handle = progress_bar(iterations, desc="resfit_residual", enabled=progress)
    start = _now()
    for _ in range(iterations):
        fitter.compute_residual_error(primitives, target_points)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (num_points * num_primitives) / per_iter_s if per_iter_s > 0 else 0.0

    return BenchResult(
        name="resfit_residual",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "num_points": num_points,
            "num_primitives": num_primitives,
            "throughput": throughput,
            "throughput_unit": "evals",
            "throughput_label": "sdf evals",
        },
    )


def bench_resfit_full(
    iterations: int,
    num_points: int,
    num_primitives: int,
    steps: int,
    progress: bool = True,
) -> BenchResult:
    try:
        from placement.resfitting import ResidualFitter
        from primitives.superfrustum import SuperFrustum
    except Exception as exc:
        return BenchResult(
            name="resfit_full",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")
    if num_points <= 0:
        raise ValueError("num_points must be >= 1")
    if num_primitives <= 0:
        raise ValueError("num_primitives must be >= 1")
    if steps <= 0:
        raise ValueError("steps must be >= 1")

    rng = np.random.default_rng(2026)
    shapes = [
        (
            "cylinder",
            _sample_cylinder_points(1.5, 3.0, num_points, rng),
        ),
        (
            "cone",
            _sample_cone_points(2.0, 0.5, 3.0, num_points, rng),
        ),
        (
            "rotated_cube",
            _sample_rotated_cube_points(1.0, num_points, rng),
        ),
        (
            "sphere",
            _sample_sphere_points(1.25, num_points, rng),
        ),
    ]

    steps_per_shape = steps + 1
    total_steps = iterations * len(shapes) * steps_per_shape
    progress_handle = progress_bar(total_steps, desc="resfit_full", enabled=progress)

    start = _now()
    for _ in range(iterations):
        for _, target_points in shapes:
            primitives = []
            for _ in range(num_primitives):
                position = tuple(rng.normal(scale=0.2, size=3))
                orientation = (
                    float(rng.uniform(0.0, 3.14)),
                    float(rng.uniform(0.0, 3.14)),
                )
                radius_bottom = float(rng.uniform(0.7, 1.8))
                radius_top = float(rng.uniform(0.4, 1.4))
                height = float(rng.uniform(1.0, 3.5))
                primitives.append(
                    SuperFrustum(
                        position=position,
                        orientation=orientation,
                        radius_bottom=radius_bottom,
                        radius_top=radius_top,
                        height=height,
                    )
                )

            fitter = ResidualFitter(
                learning_rate=0.01,
                optimization_steps=steps,
            )
            fitter.optimize_primitives(
                primitives,
                target_points,
                steps=steps,
                log_progress=progress,
                progress_callback=progress_handle.update,
            )
            fitter.compute_residual_error(primitives, target_points)
            progress_handle.update(1)
    progress_handle.close()

    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    step_units = iterations * len(shapes) * steps
    throughput = (step_units / total) if total > 0 else 0.0

    return BenchResult(
        name="resfit_full",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "num_points": num_points,
            "num_primitives": num_primitives,
            "steps": steps,
            "shape_count": len(shapes),
            "throughput": throughput,
            "throughput_unit": "steps",
            "throughput_label": "opt steps",
        },
    )


def bench_resfit_optimize(
    iterations: int,
    num_points: int,
    num_primitives: int,
    steps: int,
    progress: bool = True,
) -> BenchResult:
    try:
        from placement.resfitting import ResidualFitter
        from primitives.superfrustum import SuperFrustum
    except Exception as exc:
        return BenchResult(
            name="resfit_optimize",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")
    if num_points <= 0:
        raise ValueError("num_points must be >= 1")
    if num_primitives <= 0:
        raise ValueError("num_primitives must be >= 1")
    if steps <= 0:
        raise ValueError("steps must be >= 1")

    rng = np.random.default_rng(9001)
    target_points = rng.normal(size=(num_points, 3))

    total_steps = iterations * steps
    progress_handle = progress_bar(
        total_steps, desc="resfit_optimize", enabled=progress
    )
    start = _now()
    for _ in range(iterations):
        primitives = []
        for _ in range(num_primitives):
            position = tuple(rng.normal(scale=0.2, size=3))
            orientation = (
                float(rng.uniform(0.0, 3.14)),
                float(rng.uniform(0.0, 3.14)),
            )
            radius_bottom = float(rng.uniform(0.7, 1.8))
            radius_top = float(rng.uniform(0.4, 1.4))
            height = float(rng.uniform(1.0, 3.5))
            primitives.append(
                SuperFrustum(
                    position=position,
                    orientation=orientation,
                    radius_bottom=radius_bottom,
                    radius_top=radius_top,
                    height=height,
                )
            )

        fitter = ResidualFitter(learning_rate=0.01, optimization_steps=steps)
        fitter.optimize_primitives(
            primitives,
            target_points,
            steps=steps,
            log_progress=progress,
            progress_callback=progress_handle.update,
        )
    progress_handle.close()

    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (iterations * steps / total) if total > 0 else 0.0

    return BenchResult(
        name="resfit_optimize",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "num_points": num_points,
            "num_primitives": num_primitives,
            "steps": steps,
            "throughput": throughput,
            "throughput_unit": "steps",
            "throughput_label": "opt steps",
        },
    )
