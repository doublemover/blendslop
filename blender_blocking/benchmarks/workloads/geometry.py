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


def bench_geometry_metrics(
    iterations: int,
    resolution: int,
    progress: bool = True,
) -> BenchResult:
    try:
        from evaluation.geometry import (
            surface_distance_report,
            volumetric_iou,
        )
    except Exception as exc:
        return BenchResult(
            name="geometry_metrics",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    rng = np.random.default_rng(90210)
    points = _sample_sphere_points(1.0, max(256, resolution * resolution), rng)
    candidate = points + rng.normal(0.0, 0.0025, size=points.shape)
    grid_ref = rng.random((resolution, resolution, resolution)) > 0.65
    grid_cand = np.logical_or(
        grid_ref,
        rng.random((resolution, resolution, resolution)) > 0.995,
    )
    progress_handle = progress_bar(iterations, desc="geometry_metrics", enabled=progress)
    start = _now()
    report = None
    volume_iou = 0.0
    for _ in range(iterations):
        report = surface_distance_report(points, candidate, tolerance=0.01)
        volume_iou = volumetric_iou(grid_ref, grid_cand)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    return BenchResult(
        name="geometry_metrics",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_s * 1000.0,
        meta={
            "point_count": int(points.shape[0]),
            "resolution": resolution,
            "chamfer_l2": None if report is None else report.chamfer_l2,
            "fscore_tau": None if report is None else report.fscore_tau,
            "volumetric_iou": volume_iou,
            "throughput": (points.shape[0] + grid_ref.size) / per_iter_s if per_iter_s > 0 else 0.0,
            "throughput_unit": "sample",
            "throughput_label": "geometry samples",
        },
    )
