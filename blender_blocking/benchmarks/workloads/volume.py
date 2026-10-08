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


def bench_visual_hull(
    resolution: int,
    num_views: int,
    include_top: bool,
    repeat: int,
    progress: bool = True,
) -> BenchResult:
    try:
        from integration.multi_view.visual_hull import MultiViewVisualHull
    except Exception as exc:
        return BenchResult(
            name="visual_hull_reconstruct",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if num_views <= 0:
        raise ValueError("num_views must be >= 1")

    total = 0.0
    voxel_count = 0
    steps_per_iter = num_views + (1 if include_top else 0) + 1
    total_steps = repeat * steps_per_iter
    progress_handle = progress_bar(
        total_steps,
        desc="visual_hull",
        enabled=progress,
        mininterval=0.0,
        miniters=1,
    )
    for _ in range(repeat):
        hull = MultiViewVisualHull(
            resolution=resolution,
            bounds_min=np.array([-1.0, -1.0, -1.0]),
            bounds_max=np.array([1.0, 1.0, 1.0]),
        )

        # Lateral views
        for i in range(num_views):
            angle = float(i) * (360.0 / num_views)
            silhouette = _make_rect_silhouette(64, 64, ratio=0.4)
            hull.add_view_from_silhouette(silhouette, angle=angle, view_type="lateral")
            progress_handle.update(1)

        if include_top:
            top = _make_circle_silhouette(64, 64, ratio=0.35)
            hull.add_view_from_silhouette(top, angle=0.0, view_type="top")
            progress_handle.update(1)

        start = _now()
        voxels = hull.reconstruct(verbose=False)
        total += _now() - start
        voxel_count = int(np.sum(voxels))
        progress_handle.update(1)
    progress_handle.close()

    per_iter_s = total / max(repeat, 1)
    per_iter_ms = per_iter_s * 1000.0
    views_total = num_views + (1 if include_top else 0)
    voxels_per_iter = resolution**3 * views_total
    throughput = (voxels_per_iter / per_iter_s) if per_iter_s > 0 else 0.0
    return BenchResult(
        name="visual_hull_reconstruct",
        iterations=repeat,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "resolution": resolution,
            "num_views": num_views,
            "include_top": include_top,
            "occupied_voxels": voxel_count,
            "throughput": throughput,
            "throughput_unit": "vox",
            "throughput_label": "voxel-projections",
        },
    )


def bench_surface_voxels(
    iterations: int,
    resolution: int,
    fill_ratio: float,
    progress: bool = True,
) -> BenchResult:
    try:
        from integration.multi_view.visual_hull import MultiViewVisualHull
    except Exception as exc:
        return BenchResult(
            name="surface_voxel_extract",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")

    rng = np.random.default_rng(1337)
    grid = rng.random((resolution, resolution, resolution)) < fill_ratio

    hull = MultiViewVisualHull(resolution=resolution)
    progress_handle = progress_bar(iterations, desc="surface_voxels", enabled=progress)
    start = _now()
    surface = None
    for _ in range(iterations):
        surface = hull._extract_surface_voxels(grid)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (grid.size / per_iter_s) if per_iter_s > 0 else 0.0

    surface_count = int(surface.sum()) if surface is not None else 0
    return BenchResult(
        name="surface_voxel_extract",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "resolution": resolution,
            "fill_ratio": fill_ratio,
            "surface_voxels": surface_count,
            "throughput": throughput,
            "throughput_unit": "vox",
            "throughput_label": "voxel checks",
        },
    )


def bench_volume_surface_new(
    iterations: int,
    resolution: int,
    fill_ratio: float,
    progress: bool = True,
) -> BenchResult:
    try:
        from volume import extract_surface_voxels
    except Exception as exc:
        return BenchResult(
            name="volume_surface_vectorized",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    rng = np.random.default_rng(424242)
    grid = rng.random((resolution, resolution, resolution)) < fill_ratio
    progress_handle = progress_bar(iterations, desc="volume_surface", enabled=progress)
    start = _now()
    surface = None
    for _ in range(iterations):
        surface = extract_surface_voxels(grid, prefer_scipy=True)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    return BenchResult(
        name="volume_surface_vectorized",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_s * 1000.0,
        meta={
            "resolution": resolution,
            "fill_ratio": fill_ratio,
            "surface_voxels": int(surface.sum()) if surface is not None else 0,
            "throughput": grid.size / per_iter_s if per_iter_s > 0 else 0.0,
            "throughput_unit": "vox",
            "throughput_label": "surface checks",
        },
    )
