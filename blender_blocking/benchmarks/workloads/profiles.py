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


def bench_vertical_profile(
    iterations: int,
    image_size: int,
    num_samples: int,
    progress: bool = True,
) -> BenchResult:
    try:
        from integration.shape_matching.profile_extractor import (
            extract_vertical_profile,
        )
    except Exception as exc:
        return BenchResult(
            name="vertical_profile",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")

    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    y0 = image_size // 4
    y1 = image_size - y0
    x0 = image_size // 3
    x1 = image_size - x0
    mask[y0:y1, x0:x1] = 255

    progress_handle = progress_bar(
        iterations, desc="vertical_profile", enabled=progress
    )
    start = _now()
    for _ in range(iterations):
        extract_vertical_profile(mask, num_samples=num_samples, already_silhouette=True)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (mask.size / per_iter_s) if per_iter_s > 0 else 0.0

    return BenchResult(
        name="vertical_profile",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "image_size": image_size,
            "num_samples": num_samples,
            "throughput": throughput,
            "throughput_unit": "px",
            "throughput_label": "input pixels",
        },
    )


def bench_vertical_width_profile(
    iterations: int,
    image_size: int,
    num_samples: int,
    progress: bool = True,
) -> BenchResult:
    try:
        from geometry.dual_profile import extract_vertical_width_profile_px
    except Exception as exc:
        return BenchResult(
            name="vertical_width_profile",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")

    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    x0 = image_size // 3
    x1 = image_size - x0
    mask[:, x0:x1] = 255

    progress_handle = progress_bar(iterations, desc="vertical_width", enabled=progress)
    start = _now()
    for _ in range(iterations):
        extract_vertical_width_profile_px(
            mask,
            num_samples=num_samples,
            sample_policy="endpoints",
            fill_strategy="interp_nearest",
            smoothing_window=1,
        )
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (mask.size / per_iter_s) if per_iter_s > 0 else 0.0

    return BenchResult(
        name="vertical_width_profile",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "image_size": image_size,
            "num_samples": num_samples,
            "fill_strategy": "interp_nearest",
            "throughput": throughput,
            "throughput_unit": "px",
            "throughput_label": "input pixels",
        },
    )


def bench_profile_interpolation(
    iterations: int, num_samples: int, progress: bool = True
) -> BenchResult:
    try:
        from placement.primitive_placement import SliceAnalyzer
    except Exception as exc:
        return BenchResult(
            name="profile_interpolation",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")

    analyzer = SliceAnalyzer(
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(2.0, 2.0, 2.0),
        num_slices=2,
        vertical_profile=[(0.0, 0.2), (0.5, 0.6), (1.0, 1.0)],
    )
    z_values = np.linspace(0.0, 1.0, num_samples)

    total_steps = iterations * len(z_values)
    progress_handle = progress_bar(total_steps, desc="profile_interp", enabled=progress)
    start = _now()
    for _ in range(iterations):
        for z in z_values:
            analyzer._interpolate_profile(float(z))
            progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (num_samples / per_iter_s) if per_iter_s > 0 else 0.0

    return BenchResult(
        name="profile_interpolation",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "num_samples": num_samples,
            "throughput": throughput,
            "throughput_unit": "samples",
            "throughput_label": "interpolations",
        },
    )


def bench_combine_profiles(
    iterations: int,
    num_profiles: int,
    num_samples: int,
    method: str,
    progress: bool = True,
) -> BenchResult:
    try:
        from integration.shape_matching.mesh_profile_extractor import combine_profiles
    except Exception as exc:
        return BenchResult(
            name="combine_profiles",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")

    heights = np.linspace(0.0, 1.0, num_samples)
    profiles = []
    for i in range(num_profiles):
        radii = np.linspace(0.5, 1.0 + i * 0.01, num_samples)
        profiles.append([(float(h), float(r)) for h, r in zip(heights, radii)])

    progress_handle = progress_bar(
        iterations, desc="combine_profiles", enabled=progress
    )
    start = _now()
    for _ in range(iterations):
        combine_profiles(profiles, method=method)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (num_profiles * num_samples) / per_iter_s if per_iter_s > 0 else 0.0

    return BenchResult(
        name="combine_profiles",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "num_profiles": num_profiles,
            "num_samples": num_samples,
            "method": method,
            "throughput": throughput,
            "throughput_unit": "samples",
            "throughput_label": "profile samples",
        },
    )


def bench_slice_metrics(
    iterations: int, num_profiles: int, progress: bool = True
) -> BenchResult:
    try:
        from shape_matching.slice_shape_matcher import SliceBasedShapeMatcher
    except Exception as exc:
        return BenchResult(
            name="slice_metrics",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")

    rng = np.random.default_rng(2024)
    features1 = rng.random((num_profiles, 4))
    features2 = rng.random((num_profiles, 4))
    matcher = SliceBasedShapeMatcher(num_slices=2)

    steps_per_iter = 4
    progress_handle = progress_bar(
        iterations * steps_per_iter, desc="slice_metrics", enabled=progress
    )
    start = _now()
    for _ in range(iterations):
        norm1 = matcher._normalize_features(features1)
        progress_handle.update(1)
        norm2 = matcher._normalize_features(features2)
        progress_handle.update(1)
        matcher._cosine_similarity(norm1, norm2)
        progress_handle.update(1)
        matcher._correlation(features1[:, 0], features2[:, 0])
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (num_profiles / per_iter_s) if per_iter_s > 0 else 0.0

    return BenchResult(
        name="slice_metrics",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "num_profiles": num_profiles,
            "throughput": throughput,
            "throughput_unit": "profiles",
            "throughput_label": "feature rows",
        },
    )
