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


def bench_canonicalize(
    iterations: int, output_size: int, progress: bool = True
) -> BenchResult:
    try:
        from validation.silhouette_iou import canonicalize_mask
    except Exception as exc:
        return BenchResult(
            name="canonicalize_mask",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    mask = _make_rect_silhouette(96, 128, ratio=0.35)
    progress_handle = progress_bar(iterations, desc="canonicalize", enabled=progress)
    start = _now()
    for _ in range(iterations):
        canonicalize_mask(mask, output_size=output_size, padding_frac=0.1)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (mask.size / per_iter_s) if per_iter_s > 0 else 0.0
    return BenchResult(
        name="canonicalize_mask",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "output_size": output_size,
            "throughput": throughput,
            "throughput_unit": "px",
            "throughput_label": "input pixels",
        },
    )


def bench_compare_silhouettes(
    iterations: int, output_size: int, progress: bool = True
) -> BenchResult:
    try:
        from integration.shape_matching.shape_matcher import compare_silhouettes
    except Exception as exc:
        return BenchResult(
            name="compare_silhouettes",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    image = np.full((96, 96), 255, dtype=np.uint8)
    image[24:72, 36:60] = 0
    progress_handle = progress_bar(iterations, desc="compare", enabled=progress)
    start = _now()
    for _ in range(iterations):
        compare_silhouettes(image, image, output_size=output_size)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = ((image.size * 2) / per_iter_s) if per_iter_s > 0 else 0.0
    return BenchResult(
        name="compare_silhouettes",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "output_size": output_size,
            "throughput": throughput,
            "throughput_unit": "px",
            "throughput_label": "image pixels",
        },
    )


def bench_extract_silhouette(iterations: int, progress: bool = True) -> BenchResult:
    try:
        from geometry.silhouette import extract_binary_silhouette
    except Exception as exc:
        return BenchResult(
            name="extract_binary_silhouette",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    image = np.zeros((96, 96, 4), dtype=np.uint8)
    image[:, :, :3] = 255
    image[16:80, 32:64, 3] = 255

    progress_handle = progress_bar(iterations, desc="extract", enabled=progress)
    start = _now()
    for _ in range(iterations):
        extract_binary_silhouette(image, prefer_alpha=True)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (
        (image.shape[0] * image.shape[1] / per_iter_s) if per_iter_s > 0 else 0.0
    )
    return BenchResult(
        name="extract_binary_silhouette",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "image_shape": list(image.shape),
            "throughput": throughput,
            "throughput_unit": "px",
            "throughput_label": "rgba pixels",
        },
    )


def bench_silhouette_pipeline(iterations: int, progress: bool = True) -> BenchResult:
    try:
        from geometry.silhouette_pipeline import (
            build_uncertain_mask,
            canonicalize_silhouette,
            extract_silhouette_candidates,
        )
    except Exception as exc:
        return BenchResult(
            name="silhouette_pipeline",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    image = np.full((128, 128, 4), 255, dtype=np.uint8)
    image[:, :, 3] = 255
    image[28:110, 38:91, :3] = 0
    progress_handle = progress_bar(iterations, desc="silhouette_pipeline", enabled=progress)
    start = _now()
    candidate_count = 0
    for _ in range(iterations):
        candidates = extract_silhouette_candidates(image)
        uncertain = build_uncertain_mask(image)
        canonicalize_silhouette(uncertain.hard_mask, output_size=256)
        candidate_count = len(candidates)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    return BenchResult(
        name="silhouette_pipeline",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_s * 1000.0,
        meta={
            "image_shape": list(image.shape),
            "candidate_count": candidate_count,
            "throughput": image.shape[0] * image.shape[1] / per_iter_s
            if per_iter_s > 0
            else 0.0,
            "throughput_unit": "px",
            "throughput_label": "pipeline pixels",
        },
    )
