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


def bench_target_builder(iterations: int, progress: bool = True) -> BenchResult:
    try:
        from config import BlockingConfig
        from reconstruction.target_builder import build_target_from_images
    except Exception as exc:
        return BenchResult(
            name="target_builder",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    cfg = BlockingConfig()
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    image[20:82, 30:68, :] = 0
    views = {"front": image, "side": image}
    progress_handle = progress_bar(iterations, desc="target_builder", enabled=progress)
    start = _now()
    for _ in range(iterations):
        build_target_from_images(
            views,
            config=cfg,
            bounds_minmax=((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
            profile_samples=32,
        )
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    return BenchResult(
        name="target_builder",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_s * 1000.0,
        meta={
            "views": 2,
            "profile_samples": 32,
            "throughput": image.size * 2 / per_iter_s if per_iter_s > 0 else 0.0,
            "throughput_unit": "px",
            "throughput_label": "target pixels",
        },
    )
