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


def bench_shape_program_build(iterations: int, progress: bool = True) -> BenchResult:
    try:
        from reconstruction.backends.shape_program import build_shape_program_from_target
        from reconstruction.types import (
            Bounds3D,
            OrthographicCameraSpec,
            ProfileBand,
            ProfileIntervalPx,
            ReconstructionTarget,
            ViewConstraint,
        )
    except Exception as exc:
        return BenchResult(
            name="shape_program_build",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    bands = tuple(
        ProfileBand(
            t=index / 63.0,
            intervals=(
                ProfileIntervalPx(
                    x0=28.0 + (index % 5) * 0.25,
                    x1=68.0 - (index % 7) * 0.2,
                    confidence=0.9,
                ),
            ),
            center_x=48.0,
            width_px=40.0,
            confidence=0.9,
            source_view="front",
        )
        for index in range(64)
    )
    mask = np.zeros((96, 96), dtype=bool)
    mask[16:80, 28:68] = True
    target = ReconstructionTarget(
        constraints=(
            ViewConstraint(
                view="front",
                mask=mask,
                camera=OrthographicCameraSpec("front", "front", resolution=(96, 96)),
            ),
            ViewConstraint(
                view="side",
                mask=mask,
                camera=OrthographicCameraSpec("side", "side", resolution=(96, 96)),
            ),
        ),
        profile_bands={"front": bands},
        bounds=Bounds3D(-1.0, 1.0, -0.75, 0.75, 0.0, 2.0),
    )
    config = {
        "root_strategy": "hybrid_profile_bounds",
        "residual_policy": "suggest_patches",
        "max_nodes": 64,
    }
    progress_handle = progress_bar(iterations, desc="shape_program", enabled=progress)
    start = _now()
    program = None
    diagnostics = {}
    for index in range(iterations):
        program, diagnostics = build_shape_program_from_target(
            target,
            config=config,
            program_id=f"bench_shape_program_{index}",
        )
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    return BenchResult(
        name="shape_program_build",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_s * 1000.0,
        meta={
            "node_count": 0 if program is None else program.node_count(),
            "residual_patch_count": 0 if program is None else program.residual_patch_count(),
            "dominant_profile_band_count": diagnostics.get("dominant_profile_band_count", 0),
            "throughput": len(bands) / per_iter_s if per_iter_s > 0 else 0.0,
            "throughput_unit": "band",
            "throughput_label": "profile bands",
        },
    )
