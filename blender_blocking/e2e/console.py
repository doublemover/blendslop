# ruff: noqa: E402,F401,F403
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from blender_blocking.verify_setup import configure_dependency_paths
configure_dependency_paths()

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    BLENDER_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

from blender_blocking.config import BlockingConfig
from blender_blocking.config import CandidateConfig as ConfigCandidateConfig
from blender_blocking.config import RenderConfig
from blender_blocking.evaluation.cost_model import CostRecorder
from blender_blocking.evaluation.silhouette_eval import (
    SilhouetteGateConfig,
    evaluate_silhouette_pair,
    missing_silhouette_view,
    summarize_silhouette_views,
)
from blender_blocking.integration.blender_ops.render_utils import (
    parse_orbit_view_degrees,
    render_orthogonal_views,
)
from blender_blocking.integration.image_processing.image_loader import load_image
from blender_blocking.utils.generation_context import GenerationContext
from blender_blocking.utils.progress import progress_bar
from blender_blocking.validation.silhouette_iou import canonicalize_mask, mask_from_image_array
from blender_blocking.e2e.constants import *


def _status_icon(ok: bool) -> str:
    return "✓" if ok else "✗"

def _print_rule(title: str = "", width: int = 72) -> None:
    if title:
        label = f" {title} "
        fill = max(0, width - len(label))
        left = fill // 2
        right = fill - left
        print("=" * left + label + "=" * right)
    else:
        print("=" * width)

def _print_section(title: str) -> None:
    print()
    _print_rule(title, width=72)

def _print_kv_table(rows: Iterable[Tuple[str, object]], *, title: str = "") -> None:
    rows = tuple((str(key), value) for key, value in rows if value is not None)
    if not rows:
        return
    if title:
        print(title)
    key_width = max(len(key) for key, _ in rows)
    for key, value in rows:
        print(f"  {key:<{key_width}} : {value}")

def _print_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    if not rows:
        return
    widths = {
        column: max(
            len(column),
            *[len(str(row.get(column, ""))) for row in rows],
        )
        for column in columns
    }
    header = "  " + "  ".join(f"{column:<{widths[column]}}" for column in columns)
    print(header)
    print("  " + "  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print(
            "  "
            + "  ".join(
                f"{str(row.get(column, '')):<{widths[column]}}" for column in columns
            )
        )

def _print_result_table(rows: Sequence[Mapping[str, object]]) -> None:
    _print_table(rows, ("view", "iou", "threshold", "status", "intersection", "union"))

def _print_candidate_table(rows: Sequence[Mapping[str, object]]) -> None:
    _print_table(
        rows,
        ("candidate", "backend", "status", "score", "warnings", "errors", "artifact"),
    )

def _format_metric(value: object, *, precision: int) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isinf(numeric):
        return "inf"
    return f"{numeric:.{precision}f}"
