# ruff: noqa: E402,F401,F403
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
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


REPO_ROOT = Path(__file__).resolve().parents[2]


_ANSI_COLORS = {
    "blue": "34",
    "cyan": "36",
    "green": "32",
    "magenta": "35",
    "red": "31",
    "yellow": "33",
    "dim": "2",
}


def _color_enabled(stream: object | None = None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def _style(
    text: str,
    color: str | None = None,
    *,
    bold: bool = False,
    stream: object | None = None,
) -> str:
    if not _color_enabled(stream):
        return text
    codes: list[str] = []
    if bold:
        codes.append("1")
    if color:
        code = _ANSI_COLORS.get(color)
        if code:
            codes.append(code)
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def _console_print(text: str = "", *, color: str | None = None, bold: bool = False) -> None:
    print(_style(text, color, bold=bold), flush=True)


def _stream_supports_unicode() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or ""
    return encoding.lower().replace("-", "") in {"utf8", "utf8sig", "utf16"}


def _status_icon(ok: bool) -> str:
    if _stream_supports_unicode():
        return "✓" if ok else "✗"
    return "OK" if ok else "FAIL"


def _print_rule(title: str = "", width: int = 72) -> None:
    if title:
        label = f" {title} "
        fill = max(0, width - len(label))
        left = fill // 2
        right = fill - left
        _console_print("=" * left + label + "=" * right, color="magenta", bold=True)
    else:
        _console_print("=" * width, color="magenta", bold=True)

def _print_section(title: str) -> None:
    _console_print()
    _print_rule(title, width=72)

def _display_path(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    try:
        path = Path(text)
    except (TypeError, ValueError):
        return text
    try:
        resolved = path.resolve(strict=False)
        return resolved.relative_to(REPO_ROOT).as_posix()
    except (OSError, ValueError):
        if path.is_absolute():
            return path.name
        return path.as_posix()

def _display_value(value: object) -> str:
    if isinstance(value, Path):
        return _display_path(value)
    if isinstance(value, str):
        if re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith(("/", "\\")):
            return _display_path(value)
    return str(value)

def _artifact_line(label: str, path: object) -> None:
    display = _display_path(path)
    if display:
        _console_print(f"{label}: {_style(display, 'dim')}")

def _render_summary(label: str, paths: Mapping[str, object]) -> None:
    items = [(view, path) for view, path in paths.items() if path]
    if not items:
        return
    views = " ".join(view for view, _path in items)
    parents = {Path(str(path)).parent for _view, path in items}
    suffix = ""
    if len(parents) == 1:
        suffix = f" -> {_style(_display_path(next(iter(parents))) + '/', 'dim')}"
    _console_print(f"{label}: {views}{suffix}", color="cyan")

def _print_kv_table(rows: Iterable[Tuple[str, object]], *, title: str = "") -> None:
    rows = tuple((str(key), value) for key, value in rows if value is not None)
    if not rows:
        return
    if title:
        _console_print(title)
    key_width = max(len(key) for key, _ in rows)
    for key, value in rows:
        _console_print(f"  {key:<{key_width}} : {_display_value(value)}")

def _print_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    if not rows:
        return
    widths = {
        column: max(
            len(column),
            *[len(_display_value(row.get(column, ""))) for row in rows],
        )
        for column in columns
    }
    header = "  " + "  ".join(f"{column:<{widths[column]}}" for column in columns)
    _console_print(header, color="dim", bold=True)
    _console_print("  " + "  ".join("-" * widths[column] for column in columns), color="dim")
    for row in rows:
        _console_print(
            "  "
            + "  ".join(
                f"{_display_value(row.get(column, '')):<{widths[column]}}" for column in columns
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
