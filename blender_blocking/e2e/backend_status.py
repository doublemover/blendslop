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
from blender_blocking.e2e.console import (
    _display_path,
    _format_metric,
    _print_section,
    _status_icon,
    _console_print,
)
from blender_blocking.e2e.payloads import _e2e_payload_with_evaluation


def _backend_status_ok(status: object) -> bool:
    return str(status) in BACKEND_STATUS_OK


def _candidate_status_payload(result: object) -> Tuple[str, Dict[str, Any]]:
    if result is None:
        return "missing", {}
    if isinstance(result, Mapping):
        data = dict(result)
        status = str(data.get("status", "unstructured"))
        return status, data
    if hasattr(result, "selected") and hasattr(result, "candidates"):
        data = result.to_dict() if hasattr(result, "to_dict") else {}
        data = _e2e_payload_with_evaluation(data, result)
        selected = getattr(result, "selected", None)
        status = getattr(selected, "status", "missing") if selected else "missing"
        return str(status), data
    if hasattr(result, "status"):
        data = result.to_dict() if hasattr(result, "to_dict") else {}
        data = _e2e_payload_with_evaluation(data, result)
        return str(getattr(result, "status")), data
    return "unstructured", {"type": type(result).__name__, "repr": repr(result)}

def _print_backend_summary(result: object) -> bool:
    status, data = _candidate_status_payload(result)
    selected = data.get("selected") if isinstance(data, dict) else None
    if selected:
        summary_source = selected
    else:
        summary_source = data
    passed = _backend_status_ok(status)

    _print_section("Backend Result")
    details = [
        f"status={_status_icon(passed)} {status}",
    ]
    if summary_source:
        if summary_source.get("backend_name"):
            details.append(f"backend={summary_source.get('backend_name')}")
        if summary_source.get("candidate_id"):
            details.append(f"candidate={summary_source.get('candidate_id')}")
    _console_print("  " + " | ".join(details), color="green" if passed else "red")

    artifacts = []
    if summary_source:
        for label, key in (
            ("mesh", "mesh_path"),
            ("primitives", "primitive_path"),
            ("volume", "volume_path"),
        ):
            value = summary_source.get(key)
            if value:
                artifacts.append(f"{label}={_display_path(value)}")
    if artifacts:
        _console_print("  artifacts " + " | ".join(artifacts), color="dim")

    warnings = len(summary_source.get("warnings", [])) if summary_source else 0
    errors = len(summary_source.get("errors", [])) if summary_source else 0

    if data.get("candidates"):
        counts: dict[str, int] = {}
        ids: list[str] = []
        for candidate in data["candidates"]:
            candidate_status = str(candidate.get("status", "unknown"))
            counts[candidate_status] = counts.get(candidate_status, 0) + 1
            if candidate.get("candidate_id"):
                ids.append(str(candidate.get("candidate_id")))
        count_text = " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        shown_ids = ",".join(ids[:4])
        if len(ids) > 4:
            shown_ids += f",+{len(ids) - 4}"
        candidate_text = f"  candidates total={len(data['candidates'])}"
        if count_text:
            candidate_text += f" | {count_text}"
        if shown_ids:
            candidate_text += f" | ids={shown_ids}"
        _console_print(candidate_text)

    metrics = summary_source.get("metric_result", {}) if summary_source else {}
    extras = metrics.get("extras", {}) if isinstance(metrics, dict) else {}
    if metrics:
        metric_parts = []
        for label, value, precision in (
            ("area_mean", metrics.get("area_iou_mean"), 4),
            ("area_min", metrics.get("area_iou_min"), 4),
            ("topology", metrics.get("topology_score"), 4),
            ("editability", metrics.get("editability_score"), 4),
            ("complexity", metrics.get("complexity_penalty"), 4),
            ("elapsed", metrics.get("elapsed_s"), 2),
            ("voxels", extras.get("occupied_voxels"), 0),
            ("primitives", extras.get("primitive_count"), 0),
            ("coverage", extras.get("coverage_score"), 4),
            ("loss", extras.get("loss_total"), 4),
        ):
            if value is not None:
                metric_parts.append(f"{label}={_format_metric(value, precision=precision)}")
        health = f"warnings={warnings} errors={errors}"
        if metric_parts:
            _console_print("  metrics " + " | ".join(metric_parts) + f" | {health}")
        else:
            _console_print(f"  {health}")
    elif warnings or errors:
        _console_print(
            f"  warnings={warnings} errors={errors}",
            color="yellow" if warnings and not errors else "red",
        )
    return passed
