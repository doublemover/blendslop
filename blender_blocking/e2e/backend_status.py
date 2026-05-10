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
from blender_blocking.e2e.console import _print_candidate_table, _print_kv_table, _print_section, _status_icon
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
    _print_kv_table(
        (
            ("status", f"{_status_icon(passed)} {status}"),
            ("backend", summary_source.get("backend_name") if summary_source else None),
            (
                "candidate",
                summary_source.get("candidate_id") if summary_source else None,
            ),
            ("mesh", summary_source.get("mesh_path") if summary_source else None),
            (
                "primitives",
                summary_source.get("primitive_path") if summary_source else None,
            ),
            ("volume", summary_source.get("volume_path") if summary_source else None),
            (
                "warnings",
                len(summary_source.get("warnings", [])) if summary_source else None,
            ),
            (
                "errors",
                len(summary_source.get("errors", [])) if summary_source else None,
            ),
        )
    )

    if data.get("candidates"):
        print("\nCandidates:")
        rows = []
        for candidate in data["candidates"]:
            metric_result = candidate.get("metric_result", {}) or {}
            mesh_path = candidate.get("mesh_path") or candidate.get(
                "artifacts", {}
            ).get("mesh_obj", "")
            rows.append(
                {
                    "candidate": candidate.get("candidate_id", ""),
                    "backend": candidate.get("backend_name", ""),
                    "status": candidate.get("status", ""),
                    "score": metric_result.get("scalar_score", ""),
                    "warnings": len(candidate.get("warnings", [])),
                    "errors": len(candidate.get("errors", [])),
                    "artifact": mesh_path,
                }
            )
        _print_candidate_table(rows)

    metrics = summary_source.get("metric_result", {}) if summary_source else {}
    extras = metrics.get("extras", {}) if isinstance(metrics, dict) else {}
    if metrics:
        _print_kv_table(
            (
                ("area_iou_mean", metrics.get("area_iou_mean")),
                ("area_iou_min", metrics.get("area_iou_min")),
                ("topology_score", metrics.get("topology_score")),
                ("editability_score", metrics.get("editability_score")),
                ("complexity_penalty", metrics.get("complexity_penalty")),
                ("elapsed_s", metrics.get("elapsed_s")),
                ("occupied_voxels", extras.get("occupied_voxels")),
                ("primitive_count", extras.get("primitive_count")),
                ("coverage_score", extras.get("coverage_score")),
                ("loss_total", extras.get("loss_total")),
            ),
            title="\nMetrics:",
        )
    return passed
