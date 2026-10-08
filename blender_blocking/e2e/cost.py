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
from blender_blocking.metrics.values import optional_float
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
from blender_blocking.e2e.payloads import _evaluation_outputs_from_payload


def _backend_cost_report_from_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    selected = payload.get("selected")
    source = selected if isinstance(selected, Mapping) else payload
    metric_result = source.get("metric_result") if isinstance(source, Mapping) else None
    extras = metric_result.get("extras", {}) if isinstance(metric_result, Mapping) else {}
    if isinstance(extras, Mapping):
        explicit = extras.get("cost_report") or extras.get("cost")
        if isinstance(explicit, Mapping):
            return dict(explicit)
    nested = payload.get("backend_result")
    if isinstance(nested, Mapping):
        return _backend_cost_report_from_payload(nested)
    return {}

def _cost_payload(
    validation_cost: Mapping[str, Any],
    backend_payload: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    backend_cost = (
        _backend_cost_report_from_payload(backend_payload)
        if backend_payload is not None
        else {}
    )
    payload: Dict[str, Any] = {
        "schema_version": "e2e_cost_report_v2",
        "backend_is_nested_in_validation": True,
        "validation": dict(validation_cost),
    }
    if backend_cost:
        payload["backend"] = backend_cost
    # The measured validation workflow already contains the backend. A missing
    # outer measurement cannot be reconstructed by adding its nested children.
    payload["combined_total_wall_ms"] = optional_float(validation_cost.get("total_wall_ms"))
    return payload

def _cost_gate_report(
    cost_payload: Mapping[str, Any],
    *,
    max_wall_ms: Optional[float] = None,
    max_backend_wall_ms: Optional[float] = None,
) -> Dict[str, Any]:
    failures: list[str] = []
    validation = cost_payload.get("validation")
    backend = cost_payload.get("backend")
    validation_wall_ms = optional_float(validation.get("total_wall_ms")) if isinstance(validation, Mapping) else None
    backend_wall_ms = optional_float(backend.get("total_wall_ms")) if isinstance(backend, Mapping) else None
    combined = optional_float(cost_payload.get("combined_total_wall_ms", validation_wall_ms))
    for name, value, limit in (("combined_total_wall_ms", combined, max_wall_ms),
                               ("backend_total_wall_ms", backend_wall_ms, max_backend_wall_ms)):
        if limit is None:
            continue
        if value is None:
            failures.append(f"{name} unavailable; cannot verify {float(limit):.3f} limit")
        elif value > float(limit):
            failures.append(f"{name} {value:.3f} exceeds {float(limit):.3f}")
    return {
        "schema_version": "e2e_cost_gate_v1",
        "passed": not failures,
        "failures": failures,
        "thresholds": {
            "max_wall_ms": max_wall_ms,
            "max_backend_wall_ms": max_backend_wall_ms,
        },
        "metrics": {
            "validation_total_wall_ms": validation_wall_ms,
            "backend_total_wall_ms": backend_wall_ms,
            "combined_total_wall_ms": combined,
        },
    }

def _matrix_cost_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cost_rows: list[Dict[str, Any]] = []
    failed_gates = 0
    for row in rows:
        cost = row.get("cost_report", {})
        cost = cost if isinstance(cost, Mapping) else {}
        gate = row.get("cost_gate")
        validation = cost.get("validation")
        backend = cost.get("backend")
        validation_wall_ms = optional_float(validation.get("total_wall_ms")) if isinstance(validation, Mapping) else None
        backend_wall_ms = optional_float(backend.get("total_wall_ms")) if isinstance(backend, Mapping) else None
        combined_wall_ms = optional_float(cost.get("combined_total_wall_ms", validation_wall_ms))
        gate_passed = bool(gate.get("passed", False)) if isinstance(gate, Mapping) else None
        if gate_passed is False:
            failed_gates += 1
        cost_rows.append({
            "case": row.get("case", ""), "name": row.get("name", ""),
            "mode": row.get("mode", ""), "status": row.get("status", ""),
            "passed": bool(row.get("passed", False)), "gate_passed": gate_passed,
            "validation_total_wall_ms": validation_wall_ms,
            "backend_total_wall_ms": backend_wall_ms,
            "combined_total_wall_ms": combined_wall_ms,
        })
    def complete_values(key):
        values = [row[key] for row in cost_rows]
        return values if values and all(value is not None for value in values) else None
    validation_values = complete_values("validation_total_wall_ms")
    backend_values = complete_values("backend_total_wall_ms")
    combined_values = complete_values("combined_total_wall_ms")
    return {
        "schema_version": "e2e_matrix_cost_report_v2", "count": len(cost_rows),
        "passed": failed_gates == 0, "failed_gate_count": failed_gates,
        "missing_cost_count": sum(row["combined_total_wall_ms"] is None for row in cost_rows),
        "total_validation_wall_ms": sum(validation_values) if validation_values is not None else None,
        "total_backend_wall_ms": sum(backend_values) if backend_values is not None else None,
        "total_combined_wall_ms": sum(combined_values) if combined_values is not None else None,
        "mean_combined_wall_ms": sum(combined_values) / len(combined_values) if combined_values is not None else None,
        "max_combined_wall_ms": max(combined_values) if combined_values is not None else None,
        "max_backend_wall_ms": max(backend_values) if backend_values is not None else None,
        "rows": cost_rows,
    }
