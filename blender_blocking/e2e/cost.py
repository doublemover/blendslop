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
        "schema_version": "e2e_cost_report_v1",
        "validation": dict(validation_cost),
    }
    if backend_cost:
        payload["backend"] = backend_cost
        payload["combined_total_wall_ms"] = float(
            validation_cost.get("total_wall_ms", 0.0) or 0.0
        ) + float(backend_cost.get("total_wall_ms", 0.0) or 0.0)
    else:
        payload["combined_total_wall_ms"] = float(
            validation_cost.get("total_wall_ms", 0.0) or 0.0
        )
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
    validation_wall_ms = (
        float(validation.get("total_wall_ms", 0.0) or 0.0)
        if isinstance(validation, Mapping)
        else 0.0
    )
    backend_wall_ms = (
        float(backend.get("total_wall_ms", 0.0) or 0.0)
        if isinstance(backend, Mapping)
        else 0.0
    )
    combined = float(cost_payload.get("combined_total_wall_ms", validation_wall_ms) or 0.0)
    if max_wall_ms is not None and combined > float(max_wall_ms):
        failures.append(
            f"combined_total_wall_ms {combined:.3f} exceeds {float(max_wall_ms):.3f}"
        )
    if max_backend_wall_ms is not None and backend_wall_ms > float(max_backend_wall_ms):
        failures.append(
            f"backend_total_wall_ms {backend_wall_ms:.3f} exceeds {float(max_backend_wall_ms):.3f}"
        )
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
    combined_total = 0.0
    backend_total = 0.0
    validation_total = 0.0
    max_combined = 0.0
    max_backend = 0.0
    failed_gates = 0
    for row in rows:
        cost = row.get("cost_report")
        if not isinstance(cost, Mapping):
            continue
        gate = row.get("cost_gate")
        validation = cost.get("validation")
        backend = cost.get("backend")
        validation_wall_ms = (
            float(validation.get("total_wall_ms", 0.0) or 0.0)
            if isinstance(validation, Mapping)
            else 0.0
        )
        backend_wall_ms = (
            float(backend.get("total_wall_ms", 0.0) or 0.0)
            if isinstance(backend, Mapping)
            else 0.0
        )
        combined_wall_ms = float(
            cost.get("combined_total_wall_ms", validation_wall_ms + backend_wall_ms)
            or 0.0
        )
        combined_total += combined_wall_ms
        backend_total += backend_wall_ms
        validation_total += validation_wall_ms
        max_combined = max(max_combined, combined_wall_ms)
        max_backend = max(max_backend, backend_wall_ms)
        gate_passed = None
        if isinstance(gate, Mapping):
            gate_passed = bool(gate.get("passed", False))
            if not gate_passed:
                failed_gates += 1
        cost_rows.append(
            {
                "case": row.get("case", ""),
                "name": row.get("name", ""),
                "mode": row.get("mode", ""),
                "status": row.get("status", ""),
                "passed": bool(row.get("passed", False)),
                "gate_passed": gate_passed,
                "validation_total_wall_ms": validation_wall_ms,
                "backend_total_wall_ms": backend_wall_ms,
                "combined_total_wall_ms": combined_wall_ms,
            }
        )
    count = len(cost_rows)
    return {
        "schema_version": "e2e_matrix_cost_report_v1",
        "count": count,
        "passed": failed_gates == 0,
        "failed_gate_count": failed_gates,
        "total_validation_wall_ms": validation_total,
        "total_backend_wall_ms": backend_total,
        "total_combined_wall_ms": combined_total,
        "mean_combined_wall_ms": combined_total / count if count else 0.0,
        "max_combined_wall_ms": max_combined,
        "max_backend_wall_ms": max_backend,
        "rows": cost_rows,
    }
