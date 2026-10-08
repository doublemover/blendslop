from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from blender_blocking.utils.optional_deps import dependency_report, probe_dependency
except Exception:  # pragma: no cover - script-style imports
    from utils.optional_deps import dependency_report, probe_dependency


def _record_retopology_policy(
    *,
    mesh_metrics: dict[str, Any],
    per_view_metrics: Mapping[str, Any],
    editability_score: float,
    config: Mapping[str, Any],
    warnings: list[str],
) -> Any:
    topology = mesh_metrics.get("topology")
    if not isinstance(topology, Mapping):
        return None
    try:
        from metrics.retopology_policy import retopology_decision_from_metrics

        decision = retopology_decision_from_metrics(
            topology,
            editability_score=editability_score,
            min_required_iou=_min_area_iou(per_view_metrics),
            signed_distance_loss=_mean_signed_distance_loss(per_view_metrics),
            config=config,
        )
        mesh_metrics["retopology_policy"] = decision.to_dict()
        if not decision.accepted_for_editing:
            warnings.append(f"retopology policy: {decision.reason}")
        return decision
    except Exception as exc:
        mesh_metrics["retopology_policy"] = {
            "status": "failed",
            "message": str(exc),
            "error_type": type(exc).__name__,
        }
        warnings.append(f"retopology policy failed: {exc}")
        return None

def _mesh_metadata(mesh_result: Any) -> dict[str, Any]:
    return {
        "available": bool(mesh_result.available),
        "status": mesh_result.status,
        "method": mesh_result.method,
        "requested_method": getattr(mesh_result, "requested_method", mesh_result.method),
        "vertex_count": int(len(mesh_result.vertices)),
        "face_count": int(len(mesh_result.faces)),
        "has_faces": bool(len(mesh_result.faces)),
        "has_normals": mesh_result.normals is not None,
    }

def _openvdb_required(config: Mapping[str, Any]) -> bool:
    return bool(
        config.get("require_openvdb")
        or config.get("openvdb_required")
        or config.get("fail_on_openvdb_skip")
        or config.get("export_openvdb_required")
    )

def _retopology_required(config: Mapping[str, Any]) -> bool:
    return bool(
        config.get("retopology_required")
        or config.get("fail_on_retopology_required")
        or config.get("require_editable_mesh")
    )

def _min_area_iou(per_view_metrics: Mapping[str, Any]) -> float | None:
    values: list[float] = []
    for payload in per_view_metrics.values():
        if not isinstance(payload, Mapping):
            continue
        parsed = _optional_float_value(payload.get("area_iou", payload.get("iou")))
        if parsed is not None:
            values.append(parsed)
    return min(values) if values else None

def _mean_signed_distance_loss(per_view_metrics: Mapping[str, Any]) -> float | None:
    values: list[float] = []
    for payload in per_view_metrics.values():
        if not isinstance(payload, Mapping):
            continue
        parsed = _optional_float_value(payload.get("signed_distance_loss"))
        if parsed is not None:
            values.append(parsed)
    return float(sum(values) / len(values)) if values else None

def _optional_float_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
