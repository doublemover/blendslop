"""Metric, status, and ambiguity signal readers for adaptive refinement."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


_DIRECT_METRIC_ALIASES = {
    "area_iou_min": "silhouette.min_view_iou",
    "area_iou_mean": "silhouette.average_iou",
    "average_iou": "silhouette.average_iou",
    "boundary_iou_mean": "silhouette.mean_boundary_iou",
    "signed_distance_loss_mean": "silhouette.mean_signed_distance_loss",
    "topology_score": "topology.score",
    "topology_penalty": "topology.penalty",
    "editability_score": "editability.editable_reconstruction_index",
    "complexity_penalty": "editability.complexity_penalty",
    "geometry_fscore_tau": "geometry.fscore_tau",
    "geometry_volumetric_iou": "geometry.volumetric_iou",
    "geometry_chamfer_l2": "geometry.chamfer_l2",
    "geometry_surface_coverage": "geometry.surface_coverage",
    "appearance_uv_valid": "appearance.uv_valid",
    "appearance_pbr_channel_coverage_ratio": "appearance.pbr_channel_coverage_ratio",
    "appearance_texture_only_detail_score": "appearance.attribution_texture_only_detail_score",
    "appearance_geometry_detail_score": "appearance.attribution_geometry_detail_score",
    "appearance_texture_memory_mb": "appearance.texture_memory_mb",
}


def _metric_index(bundle: Any) -> Mapping[str, float]:
    if hasattr(bundle, "metric_index"):
        return {
            key: _float(getattr(metric, "value", None))
            for key, metric in bundle.metric_index().items()
        }
    if isinstance(bundle, Mapping):
        metrics: dict[str, float] = {}
        direct = bundle.get("metrics")
        if isinstance(direct, Mapping):
            for key, value in direct.items():
                metric_name = str(key)
                numeric = _float(value)
                metrics[metric_name] = numeric
                alias = _DIRECT_METRIC_ALIASES.get(metric_name)
                if alias:
                    metrics.setdefault(alias, numeric)
        for group in bundle.get("metric_groups", ()) or ():
            if not isinstance(group, Mapping):
                continue
            for metric in group.get("metrics", ()) or ():
                if isinstance(metric, Mapping):
                    metrics[str(metric.get("name", ""))] = _float(metric.get("value"))
        return metrics
    return {}


def _failure_codes(bundle: Any) -> tuple[str, ...]:
    failures = getattr(bundle, "failures", None)
    if failures is None and isinstance(bundle, Mapping):
        failures = bundle.get("failures", ())
    codes = []
    for failure in failures or ():
        if hasattr(failure, "code"):
            codes.append(str(failure.code))
        elif isinstance(failure, Mapping):
            codes.append(str(failure.get("code", "")))
        else:
            codes.append(str(failure))
    return tuple(code for code in codes if code)


def _status(bundle: Any) -> str:
    if hasattr(bundle, "status"):
        return str(bundle.status)
    if isinstance(bundle, Mapping):
        return str(bundle.get("status", ""))
    return ""


def _metric(metrics: Mapping[str, float], name: str, default: float = 0.0) -> float:
    return _float(metrics.get(name, default), default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default


def ambiguity_signal(
    metrics: Mapping[str, float],
    failures: Sequence[str],
) -> bool:
    ambiguity = _metric(metrics, "geometry.ambiguity_gap")
    fscore = _metric(metrics, "geometry.fscore_tau", default=1.0)
    min_iou = _metric(metrics, "silhouette.min_view_iou")
    return (
        ambiguity > 0.1
        or fscore < 0.5
        or min_iou < 0.45
        or any(
            "ambiguous" in failure or failure.startswith("geometry_")
            for failure in failures
        )
    )
