"""First-class silhouette evaluation and required-view gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from metrics.silhouette import silhouette_metric_result
except Exception:  # pragma: no cover - package import fallback
    from blender_blocking.metrics.silhouette import silhouette_metric_result


DEFAULT_REQUIRED_VIEWS = ("front", "side", "top")


@dataclass(frozen=True)
class SilhouetteGateConfig:
    min_area_iou: float = 0.7
    per_view_min_area_iou: Mapping[str, float] = field(default_factory=dict)
    min_boundary_iou: float | None = None
    max_signed_distance_loss: float | None = None
    required_views: tuple[str, ...] = DEFAULT_REQUIRED_VIEWS
    boundary_dilation_radius: int = 2

    def threshold_for_view(self, view: str) -> float:
        return float(self.per_view_min_area_iou.get(view, self.min_area_iou))


def evaluate_silhouette_pair(
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    view: str,
    config: SilhouetteGateConfig | None = None,
    required: bool = True,
) -> dict[str, object]:
    """Evaluate one canonical reference/candidate silhouette pair."""
    config = config or SilhouetteGateConfig()
    threshold = config.threshold_for_view(view)
    result = silhouette_metric_result(
        reference_mask,
        candidate_mask,
        view=view,
        required=required,
        min_area_iou=threshold,
        min_boundary_iou=config.min_boundary_iou,
        max_signed_distance_loss=config.max_signed_distance_loss,
        boundary_dilation_radius=config.boundary_dilation_radius,
    )
    payload = result.to_dict()
    payload["iou"] = result.area_iou
    payload["threshold"] = threshold
    payload["precision"] = _precision(
        result.intersection,
        result.render_area,
        result.ref_area,
    )
    payload["recall"] = _recall(result.intersection, result.ref_area, result.render_area)
    payload["area_ratio"] = _area_ratio(result.render_area, result.ref_area)
    payload["centroid_delta_px"] = _centroid_delta(reference_mask, candidate_mask)
    payload["pixel_difference"] = float(
        np.abs(
            np.asarray(reference_mask, dtype=float)
            - np.asarray(candidate_mask, dtype=float)
        ).mean()
    )
    payload["thresholds"] = {
        "min_area_iou": threshold,
        "min_boundary_iou": config.min_boundary_iou,
        "max_signed_distance_loss": config.max_signed_distance_loss,
    }
    return payload


def missing_silhouette_view(
    view: str,
    *,
    reason: str,
    config: SilhouetteGateConfig | None = None,
    required: bool = True,
) -> dict[str, object]:
    """Create a failed metric payload for a required view that could not run."""
    config = config or SilhouetteGateConfig()
    threshold = config.threshold_for_view(view)
    return {
        "view": view,
        "iou": 0.0,
        "area_iou": 0.0,
        "boundary_iou": None,
        "soft_iou": None,
        "signed_distance_loss": None,
        "intersection": 0,
        "union": 0,
        "ref_area": 0,
        "render_area": 0,
        "required": required,
        "passed": not required,
        "pass": not required,
        "reason": reason,
        "warnings": [reason],
        "threshold": threshold,
        "precision": None,
        "recall": None,
        "area_ratio": None,
        "centroid_delta_px": None,
        "pixel_difference": 1.0,
        "thresholds": {
            "min_area_iou": threshold,
            "min_boundary_iou": config.min_boundary_iou,
            "max_signed_distance_loss": config.max_signed_distance_loss,
        },
    }


def summarize_silhouette_views(
    views: Mapping[str, Mapping[str, Any]],
    *,
    config: SilhouetteGateConfig | None = None,
) -> dict[str, object]:
    """Summarize per-view metrics without letting mean IoU hide failures."""
    config = config or SilhouetteGateConfig()
    required_views = tuple(config.required_views)
    completed_required: list[float] = []
    completed_all: list[float] = []
    boundary_values: list[float] = []
    sdf_values: list[float] = []
    failed_required_views: list[str] = []
    missing_required_views: list[str] = []
    missing_required_metric_count = 0

    for view in required_views:
        payload = views.get(view)
        if not isinstance(payload, Mapping):
            missing_required_views.append(view)
            failed_required_views.append(view)
            continue
        if not bool(payload.get("required", True)):
            continue
        area = _float_or_none(payload.get("area_iou", payload.get("iou")))
        if area is None:
            missing_required_metric_count += 1
            failed_required_views.append(view)
        else:
            completed_required.append(area)
        boundary = _float_or_none(payload.get("boundary_iou"))
        if boundary is None:
            missing_required_metric_count += 1
        else:
            boundary_values.append(boundary)
        sdf = _float_or_none(payload.get("signed_distance_loss"))
        if sdf is None:
            missing_required_metric_count += 1
        else:
            sdf_values.append(sdf)
        if not bool(payload.get("passed", payload.get("pass", False))):
            failed_required_views.append(view)
        if str(payload.get("reason", "")).startswith("missing"):
            missing_required_views.append(view)

    for payload in views.values():
        if not isinstance(payload, Mapping):
            continue
        area = _float_or_none(payload.get("area_iou", payload.get("iou")))
        if area is not None:
            completed_all.append(area)

    failed_required_views = list(dict.fromkeys(failed_required_views))
    missing_required_views = list(dict.fromkeys(missing_required_views))
    average_iou = float(sum(completed_all) / len(completed_all)) if completed_all else 0.0
    min_view_iou = min(completed_required) if completed_required else 0.0
    mean_boundary = (
        float(sum(boundary_values) / len(boundary_values)) if boundary_values else None
    )
    min_boundary = min(boundary_values) if boundary_values else None
    mean_sdf = float(sum(sdf_values) / len(sdf_values)) if sdf_values else None
    average_threshold_passed = average_iou >= float(config.min_area_iou)
    required_views_passed = not failed_required_views and not missing_required_views
    passed = bool(average_threshold_passed and required_views_passed)
    return {
        "passed": passed,
        "average_threshold_passed": average_threshold_passed,
        "required_views_passed": required_views_passed,
        "average_iou": average_iou,
        "min_view_iou": float(min_view_iou),
        "mean_boundary_iou": mean_boundary,
        "min_boundary_iou": min_boundary,
        "mean_signed_distance_loss": mean_sdf,
        "required_views": list(required_views),
        "required_view_count": len(required_views),
        "compared_view_count": len(completed_all),
        "failed_required_view_count": len(failed_required_views),
        "missing_required_view_count": len(missing_required_views),
        "missing_required_metric_count": missing_required_metric_count,
        "failed_required_views": failed_required_views,
        "missing_required_views": missing_required_views,
        "min_area_iou_threshold": float(config.min_area_iou),
        "per_view_min_area_iou": dict(config.per_view_min_area_iou),
        "min_boundary_iou_threshold": config.min_boundary_iou,
        "max_signed_distance_loss_threshold": config.max_signed_distance_loss,
    }


def _precision(intersection: int, candidate_area: int, ref_area: int) -> float:
    if candidate_area:
        return float(intersection / candidate_area)
    return 1.0 if ref_area == 0 else 0.0


def _recall(intersection: int, ref_area: int, candidate_area: int) -> float:
    if ref_area:
        return float(intersection / ref_area)
    return 1.0 if candidate_area == 0 else 0.0


def _area_ratio(candidate_area: int, ref_area: int) -> float | None:
    if ref_area == 0:
        return None
    return float(candidate_area / ref_area)


def _centroid_delta(
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> list[float] | None:
    ref = _centroid(reference_mask)
    cand = _centroid(candidate_mask)
    if ref is None or cand is None:
        return None
    return [float(cand[0] - ref[0]), float(cand[1] - ref[1])]


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(np.asarray(mask).astype(bool, copy=False))
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
