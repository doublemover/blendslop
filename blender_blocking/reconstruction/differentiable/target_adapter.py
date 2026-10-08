from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence

import numpy as np

try:
    from utils.optional_deps import optional_policy_decision, probe_dependency
except Exception:  # pragma: no cover
    from ...utils.optional_deps import optional_policy_decision, probe_dependency

try:
    from primitives.analytic_primitives import (
        AnisotropicGaussianPrimitive,
        EllipsoidPrimitive,
        SuperquadricPrimitive,
    )
    from primitives.primitive_protocol import MeshData
    from primitives.soft_silhouette import (
        OrthographicCamera,
        render_projected_soft_silhouette,
        soft_mask_metrics,
    )
    from primitives.superfrustum import SuperFrustum
    from reconstruction.artifacts import write_json
except ImportError:  # pragma: no cover - package import path.
    from ...primitives.analytic_primitives import (
        AnisotropicGaussianPrimitive,
        EllipsoidPrimitive,
        SuperquadricPrimitive,
    )
    from ...primitives.primitive_protocol import MeshData
    from ...primitives.soft_silhouette import (
        OrthographicCamera,
        render_projected_soft_silhouette,
        soft_mask_metrics,
    )
    from ...primitives.superfrustum import SuperFrustum
    from ...reconstruction.artifacts import write_json
from .contracts import RenderablePrimitive


def _collect_target_view_signal_weights(
    target: object,
) -> tuple[dict[str, float], dict[str, dict[str, object]], tuple[str, ...]]:
    """Collect per-view optional weights from available target signals."""
    constraints = tuple(getattr(target, "constraints", ()))
    extras = getattr(target, "extras", {})
    if not isinstance(extras, Mapping):
        extras = {}
    weights: dict[str, float] = {}
    details: dict[str, dict[str, object]] = {}
    warnings: list[str] = []

    extra_view_diagnostics = extras.get("view_diagnostics")
    if not isinstance(extra_view_diagnostics, Mapping):
        extra_view_diagnostics = {}

    for constraint in constraints:
        view = str(getattr(constraint, "view", ""))
        if not view:
            continue
        values: list[float] = []
        source_count = 0
        source_values: list[float] = []
        source_details: dict[str, object] = {
            "sources": [],
            "available": False,
        }

        uncertainty = getattr(constraint, "uncertainty", None)
        if uncertainty is not None:
            raw_confidence = np.asarray(getattr(uncertainty, "confidence", ()))
            if getattr(constraint,'mask',None) is not None:
                from reconstruction.visibility import valid_evidence
                valid=valid_evidence(constraint)
                if raw_confidence.shape==valid.shape:raw_confidence=raw_confidence[valid]
            if raw_confidence.size:
                mean_conf = float(np.mean(raw_confidence))
                if math.isfinite(mean_conf):
                    values.append(float(np.clip(mean_conf, 0.0, 1.0)))
                    source_count += 1
                    source_values.append(float(np.clip(mean_conf, 0.0, 1.0)))
                    source_details["sources"].append("constraint_uncertainty")
                    source_details["constraint_confidence_mean"] = source_values[-1]

        view_diag = extra_view_diagnostics.get(view)
        if isinstance(view_diag, Mapping):
            uncertainty_diag = view_diag.get("uncertainty")
            if isinstance(uncertainty_diag, Mapping):
                conf = uncertainty_diag.get("confidence_mean")
                if conf is not None:
                    try:
                        parsed = float(conf)
                    except (TypeError, ValueError):
                        warnings.append(
                            f"invalid confidence_mean signal for view {view!r}: {conf!r}"
                        )
                    else:
                        if math.isfinite(parsed):
                            parsed = float(np.clip(parsed, 0.0, 1.0))
                            values.append(parsed)
                            source_count += 1
                            source_values.append(parsed)
                            source_details["sources"].append(
                                "extras.view_diagnostics.uncertainty.confidence_mean"
                            )
                            source_details["view_diagnostics_confidence_mean"] = parsed

        explicit = extras.get("view_signals")
        if isinstance(explicit, Mapping):
            explicit_value = explicit.get(view)
            if explicit_value is not None:
                try:
                    parsed = float(explicit_value)
                except (TypeError, ValueError):
                    warnings.append(
                        f"invalid explicit view signal for {view!r}: {explicit_value!r}"
                    )
                else:
                    if math.isfinite(parsed):
                        parsed = float(np.clip(parsed, 0.0, 1.0))
                        values.append(parsed)
                        source_count += 1
                        source_values.append(parsed)
                        source_details["sources"].append("extras.view_signals")
                        source_details["view_signal_value"] = parsed

        weight = 1.0
        if values:
            weight = float(np.mean(values))
            source_details["available"] = True
            source_details["source_count"] = source_count
            source_details["source_values"] = tuple(source_values)
        source_details["count"] = source_count
        details[view] = source_details
        if 0.0 <= weight <= 1.0:
            weights[view] = weight

    return weights, details, tuple(warnings)

def _coerce_view_weight(
    value: object,
    *,
    view_name: str,
    warnings: list[str],
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        warnings.append(f"invalid signal weight for view {view_name!r}: {value!r}")
        return 1.0
    if not math.isfinite(parsed):
        warnings.append(f"non-finite signal weight for view {view_name!r}: {parsed}")
        return 1.0
    return float(np.clip(parsed, 0.0, 1.0))

def _weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for raw, weight in zip(values, weights):
        numerator += float(raw) * float(weight)
        denominator += float(weight)
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)

def _silhouette_view_history(
    name: str,
    predicted: np.ndarray,
    target: np.ndarray,
    metrics: Mapping[str, float],
) -> dict[str, object]:
    predicted_array = np.asarray(predicted, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    intersection = float(np.minimum(predicted_array, target_array).sum())
    union = float(np.maximum(predicted_array, target_array).sum())
    return {
        "view": name,
        "shape": tuple(int(value) for value in predicted_array.shape),
        "pred_area_ratio": float(predicted_array.mean()),
        "target_area_ratio": float(target_array.mean()),
        "intersection_area": intersection,
        "union_area": union,
        "soft_l1": float(metrics.get("soft_l1", 0.0)),
        "soft_l2": float(metrics.get("soft_l2", 0.0)),
        "soft_iou_loss": float(metrics.get("soft_iou_loss", 0.0)),
        "area_iou_loss": float(metrics.get("area_iou_loss", 0.0)),
        "boundary_iou": float(metrics.get("boundary_iou", 0.0)),
        "boundary_iou_loss": float(metrics.get("boundary_iou_loss", 1.0)),
        "signed_distance_loss": float(metrics.get("signed_distance_loss", 0.0)),
    }

def _candidate_per_view_metrics(
    per_view: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, object]]:
    """Convert soft-renderer losses into the shared CandidateMetrics contract."""
    converted: dict[str, dict[str, object]] = {}
    for view, values in per_view.items():
        area_loss = _metric_float(values, "area_iou_loss", 1.0)
        soft_loss = _metric_float(values, "soft_iou_loss", area_loss)
        signed_distance = max(
            0.0,
            _metric_float(
                values,
                "signed_distance_loss",
                _metric_float(values, "soft_l2", soft_loss),
            ),
        )
        area_iou = _clamp01(1.0 - area_loss)
        soft_iou = _clamp01(1.0 - soft_loss)
        boundary_iou = _clamp01(
            _metric_float(values, "boundary_iou", soft_iou)
        )
        passed = bool(
            area_iou >= 0.5
            and boundary_iou > 0.0
            and math.isfinite(signed_distance)
        )
        reason = ""
        if not passed:
            reasons: list[str] = []
            if area_iou < 0.5:
                if soft_iou > 0.0:
                    reasons.append(
                        "soft silhouette hard-IoU gate failed despite nonzero soft overlap"
                    )
                else:
                    reasons.append("silhouette has no measurable overlap")
            if boundary_iou <= 0.0:
                reasons.append("boundary IoU is zero")
            if not math.isfinite(signed_distance):
                reasons.append("signed-distance loss is non-finite")
            reason = "; ".join(reasons) or "soft silhouette did not satisfy required view gate"
        converted[str(view)] = {
            **dict(values),
            "area_iou": area_iou,
            "soft_iou": soft_iou,
            "boundary_iou": boundary_iou,
            "signed_distance_loss": signed_distance,
            "required": True,
            "passed": passed,
            "pass": passed,
            "reason": reason,
        }
    return converted

def _mean_candidate_metric(
    per_view: Mapping[str, Mapping[str, object]],
    name: str,
    fallback: float,
) -> float:
    values = []
    for payload in per_view.values():
        value = payload.get(name)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return fallback
    return float(np.mean(values))

def _min_candidate_metric(
    per_view: Mapping[str, Mapping[str, object]],
    name: str,
    fallback: float,
) -> float:
    values = []
    for payload in per_view.values():
        value = payload.get(name)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return fallback
    return float(min(values))

def _metric_float(
    values: Mapping[str, float],
    name: str,
    default: float,
) -> float:
    try:
        value = float(values.get(name, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value

def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))

def primitive_from_renderable(renderable: RenderablePrimitive) -> object:
    """Create a known primitive object from a renderable primitive record."""
    ptype = renderable.primitive_type.lower()
    params = dict(renderable.parameters)
    if ptype == "ellipsoid":
        return EllipsoidPrimitive.from_dict(params)
    if ptype in ("gaussian", "anisotropic_gaussian"):
        return AnisotropicGaussianPrimitive.from_dict(params)
    if ptype == "deformed_superquadric":
        from primitives.deformed_superquadric import DeformedSuperquadricPrimitive
        return DeformedSuperquadricPrimitive.from_dict(params)
    if ptype == "superquadric":
        return SuperquadricPrimitive.from_dict(params)
    if ptype == "superfrustum":
        return SuperFrustum.from_dict(params)
    raise ValueError(f"unsupported renderable primitive type: {renderable.primitive_type}")

def renderable_from_primitive(primitive: object) -> RenderablePrimitive:
    """Serialize a known primitive into the renderable scene contract."""
    if not hasattr(primitive, "to_dict"):
        raise TypeError(f"primitive lacks to_dict: {type(primitive)!r}")
    params = primitive.to_dict()
    ptype = str(params.get("type", type(primitive).__name__.lower()))
    mesh = primitive.to_mesh_data(32) if hasattr(primitive, "to_mesh_data") else None
    return RenderablePrimitive(primitive_type=ptype, parameters=params, mesh_proxy=mesh)
