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
from .contracts import LossResult, LossWeights, ReconstructionTarget, RenderBatch
from .target_adapter import _coerce_view_weight, _weighted_average


def evaluate_render_loss(
    render_batch: RenderBatch,
    target: ReconstructionTarget,
    weights: LossWeights = LossWeights(),
    view_weights: Mapping[str, float] | None = None,
) -> LossResult:
    """Compute soft silhouette/depth losses for a rendered batch."""
    terms: Dict[str, float] = {}
    per_view: Dict[str, Mapping[str, float]] = {}
    warnings: list[str] = []
    configured_view_weights = dict(view_weights or {})

    silhouette_weight_sum = 0.0
    depth_terms_sum: float = 0.0
    depth_weight_sum: float = 0.0

    for name, predicted in render_batch.silhouettes.items():
        target_mask = target.silhouettes.get(name)
        if target_mask is None:
            warnings.append(f"missing target silhouette for view {name}")
            continue
        metrics = dict(soft_mask_metrics(predicted, target_mask))
        weight = _coerce_view_weight(
            configured_view_weights.get(name, 1.0),
            view_name=name,
            warnings=warnings,
        )
        per_view[name] = {
            **metrics,
            "weight": weight,
        }
        silhouette_weight_sum += weight
        for key, value in metrics.items():
            terms[f"{name}_{key}"] = float(value)

    silhouette_l2 = (
        _weighted_average(
            [m["soft_l2"] for m in per_view.values()],
            [m["weight"] for m in per_view.values()],
        )
        if per_view
        else 0.0
    )
    if silhouette_weight_sum <= 0.0 and per_view:
        warnings.append(
            "all silhouette view weights were zero; fell back to unweighted aggregation"
        )
        silhouette_l2 = float(np.mean([m["soft_l2"] for m in per_view.values()]))
        soft_iou = float(np.mean([m["soft_iou_loss"] for m in per_view.values()]))
        area_iou = float(np.mean([m["area_iou_loss"] for m in per_view.values()]))
    else:
        soft_iou = float(
            _weighted_average(
                [m["soft_iou_loss"] for m in per_view.values()],
                [m["weight"] for m in per_view.values()],
            )
            if per_view
            else 0.0
        )
        area_iou = float(
            _weighted_average(
                [m["area_iou_loss"] for m in per_view.values()],
                [m["weight"] for m in per_view.values()],
            )
            if per_view
            else 0.0
        )
        silhouette_l2 = float(silhouette_l2)

    if silhouette_weight_sum <= 0.0 and per_view:
        boundary_iou = float(
            np.mean([m.get("boundary_iou_loss", 1.0) for m in per_view.values()])
        )
        signed_distance = float(
            np.mean([m.get("signed_distance_loss", 0.0) for m in per_view.values()])
        )
    else:
        boundary_iou = float(
            _weighted_average(
                [m.get("boundary_iou_loss", 1.0) for m in per_view.values()],
                [m["weight"] for m in per_view.values()],
            )
            if per_view
            else 0.0
        )
        signed_distance = float(
            _weighted_average(
                [m.get("signed_distance_loss", 0.0) for m in per_view.values()],
                [m["weight"] for m in per_view.values()],
            )
            if per_view
            else 0.0
        )

    terms["silhouette_l2"] = silhouette_l2
    terms["soft_iou"] = soft_iou
    terms["area_iou"] = area_iou
    terms["boundary_iou"] = boundary_iou
    terms["boundary_iou_loss"] = boundary_iou
    terms["boundary_iou_score"] = float(1.0 - boundary_iou)
    terms["signed_distance"] = signed_distance
    terms["signed_distance_loss"] = signed_distance
    terms["silhouette_weight_sum"] = float(silhouette_weight_sum)

    depth_losses = []
    for name, predicted_depth in render_batch.depths.items():
        target_depth = target.depths.get(name)
        if target_depth is None:
            continue
        pred = np.asarray(predicted_depth, dtype=np.float64)
        tgt = np.asarray(target_depth, dtype=np.float64)
        if pred.shape != tgt.shape:
            warnings.append(f"depth shape mismatch for view {name}")
            continue
        weight = _coerce_view_weight(
            configured_view_weights.get(name, 1.0),
            view_name=name,
            warnings=warnings,
        )
        depth_value = float(np.mean((pred - tgt) ** 2))
        depth_losses.append(depth_value)
        if weight > 0.0:
            depth_terms_sum += depth_value * weight
            depth_weight_sum += weight

    if depth_weight_sum > 0.0:
        depth_l2 = float(depth_terms_sum / depth_weight_sum)
    elif depth_losses:
        warnings.append(
            "all depth view weights were zero; fell back to unweighted depth aggregation"
        )
        depth_l2 = float(np.mean(depth_losses))
    else:
        depth_l2 = 0.0
    terms["depth_l2"] = depth_l2
    terms["depth_weight_sum"] = float(depth_weight_sum)

    total = (
        weights.silhouette_l2 * silhouette_l2
        + weights.soft_iou * soft_iou
        + weights.area_iou * area_iou
        + weights.boundary_iou * boundary_iou
        + weights.signed_distance * signed_distance
        + weights.depth_l2 * depth_l2
    )
    return LossResult(
        total=float(total),
        terms=terms,
        per_view=per_view,
        warnings=tuple(warnings),
    )
