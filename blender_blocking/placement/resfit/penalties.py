from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..resfit_initialization import (
    PrimitiveInitializationConfig,
    initialize_ellipsoids_from_points,
    initialize_from_profile_bands,
    initialize_gaussians_from_points,
    initialize_superfrusta_from_points,
)
from ..resfit_objective import (
    PenaltyHook,
    ResFitLossWeights,
    ResFitObjectiveResult,
    SilhouetteHook,
    evaluate_resfit_objective,
)
from ..resfit_optimizer import (
    CoordinateDescentConfig,
    OptimizationRecord,
    coordinate_descent_optimize,
)

from .config import by_view_total
from .profiles import _bounds_z_range


def _build_profile_silhouette_hook(
    *,
    profile_rows: Sequence[Mapping[str, Any]],
    uncertainty_by_view: Mapping[str, Mapping[str, Any]],
) -> Callable[[Sequence[object]], Mapping[str, float]]:
    def hook(primitives: Sequence[object]) -> Mapping[str, float]:
        terms: dict[str, float] = {}
        if not profile_rows:
            return terms

        by_view: dict[str, list[float]] = {}
        for row in profile_rows:
            view = str(row.get("view", "generic"))
            z_world = float(row.get("z_world", 0.0))
            target_width = float(row.get("width_world", 0.0))
            target_center = float(row.get("center_x_world", 0.0))
            base_conf = float(row.get("confidence", 1.0))
            view_signal = uncertainty_by_view.get(view, {})
            confidence = float(view_signal.get("mean_confidence", 1.0))
            weight = float(base_conf * confidence)
            if weight <= 0.0:
                continue

            predicted_width = 0.0
            predicted_centers: list[float] = []
            for primitive in primitives:
                if not hasattr(primitive, "profile_width_at_world_z"):
                    continue
                predicted_width = max(
                    predicted_width,
                    float(primitive.profile_width_at_world_z(z_world)),
                )
                center_attr = "position" if hasattr(primitive, "position") else "center"
                position = np.asarray(getattr(primitive, center_attr, (0.0, 0.0, 0.0)))
                if position.size == 3:
                    predicted_centers.append(float(position[0]))
            if not predicted_centers:
                if predicted_width == 0.0:
                    predicted_width = 0.0
                predicted_center = 0.0
            else:
                predicted_center = float(np.mean(predicted_centers))

            if target_width <= 0.0:
                width_term = min(1.0, abs(predicted_width) * 0.1)
            else:
                normalized_width = (predicted_width - target_width) / target_width
                width_term = normalized_width * normalized_width

            width_scale = max(1e-3, target_width)
            center_term = ((predicted_center - target_center) / width_scale) ** 2
            row_term = width_term + 0.5 * center_term
            by_view.setdefault(view, []).append(weight * row_term)

        for view, values in by_view.items():
            if not values:
                terms[f"profile_{view}"] = 0.0
            else:
                terms[f"profile_{view}"] = float(np.mean(values))
        terms["profile"] = float(sum(by_view_total(values) for values in by_view.values()))
        return terms

    return hook


def _build_topology_penalty_hook(topology_signal: Mapping[str, Any]) -> PenaltyHook:
    topology_target = float(topology_signal.get("score", 1.0))
    complexity = float(topology_signal.get("complexity", 0.0))

    def hook(primitives: Sequence[object]) -> float:
        if topology_target <= 0.0 and not complexity:
            return 0.0
        count = float(len(primitives))
        return float((1.0 - topology_target) + 0.01 * complexity * count)

    return hook


def _build_constraint_penalty_hook(
    *,
    constraint_signal: Mapping[str, Any],
    bounds: Any | None,
) -> PenaltyHook:
    score = float(constraint_signal.get("score", 1.0))
    count = int(constraint_signal.get("constraint_count", 0))
    has_bounds = bounds is not None
    bounds_min = None
    bounds_max = None
    if has_bounds:
        bounds_min = np.array([bounds.min_x, bounds.min_y, bounds.min_z], dtype=float)
        bounds_max = np.array([bounds.max_x, bounds.max_y, bounds.max_z], dtype=float)
        extents = bounds_max - bounds_min
        extents = np.where(extents > 0.0, extents, 1.0)

    def hook(primitives: Sequence[object]) -> float:
        base = 0.0
        for primitive in primitives:
            center_attr = "position" if hasattr(primitive, "position") else "center"
            if not hasattr(primitive, center_attr):
                continue
            if has_bounds:
                position = np.asarray(getattr(primitive, center_attr), dtype=float)
                if position.size == 3:
                    below = np.maximum(bounds_min - position, 0.0)
                    above = np.maximum(position - bounds_max, 0.0)
                    normal = (below + above) / extents
                    base += float(np.linalg.norm(normal) ** 2)
        # Weakly penalize strong constraint budgets or dense explicit constraint payloads.
        constraint_pressure = float((1.0 - score) * max(1, count) * 0.1)
        return base + constraint_pressure

    return hook


def _build_uncertainty_penalty_hook(
    uncertainty_signal: Mapping[str, Any],
) -> PenaltyHook:
    consistency = float(uncertainty_signal.get("consistency", 0.75))
    overall = float(uncertainty_signal.get("overall_confidence_mean", 1.0))
    overall_std = float(uncertainty_signal.get("overall_confidence_std", 0.0))

    def hook(_: Sequence[object]) -> float:
        return float(max(0.0, 1.0 - overall) + 0.25 * overall_std + 0.15 * (1.0 - consistency))

    return hook
