"""
Objective terms for modular residual primitive fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Sequence

import numpy as np


SilhouetteHook = Callable[[Sequence[object]], Mapping[str, float]]
PenaltyHook = Callable[[Sequence[object]], float]

def _validate_non_negative_weight(value: float, name: str, errors: list[str]) -> float:
    if not np.isfinite(value):
        errors.append(f"{name} must be finite, got {value!r}")
        return 0.0
    if value < 0.0:
        errors.append(f"{name} must be >= 0.0, got {value!r}")
        return 0.0
    return float(value)


@dataclass(frozen=True)
class ResFitLossWeights:
    """Weights for deterministic ResFit objective terms."""

    surface_residual: float = 1.0
    visual_hull_occupancy: float = 0.0
    primitive_count: float = 0.01
    overlap_penalty: float = 0.05
    silhouette: float = 0.0
    topology_penalty: float = 0.0
    constraint_penalty: float = 0.05
    uncertainty_penalty: float = 0.0

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        _validate_non_negative_weight(self.surface_residual, "surface_residual", errors)
        _validate_non_negative_weight(
            self.visual_hull_occupancy, "visual_hull_occupancy", errors
        )
        _validate_non_negative_weight(self.primitive_count, "primitive_count", errors)
        _validate_non_negative_weight(self.overlap_penalty, "overlap_penalty", errors)
        _validate_non_negative_weight(self.silhouette, "silhouette", errors)
        _validate_non_negative_weight(self.topology_penalty, "topology_penalty", errors)
        _validate_non_negative_weight(self.constraint_penalty, "constraint_penalty", errors)
        _validate_non_negative_weight(
            self.uncertainty_penalty, "uncertainty_penalty", errors
        )
        return tuple(errors)


@dataclass(frozen=True)
class ResFitObjectiveResult:
    """Loss decomposition returned by objective evaluation."""

    total: float
    terms: Mapping[str, float]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    return points


def primitive_sdf_matrix(primitives: Sequence[object], points: np.ndarray) -> np.ndarray:
    """Return SDF values with shape (primitive_count, point_count)."""
    points = _validate_points(points, "points")
    rows = []
    for primitive in primitives:
        if not hasattr(primitive, "sdf_batch"):
            raise TypeError(f"primitive lacks sdf_batch: {type(primitive)!r}")
        rows.append(np.asarray(primitive.sdf_batch(points), dtype=np.float64))
    if not rows:
        return np.zeros((0, len(points)), dtype=np.float64)
    return np.vstack(rows)


def surface_residual(primitives: Sequence[object], target_points: np.ndarray) -> float:
    """Mean squared nearest-surface SDF residual."""
    target_points = _validate_points(target_points, "target_points")
    if len(target_points) == 0 or not primitives:
        return 0.0
    sdf = primitive_sdf_matrix(primitives, target_points)
    nearest = np.min(np.abs(sdf), axis=0)
    return float(np.mean(nearest * nearest))


def visual_hull_occupancy_penalty(
    primitives: Sequence[object],
    occupied_points: np.ndarray | None,
) -> float:
    """Penalty for visual-hull occupied samples outside every primitive."""
    if occupied_points is None:
        return 0.0
    occupied_points = _validate_points(occupied_points, "occupied_points")
    if len(occupied_points) == 0 or not primitives:
        return 0.0
    sdf = primitive_sdf_matrix(primitives, occupied_points)
    nearest = np.min(sdf, axis=0)
    outside = np.maximum(nearest, 0.0)
    return float(np.mean(outside * outside))


def primitive_overlap_penalty(
    primitives: Sequence[object],
    samples_per_primitive: int = 64,
) -> float:
    """Estimate overlap by sampling each primitive surface inside others."""
    if len(primitives) < 2:
        return 0.0
    penalties = []
    for idx, primitive in enumerate(primitives):
        if not hasattr(primitive, "sample_surface"):
            continue
        samples = np.asarray(primitive.sample_surface(samples_per_primitive), dtype=np.float64)
        if len(samples) == 0:
            continue
        for other_idx, other in enumerate(primitives):
            if idx == other_idx or not hasattr(other, "sdf_batch"):
                continue
            inside_depth = np.maximum(-np.asarray(other.sdf_batch(samples)), 0.0)
            if inside_depth.size:
                penalties.append(float(np.mean(inside_depth * inside_depth)))
    return float(np.mean(penalties)) if penalties else 0.0


def default_constraint_penalty(primitives: Sequence[object]) -> float:
    """Softly penalize invalid or unstable primitive parameters."""
    penalty = 0.0
    for primitive in primitives:
        if hasattr(primitive, "radii"):
            radii = np.asarray(getattr(primitive, "radii"), dtype=np.float64)
            penalty += float(np.sum(np.maximum(1e-4 - radii, 0.0) ** 2))
        for attr in ("radius_bottom", "radius_top", "height"):
            if hasattr(primitive, attr):
                value = float(getattr(primitive, attr))
                floor = 1e-4 if attr != "height" else 1e-3
                penalty += float(max(floor - value, 0.0) ** 2)
        for attr in ("epsilon1", "epsilon2"):
            if hasattr(primitive, attr):
                value = float(getattr(primitive, attr))
                penalty += float(max(0.05 - value, 0.0) ** 2)
                penalty += float(max(value - 4.0, 0.0) ** 2)
        if hasattr(primitive, "covariance"):
            eigvals = np.linalg.eigvalsh(np.asarray(getattr(primitive, "covariance")))
            penalty += float(np.sum(np.maximum(1e-8 - eigvals, 0.0) ** 2))
    return penalty


def evaluate_resfit_objective(
    primitives: Sequence[object],
    target_points: np.ndarray,
    weights: ResFitLossWeights = ResFitLossWeights(),
    occupied_points: np.ndarray | None = None,
    silhouette_hook: SilhouetteHook | None = None,
    topology_penalty_hook: PenaltyHook | None = None,
    constraint_penalty_hook: PenaltyHook | None = None,
    uncertainty_penalty_hook: PenaltyHook | None = None,
) -> ResFitObjectiveResult:
    """Evaluate weighted objective and return a loss decomposition."""
    warnings = []
    target_points = _validate_points(target_points, "target_points")
    if len(target_points) == 0:
        warnings.append("target_points is empty")
    if not primitives:
        warnings.append("primitive set is empty")

    terms: Dict[str, float] = {}
    terms["surface_residual"] = surface_residual(primitives, target_points)
    terms["visual_hull_occupancy"] = visual_hull_occupancy_penalty(
        primitives, occupied_points
    )
    terms["primitive_count"] = float(len(primitives))
    terms["overlap_penalty"] = primitive_overlap_penalty(primitives)

    if silhouette_hook is not None:
        silhouette_terms = dict(silhouette_hook(primitives))
        for key, value in silhouette_terms.items():
            terms[f"silhouette_{key}"] = float(value)
        terms["silhouette"] = float(sum(silhouette_terms.values()))
    else:
        terms["silhouette"] = 0.0

    terms["topology_penalty"] = (
        float(topology_penalty_hook(primitives)) if topology_penalty_hook else 0.0
    )
    constraint_hook = constraint_penalty_hook or default_constraint_penalty
    terms["constraint_penalty"] = float(constraint_hook(primitives))
    terms["uncertainty_penalty"] = (
        float(uncertainty_penalty_hook(primitives))
        if uncertainty_penalty_hook is not None
        else 0.0
    )

    total = (
        weights.surface_residual * terms["surface_residual"]
        + weights.visual_hull_occupancy * terms["visual_hull_occupancy"]
        + weights.primitive_count * terms["primitive_count"]
        + weights.overlap_penalty * terms["overlap_penalty"]
        + weights.silhouette * terms["silhouette"]
        + weights.topology_penalty * terms["topology_penalty"]
        + weights.constraint_penalty * terms["constraint_penalty"]
        + weights.uncertainty_penalty * terms["uncertainty_penalty"]
    )
    return ResFitObjectiveResult(total=float(total), terms=terms, warnings=tuple(warnings))
