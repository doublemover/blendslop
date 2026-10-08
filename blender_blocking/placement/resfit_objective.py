"""
Objective terms for modular residual primitive fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
import hashlib
import json
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
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain finite coordinates")
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
    """Mean squared signed union-field residual, not an exact distance.

    Taking abs after the signed minimum prevents a buried component surface
    from satisfying an exterior target sample. Primitive fields may themselves
    be distance approximations, especially within overlapping components.
    """
    target_points = _validate_points(target_points, "target_points")
    if len(target_points) == 0 or not primitives:
        return 0.0
    sdf = primitive_sdf_matrix(primitives, target_points)
    nearest = np.min(sdf, axis=0)
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


def default_constraint_penalty(primitives: Sequence[object], *, length_scale: float = 1.) -> float:
    """Softly penalize invalid or unstable primitive parameters."""
    penalty = 0.0
    for primitive in primitives:
        if hasattr(primitive, "radii"):
            radii = np.asarray(getattr(primitive, "radii"), dtype=np.float64) / length_scale
            penalty += float(np.sum(np.maximum(1e-4 - radii, 0.0) ** 2))
        for attr in ("radius_bottom", "radius_top", "height"):
            if hasattr(primitive, attr):
                value = float(getattr(primitive, attr)) / length_scale
                floor = 1e-4 if attr != "height" else 1e-3
                penalty += float(max(floor - value, 0.0) ** 2)
        for attr in ("epsilon1", "epsilon2"):
            if hasattr(primitive, attr):
                value = float(getattr(primitive, attr))
                penalty += float(max(0.05 - value, 0.0) ** 2)
                penalty += float(max(value - 4.0, 0.0) ** 2)
        covariance = getattr(primitive, "covariance", None)
        if covariance is not None:
            covariance = covariance() if callable(covariance) else covariance
            eigvals = np.linalg.eigvalsh(np.asarray(covariance)) / (length_scale * length_scale)
            penalty += float(np.sum(np.maximum(1e-8 - eigvals, 0.0) ** 2))
    return penalty


@dataclass(frozen=True)
class _PartEvaluation:
    surface_sdf: np.ndarray
    occupancy_sdf: np.ndarray | None
    samples: np.ndarray
    area_weights: np.ndarray
    reverse_squared_distances: np.ndarray


def _geometry_key(primitive: object) -> str | None:
    """Content key for the serializable analytic families used by this fitter."""
    if not hasattr(primitive, "to_dict"):
        # Unknown duck-typed primitives have no guaranteed geometry contract.
        return None
    payload = primitive.to_dict()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         default=lambda value: np.asarray(value).tolist(),
                         allow_nan=False).encode("utf-8")
    name = f"{type(primitive).__module__}.{type(primitive).__qualname__}".encode("utf-8")
    return hashlib.sha256(name + encoded).hexdigest()


def _checked_sdf(primitive: object, points: np.ndarray) -> np.ndarray:
    if not hasattr(primitive, "sdf_batch"):
        raise TypeError(f"primitive lacks sdf_batch: {type(primitive)!r}")
    values = np.asarray(primitive.sdf_batch(points), dtype=np.float64)
    if values.shape != (len(points),) or not np.isfinite(values).all():
        raise ValueError("primitive sdf_batch must return one finite value per point")
    return values


class ResFitObjectiveEvaluator:
    """One fit's exterior objective with bounded changed-part evaluation reuse.

    Input-derived target arrays are captured once. Only geometry-identical part
    rows, surface samples, and pairwise sample fields are reused. Hooks still
    receive the complete candidate, and all acceptance scores are full weighted
    objectives. This evaluator contains no Blender or global shared state.
    """

    def __init__(self, target_points: np.ndarray,
                 weights: ResFitLossWeights = ResFitLossWeights(),
                 occupied_points: np.ndarray | None = None,
                 silhouette_hook: SilhouetteHook | None = None,
                 topology_penalty_hook: PenaltyHook | None = None,
                 constraint_penalty_hook: PenaltyHook | None = None,
                 uncertainty_penalty_hook: PenaltyHook | None = None,
                 *, surface_samples: int = 64, exterior_tolerance: float = 1e-7,
                 max_cached_parts: int = 64, objective_mode: str = "legacy_world_squared",
                 length_scale: float | None = None):
        errors = weights.validate()
        if errors:
            raise ValueError("invalid objective weights: " + "; ".join(errors))
        if surface_samples < 1 or max_cached_parts < 1:
            raise ValueError("surface_samples and max_cached_parts must be positive")
        if exterior_tolerance < 0 or not np.isfinite(exterior_tolerance):
            raise ValueError("exterior_tolerance must be finite and nonnegative")
        self.target_points = _validate_points(target_points, "target_points").copy()
        self.occupied_points = (None if occupied_points is None else
                                _validate_points(occupied_points, "occupied_points").copy())
        self.target_points.setflags(write=False)
        if self.occupied_points is not None:
            self.occupied_points.setflags(write=False)
        if objective_mode not in {"legacy_world_squared", "normalized_area_v1"}:
            raise ValueError("unsupported ResFit objective_mode")
        self.objective_mode = objective_mode
        inferred_scale = float(np.max(np.ptp(self.target_points, axis=0))) if len(self.target_points) > 1 else 1.
        self.length_scale = float(length_scale if length_scale is not None else inferred_scale)
        if not np.isfinite(self.length_scale) or self.length_scale <= 0.:
            if objective_mode == "normalized_area_v1":
                raise ValueError("normalized objective requires a positive finite object length scale")
            self.length_scale = 1.
        self.distance_squared_scale = self.length_scale ** 2 if objective_mode == "normalized_area_v1" else 1.
        self.weights = weights
        self.silhouette_hook = silhouette_hook
        self.topology_penalty_hook = topology_penalty_hook
        self.constraint_penalty_hook = constraint_penalty_hook
        self.uncertainty_penalty_hook = uncertainty_penalty_hook
        self.surface_samples = int(surface_samples)
        self.exterior_tolerance = float(exterior_tolerance) * (self.length_scale if objective_mode == "normalized_area_v1" else 1.)
        self.max_cached_parts = int(max_cached_parts)
        self._parts: OrderedDict[str, _PartEvaluation] = OrderedDict()
        self._sample_fields: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._target_tree = None
        if len(self.target_points) and weights.surface_residual > 0:
            try:
                from scipy.spatial import cKDTree
                self._target_tree = cKDTree(self.target_points)
            except ImportError:
                pass

    def _nearest_target_squared(self, samples: np.ndarray) -> np.ndarray:
        if not len(samples) or not len(self.target_points):
            return np.zeros(len(samples), dtype=np.float64)
        if self._target_tree is not None:
            distances, _ = self._target_tree.query(samples, workers=1)
            return np.asarray(distances, dtype=np.float64) ** 2
        # Bound the NumPy fallback's temporary array even for large input sets.
        nearest = np.full(len(samples), np.inf)
        for start in range(0, len(self.target_points), 1024):
            delta = samples[:, None, :] - self.target_points[None, start:start + 1024, :]
            nearest = np.minimum(nearest, np.min(np.sum(delta * delta, axis=2), axis=1))
        return nearest

    def _part(self, primitive: object, key: str | None) -> _PartEvaluation:
        if key is not None and key in self._parts:
            self._parts.move_to_end(key)
            return self._parts[key]
        samples = (np.asarray(primitive.sample_surface(self.surface_samples), dtype=np.float64)
                   if (self.weights.surface_residual > 0 or self.weights.overlap_penalty > 0)
                   and hasattr(primitive, "sample_surface") else np.empty((0, 3)))
        samples = _validate_points(samples, "primitive surface samples")
        if self.objective_mode == "normalized_area_v1" and len(samples):
            from .surface_quadrature import surface_area_weights
            area_weights = surface_area_weights(primitive, samples)
        else:
            area_weights = np.ones(len(samples))
        record = _PartEvaluation(
            surface_sdf=(_checked_sdf(primitive, self.target_points)
                         if self.weights.surface_residual > 0 else np.empty(0)),
            occupancy_sdf=(None if self.occupied_points is None or self.weights.visual_hull_occupancy == 0 else
                           _checked_sdf(primitive, self.occupied_points)),
            samples=samples.copy(),
            area_weights=area_weights,
            reverse_squared_distances=(self._nearest_target_squared(samples)
                                       if self.weights.surface_residual > 0 else np.empty(0)),
        )
        if key is not None:
            self._parts[key] = record
            while len(self._parts) > self.max_cached_parts:
                self._parts.popitem(last=False)
        return record

    def _sample_field(self, source_key: str | None, field_key: str | None,
                      samples: np.ndarray, primitive: object) -> np.ndarray:
        key = None if source_key is None or field_key is None else (source_key, field_key)
        if key is not None and key in self._sample_fields:
            self._sample_fields.move_to_end(key)
            return self._sample_fields[key]
        values = _checked_sdf(primitive, samples)
        if key is not None:
            self._sample_fields[key] = values
            while len(self._sample_fields) > self.max_cached_parts ** 2:
                self._sample_fields.popitem(last=False)
        return values

    def residuals(self, primitives: Sequence[object]) -> np.ndarray:
        """Per-input-point union-field magnitudes used to localize proposals."""
        if not primitives:
            return np.full(len(self.target_points), np.inf)
        rows = [(self._part(part, _geometry_key(part)).surface_sdf
                 if self.weights.surface_residual > 0 else _checked_sdf(part, self.target_points))
                for part in primitives]
        return np.abs(np.min(np.stack(rows), axis=0))

    def evaluate_batch(self, candidates, *, budget):
        """Score independent alternatives using one fit's unchanged-part rows.

        The completed prefix is admitted in order. Numerical rows and pair
        fields are local to this evaluator; no candidate states are combined.
        """
        from .resfit_optimizer import OptimizationBudgetExhausted
        for candidate in candidates:
            if budget.reason() is not None:
                break
            try:
                yield budget.evaluate(self, candidate)
            except OptimizationBudgetExhausted:
                break

    def residual_vector(self, primitives, *, evaluated=None):
        """Fixed-size same-objective residuals for a fixed primitive topology.

        Numeric step proposals reuse the existing part/target caches. Visibility
        changes zero weights rather than removing rows, so Jacobian dimensions
        cannot change while fitting one assembly.
        """
        result = self(primitives) if evaluated is None else evaluated
        keys = [_geometry_key(p) for p in primitives]
        records = [self._part(p, key) for p, key in zip(primitives, keys)]
        residuals = []
        scale = np.sqrt(self.distance_squared_scale)
        if records and len(self.target_points) and self.weights.surface_residual > 0.:
            union = np.min(np.stack([r.surface_sdf for r in records]), axis=0)
            residuals.append(union * np.sqrt(self.weights.surface_residual/len(union)) / scale)
        if records and self.occupied_points is not None and len(self.occupied_points) and self.weights.visual_hull_occupancy > 0.:
            union = np.min(np.stack([r.occupancy_sdf for r in records]), axis=0)
            residuals.append(np.maximum(union, 0.) * np.sqrt(self.weights.visual_hull_occupancy/len(union)) / scale)
        reverse_distances, reverse_weights, pair_rows = [], [], []
        for index, record in enumerate(records):
            if not len(record.samples):
                continue
            visible = np.ones(len(record.samples), bool)
            for other_index, other in enumerate(primitives):
                if index == other_index:
                    continue
                field = self._sample_field(keys[index], keys[other_index], record.samples, other)
                visible &= field >= -self.exterior_tolerance
                if self.weights.overlap_penalty > 0.:
                    pair_rows.append(np.maximum(-field, 0.) * np.sqrt(record.area_weights/record.area_weights.sum()))
            if len(self.target_points) and self.weights.surface_residual > 0.:
                reverse_distances.extend(np.sqrt(record.reverse_squared_distances))
                reverse_weights.extend(record.area_weights * visible)
        if reverse_distances:
            weights = np.asarray(reverse_weights)
            denominator = weights.sum()
            residuals.append(np.asarray(reverse_distances) * np.sqrt(
                self.weights.surface_residual * weights / max(denominator, np.finfo(float).tiny)) / scale)
        if pair_rows:
            factor = 1. if self.objective_mode == "normalized_area_v1" else 1./len(pair_rows)
            residuals.append(np.concatenate(pair_rows) * np.sqrt(self.weights.overlap_penalty * factor) / scale)
        for name in ("primitive_count", "silhouette", "topology_penalty", "constraint_penalty", "uncertainty_penalty"):
            value = float(result.terms[name]) * float(getattr(self.weights, name))
            if value < 0.:
                raise ValueError("least-squares residual terms must be nonnegative")
            residuals.append(np.array([np.sqrt(value)]))
        return np.concatenate(residuals) if residuals else np.zeros(1)

    def __call__(self, primitives: Sequence[object]) -> ResFitObjectiveResult:
        keys = [_geometry_key(part) for part in primitives]
        records = [self._part(part, key) for part, key in zip(primitives, keys)]
        warnings = []
        if not len(self.target_points):
            warnings.append("target_points is empty")
        if not primitives:
            warnings.append("primitive set is empty")
        forward = 0.0
        occupancy = 0.0
        if records and len(self.target_points) and self.weights.surface_residual > 0:
            union = np.min(np.stack([record.surface_sdf for record in records]), axis=0)
            forward = float(np.mean(union * union))
        if records and self.occupied_points is not None and len(self.occupied_points) and self.weights.visual_hull_occupancy > 0:
            union = np.min(np.stack([record.occupancy_sdf for record in records]), axis=0)
            occupancy = float(np.mean(np.maximum(union, 0.0) ** 2))
        reverse_values, reverse_weights = [], []
        overlap = []
        for index, record in enumerate(records):
            if not len(record.samples):
                continue
            visible = np.ones(len(record.samples), dtype=bool)
            for other_index, other in enumerate(primitives):
                if index == other_index:
                    continue
                field = self._sample_field(keys[index], keys[other_index], record.samples, other)
                visible &= field >= -self.exterior_tolerance
                if self.weights.overlap_penalty > 0:
                    overlap.append(float(np.average(np.maximum(-field, 0.0) ** 2,
                                                    weights=record.area_weights)))
            if len(self.target_points) and self.weights.surface_residual > 0:
                reverse_values.extend(record.reverse_squared_distances[visible])
                reverse_weights.extend(record.area_weights[visible])
        reverse = (float(np.average(reverse_values, weights=reverse_weights))
                   if len(reverse_values) and np.sum(reverse_weights) > 0. else 0.)
        forward /= self.distance_squared_scale
        reverse /= self.distance_squared_scale
        occupancy /= self.distance_squared_scale
        overlap_value = (float(np.sum(overlap)) if self.objective_mode == "normalized_area_v1"
                         else float(np.mean(overlap))) if overlap else 0.
        terms: Dict[str, float] = {
            "surface_union_field": forward,
            "surface_exterior_samples": reverse,
            "surface_residual": forward + reverse,
            "visual_hull_occupancy": occupancy,
            "primitive_count": float(len(primitives)),
            "overlap_penalty": overlap_value / self.distance_squared_scale,
        }
        if self.silhouette_hook is not None and self.weights.silhouette > 0:
            silhouette_terms = dict(self.silhouette_hook(primitives))
            terms.update({f"silhouette_{key}": float(value) for key, value in silhouette_terms.items()})
            terms["silhouette"] = float(sum(silhouette_terms.values()))
        else:
            terms["silhouette"] = 0.0
        terms["topology_penalty"] = (float(self.topology_penalty_hook(primitives))
                                     if self.topology_penalty_hook and self.weights.topology_penalty > 0 else 0.0)
        if self.weights.constraint_penalty <= 0:
            terms["constraint_penalty"] = 0.
        elif self.constraint_penalty_hook is not None:
            # External hooks declare their own units; backend extent hooks are dimensionless.
            terms["constraint_penalty"] = float(self.constraint_penalty_hook(primitives))
        else:
            terms["constraint_penalty"] = default_constraint_penalty(primitives,
                length_scale=self.length_scale if self.objective_mode == "normalized_area_v1" else 1.)
        terms["uncertainty_penalty"] = (float(self.uncertainty_penalty_hook(primitives))
                                        if self.uncertainty_penalty_hook and self.weights.uncertainty_penalty > 0 else 0.0)
        if not all(np.isfinite(value) for value in terms.values()):
            raise ValueError("objective terms must be finite")
        total = sum(float(getattr(self.weights, name)) * terms[name]
                    for name in self.weights.__dataclass_fields__)
        return ResFitObjectiveResult(float(total), terms, tuple(warnings))


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
    """Evaluate the complete exterior objective without retaining trial state.

    A fitting loop should construct ResFitObjectiveEvaluator once to reuse the
    unchanged parts. The surface term combines a signed union-field surrogate
    with input-point distances from sampled, non-buried component surfaces.
    """
    return ResFitObjectiveEvaluator(
        target_points, weights, occupied_points, silhouette_hook,
        topology_penalty_hook, constraint_penalty_hook, uncertainty_penalty_hook,
    )(primitives)
