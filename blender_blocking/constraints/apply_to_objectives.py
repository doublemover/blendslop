"""Constraint terms for candidate objective score dictionaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .model import (
    AxisConstraint,
    BBoxConstraint,
    ConstraintSatisfaction,
    ConstraintSet,
    DimensionConstraint,
    PrimitiveFamilyConstraint,
    SymmetryConstraint,
    satisfaction_for_constraint,
    summarize_satisfaction,
)


@dataclass(frozen=True)
class ObjectiveConstraintResult:
    scores: Dict[str, float]
    satisfaction: Tuple[ConstraintSatisfaction, ...]
    hard_failed: bool
    report: Dict[str, Any]


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _bounds_dimensions(candidate: Any) -> Dict[str, float]:
    dimensions = _get_field(candidate, "dimensions")
    if isinstance(dimensions, Mapping):
        return {str(key).lower(): float(value) for key, value in dimensions.items()}

    bounds_min = _get_field(candidate, "bounds_min")
    bounds_max = _get_field(candidate, "bounds_max")
    bounds = _get_field(candidate, "bounds")
    if bounds_min is None and isinstance(bounds, Mapping):
        bounds_min = bounds.get("min")
        bounds_max = bounds.get("max")
    if bounds_min is None or bounds_max is None:
        return {}
    min_vec = np.asarray(bounds_min, dtype=np.float64)
    max_vec = np.asarray(bounds_max, dtype=np.float64)
    if min_vec.shape != (3,) or max_vec.shape != (3,):
        return {}
    sizes = max_vec - min_vec
    return {
        "width": float(sizes[0]),
        "x": float(sizes[0]),
        "depth": float(sizes[1]),
        "y": float(sizes[1]),
        "height": float(sizes[2]),
        "z": float(sizes[2]),
    }


def _candidate_bbox(candidate: Any, view: str) -> Optional[Tuple[float, float, float, float]]:
    bboxes = _get_field(candidate, "bboxes")
    if isinstance(bboxes, Mapping) and view in bboxes:
        values = bboxes[view]
    else:
        values = _get_field(candidate, "bbox_px")
        if isinstance(values, Mapping):
            values = values.get(view)
    if values is None:
        masks = _get_field(candidate, "masks")
        mask = masks.get(view) if isinstance(masks, Mapping) else None
        if mask is not None:
            array = np.asarray(mask).astype(bool)
            if array.ndim == 2 and array.any():
                ys, xs = np.nonzero(array)
                return (
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max() + 1),
                    float(ys.max() + 1),
                )
        return None
    try:
        x0, y0, x1, y1 = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    intersection = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def _candidate_axis(candidate: Any, axis_name: str) -> Optional[np.ndarray]:
    axes = _get_field(candidate, "axes")
    if isinstance(axes, Mapping) and axis_name in axes:
        return np.asarray(axes[axis_name], dtype=np.float64)
    axis_constraints = _get_field(candidate, "axis_constraints")
    if isinstance(axis_constraints, Mapping) and axis_name in axis_constraints:
        return np.asarray(axis_constraints[axis_name], dtype=np.float64)
    return None


def _dimension_satisfaction(
    dimension: DimensionConstraint, candidate: Any
) -> ConstraintSatisfaction:
    dims = _bounds_dimensions(candidate)
    current = dims.get(dimension.name)
    if current is None:
        return satisfaction_for_constraint(
            dimension,
            satisfied=False,
            score=0.0,
            message="candidate has no matching dimension",
            details={"name": dimension.name},
        )
    error = abs(current - dimension.value_u)
    satisfied = error <= dimension.tolerance_u
    score = 1.0 if satisfied else max(0.0, 1.0 - error / max(dimension.value_u, 1e-12))
    return satisfaction_for_constraint(
        dimension,
        satisfied=satisfied,
        score=score,
        message="candidate dimension checked",
        details={
            "name": dimension.name,
            "candidate_u": current,
            "target_u": dimension.value_u,
            "tolerance_u": dimension.tolerance_u,
        },
    )


def _bbox_satisfaction(bbox: BBoxConstraint, candidate: Any) -> ConstraintSatisfaction:
    candidate_bbox = _candidate_bbox(candidate, bbox.view)
    if candidate_bbox is None:
        return satisfaction_for_constraint(
            bbox,
            satisfied=False,
            score=0.0,
            message="candidate has no matching bbox or mask",
            details={"view": bbox.view},
        )
    errors = [abs(candidate_bbox[i] - bbox.bbox_px[i]) for i in range(4)]
    max_error = max(errors)
    satisfied = max_error <= bbox.tolerance_px
    score = 1.0 if satisfied else _bbox_iou(candidate_bbox, bbox.bbox_px)
    return satisfaction_for_constraint(
        bbox,
        satisfied=satisfied,
        score=score,
        message="candidate bbox checked",
        details={
            "view": bbox.view,
            "candidate_bbox_px": candidate_bbox,
            "target_bbox_px": bbox.bbox_px,
            "max_error_px": max_error,
            "tolerance_px": bbox.tolerance_px,
        },
    )


def _symmetry_satisfaction(
    symmetry: SymmetryConstraint, candidate: Any
) -> ConstraintSatisfaction:
    symmetry_scores = _get_field(candidate, "symmetry_scores")
    if isinstance(symmetry_scores, Mapping) and symmetry.plane in symmetry_scores:
        score = float(np.clip(symmetry_scores[symmetry.plane], 0.0, 1.0))
        return satisfaction_for_constraint(
            symmetry,
            satisfied=score >= max(0.0, symmetry.confidence),
            score=score,
            message="candidate symmetry score checked",
            details={"plane": symmetry.plane},
        )

    symmetry_penalties = _get_field(candidate, "symmetry_penalties")
    if isinstance(symmetry_penalties, Mapping) and symmetry.plane in symmetry_penalties:
        penalty = max(0.0, float(symmetry_penalties[symmetry.plane]))
        score = float(np.clip(1.0 - penalty, 0.0, 1.0))
        return satisfaction_for_constraint(
            symmetry,
            satisfied=score >= symmetry.confidence,
            score=score,
            message="candidate symmetry penalty checked",
            details={"plane": symmetry.plane, "penalty": penalty},
        )

    return satisfaction_for_constraint(
        symmetry,
        satisfied=False,
        score=0.5,
        message="candidate has no symmetry metric; treating soft hint as neutral",
        details={"plane": symmetry.plane},
    )


def _axis_satisfaction(axis: AxisConstraint, candidate: Any) -> ConstraintSatisfaction:
    candidate_axis = _candidate_axis(candidate, axis.axis_name)
    if candidate_axis is None or candidate_axis.shape != (3,):
        return satisfaction_for_constraint(
            axis,
            satisfied=False,
            score=0.5,
            message="candidate has no matching axis vector; treating soft hint as neutral",
            details={"axis_name": axis.axis_name},
        )
    norm = float(np.linalg.norm(candidate_axis))
    if norm <= 0:
        score = 0.0
    else:
        score = float(abs(np.dot(candidate_axis / norm, np.asarray(axis.direction_world))))
    return satisfaction_for_constraint(
        axis,
        satisfied=score >= axis.confidence,
        score=score,
        message="candidate axis alignment checked",
        details={"axis_name": axis.axis_name, "alignment": score},
    )


def _primitive_family_satisfaction(
    family: PrimitiveFamilyConstraint, candidate: Any
) -> ConstraintSatisfaction:
    candidate_family = _get_field(candidate, "primitive_family")
    if candidate_family is None:
        candidate_family = _get_field(candidate, "family")
    if candidate_family is None:
        primitive_families = _get_field(candidate, "primitive_families")
        if isinstance(primitive_families, Sequence) and not isinstance(
            primitive_families, (str, bytes)
        ):
            candidate_family = primitive_families[0] if primitive_families else None
    if candidate_family is None:
        return satisfaction_for_constraint(
            family,
            satisfied=False,
            score=0.5,
            message="candidate has no primitive family; treating soft hint as neutral",
            details={"family": family.family},
        )
    match = str(candidate_family).lower() == family.family.lower()
    return satisfaction_for_constraint(
        family,
        satisfied=match,
        score=1.0 if match else 0.0,
        message="candidate primitive family checked",
        details={"expected": family.family, "actual": str(candidate_family)},
    )


def evaluate_constraint_satisfaction(
    constraints: ConstraintSet,
    candidate: Any,
) -> Tuple[ConstraintSatisfaction, ...]:
    """Evaluate objective-level constraints against a candidate-like object."""
    satisfaction = []
    satisfaction.extend(
        _dimension_satisfaction(dimension, candidate)
        for dimension in constraints.dimensions
    )
    satisfaction.extend(_bbox_satisfaction(bbox, candidate) for bbox in constraints.bboxes)
    satisfaction.extend(
        _symmetry_satisfaction(symmetry, candidate)
        for symmetry in constraints.symmetries
    )
    satisfaction.extend(_axis_satisfaction(axis, candidate) for axis in constraints.axes)
    satisfaction.extend(
        _primitive_family_satisfaction(family, candidate)
        for family in constraints.primitive_families
    )
    return tuple(satisfaction)


def apply_constraints_to_objective_scores(
    objective_scores: Mapping[str, Any],
    constraints: ConstraintSet,
    *,
    candidate: Any = None,
    constraint_weight: float = 1.0,
    fail_on_unsatisfied_hard_constraints: bool = True,
    total_key: Optional[str] = None,
) -> ObjectiveConstraintResult:
    """Add constraint satisfaction terms to an objective score dictionary.

    Existing scores are copied. A lower-is-better `constraint_penalty` is added,
    plus a higher-is-better `constraint_satisfaction` score for reporting.
    """
    scores: Dict[str, float] = {}
    for key, value in objective_scores.items():
        try:
            scores[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    satisfaction = evaluate_constraint_satisfaction(constraints, candidate)
    report = summarize_satisfaction(satisfaction)
    penalty = (1.0 - float(report["score"])) * float(constraint_weight)
    scores["constraint_satisfaction"] = float(report["score"])
    scores["constraint_penalty"] = penalty

    if total_key is not None and total_key in scores:
        scores[total_key] = scores[total_key] + penalty
    elif "total" in scores:
        scores["total"] = scores["total"] + penalty
    elif "score" in scores:
        scores["score"] = scores["score"] + penalty

    hard_failed = bool(report["hard_failed"]) and fail_on_unsatisfied_hard_constraints
    report["constraint_weight"] = float(constraint_weight)
    report["constraint_penalty"] = penalty
    report["hard_failed"] = hard_failed
    return ObjectiveConstraintResult(
        scores=scores,
        satisfaction=satisfaction,
        hard_failed=hard_failed,
        report=report,
    )
