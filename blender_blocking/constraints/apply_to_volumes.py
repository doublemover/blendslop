"""Apply constraints to volume-like reconstruction inputs and outputs."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .apply_to_masks import MaskConstraintResult, apply_constraints_to_masks
from .model import (
    AxisConstraint,
    ConstraintSatisfaction,
    ConstraintSet,
    DimensionConstraint,
    PlaneConstraint,
    satisfaction_for_constraint,
    summarize_satisfaction,
)


@dataclass(frozen=True)
class VolumeConstraintResult:
    volume: Any
    bounds_min: Optional[Tuple[float, float, float]]
    bounds_max: Optional[Tuple[float, float, float]]
    view_masks: Dict[str, MaskConstraintResult]
    satisfaction: Tuple[ConstraintSatisfaction, ...]
    report: Dict[str, Any]


def _as_vec3(value: Any, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3-value sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _get_bounds(volume: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if volume is None:
        return None, None
    if isinstance(volume, Mapping):
        if "bounds_min" in volume or "bounds_max" in volume:
            return _as_vec3(volume.get("bounds_min"), "bounds_min"), _as_vec3(
                volume.get("bounds_max"), "bounds_max"
            )
        bounds = volume.get("bounds")
        if isinstance(bounds, Mapping):
            return _as_vec3(bounds.get("min"), "bounds.min"), _as_vec3(
                bounds.get("max"), "bounds.max"
            )
    return _as_vec3(getattr(volume, "bounds_min", None), "bounds_min"), _as_vec3(
        getattr(volume, "bounds_max", None), "bounds_max"
    )


def _set_volume_fields(volume: Any, updates: Mapping[str, Any], *, copy: bool = True) -> Any:
    if isinstance(volume, MutableMapping):
        target = dict(volume) if copy else volume
        for key, value in updates.items():
            target[key] = value
        return target
    if volume is None:
        return dict(updates)
    for key, value in updates.items():
        if hasattr(volume, key):
            try:
                setattr(volume, key, value)
            except Exception:
                pass
    return volume


def _bounds_from_dimensions(constraints: ConstraintSet) -> Tuple[np.ndarray, np.ndarray]:
    sizes = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    for dimension in constraints.dimensions:
        if dimension.axis_index is not None:
            sizes[dimension.axis_index] = dimension.value_u
    return np.array([-sizes[0] / 2.0, -sizes[1] / 2.0, 0.0]), np.array(
        [sizes[0] / 2.0, sizes[1] / 2.0, sizes[2]]
    )


def apply_dimension_constraints_to_bounds(
    bounds_min: Optional[Sequence[float]],
    bounds_max: Optional[Sequence[float]],
    constraints: ConstraintSet,
) -> Tuple[np.ndarray, np.ndarray, Tuple[ConstraintSatisfaction, ...]]:
    """Resize bounds to satisfy known width/depth/height constraints."""
    min_vec = _as_vec3(bounds_min, "bounds_min") if bounds_min is not None else None
    max_vec = _as_vec3(bounds_max, "bounds_max") if bounds_max is not None else None
    if min_vec is None or max_vec is None:
        min_vec, max_vec = _bounds_from_dimensions(constraints)

    if np.any(max_vec <= min_vec):
        raise ValueError("bounds_max must be greater than bounds_min")

    satisfaction = []
    for dimension in constraints.dimensions:
        axis = dimension.axis_index
        if axis is None:
            continue
        current = float(max_vec[axis] - min_vec[axis])
        center = float((max_vec[axis] + min_vec[axis]) / 2.0)
        if dimension.name in {"height", "z"}:
            min_vec[axis] = min_vec[axis]
            max_vec[axis] = min_vec[axis] + dimension.value_u
        else:
            min_vec[axis] = center - dimension.value_u / 2.0
            max_vec[axis] = center + dimension.value_u / 2.0

        error = abs(current - dimension.value_u)
        satisfied = error <= dimension.tolerance_u
        score = 1.0 if satisfied else max(0.0, 1.0 - error / max(dimension.value_u, 1e-12))
        satisfaction.append(
            satisfaction_for_constraint(
                dimension,
                satisfied=satisfied,
                score=score,
                message="dimension constraint applied to volume bounds",
                details={
                    "name": dimension.name,
                    "axis_index": axis,
                    "current_u": current,
                    "target_u": dimension.value_u,
                    "tolerance_u": dimension.tolerance_u,
                },
            )
        )
    return min_vec, max_vec, tuple(satisfaction)


def axis_transform_hints(constraints: ConstraintSet) -> Dict[str, Tuple[float, float, float]]:
    """Return axis-name to normalized world direction hints."""
    return {axis.axis_name: axis.direction_world for axis in constraints.axes}


def view_role_hints(constraints: ConstraintSet) -> Dict[str, str]:
    """Return view-name to confirmed reconstruction role hints."""
    roles: Dict[str, str] = {}
    confidence_by_view: Dict[str, float] = {}
    for view_role in constraints.view_roles:
        current = confidence_by_view.get(view_role.view, -1.0)
        if view_role.confidence >= current:
            roles[view_role.view] = view_role.role
            confidence_by_view[view_role.view] = view_role.confidence
    return roles


def _axis_satisfaction(axis: AxisConstraint) -> ConstraintSatisfaction:
    return satisfaction_for_constraint(
        axis,
        satisfied=True,
        score=axis.confidence,
        message="axis hint attached to volume metadata",
        details={"axis_name": axis.axis_name, "direction_world": axis.direction_world},
    )


def _plane_satisfaction(plane: PlaneConstraint, bounds_min: np.ndarray) -> ConstraintSatisfaction:
    current = float(bounds_min[plane.axis_index])
    error = abs(current - plane.value_u)
    satisfied = error <= plane.tolerance_u
    score = 1.0 if satisfied else max(0.0, 1.0 - error / max(abs(plane.value_u), 1.0))
    return satisfaction_for_constraint(
        plane,
        satisfied=satisfied,
        score=score,
        message="fixed plane checked against volume bounds",
        details={
            "kind": plane.kind,
            "axis": plane.axis,
            "current_u": current,
            "target_u": plane.value_u,
            "tolerance_u": plane.tolerance_u,
        },
    )


def apply_constraints_to_volume(
    volume: Any,
    constraints: ConstraintSet,
    *,
    view_masks: Optional[Mapping[str, np.ndarray]] = None,
    view_confidences: Optional[Mapping[str, np.ndarray]] = None,
    copy: bool = True,
) -> VolumeConstraintResult:
    """Apply dimension, axis, plane, and scribble mask constraints to a volume-like object."""
    bounds_min, bounds_max = _get_bounds(volume)
    bounds_min, bounds_max, dimension_satisfaction = apply_dimension_constraints_to_bounds(
        bounds_min, bounds_max, constraints
    )

    mask_results: Dict[str, MaskConstraintResult] = {}
    if view_masks is not None:
        mask_results = apply_constraints_to_masks(
            view_masks, constraints, confidences=view_confidences
        )

    satisfaction = list(dimension_satisfaction)
    satisfaction.extend(_axis_satisfaction(axis) for axis in constraints.axes)
    satisfaction.extend(_plane_satisfaction(plane, bounds_min) for plane in constraints.planes)
    for result in mask_results.values():
        satisfaction.extend(result.satisfaction)

    updates: Dict[str, Any] = {
        "bounds_min": tuple(float(v) for v in bounds_min),
        "bounds_max": tuple(float(v) for v in bounds_max),
        "axis_constraints": axis_transform_hints(constraints),
        "view_roles": view_role_hints(constraints),
    }
    updated_volume = _set_volume_fields(volume, updates, copy=copy)

    report = summarize_satisfaction(satisfaction)
    report["bounds"] = {
        "min": [float(v) for v in bounds_min],
        "max": [float(v) for v in bounds_max],
    }
    report["axis_constraints"] = updates["axis_constraints"]
    report["view_roles"] = updates["view_roles"]
    report["view_masks"] = {view: result.report for view, result in mask_results.items()}
    return VolumeConstraintResult(
        volume=updated_volume,
        bounds_min=tuple(float(v) for v in bounds_min),
        bounds_max=tuple(float(v) for v in bounds_max),
        view_masks=mask_results,
        satisfaction=tuple(satisfaction),
        report=report,
    )
