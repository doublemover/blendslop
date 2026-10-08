"""Apply human constraints to profile-like data structures."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .model import (
    CenterlineConstraint,
    ConstraintSatisfaction,
    ConstraintSet,
    DimensionConstraint,
    satisfaction_for_constraint,
    summarize_satisfaction,
)


@dataclass(frozen=True)
class ProfileConstraintResult:
    profile: Any
    satisfaction: Tuple[ConstraintSatisfaction, ...]
    cleanup_options: Dict[str, Any]
    report: Dict[str, Any]


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set_fields(obj: Any, updates: Mapping[str, Any], *, copy: bool = True) -> Any:
    if isinstance(obj, MutableMapping):
        target = dict(obj) if copy else obj
        target.update(updates)
        return target
    if is_dataclass(obj):
        valid = {field.name for field in fields(obj)}
        replace_updates = {key: value for key, value in updates.items() if key in valid}
        if replace_updates:
            return replace(obj, **replace_updates)
        return obj
    target = obj
    for key, value in updates.items():
        if hasattr(target, key):
            try:
                setattr(target, key, value)
            except Exception:
                pass
    return target


def _profile_bbox_height(profile: Any) -> Optional[float]:
    bbox = _get_field(profile, "bbox")
    if bbox is None:
        return None
    height = _get_field(bbox, "h")
    if height is not None:
        return float(height)
    y0 = _get_field(bbox, "y0")
    y1 = _get_field(bbox, "y1")
    if y0 is not None and y1 is not None:
        return float(y1) - float(y0)
    return None


def _centerline_samples(
    centerline: CenterlineConstraint, profile: Any
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(centerline.points_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("centerline points must be Nx2")

    bbox = _get_field(profile, "bbox")
    if bbox is not None:
        y0 = float(_get_field(bbox, "y0", 0.0))
        height = max(_profile_bbox_height(profile) or 1.0, 1.0)
        t_values = 1.0 - ((points[:, 1] - y0) / max(height - 1.0, 1.0))
    else:
        y_min = float(points[:, 1].min())
        y_max = float(points[:, 1].max())
        y_range = max(y_max - y_min, 1.0)
        t_values = 1.0 - ((points[:, 1] - y_min) / y_range)
    t_values = np.clip(t_values, 0.0, 1.0)
    order = np.argsort(t_values)
    return t_values[order], points[:, 0][order]


def apply_centerline_to_profile(
    profile: Any,
    centerline: CenterlineConstraint,
    *,
    copy: bool = True,
) -> Tuple[Any, ConstraintSatisfaction]:
    """Adjust center_x-like profile fields toward a centerline hint."""
    heights_t = _get_field(profile, "heights_t")
    center_x = _get_field(profile, "center_x")
    if heights_t is None or center_x is None:
        return (
            profile,
            satisfaction_for_constraint(
                centerline,
                satisfied=False,
                score=0.0,
                message="profile has no heights_t/center_x fields",
                details={"view": centerline.view},
            ),
        )

    heights = np.asarray(heights_t, dtype=np.float64)
    centers = np.asarray(center_x, dtype=np.float64)
    if heights.shape != centers.shape or heights.ndim != 1:
        return (
            profile,
            satisfaction_for_constraint(
                centerline,
                satisfied=False,
                score=0.0,
                message="profile center_x and heights_t must be 1D arrays of equal length",
                details={"view": centerline.view},
            ),
        )

    hint_t, hint_x = _centerline_samples(centerline, profile)
    target_centers = np.interp(heights, hint_t, hint_x)
    delta = target_centers - centers
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    tolerance = max(float(centerline.tolerance_px), 1e-9)
    score = max(0.0, 1.0 - (max_abs_error / tolerance)) if tolerance > 0 else 0.0

    updates: Dict[str, Any] = {"center_x": target_centers.astype(np.float32)}
    left_x = _get_field(profile, "left_x")
    right_x = _get_field(profile, "right_x")
    if left_x is not None and right_x is not None:
        half_width = (np.asarray(right_x, dtype=np.float64) - np.asarray(left_x, dtype=np.float64)) / 2.0
        updates["left_x"] = (target_centers - half_width).astype(np.float32)
        updates["right_x"] = (target_centers + half_width).astype(np.float32)

    updated_profile = _set_fields(profile, updates, copy=copy)
    return (
        updated_profile,
        satisfaction_for_constraint(
            centerline,
            satisfied=max_abs_error <= centerline.tolerance_px,
            score=1.0 if max_abs_error <= centerline.tolerance_px else score,
            message="centerline applied to profile centers",
            details={
                "view": centerline.view,
                "max_abs_error_px": max_abs_error,
                "tolerance_px": centerline.tolerance_px,
            },
        ),
    )


def _apply_dimension_to_profile(
    profile: Any,
    dimension: DimensionConstraint,
    *,
    copy: bool = True,
) -> Tuple[Any, ConstraintSatisfaction]:
    current = None
    if dimension.name == "height":
        current = _get_field(profile, "world_height")
    elif dimension.name in {"width", "x"}:
        rx = _get_field(profile, "rx")
        if rx is not None:
            current = float(np.nanmax(np.asarray(rx, dtype=np.float64))) * 2.0
    elif dimension.name in {"depth", "y"}:
        ry = _get_field(profile, "ry")
        if ry is not None:
            current = float(np.nanmax(np.asarray(ry, dtype=np.float64))) * 2.0

    updates: Dict[str, Any] = {}
    if dimension.name == "height":
        updates["world_height"] = dimension.value_u
    elif dimension.name in {"width", "x"}:
        rx = _get_field(profile, "rx")
        if rx is not None:
            updates["rx"] = np.asarray(rx, dtype=np.float64) * (
                dimension.value_u / max(float(current or dimension.value_u), 1e-12)
            )
    elif dimension.name in {"depth", "y"}:
        ry = _get_field(profile, "ry")
        if ry is not None:
            updates["ry"] = np.asarray(ry, dtype=np.float64) * (
                dimension.value_u / max(float(current or dimension.value_u), 1e-12)
            )

    updated = _set_fields(profile, updates, copy=copy) if updates else profile
    if current is None:
        return (
            updated,
            satisfaction_for_constraint(
                dimension,
                satisfied=False,
                score=0.0,
                message="profile lacks matching dimension fields",
                details={"name": dimension.name},
            ),
        )

    error = abs(float(current) - dimension.value_u)
    satisfied = error <= dimension.tolerance_u
    score = 1.0 if satisfied else max(0.0, 1.0 - error / max(dimension.value_u, 1e-12))
    return (
        updated,
        satisfaction_for_constraint(
            dimension,
            satisfied=satisfied,
            score=score,
            message="dimension constraint applied to profile scale",
            details={
                "name": dimension.name,
                "current_u": float(current),
                "target_u": dimension.value_u,
                "tolerance_u": dimension.tolerance_u,
            },
        ),
    )


def profile_cleanup_hints(constraints: ConstraintSet, view: Optional[str] = None) -> Dict[str, Any]:
    """Return profile extraction cleanup choices implied by component constraints."""
    hints: Dict[str, Any] = {}
    relevant = [
        item for item in constraints.components if view is None or item.view in {None, view}
    ]
    if any(item.kind == "preserve_holes" for item in relevant):
        hints["preserve_holes"] = True
        hints["fill_holes"] = False
    if any(item.kind == "preserve_components" for item in relevant):
        hints["preserve_components"] = True
        hints["largest_component_only"] = False
    if any(item.kind == "force_separate_parts" for item in relevant):
        hints["force_separate_parts"] = True
    ignore_crops = [item.crop_px for item in relevant if item.kind == "ignore_crop" and item.crop_px]
    if ignore_crops:
        hints["ignore_crop_px"] = ignore_crops
    return hints


def apply_constraints_to_profile(
    profile: Any,
    constraints: ConstraintSet,
    *,
    view: Optional[str] = None,
    copy: bool = True,
) -> ProfileConstraintResult:
    """Apply centerline and dimension constraints to a profile-like object."""
    source_view = view or _get_field(profile, "source_view")
    updated = profile
    satisfaction = []

    for centerline in constraints.centerlines:
        if source_view is not None and centerline.view != source_view:
            continue
        updated, result = apply_centerline_to_profile(updated, centerline, copy=copy)
        satisfaction.append(result)

    for dimension in constraints.dimensions:
        if dimension.name in {"height", "width", "depth", "x", "y", "z"}:
            updated, result = _apply_dimension_to_profile(updated, dimension, copy=copy)
            satisfaction.append(result)

    cleanup_options = profile_cleanup_hints(constraints, source_view)
    report = summarize_satisfaction(satisfaction)
    report["view"] = source_view
    report["cleanup_options"] = cleanup_options
    return ProfileConstraintResult(
        profile=updated,
        satisfaction=tuple(satisfaction),
        cleanup_options=cleanup_options,
        report=report,
    )
