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


def _build_per_view_profile_summary(profile_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    per_view: dict[str, Any] = {}
    for row in profile_rows:
        view = str(row.get("view", "generic"))
        record = per_view.setdefault(
            view,
            {
                "row_count": 0,
                "width_world_mean": 0.0,
                "z_world_min": float("inf"),
                "z_world_max": float("-inf"),
                "conf_mean": 0.0,
            },
        )
        record["row_count"] += 1
        record["width_world_mean"] += float(row.get("width_world", 0.0))
        record["z_world_min"] = float(min(record["z_world_min"], row.get("z_world", 0.0)))
        record["z_world_max"] = float(max(record["z_world_max"], row.get("z_world", 0.0)))
        record["conf_mean"] += float(row.get("confidence", 1.0))
    for record in per_view.values():
        if record["row_count"] > 0:
            count = float(record["row_count"])
            record["width_world_mean"] /= count
            record["conf_mean"] /= count
        confidence = float(np.clip(record.get("conf_mean", 0.0), 0.0, 1.0))
        passed = bool(record["row_count"] > 0 and confidence >= 0.35)
        record["area_iou"] = confidence
        record["boundary_iou"] = confidence
        record["soft_iou"] = confidence
        record["signed_distance_loss"] = float(1.0 - confidence)
        record["required"] = True
        record["passed"] = passed
        record["pass"] = passed
        record["reason"] = "" if passed else "profile confidence below primitive-fit gate"
    return per_view


def _collect_profile_rows(
    profile_signal: Mapping[str, Any],
    bounds: Any | None,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in tuple(profile_signal.get("rows", ()))]
    if not rows:
        return []
    width_scale = _profile_width_scale(profile_signal, bounds)
    z_min, z_max = _bounds_z_range(bounds)
    for row in rows:
        t = float(row.get("t", 0.0))
        if 0.0 <= t <= 1.0:
            row["z_world"] = float(z_min + t * (z_max - z_min))
        else:
            row["z_world"] = float(z_min + (0.5 + t / max(1.0, profile_signal.get("band_samples", 1))) * (z_max - z_min))
        width_px = float(row.get("width_px", 0.0))
        row["width_world"] = max(width_px * width_scale, 1e-6)
        row["half_width_world"] = row["width_world"] / 2.0
        if bounds is not None:
            bounds_min_x = float(bounds.min_x)
            bounds_max_x = float(bounds.max_x)
            bounds_span_x = max(1e-6, bounds_max_x - bounds_min_x)
            center_px = float(row.get("center_x_px", 0.0))
            reference = float(max(profile_signal.get("max_width", width_px), 1.0))
            row["center_x_world"] = float(
                (bounds_min_x + bounds_max_x) / 2.0
                + (center_px - 0.5 * reference) * (bounds_span_x / max(1e-6, reference))
            )
        else:
            row["center_x_world"] = 0.0
    return rows


def _profile_rows_to_slices(
    profile_rows: Sequence[Mapping[str, Any]],
    init_config: PrimitiveInitializationConfig,
) -> list[dict[str, Any]]:
    if not profile_rows:
        return []
    rows = list(profile_rows)
    rows.sort(key=lambda row: float(row.get("z_world", 0.0)))
    if init_config.target_point_count <= 0:
        sample_indices = range(len(rows))
    else:
        step = max(1, int(np.ceil(len(rows) / max(1, init_config.primitive_count))))
        sample_indices = range(0, len(rows), step)

    slice_data: list[dict[str, Any]] = []
    for idx in sample_indices:
        row = rows[int(idx)]
        row_center_x = float(row.get("center_x_world", 0.0))
        row_z = float(row.get("z_world", 0.0))
        row_radius = float(row.get("half_width_world", init_config.min_radius))
        if row_radius <= init_config.min_radius:
            row_radius = float(init_config.min_radius)
        slice_data.append(
            {
                "center": (row_center_x, 0.0, row_z),
                "radius": row_radius,
                "t": float(row.get("t", 0.0)),
                "view": str(row.get("view", "generic")),
            }
        )
    return slice_data


def _profile_width_scale(
    profile_signal: Mapping[str, Any],
    bounds: Any | None,
) -> float:
    max_width = float(profile_signal.get("max_width", 0.0))
    if max_width <= 0.0:
        return 1.0
    if bounds is None:
        return 1.0
    return max(1e-6, float(bounds.max_x - bounds.min_x)) / max_width


def _bounds_z_range(bounds: Any | None) -> tuple[float, float]:
    if bounds is None:
        return 0.0, 1.0
    return float(bounds.min_z), float(bounds.max_z)


def _dominant_interval(
    intervals: tuple[Any, ...] | Sequence[Any],
) -> Mapping[str, Any] | None:
    if not intervals:
        return None
    return max(intervals, key=lambda interval: float(getattr(interval, "width", 0.0)))
