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
        record["evidence_weight"] = float(np.clip(record["conf_mean"], 0, 1))
    return per_view


def _collect_profile_rows(
    profile_signal: Mapping[str, Any],
    bounds: Any | None,
    target: Any | None = None,
) -> list[dict[str, Any]]:
    if target is not None and target.constraints:
        from blender_blocking.reconstruction.projection_contract import pixel_cell_viewport
        from blender_blocking.reconstruction.visibility import valid_evidence
        rows = []
        for constraint in target.constraints:
            mask = np.asarray(getattr(constraint.mask, "mask", constraint.mask), bool)
            h, w = mask.shape
            valid = valid_evidence(constraint)
            axes, (u0, u1, v0, v1) = pixel_cell_viewport(target, constraint)
            bands = tuple(target.profile_bands.get(constraint.view, ()))
            if not bands:
                from blender_blocking.reconstruction.target_builder import mask_to_profile_bands
                bands = mask_to_profile_bands(mask, sample_count=min(h, 64), view=constraint.view)
            center = np.asarray(bounds.center if bounds is not None else (0, 0, 0), float)
            for band in bands:
                pixel_row = int(np.clip(round((1 - band.t) * (h - 1)), 0, h - 1))
                vertical = v1 - (pixel_row + .5) / h * (v1 - v0)
                padded = np.pad(mask[pixel_row] & valid[pixel_row], (1, 1)).astype(int)
                changes = np.diff(padded)
                intervals = [(u0 + a / w * (u1 - u0), u0 + b / w * (u1 - u0))
                             for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]
                known_changes = np.diff(np.pad(valid[pixel_row], (1, 1)).astype(int))
                known = [(u0 + a / w * (u1 - u0), u0 + b / w * (u1 - u0))
                         for a, b in zip(np.flatnonzero(known_changes == 1), np.flatnonzero(known_changes == -1))]
                if not known:
                    continue
                width = sum(b - a for a, b in intervals)
                horizontal = sum((a + b) * .5 * (b - a) for a, b in intervals) / width if width else center[axes[0]]
                world_center = center.copy()
                world_center[axes[0]], world_center[axes[1]] = horizontal, vertical
                from blender_blocking.reconstruction.profile_evidence import measure_profile_row
                measurement = measure_profile_row(mask[pixel_row], valid[pixel_row])
                rows.append({"view": constraint.view, "t": band.t, "axes": axes,
                    "profile_evidence": measurement,
                    "profile_width_exact": measurement["exact_width_px"] is not None,
                    "vertical_world": vertical, "z_world": world_center[2], "center_x_world": horizontal,
                    "world_center": world_center.tolist(), "intervals_world": intervals, "known_intervals_world": known,
                    "viewport_world": (u0, u1, v0, v1), "pixel_world": (u1 - u0) / w,
                    "width_world": width, "half_width_world": width * .5, "confidence": band.confidence if intervals else 1.0})
        return rows
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
    rows = [row for row in profile_rows if tuple(row.get("axes", (0, 2)))[1] == 2 and row.get("width_world", 0) > 0]
    front = [row for row in rows if row.get("view") == "front"]
    side = sorted([row for row in rows if row.get("view") == "side"],
                  key=lambda row: float(row.get("z_world", 0.)))
    rows = front or rows
    rows = [row for row in rows if row.get("profile_width_exact", True)]
    rows.sort(key=lambda row: float(row.get("z_world", 0.0)))
    # Keep every measured knot. The initializer partitions adjacent intervals
    # exactly once; pre-downsampling to the part count loses endpoint coverage.
    sample_indices = range(len(rows))

    slice_data: list[dict[str, Any]] = []
    for idx in sample_indices:
        row = rows[int(idx)]
        row_center_x = float(row.get("center_x_world", 0.0))
        row_z = float(row.get("z_world", 0.0))
        row_radius = float(row.get("half_width_world", init_config.min_radius))
        if row_radius <= init_config.min_radius:
            row_radius = float(init_config.min_radius)
        center = np.asarray(row.get("world_center", (row_center_x, 0.0, row_z)), float).copy()
        if front and side:
            exact_side = [entry for entry in side if entry.get("profile_width_exact", True)]
            if not exact_side:
                continue
            side_z = [float(entry.get("z_world", 0.)) for entry in exact_side]
            if row_z < side_z[0] or row_z > side_z[-1]:
                continue
            side_radius = float(np.interp(row_z, side_z,
                                [entry.get("half_width_world", init_config.min_radius) for entry in exact_side]))
            # A circular seed cannot faithfully describe strongly elliptical
            # cross-sections. Leave those to the existing ellipsoid family.
            if max(row_radius, side_radius) > 1.15 * max(min(row_radius, side_radius), 1e-12):
                return []
            center[1] = np.interp(row_z, side_z,
                                 [entry.get("center_x_world", 0.) for entry in exact_side])
            row_radius = .5 * (row_radius + side_radius)
        slice_data.append(
            {
                "center": tuple(center),
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
