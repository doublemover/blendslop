"""Shared extraction of target signals used by reconstruction backends."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def collect_target_signals(
    target: object,
    *,
    include_profile_rows: bool = False,
    include_surface_density: bool = True,
    include_constraint_kinds: bool = True,
    include_topology_details: bool = True,
) -> dict[str, Mapping[str, Any]]:
    constraints = tuple(getattr(target, "constraints", ()) or ())
    extras = getattr(target, "extras", {}) or {}
    if not isinstance(extras, Mapping):
        extras = {}
    profile_bands = getattr(target, "profile_bands", {})
    if not isinstance(profile_bands, Mapping):
        profile_bands = {}
    profile_signal = collect_profile_signal(
        profile_bands,
        include_rows=include_profile_rows,
    )
    constraint_signal = collect_constraint_signal(
        constraints,
        extras,
        include_kinds=include_constraint_kinds,
    )
    return {
        "surface": collect_surface_signal(
            constraints=constraints,
            target_extras=extras,
            target_points=getattr(target, "points", None),
            include_density=include_surface_density,
        ),
        "profile": profile_signal,
        "constraints": constraint_signal,
        "uncertainty": collect_uncertainty_signal(constraints),
        "topology": collect_topology_signal(
            profile_signal=profile_signal,
            constraint_signal=constraint_signal,
            extras=extras,
            include_details=include_topology_details,
        ),
    }


def collect_surface_signal(
    *,
    constraints: Sequence[Any],
    target_extras: Mapping[str, Any],
    target_points: Any | None,
    include_density: bool = True,
) -> dict[str, Any]:
    points = target_extras.get("surface_points")
    if points is not None:
        try:
            point_count = int(np.asarray(points).shape[0])
        except Exception:
            point_count = 0
    else:
        try:
            point_count = int(target_extras.get("surface_point_count", 0))
        except Exception:
            point_count = 0
    if point_count <= 0 and target_points is not None:
        try:
            point_count = int(len(target_points))
        except Exception:
            point_count = 0
    signal = {
        "available": point_count > 0 or bool(constraints),
        "constraint_count": len(constraints),
        "surface_point_count": point_count,
        "target_views": tuple(str(getattr(constraint, "view", "")) for constraint in constraints),
    }
    if include_density:
        signal["density_hint"] = float(estimate_surface_density(point_count, len(constraints)))
    return signal


def estimate_surface_density(point_count: int, constraint_count: int) -> float:
    base = min(1.0, float(point_count) / 4096.0)
    if constraint_count <= 0:
        return base
    return float(np.clip(base * (1.0 - 0.08 * int(constraint_count)), 0.15, 1.0))


def collect_profile_signal(
    profile_bands: Mapping[str, Any],
    *,
    include_rows: bool = False,
) -> dict[str, Any]:
    empty = {
        "available": False,
        "view_count": 0,
        "band_samples": 0,
        "interval_count": 0,
        "hole_count": 0,
        "mean_width": 0.0,
        "max_width": 0.0,
        "complexity": 0.0,
    }
    if include_rows:
        empty["rows"] = ()
    if not isinstance(profile_bands, Mapping) or not profile_bands:
        return empty

    view_count = 0
    band_samples = 0
    interval_count = 0
    hole_count = 0
    widths: list[float] = []
    rows: list[dict[str, Any]] = []
    for view, raw_bands in profile_bands.items():
        bands = tuple(raw_bands or ())
        if not bands:
            continue
        view_count += 1
        for index, band in enumerate(bands):
            band_samples += 1
            intervals = tuple(getattr(band, "intervals", ()))
            holes = tuple(getattr(band, "holes", ()))
            interval_count += len(intervals)
            hole_count += len(holes)
            width_px = float(getattr(band, "width_px", 0.0))
            widths.append(width_px)
            if include_rows:
                interval = dominant_interval(intervals)
                if interval is not None:
                    interval_width = float(interval.width)
                    if interval_width > 0.0:
                        rows.append(
                            {
                                "view": str(view),
                                "t": float(getattr(band, "t", 0.0)),
                                "width_px": max(interval_width, 1e-6),
                                "center_x_px": float(getattr(interval, "center", 0.0)),
                                "reference_width_px": float(width_px),
                                "confidence": float(
                                    getattr(
                                        interval,
                                        "confidence",
                                        getattr(band, "confidence", 1.0),
                                    )
                                ),
                                "index": index,
                            }
                        )
    if not widths:
        result = dict(empty)
        result.update(
            {
                "view_count": view_count,
                "band_samples": band_samples,
                "interval_count": interval_count,
                "hole_count": hole_count,
            }
        )
        return result

    widths_arr = np.asarray(widths, dtype=float)
    complexity = 0.0
    if band_samples > 0:
        complexity = float(
            np.clip((interval_count + 0.5 * hole_count) / float(band_samples), 0.0, 1.0)
        )
    result = {
        "available": True,
        "view_count": view_count,
        "band_samples": band_samples,
        "interval_count": interval_count,
        "hole_count": hole_count,
        "mean_width": float(np.mean(widths_arr)),
        "max_width": float(np.max(widths_arr)),
        "complexity": complexity,
    }
    if include_rows:
        result["rows"] = tuple(rows)
    return result


def dominant_interval(intervals: Sequence[Any]) -> Any | None:
    if not intervals:
        return None
    return max(intervals, key=lambda interval: float(getattr(interval, "width", 0.0)))


def collect_constraint_signal(
    constraints: Sequence[Any],
    extras: Mapping[str, Any],
    *,
    include_kinds: bool = True,
) -> dict[str, Any]:
    payload = extras.get("constraint_payload", {}) if isinstance(extras, Mapping) else {}
    payload_constraints = tuple(payload.get("constraints", ())) if isinstance(payload, Mapping) else ()
    payload_count = len(payload_constraints)
    view_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for constraint in constraints:
        view = str(getattr(constraint, "view", "generic"))
        view_counts[view] = view_counts.get(view, 0) + 1
        if include_kinds:
            constraint_kind = str(getattr(constraint, "kind", "generic")) or "generic"
            kind_counts[constraint_kind] = kind_counts.get(constraint_kind, 0) + 1
    constraint_count = len(constraints) + payload_count
    result = {
        "available": bool(constraints or payload),
        "constraint_count": constraint_count,
        "constraint_payload_count": payload_count,
        "payload_count": payload_count,
        "constraint_views": view_counts,
        "score": float(np.clip(1.0 / (1.0 + 0.25 * constraint_count), 0.0, 1.0)),
    }
    if include_kinds:
        result["constraint_kinds"] = kind_counts
    return result


def collect_uncertainty_signal(constraints: Sequence[Any]) -> dict[str, Any]:
    if not constraints:
        return {
            "available": False,
            "overall_confidence_mean": 1.0,
            "overall_confidence_std": 0.0,
            "overall_boundary_uncertainty_mean": 0.0,
            "consistency": 0.75,
            "view_details": {},
            "constraint_count": 0,
        }
    means: list[float] = []
    boundary_means: list[float] = []
    details: dict[str, dict[str, Any]] = {}
    for constraint in constraints:
        view = str(getattr(constraint, "view", "unknown"))
        uncertainty = getattr(constraint, "uncertainty", None)
        if uncertainty is None:
            continue
        confidence = np.asarray(getattr(uncertainty, "confidence", ()), dtype=float).reshape(-1)
        boundary = np.asarray(
            getattr(uncertainty, "boundary_uncertainty", ()), dtype=float
        ).reshape(-1)
        if confidence.size:
            confidence = np.clip(confidence.astype(float), 0.0, 1.0)
            means.append(float(confidence.mean()))
            details.setdefault(view, {}).update(
                {
                    "mean_confidence": float(confidence.mean()),
                    "max_confidence": float(confidence.max()),
                    "min_confidence": float(confidence.min()),
                    "std_confidence": float(confidence.std()),
                }
            )
        if boundary.size:
            boundary = np.clip(boundary.astype(float), 0.0, 1.0)
            boundary_means.append(float(boundary.mean()))
            details.setdefault(view, {}).update(
                {
                    "boundary_uncertainty_mean": float(boundary.mean()),
                    "boundary_uncertainty_max": float(boundary.max()),
                    "boundary_uncertainty_min": float(boundary.min()),
                }
            )
    if not means and not boundary_means:
        return {
            "available": False,
            "overall_confidence_mean": 1.0,
            "overall_confidence_std": 0.0,
            "overall_boundary_uncertainty_mean": 0.0,
            "consistency": 0.75,
            "view_details": {},
            "constraint_count": int(len(constraints)),
        }
    overall_conf = float(np.mean(means)) if means else 0.0
    overall_std = float(np.std(means)) if means else 0.0
    overall_boundary = float(np.mean(boundary_means)) if boundary_means else 0.0
    consistency = float(np.clip(0.6 + 0.4 * overall_conf - 0.2 * overall_std, 0.0, 1.0))
    return {
        "available": True,
        "overall_confidence_mean": overall_conf,
        "overall_confidence_std": overall_std,
        "overall_boundary_uncertainty_mean": overall_boundary,
        "consistency": consistency,
        "view_details": details,
        "constraint_count": int(len(constraints)),
    }


def collect_topology_signal(
    *,
    profile_signal: Mapping[str, Any],
    constraint_signal: Mapping[str, Any],
    extras: Mapping[str, Any],
    include_details: bool = True,
) -> dict[str, Any]:
    topology_hint = extras.get("topology", {}) if isinstance(extras, Mapping) else {}
    if not isinstance(topology_hint, Mapping):
        topology_hint = {}
    explicit_score = topology_hint.get("score")
    if isinstance(explicit_score, (int, float)) and np.isfinite(float(explicit_score)):
        result = {
            "available": True,
            "score": float(np.clip(float(explicit_score), 0.0, 1.0)),
            "complexity": float(profile_signal.get("complexity", 0.0)),
            "constraint_count": int(constraint_signal.get("constraint_count", 0)),
        }
        if include_details:
            result.update({"detail": "extras.topology", "details": dict(topology_hint)})
        return result

    complexity = float(profile_signal.get("complexity", 0.0))
    constraint_count = int(constraint_signal.get("constraint_count", 0))
    score = float(
        np.clip(
            1.0 - 0.25 * complexity - min(0.6, 0.15 * constraint_count),
            0.1,
            1.0,
        )
    )
    result = {
        "available": bool(profile_signal.get("available") or constraint_signal.get("available")),
        "score": score,
        "complexity": complexity,
        "constraint_count": constraint_count,
    }
    if include_details:
        result.update(
            {
                "detail": "profile_constraint_fallback",
                "details": {
                    "profile_complexity": complexity,
                    "constraint_count": constraint_count,
                    "topology_hint_keys": sorted(topology_hint),
                },
            }
        )
    return result


def compact_signal_summary(signals: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "surface": dict(signals.get("surface", {})),
        "profile": dict(signals.get("profile", {})),
        "constraints": dict(signals.get("constraints", {})),
        "uncertainty": dict(signals.get("uncertainty", {})),
        "topology": dict(signals.get("topology", {})),
    }
