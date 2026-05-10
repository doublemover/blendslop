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

from .profiles import _collect_profile_rows, _dominant_interval


def _collect_target_signals(target: object) -> dict[str, Mapping[str, Any]]:
    return {
        "surface": {
            "target_views": tuple(
                str(getattr(constraint, "view", ""))
                for constraint in tuple(getattr(target, "constraints", ()))
            ),
        },
        "profile": _collect_profile_signal(getattr(target, "profile_bands", {})),
        "constraints": _collect_constraint_signal(
            tuple(getattr(target, "constraints", ())),
            getattr(target, "extras", {}) or {},
        ),
        "uncertainty": _collect_uncertainty_signal(
            tuple(getattr(target, "constraints", ()))
        ),
        "topology": _collect_topology_signal(
            _collect_profile_signal(getattr(target, "profile_bands", {})),
            _collect_constraint_signal(
                tuple(getattr(target, "constraints", ())),
                getattr(target, "extras", {}) or {},
            ),
        ),
    }


def _collect_profile_signal(profile_bands: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(profile_bands, Mapping) or not profile_bands:
        return {
            "available": False,
            "view_count": 0,
            "band_samples": 0,
            "interval_count": 0,
            "hole_count": 0,
            "mean_width": 0.0,
            "max_width": 0.0,
            "complexity": 0.0,
            "rows": (),
        }

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
            interval = _dominant_interval(intervals)
            if interval is not None:
                interval_width = float(interval.width)
                if interval_width > 0.0:
                    center_px = float(interval.center) if hasattr(interval, "center") else 0.0
                    rows.append(
                        {
                            "view": str(view),
                            "t": float(getattr(band, "t", 0.0)),
                            "width_px": max(interval_width, 1e-6),
                            "center_x_px": center_px,
                            "reference_width_px": float(width_px),
                            "confidence": float(
                                getattr(interval, "confidence", getattr(band, "confidence", 1.0))
                            ),
                            "index": index,
                        }
                    )
    if not widths:
        return {
            "available": False,
            "view_count": view_count,
            "band_samples": band_samples,
            "interval_count": interval_count,
            "hole_count": hole_count,
            "mean_width": 0.0,
            "max_width": 0.0,
            "complexity": 0.0,
            "rows": (),
        }

    widths_arr = np.asarray(widths, dtype=float)
    mean_width = float(np.mean(widths_arr))
    max_width = float(np.max(widths_arr))
    complexity = 0.0
    if band_samples > 0:
        complexity = float(
            np.clip((interval_count + 0.5 * hole_count) / float(band_samples), 0.0, 1.0)
        )
    return {
        "available": True,
        "view_count": view_count,
        "band_samples": band_samples,
        "interval_count": interval_count,
        "hole_count": hole_count,
        "mean_width": mean_width,
        "max_width": max_width,
        "complexity": complexity,
        "rows": tuple(rows),
    }


def _collect_uncertainty_signal(constraints: Sequence[Any]) -> dict[str, Any]:
    if not constraints:
        return {
            "available": False,
            "overall_confidence_mean": 1.0,
            "overall_confidence_std": 0.0,
            "overall_boundary_uncertainty_mean": 0.0,
            "consistency": 0.75,
            "view_details": {},
        }

    means: list[float] = []
    boundary_means: list[float] = []
    details: dict[str, Any] = {}
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
            details.setdefault(view, {})["mean_confidence"] = float(confidence.mean())
            details.setdefault(view, {})["max_confidence"] = float(confidence.max())
            details[view]["std_confidence"] = float(confidence.std())
        if boundary.size:
            boundary = np.clip(boundary.astype(float), 0.0, 1.0)
            boundary_means.append(float(boundary.mean()))
            details.setdefault(view, {})["boundary_uncertainty_mean"] = float(
                boundary.mean()
            )
            details.setdefault(view, {})["boundary_uncertainty_max"] = float(boundary.max())
            details[view]["boundary_uncertainty_min"] = float(boundary.min())

    if not means and not boundary_means:
        return {
            "available": False,
            "overall_confidence_mean": 1.0,
            "overall_confidence_std": 0.0,
            "overall_boundary_uncertainty_mean": 0.0,
            "consistency": 0.75,
            "view_details": {},
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
    }


def _collect_constraint_signal(
    constraints: Sequence[Any],
    extras: Mapping[str, Any],
) -> dict[str, Any]:
    payload = extras.get("constraint_payload", {}) if isinstance(extras, Mapping) else {}
    constraint_count = len(constraints)
    view_counts: dict[str, int] = {}
    for constraint in constraints:
        view_counts[str(getattr(constraint, "view", "generic"))] = (
            view_counts.get(str(getattr(constraint, "view", "generic")), 0) + 1
        )
    payload_count = (
        len(payload.get("constraints", ()))
        if isinstance(payload, Mapping) and payload.get("constraints") is not None
        else 0
    )
    score = float(np.clip(1.0 / (1.0 + 0.25 * constraint_count), 0.0, 1.0))
    return {
        "available": bool(constraint_count or payload),
        "constraint_count": constraint_count,
        "constraint_payload_count": payload_count,
        "constraint_views": view_counts,
        "score": score,
    }


def _collect_topology_signal(
    profile_signal: Mapping[str, Any],
    constraint_signal: Mapping[str, Any],
) -> dict[str, Any]:
    complexity = float(profile_signal.get("complexity", 0.0))
    constraint_count = int(constraint_signal.get("constraint_count", 0))
    base = 1.0 - 0.25 * complexity - min(0.6, 0.15 * constraint_count)
    return {
        "score": float(np.clip(base, 0.1, 1.0)),
        "complexity": complexity,
        "constraint_count": constraint_count,
    }
