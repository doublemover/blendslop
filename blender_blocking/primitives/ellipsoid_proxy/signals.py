from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from metrics.topology import mesh_topology_report
from placement.resfit_initialization import (
    PrimitiveInitializationConfig,
    initialize_ellipsoids_from_points,
    initialize_gaussians_from_points,
)
from reconstruction.artifacts import write_json
from reconstruction.mesh_io import (
    combine_primitive_meshes,
    mesh_arrays_from_object,
    write_obj,
    write_primitive_set,
)
from reconstruction.point_cloud import target_surface_points
from reconstruction.types import CandidateMetrics, CandidateResult

try:
    from primitives.shape_program import ShapeNode, ShapeProgram, validate_shape_program
    from primitives.proxy_distillation import distill_proxy_field, write_proxy_field_npz
except ImportError:  # pragma: no cover - package import path
    from ..shape_program import ShapeNode, ShapeProgram, validate_shape_program
    from ..proxy_distillation import distill_proxy_field, write_proxy_field_npz

from .scoring import _topology_score_from_profile_constraint_signal


def _collect_target_signals(target: object) -> dict[str, Mapping[str, Any]]:
    constraints = tuple(getattr(target, "constraints", ()) or ())
    extras = getattr(target, "extras", {}) or {}
    if not isinstance(extras, Mapping):
        extras = {}
    profile_bands = getattr(target, "profile_bands", {})
    if not isinstance(profile_bands, Mapping):
        profile_bands = {}
    surface_signal = _collect_surface_signal(
        constraints=constraints,
        target_extras=extras,
        target_points=getattr(target, "points", None),
    )
    profile_signal = _collect_profile_signal(profile_bands)
    constraint_signal = _collect_constraint_signal(
        constraints=constraints,
        extras=extras,
    )
    uncertainty_signal = _collect_uncertainty_signal(constraints=constraints)
    topology_signal = _collect_topology_signal(
        profile_signal=profile_signal,
        constraint_signal=constraint_signal,
        extras=extras,
    )
    return {
        "surface": surface_signal,
        "profile": profile_signal,
        "constraints": constraint_signal,
        "uncertainty": uncertainty_signal,
        "topology": topology_signal,
    }


def _collect_surface_signal(
    *,
    constraints: Sequence[Any],
    target_extras: Mapping[str, Any],
    target_points: Any | None,
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
    return {
        "available": point_count > 0 or bool(constraints),
        "constraint_count": len(constraints),
        "surface_point_count": point_count,
        "density_hint": float(_estimate_surface_density(point_count, len(constraints))),
    }


def _estimate_surface_density(point_count: int, constraint_count: int) -> float:
    base = min(1.0, float(point_count) / 4096.0)
    if constraint_count <= 0:
        return base
    return float(np.clip(base * (1.0 - 0.08 * int(constraint_count)), 0.15, 1.0))


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
        }
    view_count = 0
    band_samples = 0
    interval_count = 0
    hole_count = 0
    widths: list[float] = []
    for view_bands in profile_bands.values():
        bands = tuple(view_bands or ())
        if not bands:
            continue
        view_count += 1
        band_samples += len(bands)
        for band in bands:
            intervals = tuple(getattr(band, "intervals", ()))
            holes = tuple(getattr(band, "holes", ()))
            interval_count += len(intervals)
            hole_count += len(holes)
            widths.append(float(getattr(band, "width_px", 0.0)))
    mean_width = float(np.mean(widths)) if widths else 0.0
    max_width = float(np.max(widths)) if widths else 0.0
    complexity = 0.0
    if band_samples > 0:
        complexity = float(np.clip((interval_count + 0.5 * hole_count) / float(band_samples), 0.0, 1.0))
    return {
        "available": view_count > 0,
        "view_count": view_count,
        "band_samples": band_samples,
        "interval_count": interval_count,
        "hole_count": hole_count,
        "mean_width": mean_width,
        "max_width": max_width,
        "complexity": complexity,
    }


def _collect_constraint_signal(
    constraints: Sequence[Any],
    extras: Mapping[str, Any],
) -> dict[str, Any]:
    payload = extras.get("constraint_payload", {}) if isinstance(extras, Mapping) else {}
    payload_constraints = tuple(payload.get("constraints", ())) if isinstance(payload, Mapping) else ()
    payload_count = len(payload_constraints)
    view_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for constraint in constraints:
        view = str(getattr(constraint, "view", "generic"))
        view_counts[view] = view_counts.get(view, 0) + 1
        constraint_kind = str(getattr(constraint, "kind", "generic"))
        if not constraint_kind or constraint_kind == "generic":
            constraint_kind = "generic"
        kind_counts[constraint_kind] = kind_counts.get(constraint_kind, 0) + 1
    constraint_count = len(constraints) + payload_count
    return {
        "available": bool(constraints or payload),
        "constraint_count": constraint_count,
        "payload_count": payload_count,
        "constraint_views": view_counts,
        "constraint_kinds": kind_counts,
        "score": float(np.clip(1.0 / (1.0 + 0.25 * constraint_count), 0.0, 1.0)),
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
            "constraint_count": 0,
        }
    means: list[float] = []
    boundary_means: list[float] = []
    details: dict[str, Mapping[str, Any]] = {}
    for constraint in constraints:
        view = str(getattr(constraint, "view", "unknown"))
        uncertainty = getattr(constraint, "uncertainty", None)
        if uncertainty is None:
            continue
        confidence = np.asarray(getattr(uncertainty, "confidence", ()))
        boundary = np.asarray(getattr(uncertainty, "boundary_uncertainty", ()))
        if confidence.size:
            confidence = np.clip(confidence.astype(float), 0.0, 1.0)
            means.append(float(confidence.mean()))
            view_entry = details.setdefault(view, {})
            view_entry.update(
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
    overall_conf = float(np.mean(means)) if means else 0.0
    overall_std = float(np.std(means)) if means else 0.0
    overall_boundary = float(np.mean(boundary_means)) if boundary_means else 0.0
    consistency = float(np.clip(0.6 + 0.4 * overall_conf - 0.2 * overall_std, 0.0, 1.0))
    return {
        "available": bool(means or boundary_means),
        "overall_confidence_mean": overall_conf,
        "overall_confidence_std": overall_std,
        "overall_boundary_uncertainty_mean": overall_boundary,
        "consistency": consistency,
        "view_details": details,
        "constraint_count": int(len(constraints)),
    }


def _collect_topology_signal(
    *,
    profile_signal: Mapping[str, Any],
    constraint_signal: Mapping[str, Any],
    extras: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(extras, Mapping):
        extras = {}
    topology_hint = extras.get("topology", {})
    if topology_hint is None:
        topology_hint = {}
    if not isinstance(topology_hint, Mapping):
        topology_hint = {}
    explicit_score = topology_hint.get("score")
    if isinstance(explicit_score, (int, float)) and np.isfinite(float(explicit_score)):
        return {
            "available": True,
            "score": float(np.clip(float(explicit_score), 0.0, 1.0)),
            "detail": "extras.topology",
            "details": dict(topology_hint),
        }
    return {
        "available": bool(profile_signal.get("available") or constraint_signal.get("available")),
        "score": _topology_score_from_profile_constraint_signal(
            profile_signal=profile_signal,
            constraint_signal=constraint_signal,
            topology_hint=topology_hint,
        ),
        "detail": "profile_constraint_fallback",
        "details": {
            "profile_complexity": profile_signal.get("complexity", 0.0),
            "constraint_count": constraint_signal.get("constraint_count", 0),
            "topology_hint_keys": sorted(topology_hint),
        },
    }


def _compact_signal_summary(signals: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "surface": dict(signals.get("surface", {})),
        "profile": dict(signals.get("profile", {})),
        "constraints": dict(signals.get("constraints", {})),
        "uncertainty": dict(signals.get("uncertainty", {})),
        "topology": dict(signals.get("topology", {})),
    }
