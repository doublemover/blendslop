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


def _objective_total(
    terms: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    total_weight = 0.0
    weighted_sum = 0.0
    for name in ("coverage", "complexity", "topology", "constraint", "uncertainty"):
        weight = float(weights.get(name, 0.0))
        total_weight += weight
        weighted_sum += weight * float(terms[name])
    if total_weight <= 0.0:
        return float(weighted_sum)
    return float(weighted_sum / total_weight)


def _objective_terms(
    *, coverage: float, complexity: float, topology: float, constraint: float, uncertainty: float
) -> dict[str, float]:
    return {
        "coverage": float(np.clip(coverage, 0.0, 1.0)),
        "complexity": float(np.clip(complexity, 0.0, 1.0)),
        "topology": float(np.clip(topology, 0.0, 1.0)),
        "constraint": float(np.clip(constraint, 0.0, 1.0)),
        "uncertainty": float(np.clip(uncertainty, 0.0, 1.0)),
    }


def _baseline_objective_terms(
    *,
    profile_signal: Mapping[str, Any],
    constraint_signal: Mapping[str, Any],
    uncertainty_signal: Mapping[str, Any],
    topology_signal: Mapping[str, Any],
) -> dict[str, float]:
    complexity = float(np.clip(profile_signal.get("complexity", 0.0), 0.0, 1.0))
    constraint_count = int(constraint_signal.get("constraint_count", 0))
    has_topology = bool(topology_signal.get("available"))
    topology = float(np.clip(topology_signal.get("score", 0.4 if has_topology else 0.25), 0.0, 1.0))
    uncertainty = float(
        np.clip(
            uncertainty_signal.get("consistency", uncertainty_signal.get("overall_confidence_mean", 0.75)),
            0.0,
            1.0,
        )
    )
    baseline_coverage = float(np.clip(0.35 + 0.45 * (1.0 - complexity), 0.0, 0.95))
    baseline_complexity = float(np.clip(1.0 / (1.0 + complexity + 0.12 * constraint_count), 0.0, 1.0))
    baseline_constraint = float(np.clip(constraint_signal.get("score", 1.0), 0.0, 1.0))
    return _objective_terms(
        coverage=baseline_coverage,
        complexity=baseline_complexity,
        topology=topology,
        constraint=baseline_constraint,
        uncertainty=uncertainty,
    )


def _topology_source(topology_signal: Mapping[str, Any], report: Mapping[str, Any] | None) -> str:
    if report is not None:
        return "mesh"
    if topology_signal.get("detail") == "extras.topology":
        return "extras"
    return "signals"


def _compact_config_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "proxy_variant": config.get('proxy_variant','initializer_only'),
        "proxy_fit_evaluations": config.get('proxy_fit_evaluations',192),
        "family": config.get("family"),
        "primitive_count": int(config.get("primitive_count", 0)),
        "target_point_count": int(config.get("target_point_count", 0)),
        "visual_hull_resolution": int(config.get("visual_hull_resolution", 0)),
        "distillation_resolution": int(config.get("distillation_resolution", 0)),
        "distillation_padding": float(config.get("distillation_padding", 0.0)),
        "initialization": str(config.get("initialization", "kmeans")),
        "renderer": str(config.get("renderer", "cpu_projected_ellipse")),
        "min_radius": float(config.get("min_radius", 0.0)),
        "max_radius": config.get("max_radius"),
        "opacity_range": (float(config.get("opacity_min", 0.0)), float(config.get("opacity_max", 1.0))),
        "covariance_floor": float(config.get("covariance_floor", 0.0)),
        "kmeans_iterations": int(config.get("kmeans_iterations", 0)),
        "include_bounds_proxy": bool(config.get("include_bounds_proxy", False)),
        "editable_proxy_sigma": float(config.get("editable_proxy_sigma", 1.0)),
    }


def _topology_score_from_profile_constraint_signal(
    profile_signal: Mapping[str, Any],
    constraint_signal: Mapping[str, Any],
    topology_hint: Mapping[str, Any] | None = None,
) -> float:
    profile_complexity = float(profile_signal.get("complexity", 0.0))
    constraint_count = int(constraint_signal.get("constraint_count", 0))
    topology_penalty = min(0.6, 0.15 * constraint_count)
    profile_bonus = 0.05 * float(profile_signal.get("hole_count", 0))
    if topology_hint and isinstance(topology_hint, Mapping):
        topology_penalty -= 0.05 * float(topology_hint.get("penalty", 0.0))
    raw = 1.0 - 0.25 * profile_complexity - topology_penalty + profile_bonus
    return float(np.clip(raw, 0.25, 1.0))


def _per_view_scores(
    uncertainty_signal: Mapping[str, Any],
) -> dict[str, Any]:
    per_view: dict[str, Any] = {}
    view_details = uncertainty_signal.get("view_details", {})
    for view, details in dict(view_details).items():
        mean_conf = float(details.get("mean_confidence", 0.75))
        boundary_uncertainty = float(details.get("boundary_uncertainty_mean", 0.0))
        confidence_score = float(np.clip(mean_conf * (1.0 - boundary_uncertainty), 0.0, 1.0))
        passed = mean_conf >= 0.35 and boundary_uncertainty <= 0.7
        per_view[str(view)] = {
            "required": True,
            "passed": passed,
            "pass": passed,
            "area_iou": confidence_score,
            "boundary_iou": confidence_score,
            "soft_iou": confidence_score,
            "signed_distance_loss": float(1.0 - confidence_score),
            "reason": "" if passed else "uncertainty confidence below gaussian proxy gate",
            "mean_confidence": mean_conf,
            "boundary_uncertainty_mean": boundary_uncertainty,
            "confidence_std": float(details.get("std_confidence", 0.0)),
            "constraint_count": int(uncertainty_signal.get("constraint_count", 0)),
        }
    return per_view


def _coverage_score(points: np.ndarray, primitives: tuple[object, ...]) -> float:
    if len(points) == 0 or not primitives:
        return 0.0
    sdf_rows = []
    for primitive in primitives:
        if hasattr(primitive, "sdf_batch"):
            sdf_rows.append(np.asarray(primitive.sdf_batch(points), dtype=float))
    if not sdf_rows:
        return 0.0
    nearest = np.min(np.abs(np.vstack(sdf_rows)), axis=0)
    scale = max(1e-6, float(np.percentile(np.linalg.norm(points, axis=1), 95)))
    return float(1.0 / (1.0 + np.mean(nearest) / scale))


def _ellipsoid_proxy_terms(
    primitive: object,
    *,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, str]:
    source_type = type(primitive).__name__
    proxy = primitive
    if hasattr(primitive, "to_ellipsoid") and callable(getattr(primitive, "to_ellipsoid")):
        proxy = primitive.to_ellipsoid(sigma=sigma)
    center = np.asarray(getattr(proxy, "center", (0.0, 0.0, 0.0)), dtype=float)
    radii = np.asarray(getattr(proxy, "radii", (1.0, 1.0, 1.0)), dtype=float)
    if center.shape != (3,):
        center = np.zeros((3,), dtype=float)
    if radii.shape != (3,):
        radii = np.ones((3,), dtype=float)
    radii = np.maximum(np.abs(radii), 1e-6)
    rotation = np.asarray(getattr(proxy, "rotation", np.eye(3)), dtype=float)
    if rotation.shape != (3, 3):
        rotation = np.eye(3, dtype=float)
    density = float(
        getattr(proxy, "density", getattr(primitive, "opacity", 1.0))
    )
    confidence = float(getattr(proxy, "confidence", getattr(primitive, "confidence", 1.0)))
    return center, radii, rotation, density, confidence, source_type
