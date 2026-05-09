"""Gaussian/ellipsoid proxy reconstruction backend helpers."""

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
except ImportError:  # pragma: no cover - package import path
    from .shape_program import ShapeNode, ShapeProgram, validate_shape_program


_ALLOWED_FAMILIES = {"gaussian", "gaussians", "ellipsoid", "ellipsoids"}
_ALLOWED_INITIALIZERS = {"farthest_point", "kmeans", "grid"}
_ALLOWED_RENDERERS = {"cpu_projected_ellipse", "gpu_splat"}
_DEFAULT_OBJECTIVE_WEIGHTS = {
    "coverage": 0.55,
    "complexity": 0.20,
    "topology": 0.10,
    "constraint": 0.08,
    "uncertainty": 0.07,
}


def validate_gaussian_ellipsoid_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate gaussian-ellipsoid config and return human-readable errors."""
    _, errors, _ = _normalize_gaussian_ellipsoid_config(config)
    return tuple(errors)


def run_gaussian_ellipsoid_proxy(request: object) -> CandidateResult:
    """Build editable Gaussian/ellipsoid proxies from target signals."""
    start = time.perf_counter()
    config = dict(getattr(request, "config", {}) or {})
    candidate_id = getattr(request, "candidate_id")
    backend_name = getattr(request, "backend_name", "gaussian_ellipsoid_proxy")
    target = getattr(request, "target")

    normalized, errors, warnings = _normalize_gaussian_ellipsoid_config(config)
    if errors:
        return CandidateResult(
            candidate_id=candidate_id,
            backend_name=backend_name,
            status="failed",
            errors=tuple(errors),
        )

    family = normalized["family"]
    requested_primitive_count = int(normalized["primitive_count"])
    requested_point_count = int(normalized["target_point_count"])

    signals = _collect_target_signals(target)
    surface_signal = signals["surface"]
    profile_signal = signals["profile"]
    constraint_signal = signals["constraints"]
    uncertainty_signal = signals["uncertainty"]
    topology_signal = signals["topology"]

    adaptive_point_count = _adaptive_point_count(
        requested_point_count,
        profile_signal=profile_signal,
        uncertainty_signal=uncertainty_signal,
        constraint_signal=constraint_signal,
        topology_signal=topology_signal,
        surface_signal=surface_signal,
    )
    adaptive_primitive_count = _adaptive_primitive_count(
        requested_primitive_count,
        profile_signal=profile_signal,
        constraint_signal=constraint_signal,
        topology_signal=topology_signal,
        uncertainty_signal=uncertainty_signal,
    )

    try:
        points, point_meta = target_surface_points(
            target,
            resolution=int(normalized["visual_hull_resolution"]),
            max_points=adaptive_point_count,
            chunk_size=normalized["chunk_size"],
        )
        seed_points = _prepare_initialization_points(
            points,
            method=str(normalized["initialization"]),
            primitive_count=adaptive_primitive_count,
            target_point_count=adaptive_point_count,
        )
        init_config = PrimitiveInitializationConfig(
            primitive_count=max(1, adaptive_primitive_count),
            target_point_count=max(1, len(seed_points)),
            min_radius=float(normalized["min_radius"]),
            covariance_floor=float(normalized["covariance_floor"]),
            kmeans_iterations=int(normalized["kmeans_iterations"]),
        )
        if family in {"ellipsoid", "ellipsoids"}:
            primitives = tuple(initialize_ellipsoids_from_points(seed_points, init_config))
        else:
            primitives = tuple(initialize_gaussians_from_points(seed_points, init_config))
    except Exception as exc:
        return CandidateResult(
            candidate_id=candidate_id,
            backend_name=backend_name,
            status="failed",
            errors=(str(exc),),
        )

    mesh_proxy = combine_primitive_meshes(primitives, resolution=20)
    mesh_topology = None
    mesh_topology_score = float(topology_signal.get("score", 0.4))
    editable_proxy = _editable_proxy_program_from_primitives(
        primitives,
        family=family,
        program_id=f"{candidate_id}_editable_proxy",
        sigma=float(normalized["editable_proxy_sigma"]),
    )
    editable_proxy_errors = validate_shape_program(editable_proxy)
    editable_proxy_summary = _editable_proxy_summary(
        editable_proxy,
        validation_errors=editable_proxy_errors,
        source_family=family,
    )
    if editable_proxy_errors:
        warnings.append(
            "editable proxy shape-program validation failed: "
            + "; ".join(editable_proxy_errors)
        )

    root = request.candidate_artifact_root()
    artifacts: dict[str, Any] = {}
    primitive_path = None
    mesh_path = None
    if root is not None:
        primitive_path = write_primitive_set(
            root / "primitives" / "gaussian-ellipsoid.json",
            primitives,
            metadata={
                "family": family,
                "signal_summary": _compact_signal_summary(signals),
                "point_meta": point_meta,
                "requested_primitive_count": requested_primitive_count,
                "requested_point_count": requested_point_count,
                "adaptive_point_count": adaptive_point_count,
                "adaptive_primitive_count": adaptive_primitive_count,
                "normalized_config": _compact_config_summary(normalized),
            },
        )
        artifacts["primitive_json"] = primitive_path
        editable_proxy_path = write_json(
            root / "shape-program" / "gaussian-ellipsoid-editable-proxy.json",
            editable_proxy.to_dict(),
        )
        artifacts["editable_proxy_shape_program"] = editable_proxy_path
        if bool(config.get("export_mesh_proxy", True)):
            mesh_path = write_obj(
                root / "mesh" / "gaussian-ellipsoid-proxy.obj",
                mesh_proxy,
                header=(f"candidate {candidate_id}", backend_name),
            )
            artifacts["mesh_obj"] = mesh_path

    if isinstance(mesh_proxy.vertices, tuple) and mesh_proxy.vertices:
        try:
            vertices, faces = mesh_arrays_from_object(mesh_proxy)
            if len(vertices) and len(faces):
                report = mesh_topology_report(vertices, faces)
                mesh_topology = report.to_dict()
                mesh_topology_score = float(report.topology_score)
                if root is not None:
                    topology_path = write_json(
                        root / "artifacts" / "gaussian-ellipsoid-topology.json",
                        mesh_topology,
                    )
                    artifacts["topology"] = topology_path
        except Exception:
            pass

    if normalized["max_radius"] is not None and normalized["max_radius"] < float(
        normalized["min_radius"]
    ):
        warnings.append("max_radius is smaller than min_radius")

    coverage = _coverage_score(points, primitives)
    complexity_penalty = min(
        1.0,
        len(primitives)
        / max(1.0, float(normalized["target_point_count"]) / 128.0 * float(normalized["primitive_count"])),
    )
    complexity_quality = max(0.0, 1.0 - complexity_penalty)

    uncertainty_consistency = float(uncertainty_signal.get("consistency", 0.75))
    constraint_score = float(constraint_signal.get("score", 1.0))
    topology_score = float(np.clip(mesh_topology_score, 0.0, 1.0))
    topology_source = _topology_source(topology_signal, mesh_topology)

    objective_terms = _objective_terms(
        coverage=coverage,
        complexity=complexity_quality,
        topology=topology_score,
        constraint=constraint_score,
        uncertainty=uncertainty_consistency,
    )
    baseline_terms = _baseline_objective_terms(
        profile_signal=profile_signal,
        constraint_signal=constraint_signal,
        uncertainty_signal=uncertainty_signal,
        topology_signal=topology_signal,
    )
    objective_weights = normalized["objective_weights"]
    objective_total = _objective_total(objective_terms, objective_weights)
    baseline_total = _objective_total(baseline_terms, objective_weights)
    objective_improvement = objective_total - baseline_total

    objective_history: list[dict[str, Any]] = [
        {
            "stage": "target_signals",
            "terms": _objective_terms(
                coverage=baseline_terms["coverage"],
                complexity=baseline_terms["complexity"],
                topology=baseline_terms["topology"],
                constraint=baseline_terms["constraint"],
                uncertainty=baseline_terms["uncertainty"],
            ),
            "weight_sum": float(sum(objective_weights.values())),
            "total": float(baseline_total),
            "topology_source": "signals",
        },
        {
            "stage": "adaptive_counts",
            "requested_primitive_count": requested_primitive_count,
            "requested_point_count": requested_point_count,
            "adaptive_primitive_count": adaptive_primitive_count,
            "adaptive_point_count": adaptive_point_count,
            "surface_point_count": int(point_meta.get("count", len(points))),
            "point_source": str(point_meta.get("source", "target_surface_points")),
        },
        {
            "stage": "fitted_proxy",
            "primitive_count": len(primitives),
            "complexity_penalty": complexity_penalty,
            "topology_source": topology_source,
            "terms": objective_terms,
            "total": float(objective_total),
        },
        {
            "stage": "editable_proxy_distillation",
            "shape_program_node_count": editable_proxy.node_count(),
            "validation_errors": list(editable_proxy_errors),
            "source_family": family,
        },
    ]
    objective_improvement_record = {
        "baseline_total": float(baseline_total),
        "objective_total": float(objective_total),
        "objective_improvement": float(objective_improvement),
        "objective_weights": dict(objective_weights),
        "history": objective_history,
    }

    if root is not None:
        objective_path = write_json(
            root / "artifacts" / "gaussian-ellipsoid-objectives.json",
            objective_improvement_record,
        )
        artifacts["objective_history"] = objective_path

    signal_path = write_json(
        root / "artifacts" / "gaussian-ellipsoid-signals.json",
        {
            "target_signals": signals,
            "normalized_config": _compact_config_summary(normalized),
            "adapted_counts": {
                "requested_primitive_count": requested_primitive_count,
                "requested_point_count": requested_point_count,
                "adaptive_primitive_count": adaptive_primitive_count,
                "adaptive_point_count": adaptive_point_count,
            },
        },
    ) if root is not None else None
    if signal_path is not None:
        artifacts["signal_summary"] = signal_path

    if adaptive_point_count != requested_point_count:
        warnings.append(
            f"adaptive target point budget changed from {requested_point_count} to {adaptive_point_count}"
        )
    if adaptive_primitive_count != requested_primitive_count:
        warnings.append(
            f"adaptive primitive count changed from {requested_primitive_count} to "
            f"{adaptive_primitive_count}"
        )
    if objective_improvement < 0.0:
        warnings.append(
            "objective total regressed relative to target signal baseline"
        )
    if not coverage:
        warnings.append("coverage signal is zero; surface proxy may be under-constrained")

    elapsed = time.perf_counter() - start
    metrics = CandidateMetrics(
        area_iou_min=coverage,
        area_iou_mean=coverage,
        boundary_iou_mean=coverage,
        topology_score=topology_score,
        uncertainty_consistency=uncertainty_consistency,
        constraint_score=constraint_score,
        editability_score=float(editable_proxy_summary["editability_score"]),
        complexity_penalty=complexity_penalty,
        elapsed_s=elapsed,
        extras={
            "family": family,
            "objective_terms": objective_terms,
            "baseline_objective_terms": baseline_terms,
            "objective_history": objective_history,
            "objective_improvement": objective_improvement_record,
            "primitive_count": len(primitives),
            "requested_primitive_count": requested_primitive_count,
            "requested_point_count": requested_point_count,
            "adaptive_primitive_count": adaptive_primitive_count,
            "adaptive_point_count": adaptive_point_count,
            "surface_signal": dict(surface_signal),
            "profile_signal": dict(profile_signal),
            "constraint_signal": dict(constraint_signal),
            "uncertainty_signal": dict(uncertainty_signal),
            "topology_signal": dict(topology_signal),
            "mesh_topology": mesh_topology,
            "editable_proxy": editable_proxy_summary,
            "normalized_config": _compact_config_summary(normalized),
            "warnings": tuple(warnings),
            "surface_points": point_meta,
        },
        per_view=_per_view_scores(uncertainty_signal),
    )

    return CandidateResult(
        candidate_id=candidate_id,
        backend_name=backend_name,
        status="success" if primitives else "skipped",
        primitive_path=primitive_path,
        mesh_path=mesh_path,
        metric_result=metrics,
        artifacts=artifacts,
        warnings=tuple(warnings),
        payload=primitives,
    )


def _normalize_gaussian_ellipsoid_config(
    config: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    if not isinstance(config, Mapping):
        return {}, [f"config must be a mapping, got {type(config)!r}"], []
    errors: list[str] = []
    warnings: list[str] = []
    normalized: dict[str, Any] = {}

    normalized["family"] = str(
        config.get(
            "family", config.get("proxy_family", config.get("primitive_family", "gaussian"))
        )
    ).strip().lower()
    if normalized["family"] not in _ALLOWED_FAMILIES:
        errors.append(
            f"family must be one of {sorted(_ALLOWED_FAMILIES)}, got {normalized['family']!r}"
        )

    primitive_count = _coerce_int(
        config.get("primitive_count", config.get("max_primitives", 24)),
        "primitive_count",
        min_value=1,
        max_value=1024,
        errors=errors,
        default=24,
    )
    target_point_count = _coerce_int(
        config.get("target_point_count", 4096),
        "target_point_count",
        min_value=1,
        max_value=65535,
        errors=errors,
        default=4096,
    )
    visual_hull_resolution = _coerce_int(
        config.get("visual_hull_resolution", config.get("resolution", 48)),
        "visual_hull_resolution",
        min_value=16,
        max_value=512,
        errors=errors,
        default=48,
    )
    min_radius = _coerce_float(
        config.get("min_radius", 1e-4),
        "min_radius",
        min_value=1e-9,
        errors=errors,
        default=1e-4,
    )
    max_radius = _coerce_float_or_none(config.get("max_radius"), "max_radius", errors=errors)
    if max_radius is not None and max_radius < min_radius:
        errors.append("max_radius must be >= min_radius")
    covariance_floor = _coerce_float(
        config.get("covariance_floor", 1e-4),
        "covariance_floor",
        min_value=0.0,
        errors=errors,
        default=1e-4,
    )
    kmeans_iterations = _coerce_int(
        config.get("kmeans_iterations", 8),
        "kmeans_iterations",
        min_value=1,
        max_value=64,
        errors=errors,
        default=8,
    )
    initialization = str(config.get("initialization", "kmeans")).strip().lower()
    if initialization not in _ALLOWED_INITIALIZERS:
        errors.append(
            f"initialization must be one of {sorted(_ALLOWED_INITIALIZERS)}, got {initialization!r}"
        )

    renderer = str(config.get("renderer", "cpu_projected_ellipse")).strip().lower()
    if renderer not in _ALLOWED_RENDERERS:
        errors.append(
            f"renderer must be one of {sorted(_ALLOWED_RENDERERS)}, got {renderer!r}"
        )

    chunk_size = _coerce_optional_int(
        config.get("chunk_size", None),
        "chunk_size",
        errors=errors,
        default=None,
    )
    if chunk_size is not None:
        normalized["chunk_size"] = int(chunk_size)
    else:
        normalized["chunk_size"] = None

    opacity_min = _coerce_float(
        config.get("opacity_min", 0.0),
        "opacity_min",
        min_value=0.0,
        max_value=1.0,
        errors=errors,
        default=0.0,
    )
    opacity_max = _coerce_float(
        config.get("opacity_max", 1.0),
        "opacity_max",
        min_value=0.0,
        max_value=1.0,
        errors=errors,
        default=1.0,
    )
    if opacity_max < opacity_min:
        errors.append("opacity_max must be >= opacity_min")

    objective_weights, objective_weight_warnings = _coerce_objective_weights(
        config.get("objective_weights"),
        errors=errors,
    )
    warnings.extend(objective_weight_warnings)

    normalized["primitive_count"] = primitive_count
    normalized["target_point_count"] = target_point_count
    normalized["visual_hull_resolution"] = visual_hull_resolution
    normalized["min_radius"] = min_radius
    normalized["max_radius"] = max_radius
    normalized["covariance_floor"] = covariance_floor
    normalized["kmeans_iterations"] = kmeans_iterations
    normalized["initialization"] = initialization
    normalized["renderer"] = renderer
    normalized["opacity_min"] = opacity_min
    normalized["opacity_max"] = opacity_max
    normalized["objective_weights"] = objective_weights
    normalized["export_mesh_proxy"] = bool(config.get("export_mesh_proxy", True))
    normalized["editable_proxy_sigma"] = _coerce_float(
        config.get("editable_proxy_sigma", 1.0),
        "editable_proxy_sigma",
        min_value=0.05,
        max_value=4.0,
        errors=errors,
        default=1.0,
    )
    return normalized, errors, warnings


def _coerce_int(
    value: Any,
    name: str,
    *,
    min_value: int,
    max_value: int,
    errors: list[str],
    default: int,
) -> int:
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        errors.append(f"{name} must be an integer, got {type(value)!r}")
        return default
    if parsed < min_value:
        errors.append(f"{name} must be >= {min_value}, got {parsed}")
    if parsed > max_value:
        errors.append(f"{name} must be <= {max_value}, got {parsed}")
    return parsed


def _coerce_optional_int(
    value: Any,
    name: str,
    *,
    errors: list[str],
    default: int | None = None,
) -> int | None:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        errors.append(f"{name} must be an integer, got {type(value)!r}")
        return default
    if parsed < 1:
        errors.append(f"{name} must be >= 1, got {parsed}")
        return default
    return parsed


def _coerce_float(
    value: Any,
    name: str,
    *,
    min_value: float,
    max_value: float | None = None,
    errors: list[str],
    default: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a float, got {type(value)!r}")
        return default
    if not np.isfinite(parsed):
        errors.append(f"{name} must be finite, got {parsed!r}")
        return default
    if parsed < min_value:
        errors.append(f"{name} must be >= {min_value}, got {parsed}")
    if max_value is not None and parsed > max_value:
        errors.append(f"{name} must be <= {max_value}, got {parsed}")
    return parsed


def _coerce_float_or_none(
    value: Any,
    name: str,
    errors: list[str],
) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a float or null, got {type(value)!r}")
        return None
    if not np.isfinite(parsed):
        errors.append(f"{name} must be finite or null, got {parsed!r}")
        return None
    return parsed


def _coerce_objective_weights(
    value: Any,
    errors: list[str],
) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    weights = dict(_DEFAULT_OBJECTIVE_WEIGHTS)
    if value is None:
        return weights, warnings
    if not isinstance(value, Mapping):
        errors.append(f"objective_weights must be a mapping, got {type(value)!r}")
        return weights, warnings
    for key in _DEFAULT_OBJECTIVE_WEIGHTS:
        raw = value.get(key)
        if raw is None:
            continue
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            errors.append(
                f"objective_weights[{key}] must be numeric, got {type(raw)!r}"
            )
            continue
        if not np.isfinite(parsed):
            errors.append(f"objective_weights[{key}] must be finite, got {parsed!r}")
            continue
        if parsed < 0.0:
            warnings.append(
                f"objective_weights[{key}] is negative ({parsed}); clamping to 0"
            )
            parsed = 0.0
        weights[key] = parsed
    return weights, warnings


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


def _adaptive_point_count(
    base_count: int,
    *,
    profile_signal: Mapping[str, Any],
    uncertainty_signal: Mapping[str, Any],
    constraint_signal: Mapping[str, Any],
    topology_signal: Mapping[str, Any],
    surface_signal: Mapping[str, Any],
) -> int:
    complexity = float(profile_signal.get("complexity", 0.0))
    constraint_penalty = min(1.0, 0.15 * int(constraint_signal.get("constraint_count", 0)))
    topology_headroom = 0.5 + 0.5 * float(topology_signal.get("score", 0.4))
    uncertainty_mean = float(uncertainty_signal.get("overall_confidence_mean", 0.75))
    surface_density = float(surface_signal.get("density_hint", 0.75))
    adapted = int(
        round(
            base_count
            * (0.7 + 0.5 * complexity)
            * (0.5 + 0.5 * uncertainty_mean)
            * topology_headroom
            * (1.0 - constraint_penalty * 0.35)
            * surface_density
        )
    )
    min_points = 64 if base_count >= 64 else max(16, base_count)
    max_points = max(base_count * 2, min_points)
    return int(np.clip(adapted, min_points, max_points))


def _adaptive_primitive_count(
    base_count: int,
    *,
    profile_signal: Mapping[str, Any],
    constraint_signal: Mapping[str, Any],
    topology_signal: Mapping[str, Any],
    uncertainty_signal: Mapping[str, Any],
) -> int:
    complexity = float(profile_signal.get("complexity", 0.0))
    uncertainty = float(uncertainty_signal.get("overall_confidence_mean", 0.75))
    topology = float(topology_signal.get("score", 0.4))
    constraint_count = int(constraint_signal.get("constraint_count", 0))
    adapted = int(
        round(
            base_count
            * (0.9 + 0.4 * complexity + 0.15 * (1.0 - topology))
            * (1.0 - 0.06 * constraint_count)
            * (0.8 + 0.2 * uncertainty)
        )
    )
    adapted = max(1, adapted)
    return int(np.clip(adapted, 1, max(1, base_count * 2)))


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


def _topology_source(topology_signal: Mapping[str, Any], report: Mapping[str, Any] | None) -> str:
    if report is not None:
        return "mesh"
    if topology_signal.get("detail") == "extras.topology":
        return "extras"
    return "signals"


def _compact_signal_summary(signals: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "surface": dict(signals.get("surface", {})),
        "profile": dict(signals.get("profile", {})),
        "constraints": dict(signals.get("constraints", {})),
        "uncertainty": dict(signals.get("uncertainty", {})),
        "topology": dict(signals.get("topology", {})),
    }


def _compact_config_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family": config.get("family"),
        "primitive_count": int(config.get("primitive_count", 0)),
        "target_point_count": int(config.get("target_point_count", 0)),
        "visual_hull_resolution": int(config.get("visual_hull_resolution", 0)),
        "initialization": str(config.get("initialization", "kmeans")),
        "renderer": str(config.get("renderer", "cpu_projected_ellipse")),
        "min_radius": float(config.get("min_radius", 0.0)),
        "max_radius": config.get("max_radius"),
        "opacity_range": (float(config.get("opacity_min", 0.0)), float(config.get("opacity_max", 1.0))),
        "covariance_floor": float(config.get("covariance_floor", 0.0)),
        "kmeans_iterations": int(config.get("kmeans_iterations", 0)),
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


def _prepare_initialization_points(
    points: np.ndarray,
    *,
    method: str,
    primitive_count: int,
    target_point_count: int,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points
    desired = max(int(primitive_count * 6), int(target_point_count))
    if method == "grid":
        return _sample_grid(points, desired)
    if method == "farthest_point":
        return _sample_farthest_points(points, desired)
    return points


def _sample_grid(points: np.ndarray, target_count: int) -> np.ndarray:
    if len(points) <= target_count:
        return points.copy()
    grid = int(np.ceil(len(points) / max(1, target_count)))
    return points[::grid]


def _sample_farthest_points(points: np.ndarray, target_count: int) -> np.ndarray:
    if len(points) <= target_count:
        return points.copy()
    indices = []
    selected = []
    centroid = points.mean(axis=0)
    start = int(np.argmax(np.linalg.norm(points - centroid, axis=1)))
    indices.append(start)
    selected.append(points[start])
    for _ in range(1, target_count):
        remaining = np.asarray(points)
        selected_points = np.asarray(selected)
        distances = np.min(
            np.linalg.norm(remaining[:, None, :] - selected_points[None, :, :], axis=2),
            axis=1,
        )
        next_idx = int(np.argmax(distances))
        indices.append(next_idx)
        selected.append(points[next_idx])
    selected_points = np.asarray(selected)
    if len(selected_points) <= 1:
        return points[:target_count]
    return selected_points


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


def _editable_proxy_program_from_primitives(
    primitives: tuple[object, ...],
    *,
    family: str,
    program_id: str,
    sigma: float,
) -> ShapeProgram:
    nodes = [
        _editable_proxy_node(
            primitive,
            index=index,
            family=family,
            sigma=sigma,
        )
        for index, primitive in enumerate(primitives)
    ]
    return ShapeProgram(
        schema_version="shape-program-v1",
        program_id=program_id,
        root_nodes=tuple(nodes),
        constraints=(),
        residual_patches=(),
        metadata={
            "source": "gaussian_ellipsoid_proxy_distillation",
            "family": family,
            "primitive_count": len(primitives),
            "sigma": sigma,
            "editable_output": True,
        },
    )


def _editable_proxy_node(
    primitive: object,
    *,
    index: int,
    family: str,
    sigma: float,
) -> ShapeNode:
    center, radii, rotation, density, confidence, source_type = _ellipsoid_proxy_terms(
        primitive,
        sigma=sigma,
    )
    return ShapeNode(
        node_id=f"editable_proxy_{index:03d}",
        operation="add",
        primitive_type="ellipsoid",
        name=f"{source_type.replace('_', ' ')} editable proxy {index:03d}",
        editable=True,
        parameters={
            "source_family": family,
            "source_primitive_type": source_type,
            "source_primitive_index": index,
            "location_x": float(center[0]),
            "location_y": float(center[1]),
            "location_z": float(center[2]),
            "width_world": float(2.0 * radii[0]),
            "depth_world": float(2.0 * radii[1]),
            "height_world": float(2.0 * radii[2]),
            "radius_x_world": float(radii[0]),
            "radius_y_world": float(radii[1]),
            "radius_z_world": float(radii[2]),
            "density": float(density),
            "opacity": float(density),
            "confidence": float(confidence),
            "rotation_row_major": [float(value) for value in rotation.reshape(-1)],
            "distillation": "gaussian_to_editable_ellipsoid",
        },
    )


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


def _editable_proxy_summary(
    program: ShapeProgram,
    *,
    validation_errors: Sequence[str],
    source_family: str,
) -> dict[str, Any]:
    valid = not validation_errors
    node_count = program.node_count()
    score = 0.0 if not valid else min(0.92, 0.72 + 0.20 * node_count / float(node_count + 4))
    return {
        "source": "gaussian_ellipsoid_proxy_distillation",
        "source_family": source_family,
        "shape_program": program.to_dict(),
        "node_count": node_count,
        "validation_errors": list(validation_errors),
        "editability_score": float(score),
        "editable_primitives": [
            node.primitive_type for node in program.root_nodes if node.editable
        ],
    }


def primitive_payload_summary(primitives: tuple[object, ...]) -> Mapping[str, object]:
    """Return a compact manifest-safe primitive summary."""
    return {
        "primitive_count": len(primitives),
        "types": [type(primitive).__name__ for primitive in primitives],
    }
