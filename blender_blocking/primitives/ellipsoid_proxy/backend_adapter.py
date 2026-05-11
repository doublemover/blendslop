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
    from primitives.analytic_primitives import AnisotropicGaussianPrimitive, EllipsoidPrimitive
    from primitives.shape_program import ShapeNode, ShapeProgram, validate_shape_program
    from primitives.proxy_distillation import distill_proxy_field, write_proxy_field_npz
except ImportError:  # pragma: no cover - package import path
    from ..analytic_primitives import AnisotropicGaussianPrimitive, EllipsoidPrimitive
    from ..shape_program import ShapeNode, ShapeProgram, validate_shape_program
    from ..proxy_distillation import distill_proxy_field, write_proxy_field_npz

from .config import _normalize_gaussian_ellipsoid_config
from .editable import _editable_proxy_program_from_primitives, _editable_proxy_summary
from .initialization import _adaptive_point_count, _adaptive_primitive_count, _prepare_initialization_points
from .scoring import (
    _baseline_objective_terms,
    _compact_config_summary,
    _coverage_score,
    _objective_terms,
    _objective_total,
    _per_view_scores,
    _topology_source,
)
from .signals import _collect_target_signals, _compact_signal_summary


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
        initialization_diagnostics = _initialization_diagnostics(
            seed_points,
            target=target,
            family=family,
            include_bounds_proxy=bool(normalized["include_bounds_proxy"]),
            min_radius=float(normalized["min_radius"]),
            opacity=float(normalized["opacity_max"]),
        )
        primitives = _prepend_bounds_proxy(
            primitives,
            family=family,
            diagnostics=initialization_diagnostics,
            primitive_limit=adaptive_primitive_count,
        )
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
            root / "p" / "gauss.json",
            primitives,
            metadata={
                "family": family,
                "signal_summary": _compact_signal_summary(signals),
                "point_meta": point_meta,
                "requested_primitive_count": requested_primitive_count,
                "requested_point_count": requested_point_count,
                "adaptive_point_count": adaptive_point_count,
                "adaptive_primitive_count": adaptive_primitive_count,
                "initialization_diagnostics": initialization_diagnostics,
                "normalized_config": _compact_config_summary(normalized),
            },
        )
        artifacts["primitive_json"] = primitive_path
        editable_proxy_path = write_json(
            root / "sp" / "gauss-proxy.json",
            editable_proxy.to_dict(),
        )
        artifacts["editable_proxy_shape_program"] = editable_proxy_path
        if bool(config.get("export_mesh_proxy", True)):
            mesh_path = write_obj(
                root / "m" / "gauss.obj",
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
                        root / "h" / "topology.json",
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
    distillation_report, distillation_grid, distillation_sdf, distillation_occupancy = (
        distill_proxy_field(
            primitives,
            target_points=points,
            resolution=int(normalized["distillation_resolution"]),
            padding=float(normalized["distillation_padding"]),
        )
    )
    if root is not None:
        distillation_path = write_proxy_field_npz(
            root / "vol" / "distill.npz",
            report=distillation_report,
            points=distillation_grid,
            sdf=distillation_sdf,
            occupancy=distillation_occupancy,
        )
        artifacts["proxy_distillation_npz"] = distillation_path
        artifacts["proxy_distillation"] = write_json(
            root / "h" / "distill.json",
            distillation_report.to_dict(),
        )
    complexity_penalty = min(
        1.0,
        len(primitives)
        / max(1.0, float(normalized["target_point_count"]) / 128.0 * float(normalized["primitive_count"])),
    )
    complexity_quality = max(0.0, 1.0 - complexity_penalty)

    uncertainty_consistency = float(uncertainty_signal.get("consistency", 0.75))
    constraint_score = float(constraint_signal.get("score", 1.0))
    editable_component_score = float(
        editable_proxy_summary.get("component_sanity_score", 0.0) or 0.0
    )
    topology_score = float(
        np.clip(max(mesh_topology_score, editable_component_score), 0.0, 1.0)
    )
    topology_source = _topology_source(topology_signal, mesh_topology)
    if editable_component_score > mesh_topology_score:
        topology_source = "editable_proxy_program"

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
            "stage": "bounds_intersection_seed",
            "diagnostics": initialization_diagnostics,
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
            "field_distillation": distillation_report.to_dict(),
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
            root / "h" / "objectives.json",
            objective_improvement_record,
        )
        artifacts["objective_history"] = objective_path

    signal_path = write_json(
        root / "h" / "signals.json",
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
    per_view_scores = _per_view_scores(uncertainty_signal)
    transform_diagnostics = _proxy_transform_diagnostics(
        target,
        per_view_scores=per_view_scores,
        coverage=coverage,
    )

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
            "proxy_distillation": distillation_report.to_dict(),
            "proxy_render_namespace": "backend_proxy_only",
            "transform_diagnostics": transform_diagnostics,
            "initialization_diagnostics": initialization_diagnostics,
            "render_proxy_disagreement_policy": {
                "high_proxy_iou_floor": 0.9,
                "catastrophic_render_iou_floor": 0.2,
                "action": "require_render_iou_preflight_before_parameter_fanout",
            },
            "normalized_config": _compact_config_summary(normalized),
            "warnings": tuple(warnings),
            "surface_points": point_meta,
        },
        per_view=per_view_scores,
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


def _initialization_diagnostics(
    points: np.ndarray,
    *,
    target: Any,
    family: str,
    include_bounds_proxy: bool,
    min_radius: float,
    opacity: float,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "bounds_proxy": {
            "enabled": False,
            "reason": "disabled" if not include_bounds_proxy else "insufficient_points",
        },
        "source_point_count": int(len(points)),
    }
    if len(points) == 0:
        return diagnostics
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    point_center = (mins + maxs) * 0.5
    point_radii = np.maximum((maxs - mins) * 0.5, float(min_radius))
    diagnostics["source_bounds"] = {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "center": point_center.tolist(),
        "radii": point_radii.tolist(),
        "source": "surface_points",
    }
    bbox_seed = _bbox_intersection_seed(target, min_radius=float(min_radius))
    if bbox_seed:
        diagnostics["bbox_intersection_seed"] = bbox_seed
    if not include_bounds_proxy:
        return diagnostics
    seed = bbox_seed or diagnostics["source_bounds"]
    center = np.asarray(seed.get("center", point_center), dtype=float)
    radii = np.asarray(seed.get("radii", point_radii), dtype=float)
    diagnostics["bounds_proxy"] = {
        "enabled": True,
        "family": family,
        "center": center.tolist(),
        "radii": radii.tolist(),
        "opacity": float(np.clip(opacity, 0.0, 1.0)),
        "reason": str(seed.get("reason", "surface_point_bounds_seed")),
        "source": str(seed.get("source", "surface_points")),
    }
    return diagnostics


def _bbox_intersection_seed(target: Any, *, min_radius: float) -> dict[str, Any]:
    bounds = getattr(target, "bounds", None)
    if bounds is not None:
        try:
            mins = np.asarray(
                [bounds.min_x, bounds.min_y, bounds.min_z],
                dtype=float,
            )
            maxs = np.asarray(
                [bounds.max_x, bounds.max_y, bounds.max_z],
                dtype=float,
            )
        except Exception:
            mins = np.empty((0,), dtype=float)
            maxs = np.empty((0,), dtype=float)
        if mins.shape == (3,) and maxs.shape == (3,) and np.all(maxs > mins):
            return _bounds_seed_payload(
                mins,
                maxs,
                source="target_bounds",
                reason="target_bounds_seed",
            )

    bbox_bounds = _bounds_from_constraint_bboxes(target, min_radius=min_radius)
    if bbox_bounds is None:
        return {}
    mins, maxs, details = bbox_bounds
    payload = _bounds_seed_payload(
        mins,
        maxs,
        source="per_view_bbox_intersection",
        reason="per_view_bbox_intersection_seed",
    )
    payload["view_bboxes"] = details
    return payload


def _bounds_seed_payload(
    mins: np.ndarray,
    maxs: np.ndarray,
    *,
    source: str,
    reason: str,
) -> dict[str, Any]:
    center = (mins + maxs) * 0.5
    radii = np.maximum((maxs - mins) * 0.5, 1.0e-8)
    return {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "center": center.tolist(),
        "radii": radii.tolist(),
        "source": source,
        "reason": reason,
    }


def _bounds_from_constraint_bboxes(
    target: Any,
    *,
    min_radius: float,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]] | None:
    constraints = getattr(target, "constraints", ()) or ()
    by_view: dict[str, Any] = {}
    for constraint in constraints:
        bbox = getattr(constraint, "bbox", None)
        view = str(getattr(constraint, "view", "")).lower().strip()
        if view and bbox is not None:
            by_view[view] = bbox
    if not by_view:
        return None

    scale = _target_unit_scale(target)
    front = by_view.get("front")
    side = by_view.get("side")
    top = by_view.get("top")
    width = _bbox_extent(front, "width", scale)
    depth = _bbox_extent(side, "width", scale)
    height_values = [
        value
        for value in (
            _bbox_extent(front, "height", scale),
            _bbox_extent(side, "height", scale),
        )
        if value is not None
    ]
    height = max(height_values) if height_values else None
    if top is not None:
        top_width = _bbox_extent(top, "width", scale)
        top_depth = _bbox_extent(top, "height", scale)
        width = width if width is not None else top_width
        depth = depth if depth is not None else top_depth
    if width is None and depth is not None:
        width = depth
    if depth is None and width is not None:
        depth = width
    if height is None and (width is not None or depth is not None):
        height = max(float(width or 0.0), float(depth or 0.0), scale)
    if width is None or depth is None or height is None:
        return None
    width = max(float(width), float(min_radius) * 2.0)
    depth = max(float(depth), float(min_radius) * 2.0)
    height = max(float(height), float(min_radius) * 2.0)
    mins = np.asarray([-width * 0.5, -depth * 0.5, 0.0], dtype=float)
    maxs = np.asarray([width * 0.5, depth * 0.5, height], dtype=float)
    details = {
        view: {
            "x0": float(getattr(bbox, "x0", 0.0)),
            "y0": float(getattr(bbox, "y0", 0.0)),
            "x1": float(getattr(bbox, "x1", 0.0)),
            "y1": float(getattr(bbox, "y1", 0.0)),
            "width": float(getattr(bbox, "width", 0.0)),
            "height": float(getattr(bbox, "height", 0.0)),
        }
        for view, bbox in sorted(by_view.items())
    }
    return mins, maxs, details


def _bbox_extent(bbox: Any, field: str, scale: float) -> float | None:
    if bbox is None:
        return None
    try:
        value = float(getattr(bbox, field))
    except (TypeError, ValueError):
        return None
    if value <= 0.0:
        return None
    return value * scale


def _target_unit_scale(target: Any) -> float:
    extras = getattr(target, "extras", {}) or {}
    if isinstance(extras, Mapping):
        for key in ("unit_scale", "bbox_unit_scale"):
            try:
                value = float(extras.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0.0:
                return value
    return 0.01


def _prepend_bounds_proxy(
    primitives: Sequence[object],
    *,
    family: str,
    diagnostics: Mapping[str, Any],
    primitive_limit: int,
) -> tuple[object, ...]:
    bounds_proxy = diagnostics.get("bounds_proxy")
    if not isinstance(bounds_proxy, Mapping) or not bounds_proxy.get("enabled"):
        return tuple(primitives)
    center = np.asarray(bounds_proxy.get("center", (0.0, 0.0, 0.0)), dtype=float)
    radii = np.asarray(bounds_proxy.get("radii", (1.0, 1.0, 1.0)), dtype=float)
    if center.shape != (3,) or radii.shape != (3,):
        return tuple(primitives)
    if str(family).lower().strip() in {"ellipsoid", "ellipsoids"}:
        proxy: object = EllipsoidPrimitive(
            center=center,
            radii=radii,
            density=1.0,
            confidence=1.0,
        )
    else:
        proxy = AnisotropicGaussianPrimitive(
            center=center,
            covariance=np.diag(np.maximum(radii, 1e-6) ** 2),
            opacity=float(bounds_proxy.get("opacity", 1.0) or 1.0),
            semantic_role="bounds_intersection_seed",
            confidence=1.0,
        )
    limit = max(1, int(primitive_limit))
    return (proxy, *tuple(primitives)[: max(0, limit - 1)])


def _proxy_transform_diagnostics(
    target: Any,
    *,
    per_view_scores: Mapping[str, Any],
    coverage: float,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "projected_bbox_ratios": {},
        "centroid_deltas": {},
        "axis_permutation_best_candidate": "not_evaluated_without_render_iou",
        "top_view_footprint_failure": False,
        "proxy_coverage": float(coverage),
    }
    constraints = getattr(target, "constraints", ()) or ()
    for constraint in constraints:
        view = str(getattr(constraint, "view", ""))
        bbox = getattr(constraint, "bbox", None)
        score = per_view_scores.get(view, {}) if isinstance(per_view_scores, Mapping) else {}
        confidence = (
            float(score.get("area_iou", 0.0))
            if isinstance(score, Mapping)
            else 0.0
        )
        if bbox is not None:
            width = float(getattr(bbox, "width", 0.0) or 0.0)
            height = float(getattr(bbox, "height", 0.0) or 0.0)
            diagnostics["projected_bbox_ratios"][view] = {
                "target_width": width,
                "target_height": height,
                "confidence_scaled_area": float(width * height * confidence),
            }
            diagnostics["centroid_deltas"][view] = {
                "x": 0.0,
                "y": 0.0,
                "status": "not_evaluated_without_render_iou",
            }
        if view == "top" and confidence < 0.35:
            diagnostics["top_view_footprint_failure"] = True
    return diagnostics
