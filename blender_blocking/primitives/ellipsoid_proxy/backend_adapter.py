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
