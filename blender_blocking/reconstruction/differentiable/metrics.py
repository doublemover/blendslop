from __future__ import annotations

from typing import Any, Mapping

from ..types import CandidateMetrics

from .target_adapter import (
    _candidate_per_view_metrics,
    _mean_candidate_metric,
    _min_candidate_metric,
)


def build_differentiable_candidate_metrics(
    *,
    elapsed_s: float,
    optimized_primitives: tuple[Any, ...],
    point_meta: Mapping[str, Any],
    render_batch: Any,
    loss: Any,
    baseline_loss: Any,
    initial_loss: Any,
    topology_score: float,
    topology_penalty: float,
    topology_payload: Mapping[str, Any],
    objective_improvement: float,
    zero_baseline_improvement: float,
    objective_improvement_record: Mapping[str, Any],
    objective_history: list[dict[str, Any]],
    optimization_summary: Mapping[str, Any],
    optimization_history: list[dict[str, object]],
    target_view_weights: Mapping[str, float],
    view_signal_details: Mapping[str, Any],
    target_signal_warnings: tuple[str, ...] | list[str],
    history_payload: list[dict[str, Any]],
    config_warnings: tuple[str, ...] | list[str],
    artifacts: Mapping[str, Any],
    backend_choice: str,
    initialization_diagnostics: Mapping[str, Any],
    boundary_sdf_improvement: Mapping[str, Any],
    mesh_proxy_scale: float,
) -> tuple[CandidateMetrics, Mapping[str, Mapping[str, Any]], int]:
    area_iou_mean = 1.0 - float(loss.terms.get("area_iou", 1.0))
    soft_iou_mean = 1.0 - float(loss.terms.get("soft_iou", 1.0))
    candidate_per_view = _candidate_per_view_metrics(loss.per_view)
    area_iou_mean = _mean_candidate_metric(
        candidate_per_view,
        "area_iou",
        fallback=max(0.0, area_iou_mean),
    )
    area_iou_min = _min_candidate_metric(
        candidate_per_view,
        "area_iou",
        fallback=max(0.0, min(area_iou_mean, soft_iou_mean)),
    )
    boundary_iou_mean = _mean_candidate_metric(
        candidate_per_view,
        "boundary_iou",
        fallback=max(0.0, soft_iou_mean),
    )
    failed_required_views = sum(
        1
        for payload in candidate_per_view.values()
        if bool(payload.get("required", True)) and not bool(payload.get("passed", False))
    )
    gate_summary = {
        "required_view_count": sum(
            1 for payload in candidate_per_view.values() if bool(payload.get("required", True))
        ),
        "failed_required_views": failed_required_views,
        "min_area_iou": max(0.0, area_iou_min),
        "mean_area_iou": max(0.0, area_iou_mean),
        "mean_boundary_iou": max(0.0, boundary_iou_mean),
        "mean_soft_iou": _mean_candidate_metric(
            candidate_per_view,
            "soft_iou",
            fallback=max(0.0, soft_iou_mean),
        ),
        "boundary_loss_improvement": float(
            boundary_sdf_improvement.get("boundary_loss_improvement", 0.0) or 0.0
        ),
        "signed_distance_loss_improvement": float(
            boundary_sdf_improvement.get("signed_distance_loss_improvement", 0.0)
            or 0.0
        ),
        "boundary_or_sdf_improved": bool(
            boundary_sdf_improvement.get("boundary_or_sdf_improved")
        ),
    }
    metrics = CandidateMetrics(
        area_iou_min=max(0.0, area_iou_min),
        area_iou_mean=max(0.0, area_iou_mean),
        boundary_iou_mean=max(0.0, boundary_iou_mean),
        topology_score=topology_score,
        topology_penalty=topology_penalty,
        editability_score=0.65,
        complexity_penalty=min(1.0, len(optimized_primitives) / 96.0),
        elapsed_s=elapsed_s,
        per_view=candidate_per_view,
        extras={
            "backend": backend_choice,
            "loss_total": loss.total,
            "loss_terms": loss.terms,
            "loss_warnings": loss.warnings,
            "primitive_count": len(optimized_primitives),
            "surface_points": point_meta,
            "render_metadata": dict(getattr(render_batch, "metadata", {})),
            "initialization_diagnostics": dict(initialization_diagnostics),
            "mesh_proxy_scale": float(mesh_proxy_scale),
            "gate_summary": gate_summary,
            "topology": dict(topology_payload),
            "baseline_loss": dict(baseline_loss.terms),
            "baseline_total": float(baseline_loss.total),
            "baseline_warnings": tuple(baseline_loss.warnings),
            "initial_loss": dict(initial_loss.terms),
            "initial_total": float(initial_loss.total),
            "initial_warnings": tuple(initial_loss.warnings),
            "objective_total": float(loss.total),
            "objective_improvement": float(objective_improvement),
            "boundary_sdf_improvement": dict(boundary_sdf_improvement),
            "zero_baseline_improvement": float(zero_baseline_improvement),
            "objective_improvement_record": dict(objective_improvement_record),
            "objective_history": objective_history,
            "optimization": dict(optimization_summary),
            "optimization_history": optimization_history,
            "view_signal_weights": dict(target_view_weights),
            "view_signal_details": dict(view_signal_details),
            "view_signal_warnings": tuple(target_signal_warnings),
            "history": history_payload,
            "objective_history_artifact": str(artifacts.get("objective_history", "")),
            "history_file": str(artifacts.get("refinement_history", "")),
            "validation_warnings": tuple(config_warnings),
        },
    )
    return metrics, candidate_per_view, failed_required_views
