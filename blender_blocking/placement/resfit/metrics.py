from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .config import _pipeline_config_summary
from .profiles import _build_per_view_profile_summary

try:
    from reconstruction.types import CandidateMetrics
except ImportError:  # pragma: no cover - package import path
    from ...reconstruction.types import CandidateMetrics


def build_resfit_candidate_metrics(
    *,
    elapsed_s: float,
    result: Any,
    primitive_family: str,
    mesh_metadata: Mapping[str, Any],
    topology_payload: Mapping[str, Any],
    uncertainty_signal: Mapping[str, Any],
    profile_rows: Sequence[Mapping[str, Any]],
    pipeline_config: Any,
    history_records: Sequence[Mapping[str, Any]],
    surface_meta: Mapping[str, Any],
    occupied_meta: Mapping[str, Any],
    signal_summary: Mapping[str, Any],
    initial_primitives: Sequence[Any] | None,
    max_runtime_s: float | None,
    max_objective_evaluations: int | None,
) -> tuple[CandidateMetrics, Mapping[str, Any]]:
    improved = result.initial_loss.total - result.final_loss.total
    improvement_ratio = (
        improved / result.initial_loss.total if result.initial_loss.total > 0.0 else 0.0
    )
    objective_topology_score = float(
        np.clip(1.0 - result.final_loss.terms.get("topology_penalty", 0.0), 0.0, 1.0)
    )
    topology_score = float(
        max(
            objective_topology_score,
            float(topology_payload.get("topology_score", 0.0) or 0.0),
        )
    )
    constraint_score = float(
        np.clip(1.0 - result.final_loss.terms.get("constraint_penalty", 0.0), 0.0, 1.0)
    )
    uncertainty_consistency = float(
        np.clip(uncertainty_signal.get("consistency", 0.75), 0.0, 1.0)
    )
    surface_score = float(result.final_loss.terms.get("surface_residual", 0.0))
    silhouette_score = float(result.final_loss.terms.get("silhouette", 0.0))
    surface_proxy_iou = float(1.0 / (1.0 + surface_score))
    silhouette_proxy_iou = float(1.0 / (1.0 + silhouette_score))
    area_iou = min(surface_proxy_iou, silhouette_proxy_iou)
    boundary_iou = silhouette_proxy_iou
    per_view = _build_per_view_profile_summary(profile_rows)
    budget_limited = result.optimization_termination_reason in {
        "elapsed_time_budget",
        "objective_evaluation_budget",
    }
    accepted_moves = sum(int(record.get("accepted_moves", 0) or 0) for record in history_records)
    rejected_moves = sum(int(record.get("rejected_moves", 0) or 0) for record in history_records)
    attempt_count = len(result.attempts)
    noop_attempt_count = sum(1 for attempt in result.attempts if _attempt_is_noop(attempt))
    all_attempts_noop = attempt_count > 0 and noop_attempt_count == attempt_count
    fail_reason = ""
    if not result.primitives:
        fail_reason = "no_primitives_emitted"
    elif improved <= 0.0 and accepted_moves <= 0:
        fail_reason = "optimizer_noop_objective_did_not_improve"
    elif improved <= 0.0:
        fail_reason = "objective_did_not_improve"
    elif accepted_moves <= 0:
        fail_reason = "optimizer_noop"
    budget_outcome = (
        "accepted_after_improvement"
        if budget_limited and improved > 0.0 and result.primitives
        else "incomplete"
        if budget_limited
        else "not_budget_limited"
    )

    metric = CandidateMetrics(
        area_iou_min=area_iou,
        area_iou_mean=area_iou,
        boundary_iou_mean=boundary_iou,
        topology_score=topology_score,
        uncertainty_consistency=uncertainty_consistency,
        constraint_score=constraint_score,
        editability_score=0.9,
        complexity_penalty=min(1.0, len(result.primitives) / 64.0),
        elapsed_s=elapsed_s,
        extras={
            "family": primitive_family,
            "primitive_count": len(result.primitives),
            "primitives": {
                "count": len(result.primitives),
                "family": primitive_family,
                "items": [
                    primitive.to_dict()
                    for primitive in result.primitives
                    if hasattr(primitive, "to_dict")
                ],
            },
            "mesh": dict(mesh_metadata),
            "topology": dict(topology_payload),
            "editability": {
                "object_hierarchy_score": min(1.0, 0.55 + 0.05 * len(result.primitives)),
                "primitive_score": 0.92 if result.primitives else 0.0,
                "modifier_score": 0.15,
                "mesh_density_score": max(
                    0.0,
                    1.0 - min(1.0, len(result.primitives) / 96.0),
                ),
                "semantic_part_score": min(1.0, 0.45 + 0.04 * len(result.primitives)),
                "topology_score": topology_score,
                "export_roundtrip_score": 0.0,
                "warnings": ["export round-trip not checked by primitive backend"],
                "metadata": {
                    "primitive_family": primitive_family,
                    "primitive_count": len(result.primitives),
                    "mesh_proxy": dict(mesh_metadata),
                },
            },
            "objective": {
                "initial_total": result.initial_loss.total,
                "final_total": result.final_loss.total,
                "objective_before": result.initial_loss.total,
                "objective_after": result.final_loss.total,
                "improvement": improved,
                "improvement_ratio": improvement_ratio,
                "improved": improved > 0.0,
                "accepted_move_count": accepted_moves,
                "rejected_move_count": rejected_moves,
                "attempt_count": attempt_count,
                "noop_attempt_count": noop_attempt_count,
                "all_attempts_noop": all_attempts_noop,
                "termination_reason": result.optimization_termination_reason,
                "objective_evaluations": result.objective_evaluations,
                "history_length": len(history_records),
                "selected_attempt": result.selected_attempt,
                "attempts": list(result.attempts),
                "per_view": per_view,
                "fail_reason": fail_reason,
            },
            "budget": {
                "limited": budget_limited,
                "outcome": budget_outcome,
                "termination_reason": result.optimization_termination_reason,
                "improved": improved > 0.0,
                "objective_improvement": improved,
                "max_runtime_s": max_runtime_s,
                "max_objective_evaluations": max_objective_evaluations,
                "objective_evaluations": result.objective_evaluations,
                "optimizer_elapsed_s": result.optimizer_elapsed_s,
            },
            "initial_primitive_count": (
                len(initial_primitives) if initial_primitives is not None else None
            ),
            "initial_loss": result.initial_loss.terms,
            "final_loss": result.final_loss.terms,
            "initial_total": result.initial_loss.total,
            "final_total": result.final_loss.total,
            "objective_before": result.initial_loss.total,
            "objective_after": result.final_loss.total,
            "objective_improvement": improved,
            "objective_improvement_ratio": improvement_ratio,
            "accepted_move_count": accepted_moves,
            "rejected_move_count": rejected_moves,
            "attempt_count": attempt_count,
            "noop_attempt_count": noop_attempt_count,
            "all_attempts_noop": all_attempts_noop,
            "termination_reason": result.optimization_termination_reason,
            "fail_reason": fail_reason,
            "surface_proxy_iou": surface_proxy_iou,
            "silhouette_proxy_iou": silhouette_proxy_iou,
            "optimization_termination_reason": result.optimization_termination_reason,
            "selected_attempt": result.selected_attempt,
            "optimization_attempts": list(result.attempts),
            "objective_evaluations": result.objective_evaluations,
            "optimizer_elapsed_s": result.optimizer_elapsed_s,
            "max_runtime_s": max_runtime_s,
            "max_objective_evaluations": max_objective_evaluations,
            "history": list(history_records),
            "surface_points": surface_meta,
            "occupied_points": occupied_meta,
            "signal_summary": signal_summary,
            "per_view": per_view,
            "configuration": _pipeline_config_summary(pipeline_config),
            "signal_rows_used": len(profile_rows),
        },
        per_view=per_view,
    )
    return metric, {
        "improved": improved,
        "topology_score": topology_score,
        "constraint_score": constraint_score,
        "uncertainty_consistency": uncertainty_consistency,
    }


def _attempt_is_noop(attempt: Mapping[str, Any]) -> bool:
    if str(attempt.get("status", "")) != "ok":
        return False
    try:
        accepted = int(attempt.get("accepted_moves", 0) or 0)
        improvement = float(attempt.get("improvement", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return accepted <= 0 and improvement <= 1.0e-12
