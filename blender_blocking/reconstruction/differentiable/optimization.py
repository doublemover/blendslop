from __future__ import annotations

from typing import Any, Mapping, Sequence

from .config import _minimum_positive_float
from .contracts import RenderableScene
from .target_adapter import renderable_from_primitive

try:
    from placement.resfit_objective import ResFitObjectiveResult
    from placement.resfit_optimizer import (
        CoordinateDescentConfig,
        coordinate_descent_optimize,
    )
except ImportError:  # pragma: no cover - package import path
    from ...placement.resfit_objective import ResFitObjectiveResult
    from ...placement.resfit_optimizer import (
        CoordinateDescentConfig,
        coordinate_descent_optimize,
    )


def run_differentiable_optimization(
    *,
    backend_choice: str,
    parsed_config: Mapping[str, Any],
    primitives: tuple[Any, ...],
    renderer: Any,
    cameras: Sequence[Any],
    target_record: Any,
    target_view_weights: Mapping[str, float],
    initial_render_batch: Any,
    initial_loss: Any,
    request_timeout_s: float | None,
) -> dict[str, Any]:
    optimized_primitives = primitives
    optimization_history: list[dict[str, object]] = []
    optimization_summary: dict[str, object] = {
        "enabled": False,
        "reason": "not_run",
        "objective_evaluations": 0,
        "elapsed_s": 0.0,
        "initial_total": float(initial_loss.total),
        "final_total": float(initial_loss.total),
        "objective_policy": _optimizer_objective_policy(parsed_config),
        "initial_optimizer_total": _optimizer_objective_total(
            initial_loss,
            parsed_config,
        ),
    }
    if (
        backend_choice == "cpu_soft_silhouette"
        and int(parsed_config["optimization_steps"]) > 0
        and primitives
        and cameras
    ):
        max_elapsed_s = _minimum_positive_float(
            parsed_config.get("max_runtime_s"),
            request_timeout_s,
        )
        optimizer_config = CoordinateDescentConfig(
            iterations=int(parsed_config["optimization_steps"]),
            initial_step=float(parsed_config["optimization_initial_step"]),
            step_decay=float(parsed_config["optimization_step_decay"]),
            min_step=float(parsed_config["optimization_min_step"]),
            max_objective_evaluations=parsed_config["max_objective_evaluations"],
            max_elapsed_s=max_elapsed_s,
        )

        def objective(primitives_to_score: Sequence[object]) -> ResFitObjectiveResult:
            trial_scene = RenderableScene(
                primitives=tuple(
                    renderable_from_primitive(primitive)
                    for primitive in primitives_to_score
                )
            )
            trial_batch = renderer.render(trial_scene, cameras)
            trial_loss = renderer.loss(
                trial_batch,
                target_record,
                parsed_config["loss_weights"],
                view_weights=target_view_weights,
            )
            optimizer_total = _optimizer_objective_total(
                trial_loss,
                parsed_config,
            )
            return ResFitObjectiveResult(
                total=float(optimizer_total),
                terms={
                    **dict(trial_loss.terms),
                    "weighted_render_loss_total": float(trial_loss.total),
                    "optimizer_objective_total": float(optimizer_total),
                    "optimizer_objective_policy": _optimizer_objective_policy(
                        parsed_config
                    ),
                },
                warnings=tuple(trial_loss.warnings),
            )

        optimization = coordinate_descent_optimize(
            primitives,
            objective,
            optimizer_config,
        )
        optimized_primitives = optimization.primitives
        optimization_history = [
            {
                "iteration": record.iteration,
                "total": record.total,
                "terms": dict(record.terms),
                "accepted_moves": record.accepted_moves,
                "rejected_moves": getattr(record, "rejected_moves", 0),
                "step_size": record.step_size,
                "reason": getattr(record, "reason", ""),
            }
            for record in optimization.history
        ]
        optimization_summary = {
            "enabled": True,
            "reason": optimization.termination_reason,
            "objective_evaluations": optimization.objective_evaluations,
            "elapsed_s": optimization.elapsed_s,
            "initial_total": float(initial_loss.total),
            "initial_optimizer_total": _optimizer_objective_total(
                initial_loss,
                parsed_config,
            ),
            "best_total": float(optimization.best_loss),
            "accepted_moves": sum(
                int(record.accepted_moves) for record in optimization.history
            ),
            "history_length": len(optimization_history),
            "objective_policy": _optimizer_objective_policy(parsed_config),
            "config": {
                "iterations": int(parsed_config["optimization_steps"]),
                "initial_step": float(parsed_config["optimization_initial_step"]),
                "step_decay": float(parsed_config["optimization_step_decay"]),
                "min_step": float(parsed_config["optimization_min_step"]),
                "max_objective_evaluations": parsed_config["max_objective_evaluations"],
                "max_elapsed_s": max_elapsed_s,
            },
        }
    elif int(parsed_config["optimization_steps"]) <= 0:
        optimization_summary["reason"] = "optimization_steps_zero"
    elif backend_choice != "cpu_soft_silhouette":
        optimization_summary["reason"] = f"optimizer disabled for backend {backend_choice}"
    elif not cameras:
        optimization_summary["reason"] = "no target cameras"
    elif not primitives:
        optimization_summary["reason"] = "no primitives"

    if optimization_summary.get("enabled"):
        renderables = tuple(
            renderable_from_primitive(primitive) for primitive in optimized_primitives
        )
        scene = RenderableScene(primitives=renderables)
        render_batch = renderer.render(scene, cameras)
        loss = renderer.loss(
            render_batch,
            target_record,
            parsed_config["loss_weights"],
            view_weights=target_view_weights,
        )
    else:
        render_batch = initial_render_batch
        loss = initial_loss
    optimization_summary["final_total"] = float(loss.total)
    optimization_summary["final_optimizer_total"] = _optimizer_objective_total(
        loss,
        parsed_config,
    )
    return {
        "optimized_primitives": tuple(optimized_primitives),
        "optimization_history": optimization_history,
        "optimization_summary": optimization_summary,
        "render_batch": render_batch,
        "loss": loss,
    }


def _optimizer_objective_policy(parsed_config: Mapping[str, Any]) -> str:
    return (
        "boundary_sdf_first"
        if bool(parsed_config.get("optimize_boundary_sdf_first", True))
        else "weighted_render_loss"
    )


def _optimizer_objective_total(loss: Any, parsed_config: Mapping[str, Any]) -> float:
    if not bool(parsed_config.get("optimize_boundary_sdf_first", True)):
        return float(loss.total)
    terms = getattr(loss, "terms", {}) or {}
    boundary = _term(terms, "boundary_iou_loss", _term(terms, "boundary_iou", 0.0))
    signed_distance = _term(
        terms,
        "signed_distance_loss",
        _term(terms, "signed_distance", 0.0),
    )
    worst_boundary = _term(terms, "worst_boundary_iou_loss", boundary)
    worst_signed_distance = _term(terms, "worst_signed_distance_loss", signed_distance)
    area = _term(terms, "area_iou", 0.0)
    soft = _term(terms, "soft_iou", 0.0)
    return float(
        2.0 * worst_boundary
        + 2.0 * worst_signed_distance
        + boundary
        + signed_distance
        + 0.25 * area
        + 0.1 * soft
    )


def _term(terms: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(terms.get(key, default))
    except (TypeError, ValueError):
        return float(default)
