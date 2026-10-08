from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .target_adapter import _silhouette_view_history


def build_refinement_history_payloads(
    *,
    silhouettes: Mapping[str, Any],
    render_batch: Any,
    baseline_render_batch: Any,
    initial_render_batch: Any,
    loss: Any,
    baseline_loss: Any,
    initial_loss: Any,
    target_view_weights: Mapping[str, float],
    view_signal_details: Mapping[str, Any],
    optimization_summary: Mapping[str, Any],
    optimization_history: list[dict[str, object]],
    backend_choice: str,
    parsed_config: Mapping[str, Any],
) -> dict[str, Any]:
    refinement_history: list[dict[str, object]] = []
    for name, target_mask in silhouettes.items():
        view_metrics = dict(loss.per_view.get(name, {}))
        if name in render_batch.silhouettes:
            refinement_history.append(
                _silhouette_view_history(
                    name=name,
                    predicted=np.asarray(
                        render_batch.silhouettes[name], dtype=np.float64
                    ),
                    target=np.asarray(target_mask, dtype=np.float64),
                    metrics=view_metrics,
                )
            )
    baseline_refinement_history = [
        _silhouette_view_history(
            name=name,
            predicted=np.zeros_like(target_mask, dtype=np.float64),
            target=np.asarray(target_mask, dtype=np.float64),
            metrics=dict(baseline_loss.per_view.get(name, {})),
        )
        for name, target_mask in silhouettes.items()
    ]
    objective_improvement = float(initial_loss.total - loss.total)
    zero_baseline_improvement = float(baseline_loss.total - loss.total)
    objective_history = [
        {
            "stage": "baseline_zero",
            "loss_total": float(baseline_loss.total),
            "loss_terms": dict(baseline_loss.terms),
            "loss_warnings": tuple(baseline_loss.warnings),
            "loss_view_weights": dict(target_view_weights),
            "views": baseline_refinement_history,
        },
        {
            "stage": "initial_render_and_score",
            "loss_total": float(initial_loss.total),
            "loss_terms": dict(initial_loss.terms),
            "loss_warnings": tuple(initial_loss.warnings),
            "loss_view_weights": dict(target_view_weights),
            "views": [
                _silhouette_view_history(
                    name=name,
                    predicted=np.asarray(
                        initial_render_batch.silhouettes[name], dtype=np.float64
                    ),
                    target=np.asarray(target_mask, dtype=np.float64),
                    metrics=dict(initial_loss.per_view.get(name, {})),
                )
                for name, target_mask in silhouettes.items()
                if name in initial_render_batch.silhouettes
            ],
        },
        {
            "stage": "optimized_render_and_score",
            "loss_total": float(loss.total),
            "loss_terms": dict(loss.terms),
            "loss_warnings": tuple(loss.warnings),
            "loss_view_weights": dict(target_view_weights),
            "views": refinement_history,
            "optimization": dict(optimization_summary),
        },
    ]
    objective_improvement_record = {
        "baseline_total": float(baseline_loss.total),
        "initial_candidate_total": float(initial_loss.total),
        "objective_total": float(loss.total),
        "objective_improvement": objective_improvement,
        "zero_baseline_improvement": zero_baseline_improvement,
        "weight_sum": {
            "silhouette_weight_sum": float(
                sum(weight for weight in target_view_weights.values())
            ),
            "depth_weight_sum": float(sum(weight for weight in target_view_weights.values())),
        },
        "history": objective_history,
        "view_signal_weights": dict(target_view_weights),
        "view_signal_details": dict(view_signal_details),
        "optimization": dict(optimization_summary),
    }
    history_payload = [
        {
            "step": "baseline_zero",
            "backend": backend_choice,
            "softness": float(parsed_config["softness"]),
            "min_variance": float(parsed_config["min_variance"]),
            "loss_total": float(baseline_loss.total),
            "loss_terms": dict(baseline_loss.terms),
            "loss_warnings": tuple(baseline_loss.warnings),
            "loss_view_weights": dict(target_view_weights),
            "views": baseline_refinement_history,
            "render_metadata": dict(getattr(baseline_render_batch, "metadata", {})),
        },
        {
            "step": "initial_render_and_score",
            "backend": backend_choice,
            "softness": float(parsed_config["softness"]),
            "min_variance": float(parsed_config["min_variance"]),
            "loss_total": float(initial_loss.total),
            "loss_terms": dict(initial_loss.terms),
            "loss_warnings": tuple(initial_loss.warnings),
            "loss_view_weights": dict(target_view_weights),
            "view_signal_details": dict(view_signal_details),
            "views": [
                _silhouette_view_history(
                    name=name,
                    predicted=np.asarray(
                        initial_render_batch.silhouettes[name], dtype=np.float64
                    ),
                    target=np.asarray(target_mask, dtype=np.float64),
                    metrics=dict(initial_loss.per_view.get(name, {})),
                )
                for name, target_mask in silhouettes.items()
                if name in initial_render_batch.silhouettes
            ],
            "render_metadata": dict(getattr(initial_render_batch, "metadata", {})),
        },
        {
            "step": "optimized_render_and_score",
            "backend": backend_choice,
            "softness": float(parsed_config["softness"]),
            "min_variance": float(parsed_config["min_variance"]),
            "loss_total": float(loss.total),
            "loss_terms": dict(loss.terms),
            "loss_warnings": tuple(loss.warnings),
            "loss_view_weights": dict(target_view_weights),
            "view_signal_details": dict(view_signal_details),
            "views": refinement_history,
            "render_metadata": dict(getattr(render_batch, "metadata", {})),
            "optimization": dict(optimization_summary),
            "optimization_history": optimization_history,
        },
    ]
    return {
        "refinement_history": refinement_history,
        "baseline_refinement_history": baseline_refinement_history,
        "objective_improvement": objective_improvement,
        "zero_baseline_improvement": zero_baseline_improvement,
        "objective_history": objective_history,
        "objective_improvement_record": objective_improvement_record,
        "history_payload": history_payload,
    }
