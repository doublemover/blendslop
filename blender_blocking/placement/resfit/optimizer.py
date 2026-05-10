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

from .config import ResFitPipelineConfig, ResFitPipelineResult
from .initialization import get_initializer


def fit_residual_primitives(
    target_points: np.ndarray,
    config: ResFitPipelineConfig = ResFitPipelineConfig(),
    initial_primitives: Sequence[object] | None = None,
    occupied_points: np.ndarray | None = None,
    silhouette_hook: SilhouetteHook | None = None,
    topology_penalty_hook: PenaltyHook | None = None,
    constraint_penalty_hook: PenaltyHook | None = None,
    uncertainty_penalty_hook: PenaltyHook | None = None,
) -> ResFitPipelineResult:
    """Fit primitives against target points using the modular ResFit path."""
    target_points = np.asarray(target_points, dtype=np.float64)
    if target_points.ndim != 2 or target_points.shape[1] != 3:
        raise ValueError("target_points must have shape (N, 3)")

    config_errors = config.validate()
    if config_errors:
        raise ValueError("invalid pipeline config: " + "; ".join(config_errors))

    warnings: list[str] = []
    if initial_primitives is None:
        initializer = get_initializer(config.primitive_family)
        primitives = tuple(initializer(target_points, config.initialization))
    else:
        primitives = tuple(initial_primitives)

    def objective(primitives_to_score: Sequence[object]) -> ResFitObjectiveResult:
        return evaluate_resfit_objective(
            primitives_to_score,
            target_points,
            weights=config.weights,
            occupied_points=occupied_points,
            silhouette_hook=silhouette_hook,
            topology_penalty_hook=topology_penalty_hook,
            constraint_penalty_hook=constraint_penalty_hook,
            uncertainty_penalty_hook=uncertainty_penalty_hook,
        )

    initial_loss = objective(primitives)
    optimization = coordinate_descent_optimize(primitives, objective, config.optimizer)
    final_loss = objective(optimization.primitives)

    if config.fail_on_regression and final_loss.total > initial_loss.total:
        warnings.append(
            "optimization regressed objective; returning initial primitive state"
        )
        return ResFitPipelineResult(
            primitives=primitives,
            initial_loss=initial_loss,
            final_loss=initial_loss,
            history=optimization.history,
            warnings=tuple(warnings + list(initial_loss.warnings)),
            optimization_termination_reason=optimization.termination_reason,
            objective_evaluations=optimization.objective_evaluations,
            optimizer_elapsed_s=optimization.elapsed_s,
        )

    return ResFitPipelineResult(
        primitives=optimization.primitives,
        initial_loss=initial_loss,
        final_loss=final_loss,
        history=optimization.history,
        warnings=tuple(warnings + list(final_loss.warnings)),
        optimization_termination_reason=optimization.termination_reason,
        objective_evaluations=optimization.objective_evaluations,
        optimizer_elapsed_s=optimization.elapsed_s,
    )


def fit_residual_primitives_multistart(
    target_points: np.ndarray,
    config: ResFitPipelineConfig,
    *,
    profile_primitives: Sequence[object] | None = None,
    occupied_points: np.ndarray | None = None,
    silhouette_hook: SilhouetteHook | None = None,
    topology_penalty_hook: PenaltyHook | None = None,
    constraint_penalty_hook: PenaltyHook | None = None,
    uncertainty_penalty_hook: PenaltyHook | None = None,
    max_attempts: int = 4,
) -> ResFitPipelineResult:
    """Run deterministic multi-start/schedule fitting and return the best result."""
    attempts: list[tuple[str, ResFitPipelineConfig, Sequence[object] | None]] = []
    attempts.append(("default_seed_default_step", config, None))
    if profile_primitives:
        attempts.append(("profile_seed_default_step", config, tuple(profile_primitives)))

    if config.optimizer.iterations > 0:
        smaller = replace(
            config,
            optimizer=replace(
                config.optimizer,
                initial_step=max(config.optimizer.min_step * 2.0, config.optimizer.initial_step * 0.5),
            ),
        )
        wider = replace(
            config,
            optimizer=replace(
                config.optimizer,
                initial_step=min(1.0, config.optimizer.initial_step * 1.75),
                step_decay=max(0.2, min(0.85, config.optimizer.step_decay * 0.9)),
            ),
        )
        if profile_primitives:
            attempts.append(("profile_seed_small_step", smaller, tuple(profile_primitives)))
        attempts.append(("default_seed_wide_step", wider, None))

    bounded_attempts = attempts[: max(1, int(max_attempts))]
    summaries: list[Mapping[str, Any]] = []
    best: ResFitPipelineResult | None = None
    best_label = ""
    errors: list[str] = []
    for label, attempt_config, seed in bounded_attempts:
        attempt_start = time.perf_counter()
        try:
            attempt_result = fit_residual_primitives(
                target_points,
                attempt_config,
                initial_primitives=seed,
                occupied_points=occupied_points,
                silhouette_hook=silhouette_hook,
                topology_penalty_hook=topology_penalty_hook,
                constraint_penalty_hook=constraint_penalty_hook,
                uncertainty_penalty_hook=uncertainty_penalty_hook,
            )
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            summaries.append(
                {
                    "label": label,
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_s": time.perf_counter() - attempt_start,
                }
            )
            continue
        improvement = attempt_result.initial_loss.total - attempt_result.final_loss.total
        summaries.append(
            {
                "label": label,
                "status": "ok",
                "initial_total": attempt_result.initial_loss.total,
                "final_total": attempt_result.final_loss.total,
                "improvement": improvement,
                "history_length": len(attempt_result.history),
                "accepted_moves": sum(record.accepted_moves for record in attempt_result.history),
                "rejected_moves": sum(getattr(record, "rejected_moves", 0) for record in attempt_result.history),
                "termination_reason": attempt_result.optimization_termination_reason,
                "objective_evaluations": attempt_result.objective_evaluations,
                "elapsed_s": time.perf_counter() - attempt_start,
            }
        )
        if best is None or _attempt_is_better(attempt_result, best):
            best = attempt_result
            best_label = label

    if best is None:
        raise RuntimeError("all primitive fitting attempts failed: " + "; ".join(errors))

    warnings = list(best.warnings)
    if errors:
        warnings.extend(errors)
    return ResFitPipelineResult(
        primitives=best.primitives,
        initial_loss=best.initial_loss,
        final_loss=best.final_loss,
        history=best.history,
        warnings=tuple(warnings),
        optimization_termination_reason=best.optimization_termination_reason,
        objective_evaluations=sum(
            int(summary.get("objective_evaluations", 0))
            for summary in summaries
            if isinstance(summary, Mapping)
        ),
        optimizer_elapsed_s=sum(
            float(summary.get("elapsed_s", 0.0))
            for summary in summaries
            if isinstance(summary, Mapping)
        ),
        selected_attempt=best_label or "default",
        attempts=tuple(summaries),
    )


def _attempt_is_better(
    candidate: ResFitPipelineResult,
    incumbent: ResFitPipelineResult,
) -> bool:
    candidate_improvement = candidate.initial_loss.total - candidate.final_loss.total
    incumbent_improvement = incumbent.initial_loss.total - incumbent.final_loss.total
    if candidate.final_loss.total < incumbent.final_loss.total:
        return True
    if candidate.final_loss.total > incumbent.final_loss.total:
        return False
    return candidate_improvement > incumbent_improvement
