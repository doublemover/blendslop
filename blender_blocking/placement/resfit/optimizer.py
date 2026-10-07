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
    ResFitObjectiveEvaluator,
)
from ..resfit_optimizer import (
    CoordinateDescentConfig,
    OptimizationRecord,
    OptimizationBudget,
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
    *,
    budget: OptimizationBudget | None = None,
) -> ResFitPipelineResult:
    """Fit primitives against target points using the modular ResFit path."""
    search_start = time.perf_counter()
    budget = budget or OptimizationBudget(
        max_objective_evaluations=config.optimizer.max_objective_evaluations,
        max_elapsed_s=config.optimizer.max_elapsed_s,
    )
    evaluation_start = budget.objective_evaluations
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

    objective = ResFitObjectiveEvaluator(
        target_points, weights=config.weights, occupied_points=occupied_points,
        silhouette_hook=silhouette_hook, topology_penalty_hook=topology_penalty_hook,
        constraint_penalty_hook=constraint_penalty_hook, uncertainty_penalty_hook=uncertainty_penalty_hook,
        objective_mode=config.objective_mode, length_scale=config.length_scale,
    )

    remaining_count = budget.remaining_evaluations()
    remaining_seconds = budget.remaining_seconds()
    residual_enabled = bool(config.optimizer.iterations > 0 and config.residual_rounds and config.max_residual_proposals
                            and (remaining_count is None or remaining_count >= 8))
    optimization_budget = OptimizationBudget(
        max_objective_evaluations=None if remaining_count is None else max(1, int(remaining_count * 0.75)),
        max_elapsed_s=None if remaining_seconds is None else remaining_seconds * 0.75,
        parent=budget,
    ) if residual_enabled else budget
    from blender_blocking.reconstruction.process_executor import publish_progress, progress_path

    def retain_scored(snapshot):
        value = ResFitPipelineResult(primitives=snapshot.primitives,
            initial_loss=snapshot.initial_result, final_loss=snapshot.final_result,
            history=snapshot.history, warnings=("scored partial state; completed work retained",),
            optimization_termination_reason="scored_checkpoint",
            objective_evaluations=budget.objective_evaluations - evaluation_start,
            optimizer_elapsed_s=time.perf_counter() - search_start,
            parameter_visits=snapshot.parameter_visits)
        publish_progress(value, recorded_evaluations=value.objective_evaluations)

    optimizer = coordinate_descent_optimize
    if config.refinement_strategy == "coupled_blocks":
        from ..resfit_coupled import coupled_block_optimize
        optimizer = coupled_block_optimize
    optimization = optimizer(primitives, objective, config.optimizer,
        budget=optimization_budget, on_progress=retain_scored if progress_path() is not None else None)
    initial_loss = optimization.initial_result
    final_loss = optimization.final_result
    assert initial_loss is not None and final_loss is not None
    fitted = optimization.primitives
    history = optimization.history
    residual_records = ()
    if residual_enabled and budget.reason() is None:
        from .residual_proposals import refine_residual_parts
        refinement = refine_residual_parts(
            fitted, target_points, config.primitive_family, objective, config.optimizer,
            budget=budget, initial_result=final_loss, max_parts=config.max_primitives,
            max_proposals=config.max_residual_proposals, refinement_steps=config.residual_refinement_steps,
            rounds=config.residual_rounds, initialization_config=config.initialization,
        )
        fitted, final_loss = refinement.primitives, refinement.final_result
        history = (*history, *refinement.history)
        residual_records = refinement.proposals

    if config.fail_on_regression and final_loss.total > initial_loss.total:
        warnings.append("optimization regressed objective; returning initial primitive state")
        fitted, final_loss = primitives, initial_loss
    return ResFitPipelineResult(
        primitives=tuple(fitted), initial_loss=initial_loss, final_loss=final_loss,
        history=tuple(history), warnings=tuple(warnings + list(final_loss.warnings)),
        optimization_termination_reason=budget.reason() or optimization.termination_reason,
        objective_evaluations=budget.objective_evaluations - evaluation_start,
        optimizer_elapsed_s=time.perf_counter() - search_start, residual_proposals=tuple(residual_records),
        parameter_visits=optimization.parameter_visits,
    )


def fit_residual_primitives_multistart(
    target_points: np.ndarray,
    config: ResFitPipelineConfig,
    *,
    profile_primitives: Sequence[object] | None = None,
    whole_primitives: Sequence[object] | None = None,
    occupied_points: np.ndarray | None = None,
    silhouette_hook: SilhouetteHook | None = None,
    topology_penalty_hook: PenaltyHook | None = None,
    constraint_penalty_hook: PenaltyHook | None = None,
    uncertainty_penalty_hook: PenaltyHook | None = None,
    max_attempts: int = 4,
    budget: OptimizationBudget | None = None,
    executor: Any = None,
) -> ResFitPipelineResult:
    """Run deterministic multi-start/schedule fitting and return the best result."""
    attempts: list[tuple[str, ResFitPipelineConfig, Sequence[object] | None]] = []
    attempts.append(("default_seed_default_step", config, None))
    if whole_primitives:
        attempts.insert(0, ("whole_oriented_seed", config, tuple(whole_primitives)))
    if profile_primitives:
        attempts.append(("profile_seed_default_step", config, tuple(profile_primitives)))

    if config.optimizer.iterations > 0 and not whole_primitives:
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

    budget = budget or OptimizationBudget(
        max_objective_evaluations=config.optimizer.max_objective_evaluations,
        max_elapsed_s=config.optimizer.max_elapsed_s,
    )
    evaluation_start = budget.objective_evaluations
    search_start = time.perf_counter()
    bounded_attempts = attempts[: max(1, int(max_attempts))]
    summaries: list[Mapping[str, Any]] = []
    best: ResFitPipelineResult | None = None
    best_label = ""
    errors: list[str] = []
    if whole_primitives:
        from ..resfit_objective import _geometry_key
        screening = ResFitObjectiveEvaluator(target_points,config.weights,occupied_points,
            silhouette_hook,topology_penalty_hook,constraint_penalty_hook,uncertainty_penalty_hook,
            objective_mode=config.objective_mode,length_scale=config.length_scale)
        distinct, seen = [], set()
        for label, attempt_config, seed in bounded_attempts:
            if budget.reason() is not None:
                break
            if seed is None:
                seed = tuple(get_initializer(config.primitive_family)(target_points,config.initialization))
            signature = tuple(sorted(str(_geometry_key(p)) for p in seed))
            if signature in seen:
                summaries.append({"label":label,"stage":"seed_screen","status":"skipped",
                                  "reason":"duplicate_geometry_seed","objective_evaluations":0})
                continue
            seen.add(signature)
            scoring_before = budget.objective_evaluations
            try:
                scored = budget.evaluate(screening,seed)
            except Exception as exc:
                summaries.append({"label":label,"stage":"seed_screen","status":"failed",
                                  "error":str(exc),"objective_evaluations":budget.objective_evaluations-scoring_before})
                continue
            summaries.append({"label":label,"stage":"seed_screen","status":"scored",
                              "total":scored.total,"objective_evaluations":1,"parts":len(seed)})
            distinct.append((scored.total,label,attempt_config,seed))
            if best is None or scored.total < best.final_loss.total:
                best = ResFitPipelineResult(tuple(seed),scored,scored,(),
                    ("retained screened seed; refinement may be unavailable",),selected_attempt=label)
                best_label = label
        # Refine genuinely different best seeds, rather than identical geometry
        # with several step-size schedules. All screening calls are charged.
        bounded_attempts = [(label,cfg,seed) for _,label,cfg,seed in sorted(distinct,key=lambda r:r[0])[:min(2,max_attempts)]]
    from blender_blocking.reconstruction.process_executor import current_worker_client, PersistentProcessExecutor
    from contextlib import nullcontext
    worker = current_worker_client()
    shared_executor = executor if executor is not None else worker
    manager = nullcontext(shared_executor) if shared_executor is not None else PersistentProcessExecutor(2)
    remaining_count = budget.remaining_evaluations()
    remaining_seconds = budget.remaining_seconds()
    jobs = []
    scheduled = []
    for index, (label, attempt_config, seed) in enumerate(bounded_attempts):
        count = None if remaining_count is None else remaining_count // len(bounded_attempts) + (index < remaining_count % len(bounded_attempts))
        if count == 0 or (remaining_seconds is not None and remaining_seconds <= 0):
            summaries.append({"label": label, "status": "skipped", "reason": budget.reason() or "objective_evaluation_budget",
                              "objective_evaluations": 0, "elapsed_s": 0.0})
            continue
        attempt_config = replace(attempt_config, optimizer=replace(attempt_config.optimizer,
            max_objective_evaluations=count, max_elapsed_s=remaining_seconds))
        payload = dict(target_points=target_points, config=attempt_config, initial_primitives=seed,
            occupied_points=occupied_points, silhouette_hook=silhouette_hook,
            topology_penalty_hook=topology_penalty_hook, constraint_penalty_hook=constraint_penalty_hook,
            uncertainty_penalty_hook=uncertainty_penalty_hook)
        jobs.append(("fit_start", payload, remaining_seconds))
        scheduled.append((label, count))
    with manager as executor:
        outcomes = executor.map(jobs, timeout_s=remaining_seconds)
    for (label, allowance), outcome in zip(scheduled, outcomes):
        if outcome.status != "success" and not outcome.partial:
            charged = allowance or 0
            for _ in range(charged):
                budget._record_evaluation()
            errors.append(f"{label}: {outcome.error}")
            summaries.append({"label": label, "status": outcome.status, "error": outcome.error,
                "objective_evaluations": None, "reserved_objective_evaluations": charged,
                "evaluation_count_is_reserved_upper_bound": True, "queue_s": outcome.queue_s,
                "stop_reason": outcome.stop_reason, "actual_work_unreported": True,
                "elapsed_s": outcome.elapsed_s})
            continue
        attempt_result = outcome.value
        if outcome.partial:
            attempt_result = replace(attempt_result, optimization_termination_reason=outcome.stop_reason,
                warnings=(*attempt_result.warnings, "worker interrupted; retaining scored partial state"))
            errors.append(f"{label}: {outcome.stop_reason}; scored partial state retained")
        reserved_charge = (max(attempt_result.objective_evaluations, allowance or 0)
                           if outcome.partial else attempt_result.objective_evaluations)
        for _ in range(reserved_charge):
            budget._record_evaluation()
        improvement = attempt_result.initial_loss.total - attempt_result.final_loss.total
        summaries.append({
            "label": label, "status": "ok", "initial_total": attempt_result.initial_loss.total,
            "final_total": attempt_result.final_loss.total, "improvement": improvement,
            "history_length": len(attempt_result.history),
            "accepted_moves": sum(record.accepted_moves for record in attempt_result.history),
            "rejected_moves": sum(getattr(record, "rejected_moves", 0) for record in attempt_result.history),
            "termination_reason": attempt_result.optimization_termination_reason,
            "objective_evaluations": attempt_result.objective_evaluations, "elapsed_s": outcome.elapsed_s,
            "queue_s": outcome.queue_s, "total_wall_s": outcome.total_wall_s,
            "partial": outcome.partial, "actual_work_unreported": outcome.partial,
            "parameter_visits": list(attempt_result.parameter_visits),
        })
        if best is None or _attempt_is_better(attempt_result, best):
            best = attempt_result
            best_label = label

    if best is None:
        reason = budget.reason()
        raise RuntimeError("all primitive fitting attempts failed: " + "; ".join(errors or [reason or "no scored attempt"]))

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
        objective_evaluations=sum(row.get("objective_evaluations") or 0 for row in summaries),
        optimizer_elapsed_s=time.perf_counter() - search_start,
        selected_attempt=best_label or "default",
        attempts=tuple(summaries),
        residual_proposals=best.residual_proposals,
        parameter_visits=best.parameter_visits,
        search_budget={"scope": "all_attempts", "max_objective_evaluations": budget.max_objective_evaluations,
                       "max_elapsed_s": budget.max_elapsed_s, "includes_initial_and_failed_objectives": True,
                       "deadline_is_cooperative": False, "execution": "coordinated_process_starts",
                       "recorded_objective_evaluations": sum(row.get("objective_evaluations") or 0 for row in summaries),
                       "reserved_evaluation_charge": budget.objective_evaluations - evaluation_start,
                       "unreported_work_possible": any(row.get("actual_work_unreported") for row in summaries)},
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
