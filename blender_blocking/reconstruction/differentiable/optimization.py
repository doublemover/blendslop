from __future__ import annotations

from typing import Any, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
import time

import numpy as np

from .config import _minimum_positive_float
from .contracts import RenderBatch
from .soft_objective import SoftSilhouetteObjective

try:
    from placement.resfit_objective import ResFitObjectiveResult
    from placement.resfit_optimizer import (
        CoordinateDescentConfig,
        coordinate_descent_optimize,
        OptimizationBudget,
        OptimizationBudgetExhausted,
    )
except ImportError:  # pragma: no cover - package import path
    from ...placement.resfit_objective import ResFitObjectiveResult
    from ...placement.resfit_optimizer import (
        CoordinateDescentConfig,
        coordinate_descent_optimize,
        OptimizationBudget,
        OptimizationBudgetExhausted,
    )


try:
    from primitives.soft_silhouette import ProjectedSilhouetteCache
    from placement.resfit_parameters import apply_parameter_increment, pullback_render_gradients
except ImportError:  # pragma: no cover - package import path
    from ...primitives.soft_silhouette import ProjectedSilhouetteCache
    from ...placement.resfit_parameters import apply_parameter_increment, pullback_render_gradients


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
    summary: dict[str, object] = {
        "enabled": False, "reason": "not_run", "objective_evaluations": 0,
        "elapsed_s": 0.0, "initial_total": float(initial_loss.total),
        "final_total": float(initial_loss.total),
        "objective_policy": _optimizer_objective_policy(parsed_config),
        "initial_optimizer_total": _optimizer_objective_total(initial_loss, parsed_config),
    }
    if backend_choice != "cpu_soft_silhouette":
        summary["reason"] = f"optimizer disabled for backend {backend_choice}"
    elif int(parsed_config["optimization_steps"]) <= 0:
        summary["reason"] = "optimization_steps_zero"
    elif not cameras:
        summary["reason"] = "no target cameras"
    elif not primitives:
        summary["reason"] = "no primitives"
    else:
        return _run_analytic_refinement(
            parsed_config=parsed_config, primitives=primitives, renderer=renderer,
            cameras=cameras, target_record=target_record,
            target_view_weights=target_view_weights, initial_render_batch=initial_render_batch,
            initial_loss=initial_loss, request_timeout_s=request_timeout_s,
        )
    summary["final_optimizer_total"] = _optimizer_objective_total(initial_loss, parsed_config)
    return {"optimized_primitives": primitives, "optimization_history": [],
            "optimization_summary": summary, "render_batch": initial_render_batch,
            "loss": initial_loss}


class _ProjectedRefinementObjective:
    """Changed-part forward reuse and retention of the best actual scored state."""

    def __init__(self, primitives, renderer, cameras, target, parsed_config,
                 view_weights, initial_batch, initial_loss):
        self.renderer = renderer
        self.cameras = tuple(cameras)
        self.target = target
        self.config = parsed_config
        self.view_weights = view_weights
        from blender_blocking.primitives.shape_aware_silhouette import is_ellipse, ShapeAwareSilhouetteCache
        cache_type = ProjectedSilhouetteCache if all(is_ellipse(p) for p in primitives) else ShapeAwareSilhouetteCache
        self.cache = cache_type(
            tuple(camera.to_orthographic_camera() for camera in cameras),
            softness=float(getattr(renderer, "softness", parsed_config["softness"])),
            min_variance=float(getattr(renderer, "min_variance", parsed_config["min_variance"])),
        )
        self.cache.render(primitives)
        self._last_generation = self.cache.generation
        self.last_batch = initial_batch
        self.last_loss = initial_loss
        self.last_result = self._result(initial_loss)
        if not np.isfinite(self.last_result.total):
            raise ValueError("initial geometric objective must be finite")
        self.best_parts = deepcopy(list(primitives))
        self.best_batch = initial_batch
        self.best_loss = initial_loss
        self.best_result = self.last_result
        self.failed_proposals: list[str] = []
        self.coordinate_batches = 0

    def _result(self, loss):
        return ResFitObjectiveResult(
            total=_optimizer_objective_total(loss, self.config),
            terms={**dict(loss.terms), "weighted_render_loss_total": float(loss.total),
                   "optimizer_objective_total": _optimizer_objective_total(loss, self.config),
                   "optimizer_objective_policy": _optimizer_objective_policy(self.config)},
            warnings=tuple(loss.warnings),
        )

    def restore_best(self):
        masks = self.cache.render(self.best_parts)
        self._last_generation = self.cache.generation
        self.last_batch, self.last_loss = self.best_batch, self.best_loss
        self.last_result = self.best_result
        return masks

    def __call__(self, primitives):
        try:
            masks = self.cache.render(primitives)
            if self.cache.generation == self._last_generation:
                return self.last_result
            batch = RenderBatch(silhouettes=masks, metadata={
                "backend": "cpu_soft_silhouette", "softness": self.cache.softness,
                "min_variance": self.cache.min_variance,
                "primitive_count": len(primitives), "camera_count": len(self.cameras),
                "changed_part_reuse": True,
                "contour_approximation_reports": [
                    {"view": key[0], "part_index": key[1], **report}
                    for key, report in getattr(self.cache, "approximation_reports", {}).items()],
            })
            loss = self.renderer.loss(batch, self.target, self.config["loss_weights"],
                                      view_weights=self.view_weights)
            result = self._result(loss)
            if not np.isfinite(result.total):
                raise ValueError("nonfinite geometric objective")
            self._last_generation = self.cache.generation
            self.last_batch, self.last_loss, self.last_result = batch, loss, result
            if result.total < self.best_result.total:
                self.best_parts = deepcopy(list(primitives))
                self.best_batch, self.best_loss, self.best_result = batch, loss, result
            return result
        except Exception as exc:
            # An invalid proposal cannot discard the previously scored winner.
            message = f"{type(exc).__name__}: {exc}"
            self.failed_proposals.append(message)
            return ResFitObjectiveResult(total=float("inf"), terms={}, warnings=(message,))


    def _score_batch_masks(self, primitives, masks):
        """Score a detached trial; the live cache still represents its base."""
        try:
            batch = RenderBatch(silhouettes=masks, metadata={
                "backend": "cpu_soft_silhouette", "softness": self.cache.softness,
                "min_variance": self.cache.min_variance,
                "primitive_count": len(primitives), "camera_count": len(self.cameras),
                "changed_part_reuse": True, "paired_coordinate_batch": True,
            })
            loss = self.renderer.loss(batch, self.target, self.config["loss_weights"],
                                      view_weights=self.view_weights)
            result = self._result(loss)
            if not np.isfinite(result.total):
                raise ValueError("nonfinite geometric objective")
            if result.total < self.best_result.total:
                self.best_parts = deepcopy(list(primitives))
                self.best_batch, self.best_loss, self.best_result = batch, loss, result
            return result
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.failed_proposals.append(message)
            return ResFitObjectiveResult(total=float("inf"), terms={}, warnings=(message,))

    def evaluate_batch(self, candidates, *, budget):
        """Ordered completed prefix under the optimizer's shared allowance."""
        if budget.reason() is not None:
            return
        remaining = budget.remaining_evaluations()
        count = min(2, len(candidates), remaining if remaining is not None else 2)
        proposals = candidates[:count]
        base = deepcopy(self.best_parts)
        self.restore_best()
        try:
            masks = self.cache.render_coordinate_batch(base, proposals)
        except Exception:
            # One invalid perturbation cannot suppress the other candidate.
            # Existing single-result failure handling retains the prior winner.
            for proposal in proposals:
                if budget.reason() is not None:
                    return
                yield budget.evaluate(self, proposal)
            return
        self.coordinate_batches += 1
        for proposal, candidate_masks in zip(proposals, masks):
            if budget.reason() is not None:
                return
            yield budget.evaluate(lambda parts: self._score_batch_masks(parts, candidate_masks), proposal)


def _run_analytic_refinement(
    *, parsed_config, primitives, renderer, cameras, target_record,
    target_view_weights, initial_render_batch, initial_loss, request_timeout_s,
):
    started = time.perf_counter()
    max_elapsed = _minimum_positive_float(parsed_config.get("max_runtime_s"), request_timeout_s)
    config = CoordinateDescentConfig(
        iterations=int(parsed_config["optimization_steps"]),
        initial_step=float(parsed_config["optimization_initial_step"]),
        step_decay=float(parsed_config["optimization_step_decay"]),
        min_step=float(parsed_config["optimization_min_step"]),
        max_objective_evaluations=parsed_config["max_objective_evaluations"],
        max_elapsed_s=max_elapsed,
    )
    errors = config.validate()
    if errors:
        raise ValueError("invalid optimizer config: " + ", ".join(errors))
    allowance = OptimizationBudget(max_objective_evaluations=config.max_objective_evaluations,
                                   max_elapsed_s=max_elapsed, started_at=started)
    objective = _ProjectedRefinementObjective(primitives, renderer, cameras, target_record,
        parsed_config, target_view_weights, initial_render_batch, initial_loss)
    soft_objective = SoftSilhouetteObjective(
        {camera.name: getattr(target_record,'probability_masks',{}).get(camera.name,target_record.silhouettes[camera.name]) for camera in cameras},
        parsed_config["loss_weights"], target_view_weights,
        getattr(target_record,'proposal_valid_masks',{}) or getattr(target_record,"valid_masks",{}),
        getattr(target_record,'pixel_weights',{}),
    )
    has_family_pullback = callable(getattr(objective.cache, "parameter_gradients", None))
    initial = (allowance.evaluate(objective, primitives) if allowance.reason() is None
               else objective.last_result)
    history: list[dict[str, object]] = []
    step = config.initial_step
    termination = "max_iterations"
    gradient_steps = 0
    coordinate_fallbacks = 0
    for iteration in range(config.iterations):
        reason = allowance.reason()
        if reason is not None:
            termination = reason
            break
        working = deepcopy(objective.best_parts)
        masks = objective.restore_best()
        soft_total, mask_gradients, soft_terms = soft_objective.evaluate(masks)
        family_pullback = getattr(objective.cache, "parameter_gradients", None)
        if callable(family_pullback):
            gradients = family_pullback(mask_gradients, config.bounds, budget=allowance)
        else:
            world_gradients = objective.cache.backward(mask_gradients)
            gradients = pullback_render_gradients(working, *world_gradients)
        norm = max((abs(value) for value in gradients.values()), default=0.0)
        before = objective.best_result.total
        rejected = 0
        accepted = False
        proposal_step = step
        if np.isfinite(norm) and norm > 1e-14:
            gradient_steps += 1
            while proposal_step >= config.min_step and allowance.reason() is None:
                trial_parts = deepcopy(working)
                for ref, gradient in gradients.items():
                    if gradient != 0.0:
                        apply_parameter_increment(trial_parts, ref,
                            -proposal_step * gradient / norm, config.bounds)
                try:
                    trial = allowance.evaluate(objective, trial_parts)
                except OptimizationBudgetExhausted as exc:
                    termination = str(exc)
                    break
                if trial.total < before:
                    accepted = True
                    step = proposal_step
                    break
                rejected += 1
                proposal_step *= config.step_decay
        # Hard thresholds can stall a smooth direction. A bounded coordinate
        # pass uses the same changed-part cache and the same global allowance.
        if not accepted and allowance.reason() is None:
            coordinate_fallbacks += 1
            objective.restore_best()
            fallback = coordinate_descent_optimize(objective.best_parts, objective,
                replace(config, iterations=1, initial_step=step), budget=allowance)
            rejected += sum(record.rejected_moves for record in fallback.history)
            accepted = objective.best_result.total < before
        history.append({
            "iteration": iteration, "total": float(objective.best_result.total),
            "terms": dict(objective.best_result.terms), "accepted_moves": int(accepted),
            "rejected_moves": rejected, "step_size": float(step),
            "soft_proposal_total": float(soft_total), "soft_proposal_terms": soft_terms,
            "reason": "accepted" if accepted else "no_geometric_improvement",
        })
        if allowance.reason() is not None:
            termination = allowance.reason()
            break
        if not accepted:
            step *= config.step_decay
        if step < config.min_step:
            termination = "min_step"
            break
    # Every returned state is an actual scored state. The soft objective never
    # overrides admission, and rejected/failed proposals cannot replace it.
    final_loss = objective.best_loss
    summary = {
        "enabled": True, "reason": termination,
        "method": "mixed_family_contour_silhouette" if has_family_pullback else "analytic_projected_silhouette",
        "proposal_gradient_policy": (
            "analytic_ellipse_and_bounded_contour_parameter_difference" if has_family_pullback
            else "analytic_ellipse_parameter_pullback"),
        "proposal_objective": "mask_l2_product_soft_iou_fixed_target_distance",
        "objective_policy": _optimizer_objective_policy(parsed_config),
        "objective_evaluations": allowance.objective_evaluations,
        "budgeted_forward_evaluations": allowance.objective_evaluations,
        "admission_objective_evaluations": allowance.objective_evaluations - getattr(
            objective.cache, "derivative_evaluations", 0),
        "elapsed_s": float(time.perf_counter() - started),
        "initial_total": float(initial_loss.total), "final_total": float(final_loss.total),
        "initial_optimizer_total": float(initial.total),
        "final_optimizer_total": float(objective.best_result.total),
        "best_total": float(objective.best_result.total),
        "accepted_moves": sum(int(record["accepted_moves"]) for record in history),
        "history_length": len(history),
        "analytic_gradient_steps": gradient_steps if not has_family_pullback else 0,
        "gradient_proposal_steps": gradient_steps,
        "coordinate_fallbacks": coordinate_fallbacks,
        "contour_derivative_evaluations": getattr(objective.cache, "derivative_evaluations", 0),
        "contour_derivative_failures": tuple(getattr(objective.cache, "derivative_failures", ())),
        "contour_approximation_reports": [
            {"view": key[0], "part_index": key[1], **report}
            for key, report in getattr(objective.cache, "approximation_reports", {}).items()],
        "paired_coordinate_batches": objective.coordinate_batches,
        "failed_proposals": tuple(objective.failed_proposals),
        "recomputed_silhouette_components": objective.cache.recomputed_components,
        "reused_silhouette_components": objective.cache.reused_components,
        "config": {"iterations": config.iterations, "initial_step": config.initial_step,
                   "step_decay": config.step_decay, "min_step": config.min_step,
                   "max_objective_evaluations": config.max_objective_evaluations,
                   "max_elapsed_s": max_elapsed},
    }
    return {"optimized_primitives": tuple(objective.best_parts),
            "optimization_history": history, "optimization_summary": summary,
            "render_batch": objective.best_batch, "loss": final_loss}


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
