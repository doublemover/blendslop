"""
Deterministic optimizers for duck-typed ResFit primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import time
from typing import Callable, List, Sequence, Tuple

import numpy as np

try:
    from .resfit_objective import ResFitObjectiveResult
    from .resfit_parameters import (apply_parameter_increment, clip_parameter_value,
                                    discover_primitive_parameters, parameter_value,
                                    set_parameter_value)
except ImportError:  # pragma: no cover - supports direct script execution.
    from resfit_objective import ResFitObjectiveResult
    from resfit_parameters import (apply_parameter_increment, clip_parameter_value,
                                   discover_primitive_parameters, parameter_value,
                                   set_parameter_value)


ObjectiveFn = Callable[[Sequence[object]], ResFitObjectiveResult]
ParameterRef = Tuple[int, str, int | None]


def _validate_finite_float(value: object, name: str, errors: list[str]) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a real number, got {value!r}")
        return None
    if not np.isfinite(parsed):
        errors.append(f"{name} must be finite, got {value!r}")
        return None
    return parsed


def _validate_positive_int(value: object, name: str, errors: list[str]) -> int | None:
    if isinstance(value, bool):
        errors.append(f"{name} must be an integer, got {value!r}")
        return None
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        errors.append(f"{name} must be an integer, got {value!r}")
        return None
    if parsed <= 0:
        errors.append(f"{name} must be > 0, got {value!r}")
    return parsed


def _validate_non_negative_int(value: object, name: str, errors: list[str]) -> int | None:
    if isinstance(value, bool):
        errors.append(f"{name} must be an integer, got {value!r}")
        return None
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        errors.append(f"{name} must be an integer, got {value!r}")
        return None
    if parsed < 0:
        errors.append(f"{name} must be >= 0, got {value!r}")
    return parsed


def _validate_optional_positive_int(
    value: object,
    name: str,
    errors: list[str],
) -> int | None:
    if value is None:
        return None
    return _validate_positive_int(value, name, errors)


@dataclass(frozen=True)
class ParameterBounds:
    """Conservative default bounds for editable primitive parameters."""

    min_radius: float = 1e-3
    max_radius: float = 1e4
    min_height: float = 1e-3
    max_height: float = 1e4
    min_exponent: float = 0.05
    max_exponent: float = 4.0
    min_opacity: float = 0.0
    max_opacity: float = 1.0

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        min_radius = _validate_finite_float(self.min_radius, "min_radius", errors)
        max_radius = _validate_finite_float(self.max_radius, "max_radius", errors)
        min_height = _validate_finite_float(self.min_height, "min_height", errors)
        max_height = _validate_finite_float(self.max_height, "max_height", errors)
        min_exponent = _validate_finite_float(self.min_exponent, "min_exponent", errors)
        max_exponent = _validate_finite_float(self.max_exponent, "max_exponent", errors)
        min_opacity = _validate_finite_float(self.min_opacity, "min_opacity", errors)
        max_opacity = _validate_finite_float(self.max_opacity, "max_opacity", errors)
        if len(errors):
            return tuple(errors)
        assert min_radius is not None
        assert max_radius is not None
        assert min_height is not None
        assert max_height is not None
        assert min_exponent is not None
        assert max_exponent is not None
        assert min_opacity is not None
        assert max_opacity is not None
        if min_radius < 0.0:
            errors.append(f"min_radius must be >= 0.0, got {self.min_radius!r}")
        if max_radius <= min_radius:
            errors.append(
                "max_radius must be greater than min_radius, got "
                f"{self.max_radius!r} <= {self.min_radius!r}"
            )
        if min_height < 0.0:
            errors.append(f"min_height must be >= 0.0, got {self.min_height!r}")
        if max_height <= min_height:
            errors.append(
                "max_height must be greater than min_height, got "
                f"{self.max_height!r} <= {self.min_height!r}"
            )
        if min_exponent < 0.0:
            errors.append(
                f"min_exponent must be >= 0.0, got {self.min_exponent!r}"
            )
        if max_exponent <= min_exponent:
            errors.append(
                "max_exponent must be greater than min_exponent, got "
                f"{self.max_exponent!r} <= {self.min_exponent!r}"
            )
        if min_opacity < 0.0 or min_opacity > 1.0:
            errors.append(
                f"min_opacity must be in [0.0, 1.0], got {self.min_opacity!r}"
            )
        if max_opacity <= min_opacity or max_opacity > 1.0:
            errors.append(
                "max_opacity must be within (min_opacity, 1.0], got "
                f"{self.max_opacity!r} <= {self.min_opacity!r}"
            )
        return tuple(errors)


@dataclass(frozen=True)
class CoordinateDescentConfig:
    """Derivative-free optimizer settings."""

    iterations: int = 20
    initial_step: float = 0.1
    step_decay: float = 0.5
    min_step: float = 1e-4
    bounds: ParameterBounds = field(default_factory=ParameterBounds)
    max_objective_evaluations: int | None = None
    max_elapsed_s: float | None = None

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        _validate_non_negative_int(self.iterations, "iterations", errors)
        _validate_optional_positive_int(
            self.max_objective_evaluations,
            "max_objective_evaluations",
            errors,
        )
        initial_step = _validate_finite_float(self.initial_step, "initial_step", errors)
        step_decay = _validate_finite_float(self.step_decay, "step_decay", errors)
        min_step = _validate_finite_float(self.min_step, "min_step", errors)
        max_elapsed_s = None
        if self.max_elapsed_s is not None:
            max_elapsed_s = _validate_finite_float(
                self.max_elapsed_s,
                "max_elapsed_s",
                errors,
            )
        if len(errors):
            return tuple(errors)
        assert initial_step is not None
        assert step_decay is not None
        assert min_step is not None
        if max_elapsed_s is not None and max_elapsed_s <= 0.0:
            errors.append(f"max_elapsed_s must be > 0.0, got {self.max_elapsed_s!r}")
        if initial_step <= 0.0:
            errors.append(f"initial_step must be > 0.0, got {self.initial_step!r}")
        if not (0.0 < step_decay < 1.0):
            errors.append(
                f"step_decay must be in (0.0, 1.0), got {self.step_decay!r}"
            )
        if min_step <= 0.0 or min_step >= initial_step:
            errors.append(
                "min_step must be > 0.0 and < initial_step, got "
                f"{self.min_step!r} >= {self.initial_step!r}"
            )
        errors.extend(self.bounds.validate())
        return tuple(errors)


@dataclass(frozen=True)
class OptimizationRecord:
    iteration: int
    total: float
    terms: dict[str, float]
    accepted_moves: int
    step_size: float
    rejected_moves: int = 0
    reason: str = ""


@dataclass(frozen=True)
class OptimizationResult:
    primitives: tuple[object, ...]
    history: tuple[OptimizationRecord, ...]
    best_loss: float
    termination_reason: str = "max_iterations"
    objective_evaluations: int = 0
    elapsed_s: float = 0.0
    initial_result: ResFitObjectiveResult | None = None
    final_result: ResFitObjectiveResult | None = None
    parameter_visits: tuple[ParameterRef, ...] = ()


class OptimizationBudgetExhausted(RuntimeError):
    """Raised before starting an objective that would exceed a shared budget."""


@dataclass
class OptimizationBudget:
    """Cooperative deadline and objective count shared by nested searches.

    A running objective cannot be preempted. Failed objective calls still count.
    Child allowances cap a family/attempt without resetting their parent budget.
    """

    max_objective_evaluations: int | None = None
    max_elapsed_s: float | None = None
    parent: OptimizationBudget | None = None
    started_at: float = field(default_factory=time.perf_counter)
    objective_evaluations: int = 0

    def remaining_evaluations(self) -> int | None:
        own = (None if self.max_objective_evaluations is None else
               max(0, self.max_objective_evaluations - self.objective_evaluations))
        inherited = None if self.parent is None else self.parent.remaining_evaluations()
        available = [value for value in (own, inherited) if value is not None]
        return min(available) if available else None

    def remaining_seconds(self) -> float | None:
        own = (None if self.max_elapsed_s is None else
               max(0.0, self.max_elapsed_s - (time.perf_counter() - self.started_at)))
        inherited = None if self.parent is None else self.parent.remaining_seconds()
        available = [value for value in (own, inherited) if value is not None]
        return min(available) if available else None

    def reason(self) -> str | None:
        count = self.remaining_evaluations()
        if count is not None and count <= 0:
            return "objective_evaluation_budget"
        seconds = self.remaining_seconds()
        if seconds is not None and seconds <= 0:
            return "elapsed_time_budget"
        return None

    def child(self, *, slots: int = 1) -> OptimizationBudget:
        slots = max(1, int(slots))
        count = self.remaining_evaluations()
        seconds = self.remaining_seconds()
        # An allowance smaller than the slot count can score only some starts.
        # Never manufacture one extra evaluation for each exhausted child.
        return OptimizationBudget(
            max_objective_evaluations=(None if count is None else
                                       max(1, count // slots) if count else 0),
            max_elapsed_s=None if seconds is None else seconds / slots,
            parent=self,
        )

    def _record_evaluation(self) -> None:
        self.objective_evaluations += 1
        if self.parent is not None:
            self.parent._record_evaluation()

    def evaluate(self, objective_fn: ObjectiveFn, primitives: Sequence[object]) -> ResFitObjectiveResult:
        reason = self.reason()
        if reason is not None:
            raise OptimizationBudgetExhausted(reason)
        self._record_evaluation()
        return objective_fn(primitives)


def clone_primitive(primitive: object) -> object:
    """Clone a primitive through its serialization API when available."""
    if hasattr(primitive, "to_dict") and hasattr(type(primitive), "from_dict"):
        return type(primitive).from_dict(primitive.to_dict())
    raise TypeError(f"primitive cannot be cloned through to_dict/from_dict: {type(primitive)!r}")


def clone_primitives(primitives: Sequence[object]) -> List[object]:
    return [clone_primitive(primitive) for primitive in primitives]


def discover_parameters(primitives: Sequence[object]) -> List[ParameterRef]:
    """Return family-specific scalar references, including coupled geometry."""
    return discover_primitive_parameters(primitives)


def ordered_parameter_refs(primitives, *, priority_parts=()):
    """Changed parts first; other parts share control coverage round-robin."""
    groups = {}
    for ref in discover_parameters(primitives):
        groups.setdefault(ref[0], []).append(ref)
    ordered, seen = [], set()
    for index in priority_parts:
        if index in groups and index not in seen:
            ordered.extend(groups.pop(index)); seen.add(index)
    while any(groups.values()):
        for index in sorted(groups):
            if groups[index]:
                ordered.append(groups[index].pop(0))
    return ordered


def _get_value(primitives: Sequence[object], ref: ParameterRef) -> float:
    index, attr, axis = ref
    return parameter_value(primitives[index], attr, axis)


def _clip_value(attr: str, value: float, bounds: ParameterBounds) -> float:
    return clip_parameter_value(attr, value, bounds)


def _set_value(
    primitives: Sequence[object],
    ref: ParameterRef,
    value: float,
    bounds: ParameterBounds,
) -> None:
    index, attr, axis = ref
    set_parameter_value(primitives[index], attr, axis, value, bounds)


def coordinate_descent_optimize(
    primitives: Sequence[object],
    objective_fn: ObjectiveFn,
    config: CoordinateDescentConfig = CoordinateDescentConfig(),
    *,
    budget: OptimizationBudget | None = None,
    on_progress: Callable | None = None,
    priority_parts: Sequence[int] = (),
) -> OptimizationResult:
    """Optimize with a shared budget, optional paired scoring, and retained losses.

    An objective may expose ``evaluate_batch(candidates, *, budget)``. It must
    use ``budget.evaluate`` per score and yield results in candidate order,
    returning a completed prefix if the allowance expires. Batches contain two
    competing directions of one coordinate, never conflicting admitted moves.
    """
    config_errors = config.validate()
    if config_errors:
        raise ValueError("invalid optimizer config: " + ", ".join(config_errors))
    start = time.perf_counter()
    allowance = OptimizationBudget(
        max_objective_evaluations=config.max_objective_evaluations,
        max_elapsed_s=config.max_elapsed_s,
        parent=budget,
    )
    working = clone_primitives(primitives)
    refs = ordered_parameter_refs(working, priority_parts=priority_parts)
    visited = []
    initial = allowance.evaluate(objective_fn, working)
    current = initial
    history: List[OptimizationRecord] = []
    step = float(config.initial_step)
    termination_reason = "zero_iterations" if config.iterations == 0 else "max_iterations"
    stopped = False

    def checkpoint():
        if on_progress is not None and np.isfinite(current.total):
            on_progress(OptimizationResult(primitives=tuple(clone_primitives(working)),
                history=tuple(history), best_loss=float(current.total),
                termination_reason="scored_checkpoint", objective_evaluations=allowance.objective_evaluations,
                elapsed_s=float(time.perf_counter() - start), initial_result=initial, final_result=current,
                parameter_visits=tuple(visited)))

    checkpoint()
    for iteration in range(max(0, config.iterations)):
        reason = allowance.reason()
        if reason is not None:
            termination_reason = reason
            break
        accepted = 0
        rejected = 0
        for ref in refs:
            reason = allowance.reason()
            if reason is not None:
                termination_reason = reason
                stopped = True
                break
            visited.append(ref)
            index = ref[0]
            baseline = deepcopy(working[index])
            best_primitive = baseline
            local_best = current
            # Competing directions share one immutable base. Only this
            # coordinate's winner is admitted before the next base is built.
            proposals = [deepcopy(working), deepcopy(working)]
            for proposal, direction in zip(proposals, (1.0, -1.0)):
                apply_parameter_increment(proposal, ref, direction * step, config.bounds)
            batch_evaluate = getattr(objective_fn, "evaluate_batch", None)
            completed = 0
            try:
                if callable(batch_evaluate):
                    # The optional method yields an ordered completed prefix,
                    # and must call allowance.evaluate once for each score.
                    trials = batch_evaluate(proposals, budget=allowance)
                else:
                    trials = (allowance.evaluate(objective_fn, proposal) for proposal in proposals)
                for trial in trials:
                    if completed >= len(proposals):
                        raise ValueError("objective batch returned too many results")
                    proposal = proposals[completed]
                    completed += 1
                    if trial.total < local_best.total:
                        local_best = trial
                        best_primitive = deepcopy(proposal[index])
                    else:
                        rejected += 1
            except OptimizationBudgetExhausted as exc:
                termination_reason = str(exc)
                stopped = True
            if completed < len(proposals):
                termination_reason = allowance.reason() or "incomplete_coordinate_batch"
                stopped = True
            # A first-direction winner survives a deadline/count cap that
            # prevents the second direction from being scored.
            working[index] = best_primitive
            if local_best.total < current.total:
                current = local_best
                accepted += 1
                checkpoint()
            if stopped:
                break

        history.append(
            OptimizationRecord(
                iteration=iteration,
                total=current.total,
                terms=dict(current.terms),
                accepted_moves=accepted,
                step_size=step,
                rejected_moves=rejected,
                reason=termination_reason if stopped else "accepted" if accepted else "no_coordinate_improved",
            )
        )
        if stopped:
            break
        if accepted == 0:
            step *= config.step_decay
        if step < config.min_step:
            termination_reason = "min_step"
            break

    return OptimizationResult(
        primitives=tuple(working),
        history=tuple(history),
        best_loss=float(current.total),
        termination_reason=termination_reason,
        objective_evaluations=allowance.objective_evaluations,
        elapsed_s=float(time.perf_counter() - start),
        initial_result=initial,
        final_result=current,
        parameter_visits=tuple(visited),
    )


def finite_difference_gradient(
    primitives: Sequence[object],
    objective_fn: ObjectiveFn,
    epsilon: float = 1e-5,
) -> dict[ParameterRef, float]:
    """Estimate gradients with respect to dimensionless local adapter increments."""
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    working = clone_primitives(primitives)
    refs = discover_parameters(working)
    base = objective_fn(working).total
    gradients: dict[ParameterRef, float] = {}
    bounds = ParameterBounds()
    for ref in refs:
        index = ref[0]
        baseline = deepcopy(working[index])
        apply_parameter_increment(working, ref, epsilon, bounds)
        plus = objective_fn(working).total
        working[index] = baseline
        gradients[ref] = float((plus - base) / epsilon)
    return gradients
