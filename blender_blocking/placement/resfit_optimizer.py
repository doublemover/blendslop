"""
Deterministic optimizers for duck-typed ResFit primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

import numpy as np

try:
    from .resfit_objective import ResFitObjectiveResult
except ImportError:  # pragma: no cover - supports direct script execution.
    from resfit_objective import ResFitObjectiveResult


ObjectiveFn = Callable[[Sequence[object]], ResFitObjectiveResult]
ParameterRef = Tuple[int, str, int | None]


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


@dataclass(frozen=True)
class CoordinateDescentConfig:
    """Derivative-free optimizer settings."""

    iterations: int = 20
    initial_step: float = 0.1
    step_decay: float = 0.5
    min_step: float = 1e-4
    bounds: ParameterBounds = field(default_factory=ParameterBounds)


@dataclass(frozen=True)
class OptimizationRecord:
    iteration: int
    total: float
    terms: dict[str, float]
    accepted_moves: int
    step_size: float


@dataclass(frozen=True)
class OptimizationResult:
    primitives: tuple[object, ...]
    history: tuple[OptimizationRecord, ...]
    best_loss: float


def clone_primitive(primitive: object) -> object:
    """Clone a primitive through its serialization API when available."""
    if hasattr(primitive, "to_dict") and hasattr(type(primitive), "from_dict"):
        return type(primitive).from_dict(primitive.to_dict())
    raise TypeError(f"primitive cannot be cloned through to_dict/from_dict: {type(primitive)!r}")


def clone_primitives(primitives: Sequence[object]) -> List[object]:
    return [clone_primitive(primitive) for primitive in primitives]


def discover_parameters(primitives: Sequence[object]) -> List[ParameterRef]:
    """Return mutable scalar parameter references for known primitive families."""
    refs: List[ParameterRef] = []
    for index, primitive in enumerate(primitives):
        if hasattr(primitive, "center"):
            refs.extend((index, "center", axis) for axis in range(3))
        if hasattr(primitive, "position"):
            refs.extend((index, "position", axis) for axis in range(3))
        if hasattr(primitive, "radii"):
            refs.extend((index, "radii", axis) for axis in range(3))
        for attr in ("radius_bottom", "radius_top", "height", "epsilon1", "epsilon2", "opacity"):
            if hasattr(primitive, attr):
                refs.append((index, attr, None))
    return refs


def _get_value(primitives: Sequence[object], ref: ParameterRef) -> float:
    index, attr, axis = ref
    value = getattr(primitives[index], attr)
    if axis is None:
        return float(value)
    return float(np.asarray(value, dtype=np.float64)[axis])


def _clip_value(attr: str, value: float, bounds: ParameterBounds) -> float:
    if attr in ("radii", "radius_bottom", "radius_top"):
        return float(np.clip(value, bounds.min_radius, bounds.max_radius))
    if attr == "height":
        return float(np.clip(value, bounds.min_height, bounds.max_height))
    if attr in ("epsilon1", "epsilon2"):
        return float(np.clip(value, bounds.min_exponent, bounds.max_exponent))
    if attr == "opacity":
        return float(np.clip(value, bounds.min_opacity, bounds.max_opacity))
    return float(value)


def _set_value(
    primitives: Sequence[object],
    ref: ParameterRef,
    value: float,
    bounds: ParameterBounds,
) -> None:
    index, attr, axis = ref
    value = _clip_value(attr, value, bounds)
    primitive = primitives[index]
    if axis is None:
        setattr(primitive, attr, value)
        return
    vector = np.asarray(getattr(primitive, attr), dtype=np.float64).copy()
    vector[axis] = value
    setattr(primitive, attr, vector)


def coordinate_descent_optimize(
    primitives: Sequence[object],
    objective_fn: ObjectiveFn,
    config: CoordinateDescentConfig = CoordinateDescentConfig(),
) -> OptimizationResult:
    """
    Optimize primitive parameters with bounded coordinate descent.

    This is intentionally simple and deterministic for early silhouette/surface
    refinement. It provides a stable baseline before analytic gradients exist.
    """
    working = clone_primitives(primitives)
    refs = discover_parameters(working)
    current = objective_fn(working)
    best_loss = current.total
    history: List[OptimizationRecord] = []
    step = float(config.initial_step)

    for iteration in range(max(0, config.iterations)):
        accepted = 0
        for ref in refs:
            original = _get_value(working, ref)
            best_value = original
            local_best = best_loss
            for direction in (1.0, -1.0):
                _set_value(working, ref, original + direction * step, config.bounds)
                trial = objective_fn(working)
                if trial.total < local_best:
                    local_best = trial.total
                    best_value = _get_value(working, ref)
            _set_value(working, ref, best_value, config.bounds)
            if local_best < best_loss:
                best_loss = local_best
                accepted += 1
            else:
                _set_value(working, ref, original, config.bounds)

        current = objective_fn(working)
        history.append(
            OptimizationRecord(
                iteration=iteration,
                total=current.total,
                terms=dict(current.terms),
                accepted_moves=accepted,
                step_size=step,
            )
        )
        if accepted == 0:
            step *= config.step_decay
        if step < config.min_step:
            break

    return OptimizationResult(
        primitives=tuple(working),
        history=tuple(history),
        best_loss=float(best_loss),
    )


def finite_difference_gradient(
    primitives: Sequence[object],
    objective_fn: ObjectiveFn,
    epsilon: float = 1e-5,
) -> dict[ParameterRef, float]:
    """Estimate objective gradients for discovered scalar parameters."""
    working = clone_primitives(primitives)
    refs = discover_parameters(working)
    base = objective_fn(working).total
    gradients: dict[ParameterRef, float] = {}
    bounds = ParameterBounds()
    for ref in refs:
        original = _get_value(working, ref)
        _set_value(working, ref, original + epsilon, bounds)
        plus = objective_fn(working).total
        _set_value(working, ref, original, bounds)
        gradients[ref] = float((plus - base) / epsilon)
    return gradients
