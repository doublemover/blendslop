from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from blender_blocking.config_models.coercion import (
        coerce_float as _coerce_float,
        coerce_int as _coerce_int,
        coerce_optional_float as _shared_coerce_optional_float,
        coerce_optional_int as _shared_coerce_optional_int,
    )
except ImportError:  # pragma: no cover - script-style imports
    from config_models.coercion import (
        coerce_float as _coerce_float,
        coerce_int as _coerce_int,
        coerce_optional_float as _shared_coerce_optional_float,
        coerce_optional_int as _shared_coerce_optional_int,
    )

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


@dataclass(frozen=True)
class ResFitPipelineConfig:
    """Top-level primitive fitting configuration."""

    primitive_family: str = "superfrustum"
    initialization: PrimitiveInitializationConfig = field(
        default_factory=PrimitiveInitializationConfig
    )
    optimizer: CoordinateDescentConfig = field(default_factory=CoordinateDescentConfig)
    weights: ResFitLossWeights = field(default_factory=ResFitLossWeights)
    fail_on_regression: bool = True
    max_primitives: int = 12
    residual_rounds: int = 1
    residual_refinement_steps: int = 1
    max_residual_proposals: int = 3
    objective_mode: str = "legacy_world_squared"
    length_scale: float | None = None
    refinement_strategy: str = "coordinate"

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not str(self.primitive_family).strip():
            errors.append("primitive_family is empty")
        errors.extend(self.initialization.validate())
        errors.extend(self.optimizer.validate())
        errors.extend(self.weights.validate())
        if self.objective_mode not in {"legacy_world_squared", "normalized_area_v1"}:
            errors.append("unsupported objective_mode")
        if self.refinement_strategy not in {"coordinate", "coupled_blocks"}:
            errors.append("unsupported refinement_strategy")
        if self.length_scale is not None and (not np.isfinite(self.length_scale) or self.length_scale <= 0.):
            errors.append("length_scale must be positive and finite")
        if not isinstance(self.fail_on_regression, bool):
            errors.append(f"fail_on_regression must be bool, got {type(self.fail_on_regression)!r}")
        if self.max_primitives < 1:
            errors.append("max_primitives must be positive")
        if min(self.residual_rounds, self.residual_refinement_steps, self.max_residual_proposals) < 0:
            errors.append("residual search limits must be nonnegative")
        return tuple(errors)


@dataclass(frozen=True)
class ResFitPipelineResult:
    primitives: tuple[object, ...]
    initial_loss: ResFitObjectiveResult
    final_loss: ResFitObjectiveResult
    history: tuple[OptimizationRecord, ...]
    warnings: tuple[str, ...]
    optimization_termination_reason: str = "not_run"
    objective_evaluations: int = 0
    optimizer_elapsed_s: float = 0.0
    selected_attempt: str = "default"
    attempts: tuple[Mapping[str, Any], ...] = ()
    family_attempts: tuple[Mapping[str, Any], ...] = ()
    search_budget: Mapping[str, Any] = field(default_factory=dict)
    residual_proposals: tuple[Mapping[str, Any], ...] = ()
    parameter_visits: tuple = ()

    def primitive_dicts(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            primitive.to_dict()
            for primitive in self.primitives
            if hasattr(primitive, "to_dict")
        )


def _pipeline_config_summary(config: ResFitPipelineConfig) -> Mapping[str, Any]:
    return {
        "primitive_family": config.primitive_family,
        "objective_mode": config.objective_mode,
        "refinement_strategy": config.refinement_strategy,
        "length_scale": config.length_scale,
        "max_primitives": config.max_primitives,
        "residual_rounds": config.residual_rounds,
        "residual_refinement_steps": config.residual_refinement_steps,
        "max_residual_proposals": config.max_residual_proposals,
        "initialization": {
            "primitive_count": config.initialization.primitive_count,
            "target_point_count": config.initialization.target_point_count,
            "min_radius": config.initialization.min_radius,
            "covariance_floor": config.initialization.covariance_floor,
            "kmeans_iterations": config.initialization.kmeans_iterations,
            "kmeans_seed": config.initialization.kmeans_seed,
        },
        "optimizer": {
            "iterations": config.optimizer.iterations,
            "initial_step": config.optimizer.initial_step,
            "step_decay": config.optimizer.step_decay,
            "min_step": config.optimizer.min_step,
            "max_objective_evaluations": config.optimizer.max_objective_evaluations,
            "max_elapsed_s": config.optimizer.max_elapsed_s,
            "bounds": {
                "min_radius": config.optimizer.bounds.min_radius,
                "max_radius": config.optimizer.bounds.max_radius,
                "min_height": config.optimizer.bounds.min_height,
                "max_height": config.optimizer.bounds.max_height,
                "min_exponent": config.optimizer.bounds.min_exponent,
                "max_exponent": config.optimizer.bounds.max_exponent,
                "min_opacity": config.optimizer.bounds.min_opacity,
                "max_opacity": config.optimizer.bounds.max_opacity,
            },
        },
        "weights": {
            "surface_residual": config.weights.surface_residual,
            "visual_hull_occupancy": config.weights.visual_hull_occupancy,
            "primitive_count": config.weights.primitive_count,
            "overlap_penalty": config.weights.overlap_penalty,
            "silhouette": config.weights.silhouette,
            "topology_penalty": config.weights.topology_penalty,
            "constraint_penalty": config.weights.constraint_penalty,
            "uncertainty_penalty": config.weights.uncertainty_penalty,
        },
        "fail_on_regression": config.fail_on_regression,
    }


def _coerce_optional_int(
    value: Any,
    name: str,
    *,
    min_value: int,
    max_value: int,
    errors: list[str],
) -> int | None:
    return _shared_coerce_optional_int(
        value,
        name,
        errors,
        default=min_value,
        min_value=min_value,
        max_value=max_value,
    )


def _coerce_optional_float(
    value: Any,
    name: str,
    *,
    min_value: float,
    max_value: float,
    errors: list[str],
) -> float | None:
    return _shared_coerce_optional_float(
        value,
        name,
        errors,
        default=min_value,
        min_value=min_value,
        max_value=max_value,
    )


def _coerce_weight(
    config: Mapping[str, Any],
    loss_weights: Mapping[str, Any],
    canonical: str,
    *,
    aliases: Sequence[str] = (),
    default: float,
    errors: list[str],
) -> float:
    flat_keys = (f"{canonical}_weight",) + tuple(
        f"{alias}_weight" for alias in aliases
    )
    for key in flat_keys:
        if key in config:
            return _coerce_float(
                config.get(key),
                key,
                default=default,
                min_value=0.0,
                max_value=1e6,
                errors=errors,
            )

    for key in (canonical,) + tuple(aliases):
        if key in loss_weights:
            return _coerce_float(
                loss_weights.get(key),
                f"loss_weights.{key}",
                default=default,
                min_value=0.0,
                max_value=1e6,
                errors=errors,
            )

    return default


def by_view_total(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0
