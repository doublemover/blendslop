"""
Orchestration for modular residual primitive fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from .resfit_initialization import (
        PrimitiveInitializationConfig,
        initialize_ellipsoids_from_points,
        initialize_from_profile_bands,
        initialize_gaussians_from_points,
        initialize_superfrusta_from_points,
    )
    from .resfit_objective import (
        PenaltyHook,
        ResFitLossWeights,
        ResFitObjectiveResult,
        SilhouetteHook,
        evaluate_resfit_objective,
    )
    from .resfit_optimizer import (
        CoordinateDescentConfig,
        OptimizationRecord,
        coordinate_descent_optimize,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from resfit_initialization import (
        PrimitiveInitializationConfig,
        initialize_ellipsoids_from_points,
        initialize_from_profile_bands,
        initialize_gaussians_from_points,
        initialize_superfrusta_from_points,
    )
    from resfit_objective import (
        PenaltyHook,
        ResFitLossWeights,
        ResFitObjectiveResult,
        SilhouetteHook,
        evaluate_resfit_objective,
    )
    from resfit_optimizer import (
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

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not str(self.primitive_family).strip():
            errors.append("primitive_family is empty")
        errors.extend(self.initialization.validate())
        errors.extend(self.optimizer.validate())
        errors.extend(self.weights.validate())
        if not isinstance(self.fail_on_regression, bool):
            errors.append(f"fail_on_regression must be bool, got {type(self.fail_on_regression)!r}")
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

    def primitive_dicts(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            primitive.to_dict()
            for primitive in self.primitives
            if hasattr(primitive, "to_dict")
        )


InitializerFn = Callable[[np.ndarray, PrimitiveInitializationConfig], Sequence[object]]


def get_initializer(family: str) -> InitializerFn:
    """Return the initializer for a primitive family."""
    normalized = family.lower().strip()
    if normalized in ("superfrustum", "superfrusta", "frustum"):
        return initialize_superfrusta_from_points
    if normalized in ("ellipsoid", "ellipsoids"):
        return initialize_ellipsoids_from_points
    if normalized in ("gaussian", "gaussians", "anisotropic_gaussian"):
        return initialize_gaussians_from_points
    raise ValueError(f"unknown primitive family: {family}")


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


def run_primitive_fit_pipeline(request: object) -> object:
    """CandidateResult adapter used by the reconstruction backend registry."""
    from reconstruction.artifacts import write_json
    from reconstruction.mesh_io import (
        combine_primitive_meshes,
        write_obj,
        write_primitive_set,
    )
    from reconstruction.point_cloud import target_occupied_points, target_surface_points
    from reconstruction.types import CandidateMetrics, CandidateResult

    start = time.perf_counter()
    config = dict(getattr(request, "config", {}) or {})
    target = getattr(request, "target", None)
    candidate_id = getattr(request, "candidate_id")
    backend_name = getattr(request, "backend_name", "primitive_fit_refine")

    if target is None:
        return CandidateResult(
            candidate_id=candidate_id,
            backend_name=backend_name,
            status="failed",
            errors=("missing target in request",),
        )

    primitive_family = _first_family(config)
    errors: list[str] = []
    primitive_count = _coerce_int(
        config.get("primitive_count", config.get("max_primitives", 6)),
        "primitive_count",
        default=6,
        min_value=1,
        max_value=4096,
        errors=errors,
    )
    target_point_count = _coerce_int(
        config.get("target_point_count", 4096),
        "target_point_count",
        default=4096,
        min_value=64,
        max_value=65535,
        errors=errors,
    )
    surface_resolution = _coerce_int(
        config.get("visual_hull_resolution", config.get("resolution", 48)),
        "visual_hull_resolution",
        default=48,
        min_value=8,
        max_value=1024,
        errors=errors,
    )
    occupied_resolution = _coerce_int(
        config.get("occupied_resolution", max(8, min(surface_resolution, 40))),
        "occupied_resolution",
        default=max(8, min(surface_resolution, 40)),
        min_value=8,
        max_value=512,
        errors=errors,
    )
    optimization_steps = _coerce_int(
        config.get("optimization_steps", 30),
        "optimization_steps",
        default=30,
        min_value=0,
        max_value=500,
        errors=errors,
    )
    max_runtime_s = _coerce_optional_float(
        config.get(
            "max_runtime_s",
            getattr(getattr(request, "budget", None), "timeout_s", None),
        ),
        "max_runtime_s",
        min_value=1e-3,
        max_value=24 * 60 * 60,
        errors=errors,
    )
    max_objective_evaluations = _coerce_optional_int(
        config.get("max_objective_evaluations"),
        "max_objective_evaluations",
        min_value=1,
        max_value=1_000_000,
        errors=errors,
    )
    loss_weights = config.get("loss_weights", {})
    if loss_weights is None:
        loss_weights = {}
    if not isinstance(loss_weights, Mapping):
        errors.append("loss_weights must be a mapping when provided")
        loss_weights = {}
    weights = ResFitLossWeights(
        surface_residual=_coerce_weight(
            config,
            loss_weights,
            "surface_residual",
            aliases=("surface",),
            default=1.0,
            errors=errors,
        ),
        visual_hull_occupancy=_coerce_weight(
            config,
            loss_weights,
            "visual_hull_occupancy",
            aliases=("occupancy", "volume", "visual_hull"),
            default=0.05,
            errors=errors,
        ),
        primitive_count=_coerce_weight(
            config,
            loss_weights,
            "primitive_count",
            aliases=("complexity", "count"),
            default=0.01,
            errors=errors,
        ),
        overlap_penalty=_coerce_weight(
            config,
            loss_weights,
            "overlap_penalty",
            aliases=("overlap",),
            default=0.05,
            errors=errors,
        ),
        silhouette=_coerce_weight(
            config,
            loss_weights,
            "silhouette",
            aliases=("profile",),
            default=0.0,
            errors=errors,
        ),
        topology_penalty=_coerce_weight(
            config,
            loss_weights,
            "topology_penalty",
            aliases=("topology",),
            default=0.0,
            errors=errors,
        ),
        constraint_penalty=_coerce_weight(
            config,
            loss_weights,
            "constraint_penalty",
            aliases=("constraint",),
            default=0.05,
            errors=errors,
        ),
        uncertainty_penalty=_coerce_weight(
            config,
            loss_weights,
            "uncertainty_penalty",
            aliases=("uncertainty",),
            default=0.0,
            errors=errors,
        ),
    )
    init_config = PrimitiveInitializationConfig(
        primitive_count=max(1, primitive_count),
        target_point_count=target_point_count,
        min_radius=_coerce_float(
            config.get("min_radius", 0.05),
            "min_radius",
            default=0.05,
            min_value=1e-6,
            max_value=1e3,
            errors=errors,
        ),
        covariance_floor=_coerce_float(
            config.get("covariance_floor", 1e-4),
            "covariance_floor",
            default=1e-4,
            min_value=0.0,
            max_value=1e3,
            errors=errors,
        ),
        kmeans_iterations=_coerce_int(
            config.get("kmeans_iterations", 8),
            "kmeans_iterations",
            default=8,
            min_value=1,
            max_value=128,
            errors=errors,
        ),
    )
    optimizer_config = CoordinateDescentConfig(
        iterations=optimization_steps,
        initial_step=_coerce_float(
            config.get("initial_step", 0.1),
            "initial_step",
            default=0.1,
            min_value=1e-6,
            max_value=1.0,
            errors=errors,
        ),
        step_decay=_coerce_float(
            config.get("step_decay", 0.5),
            "step_decay",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            errors=errors,
        ),
        min_step=_coerce_float(
            config.get("min_step", 1e-4),
            "min_step",
            default=1e-4,
            min_value=1e-8,
            max_value=0.1,
            errors=errors,
        ),
        max_objective_evaluations=max_objective_evaluations,
        max_elapsed_s=max_runtime_s,
    )

    pipeline_config = ResFitPipelineConfig(
        primitive_family=primitive_family,
        initialization=init_config,
        optimizer=optimizer_config,
        weights=weights,
        fail_on_regression=bool(config.get("fail_on_regression", True)),
    )
    config_validation = list(errors)
    config_validation.extend(pipeline_config.validate())
    if config_validation:
        return CandidateResult(
            candidate_id=candidate_id,
            backend_name=backend_name,
            status="failed",
            errors=tuple(dict.fromkeys(config_validation)),
        )

    try:
        surface, surface_meta = target_surface_points(
            target,
            resolution=surface_resolution,
            max_points=target_point_count,
            chunk_size=config.get("chunk_size"),
        )
        occupied, occupied_meta = target_occupied_points(
            target,
            resolution=occupied_resolution,
            max_points=target_point_count,
            chunk_size=config.get("chunk_size"),
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id=candidate_id,
            backend_name=backend_name,
            status="failed",
            errors=(str(exc),),
        )

    signal_summary = _collect_target_signals(target)
    profile_rows = _collect_profile_rows(signal_summary["profile"], getattr(target, "bounds", None))

    initial_primitives: Sequence[object] | None = None
    if _family_supports_profile_init(primitive_family) and profile_rows:
        try:
            slice_data = _profile_rows_to_slices(profile_rows, init_config)
            if slice_data:
                initial_primitives = initialize_from_profile_bands(
                    slice_data,
                    init_config,
                )
        except Exception as exc:
            # Keep behavior stable: degrade to canonical seeding if profile init fails.
            initial_primitives = None
            profile_init_warning = f"profile initialization failed: {exc}"
        else:
            profile_init_warning = ""
    else:
        profile_init_warning = ""

    uncertainty_signal = signal_summary["uncertainty"]
    constraint_signal = signal_summary["constraints"]
    topology_signal = signal_summary["topology"]

    silhouette_hook = None
    if profile_rows:
        silhouette_hook = _build_profile_silhouette_hook(
            profile_rows=profile_rows,
            uncertainty_by_view=uncertainty_signal["view_details"],
        )

    topology_penalty_hook = _build_topology_penalty_hook(topology_signal)
    constraint_penalty_hook = _build_constraint_penalty_hook(
        constraint_signal=constraint_signal,
        bounds=getattr(target, "bounds", None),
    )
    uncertainty_penalty_hook = _build_uncertainty_penalty_hook(uncertainty_signal)

    try:
        result = fit_residual_primitives(
            surface,
            pipeline_config,
            initial_primitives=initial_primitives,
            occupied_points=occupied,
            silhouette_hook=silhouette_hook,
            topology_penalty_hook=topology_penalty_hook,
            constraint_penalty_hook=constraint_penalty_hook,
            uncertainty_penalty_hook=uncertainty_penalty_hook,
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id=candidate_id,
            backend_name=backend_name,
            status="failed",
            errors=(str(exc),),
        )

    root = request.candidate_artifact_root()
    primitive_path = None
    mesh_path = None
    artifacts: dict[str, Any] = {}
    mesh_proxy = combine_primitive_meshes(result.primitives, resolution=24)
    if root is not None:
        primitive_path = write_primitive_set(
            root / "primitives" / "primitive-fit.json",
            result.primitives,
            metadata={
                "family": primitive_family,
                "initial_loss": result.initial_loss.terms,
                "final_loss": result.final_loss.terms,
                "surface_points": surface_meta,
                "occupied_points": occupied_meta,
                "signal_summary": signal_summary,
            },
        )
        mesh_path = write_obj(
            root / "mesh" / "primitive-fit.obj",
            mesh_proxy,
            header=(f"candidate {candidate_id}", backend_name),
        )
        artifacts["primitive_json"] = primitive_path
        artifacts["mesh_obj"] = mesh_path
        objective_path = write_json(
            root / "artifacts" / "primitive-fit-objective.json",
            {
                "weights": {
                    "surface_residual": pipeline_config.weights.surface_residual,
                    "visual_hull_occupancy": pipeline_config.weights.visual_hull_occupancy,
                    "primitive_count": pipeline_config.weights.primitive_count,
                    "overlap_penalty": pipeline_config.weights.overlap_penalty,
                    "silhouette": pipeline_config.weights.silhouette,
                    "topology_penalty": pipeline_config.weights.topology_penalty,
                    "constraint_penalty": pipeline_config.weights.constraint_penalty,
                    "uncertainty_penalty": pipeline_config.weights.uncertainty_penalty,
                },
                "initial_loss": result.initial_loss.terms,
                "final_loss": result.final_loss.terms,
                "improvement": result.initial_loss.total - result.final_loss.total,
                "optimization": {
                    "termination_reason": result.optimization_termination_reason,
                    "objective_evaluations": result.objective_evaluations,
                    "elapsed_s": result.optimizer_elapsed_s,
                    "max_runtime_s": max_runtime_s,
                    "max_objective_evaluations": max_objective_evaluations,
                },
                "history": [
                    {
                        "iteration": record.iteration,
                        "total": record.total,
                        "terms": dict(record.terms),
                        "accepted_moves": record.accepted_moves,
                        "step_size": record.step_size,
                    }
                    for record in result.history
                ],
                "signal_summary": signal_summary,
                "initial_primitives_from_profile": bool(initial_primitives),
            },
        )
        artifacts["objective_history"] = objective_path

    history_records = [
        {
            "iteration": record.iteration,
            "total": record.total,
            "terms": dict(record.terms),
            "accepted_moves": record.accepted_moves,
            "step_size": record.step_size,
        }
        for record in result.history
    ]
    improved = result.initial_loss.total - result.final_loss.total
    improvement_ratio = (
        improved / result.initial_loss.total
        if result.initial_loss.total > 0.0
        else 0.0
    )
    topology_score = float(
        np.clip(1.0 - result.final_loss.terms.get("topology_penalty", 0.0), 0.0, 1.0)
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

    warnings = list(result.warnings)
    if profile_init_warning:
        warnings.append(profile_init_warning)
    if improved <= 0.0:
        warnings.append("objective did not improve during refinement")
    if improved >= 0.0 and not profile_rows:
        warnings.append("using geometric seeding; no profile constraints were available")
    if result.optimization_termination_reason in {
        "elapsed_time_budget",
        "objective_evaluation_budget",
    }:
        warnings.append(
            f"optimization stopped by {result.optimization_termination_reason}"
        )
    status = "success" if result.primitives else "skipped"
    degraded = False
    errors_out: tuple[str, ...] = ()
    if result.primitives and improved <= 0.0:
        no_improvement_message = "primitive fit objective did not improve"
        if bool(
            config.get("require_objective_improvement")
            or config.get("fail_on_no_improvement")
        ):
            status = "failed"
            errors_out = (no_improvement_message,)
        else:
            status = "degraded"
            degraded = True
    if (
        result.primitives
        and result.optimization_termination_reason
        in {"elapsed_time_budget", "objective_evaluation_budget"}
        and status == "success"
    ):
        status = "degraded"
        degraded = True

    elapsed = time.perf_counter() - start
    metric = CandidateMetrics(
        area_iou_min=area_iou,
        area_iou_mean=area_iou,
        boundary_iou_mean=boundary_iou,
        topology_score=topology_score,
        uncertainty_consistency=uncertainty_consistency,
        constraint_score=constraint_score,
        editability_score=0.9,
        complexity_penalty=min(1.0, len(result.primitives) / 64.0),
        elapsed_s=elapsed,
        extras={
            "family": primitive_family,
            "primitive_count": len(result.primitives),
            "initial_primitive_count": (
                len(initial_primitives) if initial_primitives is not None else None
            ),
            "initial_loss": result.initial_loss.terms,
            "final_loss": result.final_loss.terms,
            "initial_total": result.initial_loss.total,
            "final_total": result.final_loss.total,
            "objective_improvement": improved,
            "objective_improvement_ratio": improvement_ratio,
            "surface_proxy_iou": surface_proxy_iou,
            "silhouette_proxy_iou": silhouette_proxy_iou,
            "optimization_termination_reason": result.optimization_termination_reason,
            "objective_evaluations": result.objective_evaluations,
            "optimizer_elapsed_s": result.optimizer_elapsed_s,
            "max_runtime_s": max_runtime_s,
            "max_objective_evaluations": max_objective_evaluations,
            "history": history_records,
            "surface_points": surface_meta,
            "occupied_points": occupied_meta,
            "signal_summary": signal_summary,
            "per_view": _build_per_view_profile_summary(profile_rows),
            "configuration": _pipeline_config_summary(pipeline_config),
            "signal_rows_used": len(profile_rows),
        },
        per_view=_build_per_view_profile_summary(profile_rows),
    )
    return CandidateResult(
        candidate_id=candidate_id,
        backend_name=backend_name,
        status=status,
        primitive_path=primitive_path,
        mesh_path=mesh_path,
        metric_result=metric,
        artifacts=artifacts,
        warnings=tuple(warnings),
        errors=errors_out,
        degraded=degraded,
        payload=result,
    )


def _build_profile_silhouette_hook(
    *,
    profile_rows: Sequence[Mapping[str, Any]],
    uncertainty_by_view: Mapping[str, Mapping[str, Any]],
) -> Callable[[Sequence[object]], Mapping[str, float]]:
    def hook(primitives: Sequence[object]) -> Mapping[str, float]:
        terms: dict[str, float] = {}
        if not profile_rows:
            return terms

        by_view: dict[str, list[float]] = {}
        for row in profile_rows:
            view = str(row.get("view", "generic"))
            z_world = float(row.get("z_world", 0.0))
            target_width = float(row.get("width_world", 0.0))
            target_center = float(row.get("center_x_world", 0.0))
            base_conf = float(row.get("confidence", 1.0))
            view_signal = uncertainty_by_view.get(view, {})
            confidence = float(view_signal.get("mean_confidence", 1.0))
            weight = float(base_conf * confidence)
            if weight <= 0.0:
                continue

            predicted_width = 0.0
            predicted_centers: list[float] = []
            for primitive in primitives:
                if not hasattr(primitive, "profile_width_at_world_z"):
                    continue
                predicted_width = max(
                    predicted_width,
                    float(primitive.profile_width_at_world_z(z_world)),
                )
                position = np.asarray(getattr(primitive, "position", (0.0, 0.0, 0.0)))
                if position.size == 3:
                    predicted_centers.append(float(position[0]))
            if not predicted_centers:
                if predicted_width == 0.0:
                    predicted_width = 0.0
                predicted_center = 0.0
            else:
                predicted_center = float(np.mean(predicted_centers))

            if target_width <= 0.0:
                width_term = min(1.0, abs(predicted_width) * 0.1)
            else:
                normalized_width = (predicted_width - target_width) / target_width
                width_term = normalized_width * normalized_width

            width_scale = max(1e-3, target_width)
            center_term = ((predicted_center - target_center) / width_scale) ** 2
            row_term = width_term + 0.5 * center_term
            by_view.setdefault(view, []).append(weight * row_term)

        for view, values in by_view.items():
            if not values:
                terms[f"profile_{view}"] = 0.0
            else:
                terms[f"profile_{view}"] = float(np.mean(values))
        terms["profile"] = float(sum(by_view_total(values) for values in by_view.values()))
        return terms

    return hook


def _build_topology_penalty_hook(topology_signal: Mapping[str, Any]) -> PenaltyHook:
    topology_target = float(topology_signal.get("score", 1.0))
    complexity = float(topology_signal.get("complexity", 0.0))

    def hook(primitives: Sequence[object]) -> float:
        if topology_target <= 0.0 and not complexity:
            return 0.0
        count = float(len(primitives))
        return float((1.0 - topology_target) + 0.01 * complexity * count)

    return hook


def _build_constraint_penalty_hook(
    *,
    constraint_signal: Mapping[str, Any],
    bounds: Any | None,
) -> PenaltyHook:
    score = float(constraint_signal.get("score", 1.0))
    count = int(constraint_signal.get("constraint_count", 0))
    has_bounds = bounds is not None
    bounds_min = None
    bounds_max = None
    if has_bounds:
        bounds_min = np.array([bounds.min_x, bounds.min_y, bounds.min_z], dtype=float)
        bounds_max = np.array([bounds.max_x, bounds.max_y, bounds.max_z], dtype=float)
        extents = bounds_max - bounds_min
        extents = np.where(extents > 0.0, extents, 1.0)

    def hook(primitives: Sequence[object]) -> float:
        base = 0.0
        for primitive in primitives:
            if not hasattr(primitive, "position"):
                continue
            if has_bounds:
                position = np.asarray(primitive.position, dtype=float)
                if position.size == 3:
                    below = np.maximum(bounds_min - position, 0.0)
                    above = np.maximum(position - bounds_max, 0.0)
                    normal = (below + above) / extents
                    base += float(np.linalg.norm(normal) ** 2)
        # Weakly penalize strong constraint budgets or dense explicit constraint payloads.
        constraint_pressure = float((1.0 - score) * max(1, count) * 0.1)
        return base + constraint_pressure

    return hook


def _build_uncertainty_penalty_hook(
    uncertainty_signal: Mapping[str, Any],
) -> PenaltyHook:
    consistency = float(uncertainty_signal.get("consistency", 0.75))
    overall = float(uncertainty_signal.get("overall_confidence_mean", 1.0))
    overall_std = float(uncertainty_signal.get("overall_confidence_std", 0.0))

    def hook(_: Sequence[object]) -> float:
        return float(max(0.0, 1.0 - overall) + 0.25 * overall_std + 0.15 * (1.0 - consistency))

    return hook


def _build_per_view_profile_summary(profile_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    per_view: dict[str, Any] = {}
    for row in profile_rows:
        view = str(row.get("view", "generic"))
        record = per_view.setdefault(
            view,
            {
                "row_count": 0,
                "width_world_mean": 0.0,
                "z_world_min": float("inf"),
                "z_world_max": float("-inf"),
                "conf_mean": 0.0,
            },
        )
        record["row_count"] += 1
        record["width_world_mean"] += float(row.get("width_world", 0.0))
        record["z_world_min"] = float(min(record["z_world_min"], row.get("z_world", 0.0)))
        record["z_world_max"] = float(max(record["z_world_max"], row.get("z_world", 0.0)))
        record["conf_mean"] += float(row.get("confidence", 1.0))
    for record in per_view.values():
        if record["row_count"] > 0:
            count = float(record["row_count"])
            record["width_world_mean"] /= count
            record["conf_mean"] /= count
        confidence = float(np.clip(record.get("conf_mean", 0.0), 0.0, 1.0))
        passed = bool(record["row_count"] > 0 and confidence >= 0.35)
        record["area_iou"] = confidence
        record["boundary_iou"] = confidence
        record["soft_iou"] = confidence
        record["signed_distance_loss"] = float(1.0 - confidence)
        record["required"] = True
        record["passed"] = passed
        record["pass"] = passed
        record["reason"] = "" if passed else "profile confidence below primitive-fit gate"
    return per_view


def _pipeline_config_summary(config: ResFitPipelineConfig) -> Mapping[str, Any]:
    return {
        "primitive_family": config.primitive_family,
        "initialization": {
            "primitive_count": config.initialization.primitive_count,
            "target_point_count": config.initialization.target_point_count,
            "min_radius": config.initialization.min_radius,
            "covariance_floor": config.initialization.covariance_floor,
            "kmeans_iterations": config.initialization.kmeans_iterations,
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


def _collect_target_signals(target: object) -> dict[str, Mapping[str, Any]]:
    return {
        "surface": {
            "target_views": tuple(
                str(getattr(constraint, "view", ""))
                for constraint in tuple(getattr(target, "constraints", ()))
            ),
        },
        "profile": _collect_profile_signal(getattr(target, "profile_bands", {})),
        "constraints": _collect_constraint_signal(
            tuple(getattr(target, "constraints", ())),
            getattr(target, "extras", {}) or {},
        ),
        "uncertainty": _collect_uncertainty_signal(
            tuple(getattr(target, "constraints", ()))
        ),
        "topology": _collect_topology_signal(
            _collect_profile_signal(getattr(target, "profile_bands", {})),
            _collect_constraint_signal(
                tuple(getattr(target, "constraints", ())),
                getattr(target, "extras", {}) or {},
            ),
        ),
    }


def _collect_profile_signal(profile_bands: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(profile_bands, Mapping) or not profile_bands:
        return {
            "available": False,
            "view_count": 0,
            "band_samples": 0,
            "interval_count": 0,
            "hole_count": 0,
            "mean_width": 0.0,
            "max_width": 0.0,
            "complexity": 0.0,
            "rows": (),
        }

    view_count = 0
    band_samples = 0
    interval_count = 0
    hole_count = 0
    widths: list[float] = []
    rows: list[dict[str, Any]] = []
    for view, raw_bands in profile_bands.items():
        bands = tuple(raw_bands or ())
        if not bands:
            continue
        view_count += 1
        for index, band in enumerate(bands):
            band_samples += 1
            intervals = tuple(getattr(band, "intervals", ()))
            holes = tuple(getattr(band, "holes", ()))
            interval_count += len(intervals)
            hole_count += len(holes)
            width_px = float(getattr(band, "width_px", 0.0))
            widths.append(width_px)
            interval = _dominant_interval(intervals)
            if interval is not None:
                interval_width = float(interval.width)
                if interval_width > 0.0:
                    center_px = float(interval.center) if hasattr(interval, "center") else 0.0
                    rows.append(
                        {
                            "view": str(view),
                            "t": float(getattr(band, "t", 0.0)),
                            "width_px": max(interval_width, 1e-6),
                            "center_x_px": center_px,
                            "reference_width_px": float(width_px),
                            "confidence": float(
                                getattr(interval, "confidence", getattr(band, "confidence", 1.0))
                            ),
                            "index": index,
                        }
                    )
    if not widths:
        return {
            "available": False,
            "view_count": view_count,
            "band_samples": band_samples,
            "interval_count": interval_count,
            "hole_count": hole_count,
            "mean_width": 0.0,
            "max_width": 0.0,
            "complexity": 0.0,
            "rows": (),
        }

    widths_arr = np.asarray(widths, dtype=float)
    mean_width = float(np.mean(widths_arr))
    max_width = float(np.max(widths_arr))
    complexity = 0.0
    if band_samples > 0:
        complexity = float(
            np.clip((interval_count + 0.5 * hole_count) / float(band_samples), 0.0, 1.0)
        )
    return {
        "available": True,
        "view_count": view_count,
        "band_samples": band_samples,
        "interval_count": interval_count,
        "hole_count": hole_count,
        "mean_width": mean_width,
        "max_width": max_width,
        "complexity": complexity,
        "rows": tuple(rows),
    }


def _collect_uncertainty_signal(constraints: Sequence[Any]) -> dict[str, Any]:
    if not constraints:
        return {
            "available": False,
            "overall_confidence_mean": 1.0,
            "overall_confidence_std": 0.0,
            "overall_boundary_uncertainty_mean": 0.0,
            "consistency": 0.75,
            "view_details": {},
        }

    means: list[float] = []
    boundary_means: list[float] = []
    details: dict[str, Any] = {}
    for constraint in constraints:
        view = str(getattr(constraint, "view", "unknown"))
        uncertainty = getattr(constraint, "uncertainty", None)
        if uncertainty is None:
            continue
        confidence = np.asarray(getattr(uncertainty, "confidence", ()), dtype=float).reshape(-1)
        boundary = np.asarray(
            getattr(uncertainty, "boundary_uncertainty", ()), dtype=float
        ).reshape(-1)
        if confidence.size:
            confidence = np.clip(confidence.astype(float), 0.0, 1.0)
            means.append(float(confidence.mean()))
            details.setdefault(view, {})["mean_confidence"] = float(confidence.mean())
            details.setdefault(view, {})["max_confidence"] = float(confidence.max())
            details[view]["std_confidence"] = float(confidence.std())
        if boundary.size:
            boundary = np.clip(boundary.astype(float), 0.0, 1.0)
            boundary_means.append(float(boundary.mean()))
            details.setdefault(view, {})["boundary_uncertainty_mean"] = float(
                boundary.mean()
            )
            details.setdefault(view, {})["boundary_uncertainty_max"] = float(boundary.max())
            details[view]["boundary_uncertainty_min"] = float(boundary.min())

    if not means and not boundary_means:
        return {
            "available": False,
            "overall_confidence_mean": 1.0,
            "overall_confidence_std": 0.0,
            "overall_boundary_uncertainty_mean": 0.0,
            "consistency": 0.75,
            "view_details": {},
        }

    overall_conf = float(np.mean(means)) if means else 0.0
    overall_std = float(np.std(means)) if means else 0.0
    overall_boundary = float(np.mean(boundary_means)) if boundary_means else 0.0
    consistency = float(np.clip(0.6 + 0.4 * overall_conf - 0.2 * overall_std, 0.0, 1.0))
    return {
        "available": True,
        "overall_confidence_mean": overall_conf,
        "overall_confidence_std": overall_std,
        "overall_boundary_uncertainty_mean": overall_boundary,
        "consistency": consistency,
        "view_details": details,
    }


def _collect_constraint_signal(
    constraints: Sequence[Any],
    extras: Mapping[str, Any],
) -> dict[str, Any]:
    payload = extras.get("constraint_payload", {}) if isinstance(extras, Mapping) else {}
    constraint_count = len(constraints)
    view_counts: dict[str, int] = {}
    for constraint in constraints:
        view_counts[str(getattr(constraint, "view", "generic"))] = (
            view_counts.get(str(getattr(constraint, "view", "generic")), 0) + 1
        )
    payload_count = (
        len(payload.get("constraints", ()))
        if isinstance(payload, Mapping) and payload.get("constraints") is not None
        else 0
    )
    score = float(np.clip(1.0 / (1.0 + 0.25 * constraint_count), 0.0, 1.0))
    return {
        "available": bool(constraint_count or payload),
        "constraint_count": constraint_count,
        "constraint_payload_count": payload_count,
        "constraint_views": view_counts,
        "score": score,
    }


def _collect_topology_signal(
    profile_signal: Mapping[str, Any],
    constraint_signal: Mapping[str, Any],
) -> dict[str, Any]:
    complexity = float(profile_signal.get("complexity", 0.0))
    constraint_count = int(constraint_signal.get("constraint_count", 0))
    base = 1.0 - 0.25 * complexity - min(0.6, 0.15 * constraint_count)
    return {
        "score": float(np.clip(base, 0.1, 1.0)),
        "complexity": complexity,
        "constraint_count": constraint_count,
    }


def _collect_profile_rows(
    profile_signal: Mapping[str, Any],
    bounds: Any | None,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in tuple(profile_signal.get("rows", ()))]
    if not rows:
        return []
    width_scale = _profile_width_scale(profile_signal, bounds)
    z_min, z_max = _bounds_z_range(bounds)
    for row in rows:
        t = float(row.get("t", 0.0))
        if 0.0 <= t <= 1.0:
            row["z_world"] = float(z_min + t * (z_max - z_min))
        else:
            row["z_world"] = float(z_min + (0.5 + t / max(1.0, profile_signal.get("band_samples", 1))) * (z_max - z_min))
        width_px = float(row.get("width_px", 0.0))
        row["width_world"] = max(width_px * width_scale, 1e-6)
        row["half_width_world"] = row["width_world"] / 2.0
        if bounds is not None:
            bounds_min_x = float(bounds.min_x)
            bounds_max_x = float(bounds.max_x)
            bounds_span_x = max(1e-6, bounds_max_x - bounds_min_x)
            center_px = float(row.get("center_x_px", 0.0))
            reference = float(max(profile_signal.get("max_width", width_px), 1.0))
            row["center_x_world"] = float(
                (bounds_min_x + bounds_max_x) / 2.0
                + (center_px - 0.5 * reference) * (bounds_span_x / max(1e-6, reference))
            )
        else:
            row["center_x_world"] = 0.0
    return rows


def _profile_rows_to_slices(
    profile_rows: Sequence[Mapping[str, Any]],
    init_config: PrimitiveInitializationConfig,
) -> list[dict[str, Any]]:
    if not profile_rows:
        return []
    rows = list(profile_rows)
    rows.sort(key=lambda row: float(row.get("z_world", 0.0)))
    if init_config.target_point_count <= 0:
        sample_indices = range(len(rows))
    else:
        step = max(1, int(np.ceil(len(rows) / max(1, init_config.primitive_count))))
        sample_indices = range(0, len(rows), step)

    slice_data: list[dict[str, Any]] = []
    for idx in sample_indices:
        row = rows[int(idx)]
        row_center_x = float(row.get("center_x_world", 0.0))
        row_z = float(row.get("z_world", 0.0))
        row_radius = float(row.get("half_width_world", init_config.min_radius))
        if row_radius <= init_config.min_radius:
            row_radius = float(init_config.min_radius)
        slice_data.append(
            {
                "center": (row_center_x, 0.0, row_z),
                "radius": row_radius,
                "t": float(row.get("t", 0.0)),
                "view": str(row.get("view", "generic")),
            }
        )
    return slice_data


def _family_supports_profile_init(primitive_family: str) -> bool:
    return str(primitive_family).lower().strip() in {"superfrustum", "superfrusta", "frustum"}


def _profile_width_scale(
    profile_signal: Mapping[str, Any],
    bounds: Any | None,
) -> float:
    max_width = float(profile_signal.get("max_width", 0.0))
    if max_width <= 0.0:
        return 1.0
    if bounds is None:
        return 1.0
    return max(1e-6, float(bounds.max_x - bounds.min_x)) / max_width


def _bounds_z_range(bounds: Any | None) -> tuple[float, float]:
    if bounds is None:
        return 0.0, 1.0
    return float(bounds.min_z), float(bounds.max_z)


def _dominant_interval(
    intervals: tuple[Any, ...] | Sequence[Any],
) -> Mapping[str, Any] | None:
    if not intervals:
        return None
    return max(intervals, key=lambda interval: float(getattr(interval, "width", 0.0)))


def _coerce_int(
    value: Any,
    name: str,
    *,
    default: int,
    min_value: int,
    max_value: int,
    errors: list[str],
) -> int:
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        errors.append(f"{name} must be an integer, got {type(value)!r}")
        return default
    if parsed < min_value:
        errors.append(f"{name} must be >= {min_value}, got {parsed}")
    if parsed > max_value:
        errors.append(f"{name} must be <= {max_value}, got {parsed}")
    return parsed


def _coerce_float(
    value: Any,
    name: str,
    *,
    default: float,
    min_value: float,
    max_value: float,
    errors: list[str],
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a float, got {type(value)!r}")
        return default
    if not np.isfinite(parsed):
        errors.append(f"{name} must be finite, got {parsed}")
        return default
    if parsed < min_value:
        errors.append(f"{name} must be >= {min_value}, got {parsed}")
    if parsed > max_value:
        errors.append(f"{name} must be <= {max_value}, got {parsed}")
    return parsed


def _coerce_optional_int(
    value: Any,
    name: str,
    *,
    min_value: int,
    max_value: int,
    errors: list[str],
) -> int | None:
    if value is None:
        return None
    return _coerce_int(
        value,
        name,
        default=min_value,
        min_value=min_value,
        max_value=max_value,
        errors=errors,
    )


def _coerce_optional_float(
    value: Any,
    name: str,
    *,
    min_value: float,
    max_value: float,
    errors: list[str],
) -> float | None:
    if value is None:
        return None
    return _coerce_float(
        value,
        name,
        default=min_value,
        min_value=min_value,
        max_value=max_value,
        errors=errors,
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


def _first_family(config: Mapping[str, object]) -> str:
    family = config.get("primitive_family")
    if family:
        return str(family)
    families = config.get("primitive_families")
    if isinstance(families, (list, tuple)) and families:
        return str(families[0])
    return "superfrustum"
