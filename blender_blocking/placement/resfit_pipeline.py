"""
Orchestration for modular residual primitive fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Mapping, Sequence

import numpy as np

try:
    from .resfit_initialization import (
        PrimitiveInitializationConfig,
        initialize_ellipsoids_from_points,
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


@dataclass(frozen=True)
class ResFitPipelineResult:
    primitives: tuple[object, ...]
    initial_loss: ResFitObjectiveResult
    final_loss: ResFitObjectiveResult
    history: tuple[OptimizationRecord, ...]
    warnings: tuple[str, ...]

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
) -> ResFitPipelineResult:
    """Fit primitives against target points using the modular ResFit path."""
    target_points = np.asarray(target_points, dtype=np.float64)
    if target_points.ndim != 2 or target_points.shape[1] != 3:
        raise ValueError("target_points must have shape (N, 3)")

    warnings = []
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
        )

    return ResFitPipelineResult(
        primitives=optimization.primitives,
        initial_loss=initial_loss,
        final_loss=final_loss,
        history=optimization.history,
        warnings=tuple(warnings + list(final_loss.warnings)),
    )


def run_primitive_fit_pipeline(request: object) -> object:
    """CandidateResult adapter used by the reconstruction backend registry."""
    from reconstruction.mesh_io import (
        combine_primitive_meshes,
        write_obj,
        write_primitive_set,
    )
    from reconstruction.point_cloud import target_occupied_points, target_surface_points
    from reconstruction.types import CandidateMetrics, CandidateResult

    start = time.perf_counter()
    config = dict(getattr(request, "config", {}) or {})
    target = getattr(request, "target")
    candidate_id = getattr(request, "candidate_id")
    backend_name = getattr(request, "backend_name", "primitive_fit_refine")

    primitive_family = _first_family(config)
    primitive_count = int(config.get("primitive_count", config.get("max_primitives", 6)))
    target_point_count = int(config.get("target_point_count", 4096))
    resolution = int(config.get("visual_hull_resolution", config.get("resolution", 48)))
    optimization_steps = int(config.get("optimization_steps", 30))
    occupied_weight = float(config.get("visual_hull_occupancy_weight", 0.05))

    surface, surface_meta = target_surface_points(
        target,
        resolution=resolution,
        max_points=target_point_count,
        chunk_size=config.get("chunk_size"),
    )
    occupied, occupied_meta = target_occupied_points(
        target,
        resolution=max(8, min(resolution, 40)),
        max_points=target_point_count,
        chunk_size=config.get("chunk_size"),
    )
    init_config = PrimitiveInitializationConfig(
        primitive_count=max(1, primitive_count),
        target_point_count=target_point_count,
        min_radius=float(config.get("min_radius", 0.05)),
        covariance_floor=float(config.get("covariance_floor", 1e-4)),
        kmeans_iterations=int(config.get("kmeans_iterations", 8)),
    )
    weights = ResFitLossWeights(
        surface_residual=float(config.get("surface_residual_weight", 1.0)),
        visual_hull_occupancy=occupied_weight,
        primitive_count=float(config.get("primitive_count_weight", 0.01)),
        overlap_penalty=float(config.get("overlap_penalty_weight", 0.05)),
        silhouette=float(config.get("silhouette_weight", 0.0)),
        topology_penalty=float(config.get("topology_penalty_weight", 0.0)),
        constraint_penalty=float(config.get("constraint_penalty_weight", 0.05)),
    )
    pipeline_config = ResFitPipelineConfig(
        primitive_family=primitive_family,
        initialization=init_config,
        optimizer=CoordinateDescentConfig(
            iterations=optimization_steps,
            initial_step=float(config.get("initial_step", 0.1)),
            step_decay=float(config.get("step_decay", 0.5)),
            min_step=float(config.get("min_step", 1e-4)),
        ),
        weights=weights,
        fail_on_regression=bool(config.get("fail_on_regression", True)),
    )

    try:
        result = fit_residual_primitives(
            surface,
            pipeline_config,
            occupied_points=occupied,
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
    artifacts = {}
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
            },
        )
        mesh_path = write_obj(
            root / "mesh" / "primitive-fit.obj",
            mesh_proxy,
            header=(f"candidate {candidate_id}", backend_name),
        )
        artifacts["primitive_json"] = primitive_path
        artifacts["mesh_obj"] = mesh_path

    elapsed = time.perf_counter() - start
    improvement = result.initial_loss.total - result.final_loss.total
    metric = CandidateMetrics(
        topology_score=0.7,
        constraint_score=max(0.0, 1.0 - result.final_loss.terms.get("constraint_penalty", 0.0)),
        editability_score=0.9,
        complexity_penalty=min(1.0, len(result.primitives) / 64.0),
        elapsed_s=elapsed,
        extras={
            "family": primitive_family,
            "primitive_count": len(result.primitives),
            "initial_loss": result.initial_loss.terms,
            "final_loss": result.final_loss.terms,
            "initial_total": result.initial_loss.total,
            "final_total": result.final_loss.total,
            "objective_improvement": improvement,
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
            "surface_points": surface_meta,
            "occupied_points": occupied_meta,
        },
    )
    return CandidateResult(
        candidate_id=candidate_id,
        backend_name=backend_name,
        status="success" if result.primitives else "skipped",
        primitive_path=primitive_path,
        mesh_path=mesh_path,
        metric_result=metric,
        artifacts=artifacts,
        warnings=tuple(result.warnings),
        payload=result,
    )


def _first_family(config: Mapping[str, object]) -> str:
    family = config.get("primitive_family")
    if family:
        return str(family)
    families = config.get("primitive_families")
    if isinstance(families, (list, tuple)) and families:
        return str(families[0])
    return "superfrustum"
