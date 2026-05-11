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

from .config import (
    ResFitPipelineConfig,
    _coerce_float,
    _coerce_int,
    _coerce_optional_float,
    _coerce_optional_int,
    _coerce_weight,
    _pipeline_config_summary,
)
from .initialization import (
    _candidate_families,
    _family_supports_profile_init,
)
from .artifacts import (
    build_resfit_mesh_proxy,
    resfit_history_records,
    write_resfit_artifacts,
)
from .metrics import build_resfit_candidate_metrics
from .optimizer import fit_residual_primitives_multistart
from .penalties import (
    _build_constraint_penalty_hook,
    _build_profile_silhouette_hook,
    _build_topology_penalty_hook,
    _build_uncertainty_penalty_hook,
)
from .profiles import _build_per_view_profile_summary, _collect_profile_rows, _profile_rows_to_slices
from .signals import _collect_target_signals
from .status import apply_resfit_quality_floors, resfit_candidate_status


def run_primitive_fit_pipeline(request: object) -> object:
    """CandidateResult adapter used by the reconstruction backend registry."""
    from reconstruction.point_cloud import target_occupied_points, target_surface_points
    from reconstruction.types import CandidateResult

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

    primitive_families = _candidate_families(config)
    errors: list[str] = []
    configured_max_primitives = _coerce_int(
        config.get("max_primitives", 6),
        "max_primitives",
        default=6,
        min_value=1,
        max_value=4096,
        errors=errors,
    )
    primitive_count = _coerce_int(
        config.get("primitive_count", min(configured_max_primitives, 6)),
        "primitive_count",
        default=min(configured_max_primitives, 6),
        min_value=1,
        max_value=configured_max_primitives,
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
    max_multistart_attempts = _coerce_int(
        config.get("max_multistart_attempts", 2),
        "max_multistart_attempts",
        default=2,
        min_value=1,
        max_value=16,
        errors=errors,
    )
    default_max_family_candidates = len(primitive_families)
    max_family_candidates = _coerce_int(
        config.get("max_family_candidates", default_max_family_candidates),
        "max_family_candidates",
        default=default_max_family_candidates,
        min_value=1,
        max_value=max(1, len(primitive_families)),
        errors=errors,
    )
    primitive_families = primitive_families[:max_family_candidates]

    pipeline_config = ResFitPipelineConfig(
        primitive_family=primitive_families[0],
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
        (
            result,
            primitive_family,
            pipeline_config,
            initial_primitives,
            profile_init_warning,
        ) = _fit_best_primitive_family(
            surface=surface,
            base_config=pipeline_config,
            primitive_families=primitive_families,
            profile_rows=profile_rows,
            init_config=init_config,
            occupied_points=occupied,
            silhouette_hook=silhouette_hook,
            topology_penalty_hook=topology_penalty_hook,
            constraint_penalty_hook=constraint_penalty_hook,
            uncertainty_penalty_hook=uncertainty_penalty_hook,
            max_attempts=max_multistart_attempts,
            share_budget_across_families=bool(
                config.get("share_budget_across_families", True)
            ),
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id=candidate_id,
            backend_name=backend_name,
            status="failed",
            errors=(str(exc),),
        )

    root = request.candidate_artifact_root()
    mesh_proxy, mesh_vertices, mesh_faces, mesh_metadata, topology_payload = build_resfit_mesh_proxy(
        result
    )
    history_records = resfit_history_records(result.history)
    primitive_path, mesh_path, artifacts = write_resfit_artifacts(
        root=root,
        candidate_id=candidate_id,
        backend_name=backend_name,
        primitive_family=primitive_family,
        result=result,
        mesh_proxy=mesh_proxy,
        surface_meta=surface_meta,
        occupied_meta=occupied_meta,
        signal_summary=signal_summary,
        pipeline_config=pipeline_config,
        max_runtime_s=max_runtime_s,
        max_objective_evaluations=max_objective_evaluations,
        initial_primitives=initial_primitives,
        history_records=history_records,
    )

    improved = result.initial_loss.total - result.final_loss.total
    status, degraded, errors_out, warnings = resfit_candidate_status(
        config=config,
        result=result,
        improved=improved,
        profile_rows_present=bool(profile_rows),
        profile_init_warning=profile_init_warning,
    )

    elapsed = time.perf_counter() - start
    metric, _metric_summary = build_resfit_candidate_metrics(
        elapsed_s=elapsed,
        result=result,
        primitive_family=primitive_family,
        mesh_metadata=mesh_metadata,
        topology_payload=topology_payload,
        uncertainty_signal=uncertainty_signal,
        profile_rows=profile_rows,
        pipeline_config=pipeline_config,
        history_records=history_records,
        surface_meta=surface_meta,
        occupied_meta=occupied_meta,
        signal_summary=signal_summary,
        initial_primitives=initial_primitives,
        max_runtime_s=max_runtime_s,
        max_objective_evaluations=max_objective_evaluations,
    )
    status, degraded, errors_out, warnings = apply_resfit_quality_floors(
        config=config,
        status=status,
        degraded=degraded,
        errors=errors_out,
        warnings=warnings,
        metric=metric,
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


def _fit_best_primitive_family(
    *,
    surface: np.ndarray,
    base_config: ResFitPipelineConfig,
    primitive_families: Sequence[str],
    profile_rows: Sequence[Mapping[str, Any]],
    init_config: PrimitiveInitializationConfig,
    occupied_points: np.ndarray,
    silhouette_hook: SilhouetteHook | None,
    topology_penalty_hook: PenaltyHook | None,
    constraint_penalty_hook: PenaltyHook | None,
    uncertainty_penalty_hook: PenaltyHook | None,
    max_attempts: int,
    share_budget_across_families: bool,
) -> tuple[ResFitPipelineResult, str, ResFitPipelineConfig, Sequence[object] | None, str]:
    family_count = max(1, len(primitive_families))
    summaries: list[Mapping[str, Any]] = []
    best: tuple[ResFitPipelineResult, str, ResFitPipelineConfig, Sequence[object] | None, str] | None = None
    for family in primitive_families:
        family_config = _family_pipeline_config(
            base_config,
            family=family,
            family_count=family_count,
            share_budget=share_budget_across_families,
        )
        initial_primitives, profile_init_warning = _profile_initial_primitives(
            family,
            profile_rows=profile_rows,
            init_config=init_config,
        )
        family_start = time.perf_counter()
        try:
            candidate = fit_residual_primitives_multistart(
                surface,
                family_config,
                profile_primitives=initial_primitives,
                occupied_points=occupied_points,
                silhouette_hook=silhouette_hook,
                topology_penalty_hook=topology_penalty_hook,
                constraint_penalty_hook=constraint_penalty_hook,
                uncertainty_penalty_hook=uncertainty_penalty_hook,
                max_attempts=max_attempts,
            )
        except Exception as exc:
            summaries.append(
                {
                    "family": family,
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_s": time.perf_counter() - family_start,
                    "profile_init_warning": profile_init_warning,
                }
            )
            continue
        improvement = candidate.initial_loss.total - candidate.final_loss.total
        summaries.append(
            {
                "family": family,
                "status": "ok",
                "selected_attempt": candidate.selected_attempt,
                "initial_total": candidate.initial_loss.total,
                "final_total": candidate.final_loss.total,
                "improvement": improvement,
                "primitive_count": len(candidate.primitives),
                "objective_evaluations": candidate.objective_evaluations,
                "optimizer_elapsed_s": candidate.optimizer_elapsed_s,
                "elapsed_s": time.perf_counter() - family_start,
                "profile_initialized": bool(initial_primitives),
                "profile_init_warning": profile_init_warning,
            }
        )
        if best is None or _family_result_is_better(candidate, best[0]):
            best = (candidate, family, family_config, initial_primitives, profile_init_warning)
    if best is None:
        errors = [
            str(summary.get("error", "unknown family fitting error"))
            for summary in summaries
            if summary.get("status") == "failed"
        ]
        raise RuntimeError("all primitive family attempts failed: " + "; ".join(errors))
    selected, family, family_config, initial_primitives, profile_init_warning = best
    selected = replace(selected, family_attempts=tuple(summaries))
    return selected, family, family_config, initial_primitives, profile_init_warning


def _family_pipeline_config(
    base_config: ResFitPipelineConfig,
    *,
    family: str,
    family_count: int,
    share_budget: bool,
) -> ResFitPipelineConfig:
    optimizer = base_config.optimizer
    if share_budget and family_count > 1:
        max_elapsed_s = (
            None
            if optimizer.max_elapsed_s is None
            else max(1.0e-3, float(optimizer.max_elapsed_s) / float(family_count))
        )
        max_objective_evaluations = (
            None
            if optimizer.max_objective_evaluations is None
            else max(1, int(optimizer.max_objective_evaluations) // int(family_count))
        )
        optimizer = replace(
            optimizer,
            max_elapsed_s=max_elapsed_s,
            max_objective_evaluations=max_objective_evaluations,
        )
    return replace(base_config, primitive_family=str(family), optimizer=optimizer)


def _profile_initial_primitives(
    primitive_family: str,
    *,
    profile_rows: Sequence[Mapping[str, Any]],
    init_config: PrimitiveInitializationConfig,
) -> tuple[Sequence[object] | None, str]:
    if not (_family_supports_profile_init(primitive_family) and profile_rows):
        return None, ""
    try:
        slice_data = _profile_rows_to_slices(profile_rows, init_config)
        if slice_data:
            return initialize_from_profile_bands(slice_data, init_config), ""
    except Exception as exc:
        return None, f"profile initialization failed: {exc}"
    return None, ""


def _family_result_is_better(
    candidate: ResFitPipelineResult,
    incumbent: ResFitPipelineResult,
) -> bool:
    if candidate.final_loss.total < incumbent.final_loss.total:
        return True
    if candidate.final_loss.total > incumbent.final_loss.total:
        return False
    candidate_improvement = candidate.initial_loss.total - candidate.final_loss.total
    incumbent_improvement = incumbent.initial_loss.total - incumbent.final_loss.total
    if candidate_improvement != incumbent_improvement:
        return candidate_improvement > incumbent_improvement
    return len(candidate.primitives) < len(incumbent.primitives)
