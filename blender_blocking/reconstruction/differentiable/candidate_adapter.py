from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence

import numpy as np

try:
    from utils.optional_deps import optional_policy_decision, probe_dependency
except Exception:  # pragma: no cover
    from ...utils.optional_deps import optional_policy_decision, probe_dependency

try:
    from primitives.analytic_primitives import (
        AnisotropicGaussianPrimitive,
        EllipsoidPrimitive,
        SuperquadricPrimitive,
    )
    from primitives.primitive_protocol import MeshData
    from primitives.soft_silhouette import (
        OrthographicCamera,
        render_projected_soft_silhouette,
        soft_mask_metrics,
    )
    from primitives.superfrustum import SuperFrustum
    from reconstruction.artifacts import write_json
except ImportError:  # pragma: no cover - package import path.
    from ...primitives.analytic_primitives import (
        AnisotropicGaussianPrimitive,
        EllipsoidPrimitive,
        SuperquadricPrimitive,
    )
    from ...primitives.primitive_protocol import MeshData
    from ...primitives.soft_silhouette import (
        OrthographicCamera,
        render_projected_soft_silhouette,
        soft_mask_metrics,
    )
    from ...primitives.superfrustum import SuperFrustum
    from ...reconstruction.artifacts import write_json
from .config import _minimum_positive_float, _normalize_differentiable_config
from .contracts import RenderBatch, ReconstructionTarget, RenderableScene
from .cpu_soft import CpuSoftSilhouetteBackend
from .losses import evaluate_render_loss
from .mesh_projection import _target_cameras_and_masks
from .nvdiffrast_adapter import NvdiffrastBackend
from .target_adapter import _candidate_per_view_metrics, _collect_target_view_signal_weights, _mean_candidate_metric, _metric_float, _min_candidate_metric, _silhouette_view_history, renderable_from_primitive


def run_refinement_candidate(request: object) -> object:
    """Run a CPU differentiable-rendering-inspired candidate.

    This path creates an editable ellipsoid proxy, renders soft silhouettes, and
    records loss terms through the same backend protocol that GPU
    renderers can implement.
    """
    import time

    from placement.resfit_objective import ResFitObjectiveResult
    from placement.resfit_optimizer import (
        CoordinateDescentConfig,
        coordinate_descent_optimize,
    )
    from placement.resfit_initialization import (
        PrimitiveInitializationConfig,
        initialize_ellipsoids_from_points,
    )
    from reconstruction.mesh_io import (
        combine_primitive_meshes,
        write_obj,
        write_primitive_set,
    )
    from reconstruction.point_cloud import target_surface_points
    from reconstruction.types import CandidateMetrics, CandidateResult
    from metrics.topology import mesh_topology_report

    start = time.perf_counter()
    config = dict(getattr(request, "config", {}) or {})
    parsed_config, config_errors, config_warnings = _normalize_differentiable_config(config)
    if config_errors:
        return CandidateResult(
            candidate_id=getattr(request, "candidate_id"),
            backend_name=getattr(request, "backend_name", "differentiable_refine"),
            status="failed",
            errors=config_errors,
            warnings=tuple(config_warnings),
        )
    backend_choice = str(parsed_config["backend"])
    candidate_id = getattr(request, "candidate_id")
    backend_name = getattr(request, "backend_name", "differentiable_refine")
    target = getattr(request, "target")
    request_budget = getattr(request, "budget", None)
    request_timeout_s = getattr(request_budget, "timeout_s", None)
    optional_dependency_policy = str(parsed_config["optional_dependency_policy"])
    (
        target_view_weights,
        view_signal_details,
        target_signal_warnings,
    ) = _collect_target_view_signal_weights(target)

    nvd_renderer = None
    if backend_choice == "nvdiffrast":
        nvd_renderer = NvdiffrastBackend()
        if not nvd_renderer.available:
            nvd_dependency = probe_dependency("nvdiffrast")
            decision = optional_policy_decision(
                nvd_dependency,
                policy=optional_dependency_policy,
                feature="differentiable_refine.nvdiffrast",
            )
            status = str(decision["result_status"])
            reason = nvd_renderer.unavailable_reason or str(decision["message"])
            metrics = CandidateMetrics(
                extras={
                    "optional_dependencies": getattr(nvd_renderer, "dependency_state", {}),
                    "optional_dependency_policy": decision,
                    "backend": backend_choice,
                }
            )
            return CandidateResult(
                candidate_id=candidate_id,
                backend_name=backend_name,
                status=status,
                metric_result=metrics,
                warnings=(
                    reason,
                    *config_warnings,
                    *tuple(target_signal_warnings),
                ),
                errors=(reason,) if status == "failed" else (),
            )

    try:
        points, point_meta = target_surface_points(
            target,
            resolution=int(parsed_config["visual_hull_resolution"]),
            max_points=int(parsed_config["target_point_count"]),
            chunk_size=parsed_config["chunk_size"],
        )
        primitives = tuple(
            initialize_ellipsoids_from_points(
                points,
                PrimitiveInitializationConfig(
                    primitive_count=int(parsed_config["primitive_count"]),
                    target_point_count=int(parsed_config["target_point_count"]),
                    min_radius=float(parsed_config["min_radius"]),
                    covariance_floor=float(parsed_config["covariance_floor"]),
                    kmeans_iterations=int(parsed_config["kmeans_iterations"]),
                ),
            )
        )
        renderables = tuple(renderable_from_primitive(primitive) for primitive in primitives)
        scene = RenderableScene(primitives=renderables)
        cameras, silhouettes = _target_cameras_and_masks(target)
        if backend_choice == "nvdiffrast":
            renderer = nvd_renderer or NvdiffrastBackend()
        else:
            renderer = CpuSoftSilhouetteBackend(
                softness=float(parsed_config["softness"]),
                min_variance=float(parsed_config["min_variance"]),
            )
        target_record = ReconstructionTarget(
            silhouettes=silhouettes,
            surface_points=points,
            depths=getattr(target, "depths", {}),
        )
        baseline_render_batch = RenderBatch(
            silhouettes={
                name: np.zeros_like(mask, dtype=np.float64)
                for name, mask in silhouettes.items()
            },
            metadata={"stage": "baseline"},
        )
        baseline_loss = renderer.loss(
            baseline_render_batch,
            target_record,
            parsed_config["loss_weights"],
            view_weights=target_view_weights,
        )
        initial_render_batch = renderer.render(scene, cameras)
        initial_loss = renderer.loss(
            initial_render_batch,
            target_record,
            parsed_config["loss_weights"],
            view_weights=target_view_weights,
        )
        optimized_primitives = primitives
        optimization_history: list[dict[str, object]] = []
        optimization_summary: dict[str, object] = {
            "enabled": False,
            "reason": "not_run",
            "objective_evaluations": 0,
            "elapsed_s": 0.0,
            "initial_total": float(initial_loss.total),
            "final_total": float(initial_loss.total),
        }
        if (
            backend_choice == "cpu_soft_silhouette"
            and int(parsed_config["optimization_steps"]) > 0
            and primitives
            and cameras
        ):
            max_elapsed_s = _minimum_positive_float(
                parsed_config.get("max_runtime_s"),
                request_timeout_s,
            )
            optimizer_config = CoordinateDescentConfig(
                iterations=int(parsed_config["optimization_steps"]),
                initial_step=float(parsed_config["optimization_initial_step"]),
                step_decay=float(parsed_config["optimization_step_decay"]),
                min_step=float(parsed_config["optimization_min_step"]),
                max_objective_evaluations=parsed_config["max_objective_evaluations"],
                max_elapsed_s=max_elapsed_s,
            )

            def objective(primitives_to_score: Sequence[object]) -> ResFitObjectiveResult:
                trial_scene = RenderableScene(
                    primitives=tuple(
                        renderable_from_primitive(primitive)
                        for primitive in primitives_to_score
                    )
                )
                trial_batch = renderer.render(trial_scene, cameras)
                trial_loss = renderer.loss(
                    trial_batch,
                    target_record,
                    parsed_config["loss_weights"],
                    view_weights=target_view_weights,
                )
                return ResFitObjectiveResult(
                    total=float(trial_loss.total),
                    terms=dict(trial_loss.terms),
                    warnings=tuple(trial_loss.warnings),
                )

            optimization = coordinate_descent_optimize(
                primitives,
                objective,
                optimizer_config,
            )
            optimized_primitives = optimization.primitives
            optimization_history = [
                {
                    "iteration": record.iteration,
                    "total": record.total,
                    "terms": dict(record.terms),
                    "accepted_moves": record.accepted_moves,
                    "rejected_moves": getattr(record, "rejected_moves", 0),
                    "step_size": record.step_size,
                    "reason": getattr(record, "reason", ""),
                }
                for record in optimization.history
            ]
            optimization_summary = {
                "enabled": True,
                "reason": optimization.termination_reason,
                "objective_evaluations": optimization.objective_evaluations,
                "elapsed_s": optimization.elapsed_s,
                "initial_total": float(initial_loss.total),
                "best_total": float(optimization.best_loss),
                "accepted_moves": sum(
                    int(record.accepted_moves) for record in optimization.history
                ),
                "history_length": len(optimization_history),
                "config": {
                    "iterations": int(parsed_config["optimization_steps"]),
                    "initial_step": float(parsed_config["optimization_initial_step"]),
                    "step_decay": float(parsed_config["optimization_step_decay"]),
                    "min_step": float(parsed_config["optimization_min_step"]),
                    "max_objective_evaluations": parsed_config["max_objective_evaluations"],
                    "max_elapsed_s": max_elapsed_s,
                },
            }
        elif int(parsed_config["optimization_steps"]) <= 0:
            optimization_summary["reason"] = "optimization_steps_zero"
        elif backend_choice != "cpu_soft_silhouette":
            optimization_summary["reason"] = f"optimizer disabled for backend {backend_choice}"
        elif not cameras:
            optimization_summary["reason"] = "no target cameras"
        elif not primitives:
            optimization_summary["reason"] = "no primitives"

        if optimization_summary.get("enabled"):
            renderables = tuple(
                renderable_from_primitive(primitive) for primitive in optimized_primitives
            )
            scene = RenderableScene(primitives=renderables)
            render_batch = renderer.render(scene, cameras)
            loss = renderer.loss(
                render_batch,
                target_record,
                parsed_config["loss_weights"],
                view_weights=target_view_weights,
            )
        else:
            render_batch = initial_render_batch
            loss = initial_loss
        optimization_summary["final_total"] = float(loss.total)
    except Exception as exc:
        if backend_choice == "nvdiffrast":
            dependency_state = (
                nvd_renderer.dependency_state if nvd_renderer is not None else {}
            )
            status = "failed" if optional_dependency_policy == "fail" else "skipped"
            message = f"nvdiffrast render path unavailable: {exc}"
            metrics = CandidateMetrics(
                extras={
                    "optional_dependencies": dependency_state,
                    "optional_dependency_policy": {
                        "policy": optional_dependency_policy,
                        "feature": "differentiable_refine.nvdiffrast",
                        "status": status,
                        "result_status": status,
                        "message": message,
                    },
                    "backend": backend_choice,
                }
            )
            return CandidateResult(
                candidate_id=candidate_id,
                backend_name=backend_name,
                status=status,
                metric_result=metrics,
                warnings=(
                    message,
                    *config_warnings,
                    *tuple(target_signal_warnings),
                ),
                errors=(message,) if status == "failed" else (),
            )
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
    mesh_proxy = combine_primitive_meshes(optimized_primitives, resolution=20)
    try:
        topology_report = mesh_topology_report(mesh_proxy.vertices, mesh_proxy.faces)
        topology_payload = topology_report.to_dict()
        topology_score = float(topology_payload.get("topology_score", 0.0))
        topology_penalty = float(topology_payload.get("penalty", 0.0))
    except Exception as exc:
        topology_payload = {
            "required": True,
            "passed": False,
            "pass": False,
            "reason": f"mesh topology report failed: {exc}",
            "topology_score": 0.0,
            "penalty": 1.0,
        }
        topology_score = 0.0
        topology_penalty = 1.0
    refinement_history: list[dict[str, object]] = []
    for name, target_mask in silhouettes.items():
        view_metrics = dict(loss.per_view.get(name, {}))
        if name in render_batch.silhouettes:
            refinement_history.append(
                _silhouette_view_history(
                    name=name,
                    predicted=np.asarray(render_batch.silhouettes[name], dtype=np.float64),
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
                    predicted=np.asarray(initial_render_batch.silhouettes[name], dtype=np.float64),
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
            "optimization": optimization_summary,
        },
    ]
    objective_improvement_record = {
        "baseline_total": float(baseline_loss.total),
        "initial_candidate_total": float(initial_loss.total),
        "objective_total": float(loss.total),
        "objective_improvement": float(objective_improvement),
        "zero_baseline_improvement": float(zero_baseline_improvement),
        "weight_sum": {
            "silhouette_weight_sum": float(sum(weight for weight in target_view_weights.values())),
            "depth_weight_sum": float(sum(weight for weight in target_view_weights.values())),
        },
        "history": objective_history,
        "view_signal_weights": dict(target_view_weights),
        "view_signal_details": dict(view_signal_details),
        "optimization": optimization_summary,
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
                    predicted=np.asarray(initial_render_batch.silhouettes[name], dtype=np.float64),
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
            "optimization": optimization_summary,
            "optimization_history": optimization_history,
        }
    ]
    if root is not None:
        primitive_path = write_primitive_set(
            root / "primitives" / "differentiable-refine.json",
            optimized_primitives,
            metadata={
                "backend": backend_choice,
                "loss": loss.terms,
                "per_view": loss.per_view,
                "surface_points": point_meta,
                "topology": topology_payload,
                "optimization": optimization_summary,
            },
        )
        artifacts["primitive_json"] = primitive_path
        mesh_path = write_obj(
            root / "mesh" / "differentiable-refine.obj",
            mesh_proxy,
            header=(f"candidate {candidate_id}", backend_name),
        )
        artifacts["mesh_obj"] = mesh_path
        objective_path = root / "artifacts" / "differentiable-refine-objective-history.json"
        write_json(
            objective_path,
            {
                "candidate_id": candidate_id,
                "backend": backend_choice,
                "objective_improvement": objective_improvement_record,
                "history": objective_history,
                "config": {
                    "backend": backend_choice,
                    "optional_dependency_policy": optional_dependency_policy,
                    "gradient_mode": str(parsed_config["gradient_mode"]),
                    "finite_difference_epsilon": float(parsed_config["finite_difference_epsilon"]),
                    "softness": float(parsed_config["softness"]),
                    "min_variance": float(parsed_config["min_variance"]),
                    "visual_hull_resolution": int(parsed_config["visual_hull_resolution"]),
                    "primitive_count": int(parsed_config["primitive_count"]),
                    "target_point_count": int(parsed_config["target_point_count"]),
                    "min_radius": float(parsed_config["min_radius"]),
                    "covariance_floor": float(parsed_config["covariance_floor"]),
                    "kmeans_iterations": int(parsed_config["kmeans_iterations"]),
                    "chunk_size": parsed_config["chunk_size"],
                    "optimization_steps": int(parsed_config["optimization_steps"]),
                    "optimization_initial_step": float(parsed_config["optimization_initial_step"]),
                    "optimization_step_decay": float(parsed_config["optimization_step_decay"]),
                    "optimization_min_step": float(parsed_config["optimization_min_step"]),
                    "max_objective_evaluations": parsed_config["max_objective_evaluations"],
                    "max_runtime_s": parsed_config["max_runtime_s"],
                    "loss_weights": dict(parsed_config["loss_weights_dict"]),
                    "warnings": tuple(config_warnings),
                },
                "loss": {
                    "baseline_total": float(baseline_loss.total),
                    "initial_total": float(initial_loss.total),
                    "final_total": float(loss.total),
                    "baseline_terms": dict(baseline_loss.terms),
                    "initial_terms": dict(initial_loss.terms),
                    "final_terms": dict(loss.terms),
                },
                "optimization": optimization_summary,
                "optimization_history": optimization_history,
                "view_signal_weights": dict(target_view_weights),
                "view_signal_details": dict(view_signal_details),
                "view_signal_warnings": tuple(target_signal_warnings),
            },
        )
        artifacts["objective_history"] = objective_path
        history_path = root / "artifacts" / "differentiable-refine-history.json"
        write_json(
            history_path,
            {
                "candidate_id": candidate_id,
                "backend": backend_choice,
                "config": {
                    "backend": backend_choice,
                    "optional_dependency_policy": optional_dependency_policy,
                    "gradient_mode": str(parsed_config["gradient_mode"]),
                    "finite_difference_epsilon": float(parsed_config["finite_difference_epsilon"]),
                    "softness": float(parsed_config["softness"]),
                    "min_variance": float(parsed_config["min_variance"]),
                    "visual_hull_resolution": int(parsed_config["visual_hull_resolution"]),
                    "primitive_count": int(parsed_config["primitive_count"]),
                    "target_point_count": int(parsed_config["target_point_count"]),
                    "min_radius": float(parsed_config["min_radius"]),
                    "covariance_floor": float(parsed_config["covariance_floor"]),
                    "kmeans_iterations": int(parsed_config["kmeans_iterations"]),
                    "chunk_size": parsed_config["chunk_size"],
                    "optimization_steps": int(parsed_config["optimization_steps"]),
                    "optimization_initial_step": float(parsed_config["optimization_initial_step"]),
                    "optimization_step_decay": float(parsed_config["optimization_step_decay"]),
                    "optimization_min_step": float(parsed_config["optimization_min_step"]),
                    "max_objective_evaluations": parsed_config["max_objective_evaluations"],
                    "max_runtime_s": parsed_config["max_runtime_s"],
                    "loss_weights": dict(parsed_config["loss_weights_dict"]),
                    "warnings": tuple(config_warnings),
                },
                "history": history_payload,
                "objective_improvement": objective_improvement_record,
                "objective_history": objective_history,
                "optimization": optimization_summary,
                "optimization_history": optimization_history,
                "loss": {
                    "total": float(loss.total),
                    "terms": dict(loss.terms),
                    "warnings": tuple(loss.warnings),
                },
                "per_view": {name: dict(values) for name, values in loss.per_view.items()},
            },
        )
        artifacts["refinement_history"] = history_path

    elapsed = time.perf_counter() - start
    area_iou_mean = 1.0 - float(loss.terms.get("area_iou", 1.0))
    soft_iou_mean = 1.0 - float(loss.terms.get("soft_iou", 1.0))
    candidate_per_view = _candidate_per_view_metrics(loss.per_view)
    area_iou_mean = _mean_candidate_metric(
        candidate_per_view,
        "area_iou",
        fallback=max(0.0, area_iou_mean),
    )
    area_iou_min = _min_candidate_metric(
        candidate_per_view,
        "area_iou",
        fallback=max(0.0, min(area_iou_mean, soft_iou_mean)),
    )
    boundary_iou_mean = _mean_candidate_metric(
        candidate_per_view,
        "boundary_iou",
        fallback=max(0.0, soft_iou_mean),
    )
    failed_required_views = sum(
        1
        for payload in candidate_per_view.values()
        if bool(payload.get("required", True)) and not bool(payload.get("passed", False))
    )
    metrics = CandidateMetrics(
        area_iou_min=max(0.0, area_iou_min),
        area_iou_mean=max(0.0, area_iou_mean),
        boundary_iou_mean=max(0.0, boundary_iou_mean),
        topology_score=topology_score,
        topology_penalty=topology_penalty,
        editability_score=0.65,
        complexity_penalty=min(1.0, len(optimized_primitives) / 96.0),
        elapsed_s=elapsed,
        per_view=candidate_per_view,
        extras={
            "backend": backend_choice,
            "loss_total": loss.total,
            "loss_terms": loss.terms,
            "loss_warnings": loss.warnings,
            "primitive_count": len(optimized_primitives),
            "surface_points": point_meta,
            "render_metadata": dict(getattr(render_batch, "metadata", {})),
            "topology": topology_payload,
            "baseline_loss": dict(baseline_loss.terms),
            "baseline_total": float(baseline_loss.total),
            "baseline_warnings": tuple(baseline_loss.warnings),
            "initial_loss": dict(initial_loss.terms),
            "initial_total": float(initial_loss.total),
            "initial_warnings": tuple(initial_loss.warnings),
            "objective_total": float(loss.total),
            "objective_improvement": float(objective_improvement),
            "zero_baseline_improvement": float(zero_baseline_improvement),
            "objective_improvement_record": objective_improvement_record,
            "objective_history": objective_history,
            "optimization": optimization_summary,
            "optimization_history": optimization_history,
            "view_signal_weights": dict(target_view_weights),
            "view_signal_details": dict(view_signal_details),
            "view_signal_warnings": tuple(target_signal_warnings),
            "history": history_payload,
            "objective_history_artifact": str(artifacts.get("objective_history", "")),
            "history_file": str(artifacts.get("refinement_history", "")),
            "validation_warnings": tuple(config_warnings),
        },
    )
    warnings = (
        tuple(loss.warnings)
        + tuple(initial_loss.warnings)
        + tuple(baseline_loss.warnings)
        + tuple(config_warnings)
        + tuple(target_signal_warnings)
    )
    errors: tuple[str, ...] = ()
    status = "success" if optimized_primitives else "skipped"
    degraded = False
    require_improvement = bool(
        config.get("fail_on_objective_regression")
        or config.get("require_objective_improvement")
    )
    if objective_improvement < 0.0 or (
        require_improvement and objective_improvement <= 0.0
    ):
        regression_message = (
            "objective worsened relative to initial differentiable candidate"
            if objective_improvement < 0.0
            else "objective did not improve relative to initial differentiable candidate"
        )
        warnings = warnings + (regression_message,)
        if require_improvement:
            status = "failed"
            errors = (regression_message,)
        elif optimized_primitives:
            status = "degraded"
            degraded = True
    if optimized_primitives and failed_required_views and status == "success":
        status = "degraded"
        degraded = True
        warnings = warnings + (
            f"{failed_required_views} required soft-silhouette view(s) failed metric gates",
        )
    return CandidateResult(
        candidate_id=candidate_id,
        backend_name=backend_name,
        status=status,
        primitive_path=primitive_path,
        mesh_path=mesh_path,
        metric_result=metrics,
        artifacts=artifacts,
        warnings=warnings,
        errors=errors,
        degraded=degraded,
        payload={
            "primitives": optimized_primitives,
            "loss": loss,
            "baseline_loss": baseline_loss,
            "history": history_payload,
            "objective_history": objective_history,
            "objective_improvement": objective_improvement_record,
            "render_batch": render_batch,
            "view_signal_weights": dict(target_view_weights),
        },
    )
