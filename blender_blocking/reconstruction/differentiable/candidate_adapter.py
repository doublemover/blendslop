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
from .artifacts import write_differentiable_candidate_artifacts
from .history import build_refinement_history_payloads
from .losses import evaluate_render_loss
from .metrics import build_differentiable_candidate_metrics
from .mesh_projection import _target_cameras_and_masks
from .nvdiffrast_adapter import NvdiffrastBackend
from .optimization import run_differentiable_optimization
from .status import differentiable_candidate_status
from .target_adapter import _candidate_per_view_metrics, _collect_target_view_signal_weights, _mean_candidate_metric, _metric_float, _min_candidate_metric, _silhouette_view_history, renderable_from_primitive


def run_refinement_candidate(request: object) -> object:
    """Run a CPU differentiable-rendering-inspired candidate.

    This path creates an editable ellipsoid proxy, renders soft silhouettes, and
    records loss terms through the same backend protocol that GPU
    renderers can implement.
    """
    import time

    from placement.resfit_initialization import (
        PrimitiveInitializationConfig,
        initialize_ellipsoids_from_points,
    )
    from reconstruction.mesh_io import combine_primitive_meshes
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
        optimization_result = run_differentiable_optimization(
            backend_choice=backend_choice,
            parsed_config=parsed_config,
            primitives=tuple(primitives),
            renderer=renderer,
            cameras=cameras,
            target_record=target_record,
            target_view_weights=target_view_weights,
            initial_render_batch=initial_render_batch,
            initial_loss=initial_loss,
            request_timeout_s=request_timeout_s,
        )
        optimized_primitives = optimization_result["optimized_primitives"]
        optimization_history = optimization_result["optimization_history"]
        optimization_summary = optimization_result["optimization_summary"]
        render_batch = optimization_result["render_batch"]
        loss = optimization_result["loss"]
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
    history_outputs = build_refinement_history_payloads(
        silhouettes=silhouettes,
        render_batch=render_batch,
        baseline_render_batch=baseline_render_batch,
        initial_render_batch=initial_render_batch,
        loss=loss,
        baseline_loss=baseline_loss,
        initial_loss=initial_loss,
        target_view_weights=target_view_weights,
        view_signal_details=view_signal_details,
        optimization_summary=optimization_summary,
        optimization_history=optimization_history,
        backend_choice=backend_choice,
        parsed_config=parsed_config,
    )
    objective_improvement = float(history_outputs["objective_improvement"])
    zero_baseline_improvement = float(history_outputs["zero_baseline_improvement"])
    objective_history = history_outputs["objective_history"]
    objective_improvement_record = history_outputs["objective_improvement_record"]
    history_payload = history_outputs["history_payload"]

    primitive_path, mesh_path, artifacts = write_differentiable_candidate_artifacts(
        root=root,
        candidate_id=candidate_id,
        backend_name=backend_name,
        backend_choice=backend_choice,
        optional_dependency_policy=optional_dependency_policy,
        optimized_primitives=tuple(optimized_primitives),
        mesh_proxy=mesh_proxy,
        loss=loss,
        baseline_loss=baseline_loss,
        initial_loss=initial_loss,
        point_meta=point_meta,
        topology_payload=topology_payload,
        optimization_summary=optimization_summary,
        optimization_history=optimization_history,
        objective_improvement_record=objective_improvement_record,
        objective_history=objective_history,
        history_payload=history_payload,
        parsed_config=parsed_config,
        config_warnings=config_warnings,
        target_view_weights=target_view_weights,
        view_signal_details=view_signal_details,
        target_signal_warnings=target_signal_warnings,
    )

    elapsed = time.perf_counter() - start
    metrics, candidate_per_view, failed_required_views = build_differentiable_candidate_metrics(
        elapsed_s=elapsed,
        optimized_primitives=tuple(optimized_primitives),
        point_meta=point_meta,
        render_batch=render_batch,
        loss=loss,
        baseline_loss=baseline_loss,
        initial_loss=initial_loss,
        topology_score=topology_score,
        topology_penalty=topology_penalty,
        topology_payload=topology_payload,
        objective_improvement=objective_improvement,
        zero_baseline_improvement=zero_baseline_improvement,
        objective_improvement_record=objective_improvement_record,
        objective_history=objective_history,
        optimization_summary=optimization_summary,
        optimization_history=optimization_history,
        target_view_weights=target_view_weights,
        view_signal_details=view_signal_details,
        target_signal_warnings=target_signal_warnings,
        history_payload=history_payload,
        config_warnings=config_warnings,
        artifacts=artifacts,
        backend_choice=backend_choice,
    )
    warnings = (
        tuple(loss.warnings)
        + tuple(initial_loss.warnings)
        + tuple(baseline_loss.warnings)
        + tuple(config_warnings)
        + tuple(target_signal_warnings)
    )
    status, degraded, errors, warnings = differentiable_candidate_status(
        config=config,
        optimized_primitives_present=bool(optimized_primitives),
        objective_improvement=objective_improvement,
        failed_required_views=failed_required_views,
        warnings=warnings,
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
