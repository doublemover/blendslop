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
        cameras, silhouettes = _target_cameras_and_masks(target)
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
        initialization_diagnostics = _bounds_seed_diagnostics(
            points,
            include_bounds_proxy=bool(parsed_config["include_bounds_proxy"]),
            min_radius=float(parsed_config["min_radius"]),
        )
        primitives = _prepend_bounds_ellipsoid_seed(
            primitives,
            diagnostics=initialization_diagnostics,
            primitive_limit=int(parsed_config["primitive_count"]),
        )
        primitives = _apply_soft_silhouette_opacity_floor(
            primitives,
            floor=float(parsed_config["primitive_opacity_floor"]),
        )
        if bool(parsed_config["calibrate_silhouette_bounds"]):
            primitives = _calibrate_primitives_to_silhouette_bounds(
                primitives,
                cameras=cameras,
                silhouettes=silhouettes,
                padding=float(parsed_config["silhouette_bounds_padding"]),
            )
        renderables = tuple(renderable_from_primitive(primitive) for primitive in primitives)
        scene = RenderableScene(primitives=renderables)
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
    mesh_export_primitives = _scale_primitives_for_mesh_export(
        optimized_primitives,
        scale=float(parsed_config["mesh_proxy_scale"]),
    )
    mesh_proxy = combine_primitive_meshes(mesh_export_primitives, resolution=20)
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
    objective_history.insert(
        1,
        {
            "stage": "bounds_intersection_seed",
            "diagnostics": initialization_diagnostics,
        },
    )
    boundary_sdf_improvement = _boundary_sdf_improvement_summary(
        initial_loss,
        loss,
    )

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
        initialization_diagnostics=initialization_diagnostics,
        boundary_sdf_improvement=boundary_sdf_improvement,
        mesh_proxy_scale=float(parsed_config["mesh_proxy_scale"]),
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
        boundary_or_sdf_improved=bool(
            boundary_sdf_improvement["boundary_or_sdf_improved"]
        ),
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
            "boundary_sdf_improvement": boundary_sdf_improvement,
            "mesh_proxy_scale": float(parsed_config["mesh_proxy_scale"]),
            "render_batch": render_batch,
            "view_signal_weights": dict(target_view_weights),
        },
    )


def _bounds_seed_diagnostics(
    points: np.ndarray,
    *,
    include_bounds_proxy: bool,
    min_radius: float,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "bounds_proxy": {
            "enabled": False,
            "reason": "disabled" if not include_bounds_proxy else "insufficient_points",
        },
        "source_point_count": int(len(points)),
    }
    if len(points) == 0:
        return diagnostics
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) * 0.5
    radii = np.maximum((maxs - mins) * 0.5, float(min_radius))
    diagnostics["source_bounds"] = {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "center": center.tolist(),
        "radii": radii.tolist(),
    }
    if include_bounds_proxy:
        diagnostics["bounds_proxy"] = {
            "enabled": True,
            "center": center.tolist(),
            "radii": radii.tolist(),
            "reason": "per_view_bbox_intersection_seed",
        }
    return diagnostics


def _prepend_bounds_ellipsoid_seed(
    primitives: Sequence[object],
    *,
    diagnostics: Mapping[str, Any],
    primitive_limit: int,
) -> tuple[object, ...]:
    bounds_proxy = diagnostics.get("bounds_proxy")
    if not isinstance(bounds_proxy, Mapping) or not bounds_proxy.get("enabled"):
        return tuple(primitives)
    center = np.asarray(bounds_proxy.get("center", (0.0, 0.0, 0.0)), dtype=float)
    radii = np.asarray(bounds_proxy.get("radii", (1.0, 1.0, 1.0)), dtype=float)
    if center.shape != (3,) or radii.shape != (3,):
        return tuple(primitives)
    proxy = EllipsoidPrimitive(
        center=center,
        radii=radii,
        density=1.0,
        confidence=1.0,
    )
    limit = max(1, int(primitive_limit))
    return (proxy, *tuple(primitives)[: max(0, limit - 1)])


def _boundary_sdf_improvement_summary(
    initial_loss: Any,
    loss: Any,
) -> dict[str, float | bool]:
    initial_terms = getattr(initial_loss, "terms", {}) or {}
    final_terms = getattr(loss, "terms", {}) or {}
    boundary_improvement = _loss_delta(
        initial_terms,
        final_terms,
        "boundary_iou",
    )
    signed_distance_improvement = _loss_delta(
        initial_terms,
        final_terms,
        "signed_distance",
    )
    area_improvement = _loss_delta(initial_terms, final_terms, "area_iou")
    return {
        "boundary_loss_improvement": boundary_improvement,
        "signed_distance_loss_improvement": signed_distance_improvement,
        "area_iou_loss_improvement": area_improvement,
        "boundary_or_sdf_improved": (
            boundary_improvement > 1.0e-9
            or signed_distance_improvement > 1.0e-9
        ),
    }


def _loss_delta(
    initial_terms: Mapping[str, Any],
    final_terms: Mapping[str, Any],
    key: str,
) -> float:
    try:
        initial_value = float(initial_terms.get(key, 0.0) or 0.0)
        final_value = float(final_terms.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(initial_value) or not math.isfinite(final_value):
        return 0.0
    return float(initial_value - final_value)


def _scale_primitives_for_mesh_export(
    primitives: Sequence[object],
    *,
    scale: float,
) -> tuple[object, ...]:
    parsed_scale = float(scale)
    if not math.isfinite(parsed_scale) or abs(parsed_scale - 1.0) <= 1.0e-12:
        return tuple(primitives)
    parsed_scale = max(1.0e-6, parsed_scale)
    scaled: list[object] = []
    for primitive in primitives:
        if isinstance(primitive, EllipsoidPrimitive):
            scaled.append(
                EllipsoidPrimitive(
                    center=primitive.center,
                    radii=primitive.radii * parsed_scale,
                    rotation=primitive.rotation,
                    density=primitive.density,
                    confidence=primitive.confidence,
                )
            )
        elif isinstance(primitive, AnisotropicGaussianPrimitive):
            scaled.append(
                AnisotropicGaussianPrimitive(
                    center=primitive.center,
                    covariance=primitive.covariance * (parsed_scale * parsed_scale),
                    opacity=primitive.opacity,
                    color=primitive.color,
                    semantic_role=primitive.semantic_role,
                    confidence=primitive.confidence,
                )
            )
        elif isinstance(primitive, SuperquadricPrimitive):
            scaled.append(
                SuperquadricPrimitive(
                    center=primitive.center,
                    radii=primitive.radii * parsed_scale,
                    rotation=primitive.rotation,
                    epsilon1=primitive.epsilon1,
                    epsilon2=primitive.epsilon2,
                    density=primitive.density,
                    confidence=primitive.confidence,
                )
            )
        else:
            scaled.append(primitive)
    return tuple(scaled)


def _apply_soft_silhouette_opacity_floor(
    primitives: Sequence[object],
    *,
    floor: float,
) -> tuple[object, ...]:
    """Raise render opacity for silhouette fitting without changing geometry."""
    opacity_floor = float(np.clip(floor, 0.0, 1.0))
    adjusted: list[object] = []
    for primitive in primitives:
        if isinstance(primitive, EllipsoidPrimitive):
            adjusted.append(
                EllipsoidPrimitive(
                    center=primitive.center,
                    radii=primitive.radii,
                    rotation=primitive.rotation,
                    density=max(float(primitive.density), opacity_floor),
                    confidence=primitive.confidence,
                )
            )
        elif isinstance(primitive, AnisotropicGaussianPrimitive):
            adjusted.append(
                AnisotropicGaussianPrimitive(
                    center=primitive.center,
                    covariance=primitive.covariance,
                    opacity=max(float(primitive.opacity), opacity_floor),
                    color=primitive.color,
                    semantic_role=primitive.semantic_role,
                    confidence=primitive.confidence,
                )
            )
        elif isinstance(primitive, SuperquadricPrimitive):
            adjusted.append(
                SuperquadricPrimitive(
                    center=primitive.center,
                    radii=primitive.radii,
                    rotation=primitive.rotation,
                    epsilon1=primitive.epsilon1,
                    epsilon2=primitive.epsilon2,
                    density=max(float(primitive.density), opacity_floor),
                    confidence=primitive.confidence,
                )
            )
        else:
            adjusted.append(primitive)
    return tuple(adjusted)


def _calibrate_primitives_to_silhouette_bounds(
    primitives: Sequence[object],
    *,
    cameras: Sequence[object],
    silhouettes: Mapping[str, np.ndarray],
    padding: float = 1.0,
) -> tuple[object, ...]:
    """Align primitive world extents to the target silhouette boxes.

    Visual-hull surface samples are useful seeds, but clustered ellipsoids can
    over-cover the orthographic frame when several broad covariance ellipses are
    alpha-composited.  This deterministic affine calibration keeps the editable
    primitive family while matching the per-axis image evidence before any
    objective refinement runs.
    """
    source = tuple(primitives)
    if not source or not cameras or not silhouettes:
        return source
    target_intervals = _target_axis_intervals_from_silhouettes(cameras, silhouettes)
    primitive_intervals = _primitive_axis_intervals(source)
    if not target_intervals or not primitive_intervals:
        return source

    scale = np.ones(3, dtype=np.float64)
    current_center = np.zeros(3, dtype=np.float64)
    target_center = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        current = primitive_intervals.get(axis)
        desired = target_intervals.get(axis)
        if current is None or desired is None:
            continue
        current_min, current_max = current
        desired_min, desired_max = desired
        current_extent = max(float(current_max - current_min), 1.0e-9)
        desired_extent = max(float(desired_max - desired_min) * float(padding), 1.0e-9)
        current_center[axis] = 0.5 * (float(current_min) + float(current_max))
        target_center[axis] = 0.5 * (float(desired_min) + float(desired_max))
        scale[axis] = float(np.clip(desired_extent / current_extent, 0.02, 50.0))

    transformed = [
        _affine_transform_primitive(
            primitive,
            source_center=current_center,
            target_center=target_center,
            scale=scale,
        )
        for primitive in source
    ]
    return tuple(transformed)


def _target_axis_intervals_from_silhouettes(
    cameras: Sequence[object],
    silhouettes: Mapping[str, np.ndarray],
) -> dict[int, tuple[float, float]]:
    intervals: dict[int, list[tuple[float, float]]] = {0: [], 1: [], 2: []}
    for camera in cameras:
        name = str(getattr(camera, "name", ""))
        raw_mask = silhouettes.get(name)
        if raw_mask is None:
            continue
        mask = np.asarray(raw_mask, dtype=np.float64)
        if mask.ndim != 2:
            continue
        hard = mask > 0.5
        if not hard.any():
            continue
        ys, xs = np.nonzero(hard)
        width, height = getattr(camera, "image_size", (mask.shape[1], mask.shape[0]))
        axes = tuple(int(axis) for axis in getattr(camera, "axes", (0, 2)))
        bounds = tuple(float(value) for value in getattr(camera, "world_bounds", (-1.0, 1.0, -1.0, 1.0)))
        x_interval = _pixel_interval_to_world(
            int(xs.min()),
            int(xs.max()),
            int(width),
            bounds[0],
            bounds[1],
            descending=False,
        )
        y_interval = _pixel_interval_to_world(
            int(ys.min()),
            int(ys.max()),
            int(height),
            bounds[2],
            bounds[3],
            descending=True,
        )
        intervals[axes[0]].append(x_interval)
        intervals[axes[1]].append(y_interval)

    merged: dict[int, tuple[float, float]] = {}
    for axis, axis_intervals in intervals.items():
        if not axis_intervals:
            continue
        lows = [pair[0] for pair in axis_intervals]
        highs = [pair[1] for pair in axis_intervals]
        low = float(np.mean(lows))
        high = float(np.mean(highs))
        if high > low:
            merged[axis] = (low, high)
    return merged


def _pixel_interval_to_world(
    pixel_min: int,
    pixel_max: int,
    length: int,
    world_min: float,
    world_max: float,
    *,
    descending: bool,
) -> tuple[float, float]:
    denom = max(1, int(length) - 1)
    start = float(pixel_min) / float(denom)
    end = float(pixel_max) / float(denom)
    if descending:
        w0 = float(world_max) - start * (float(world_max) - float(world_min))
        w1 = float(world_max) - end * (float(world_max) - float(world_min))
    else:
        w0 = float(world_min) + start * (float(world_max) - float(world_min))
        w1 = float(world_min) + end * (float(world_max) - float(world_min))
    return (min(w0, w1), max(w0, w1))


def _primitive_axis_intervals(
    primitives: Sequence[object],
) -> dict[int, tuple[float, float]]:
    lows = np.full(3, np.inf, dtype=np.float64)
    highs = np.full(3, -np.inf, dtype=np.float64)
    for primitive in primitives:
        center, cov = _primitive_center_covariance(primitive)
        if center is None or cov is None:
            continue
        radius = np.sqrt(np.maximum(np.diag(cov), 1.0e-12))
        lows = np.minimum(lows, center - radius)
        highs = np.maximum(highs, center + radius)
    intervals: dict[int, tuple[float, float]] = {}
    for axis in range(3):
        if np.isfinite(lows[axis]) and np.isfinite(highs[axis]) and highs[axis] > lows[axis]:
            intervals[axis] = (float(lows[axis]), float(highs[axis]))
    return intervals


def _primitive_center_covariance(
    primitive: object,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if isinstance(primitive, EllipsoidPrimitive):
        return primitive.center, primitive.covariance()
    if isinstance(primitive, AnisotropicGaussianPrimitive):
        return primitive.center, np.asarray(primitive.covariance, dtype=np.float64)
    if isinstance(primitive, SuperquadricPrimitive):
        cov = primitive.rotation @ np.diag(primitive.radii * primitive.radii) @ primitive.rotation.T
        return primitive.center, cov
    if hasattr(primitive, "center") and hasattr(primitive, "radii"):
        center = np.asarray(getattr(primitive, "center"), dtype=np.float64)
        radii = np.asarray(getattr(primitive, "radii"), dtype=np.float64)
        if center.shape == (3,) and radii.shape == (3,):
            rotation = np.asarray(getattr(primitive, "rotation", np.eye(3)), dtype=np.float64)
            cov = rotation @ np.diag(radii * radii) @ rotation.T
            return center, cov
    return None, None


def _affine_transform_primitive(
    primitive: object,
    *,
    source_center: np.ndarray,
    target_center: np.ndarray,
    scale: np.ndarray,
) -> object:
    center, cov = _primitive_center_covariance(primitive)
    if center is None or cov is None:
        return primitive
    new_center = target_center + (center - source_center) * scale
    transform = np.diag(scale)
    new_cov = transform @ cov @ transform
    new_radii, new_rotation = _decompose_covariance_to_radii_rotation(new_cov)
    if isinstance(primitive, EllipsoidPrimitive):
        return EllipsoidPrimitive(
            center=new_center,
            radii=new_radii,
            rotation=new_rotation,
            density=primitive.density,
            confidence=primitive.confidence,
        )
    if isinstance(primitive, AnisotropicGaussianPrimitive):
        return AnisotropicGaussianPrimitive(
            center=new_center,
            covariance=new_cov,
            opacity=primitive.opacity,
            color=primitive.color,
            semantic_role=primitive.semantic_role,
            confidence=primitive.confidence,
        )
    if isinstance(primitive, SuperquadricPrimitive):
        return SuperquadricPrimitive(
            center=new_center,
            radii=new_radii,
            rotation=new_rotation,
            epsilon1=primitive.epsilon1,
            epsilon2=primitive.epsilon2,
            density=primitive.density,
            confidence=primitive.confidence,
        )
    return primitive


def _decompose_covariance_to_radii_rotation(
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(covariance, dtype=np.float64)
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1.0e-12)
    eigvecs = eigvecs[:, order]
    return np.sqrt(eigvals), eigvecs
