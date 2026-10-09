from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from blender_blocking.utils.optional_deps import dependency_report, probe_dependency
except Exception:  # pragma: no cover - script-style imports
    from utils.optional_deps import dependency_report, probe_dependency
from ...backend import BackendCapabilities, BaseBackend, BackendBudget
from ...types import CandidateMetrics, CandidateRequest, CandidateResult
from .artifacts import (
    export_openvdb_artifact,
    save_sdf_volume_artifacts,
    save_volume_artifacts,
    write_visual_hull_editable_proxy_artifacts,
    write_visual_hull_mesh_artifact,
)
from .config import direct_sparse_builder_requested, visual_hull_run_config
from .dependency_policy import _visual_hull_dependency_report
from .metrics import _mesh_metadata, _record_retopology_policy
from .postprocess import _evaluate_postprocess, _postprocess_mesh
from .projection import collect_visual_hull_projection_metrics
from .status import visual_hull_result_status
from .volume_flow import _mesh_uses_sdf, _sdf_projection_enabled, _sdf_required


class VisualHullBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            name="visual_hull_voxel",
            capabilities=BackendCapabilities(
                requires_blender=False,
                supports_pure_python=True,
                supports_multi_view=True,
                supports_top_view=True,
                supports_uncertainty=True,
                outputs_volume=True,
                outputs_mesh=True,
                optional_dependencies=("skimage", "open3d", "openvdb"),
                editability_score=0.15,
            ),
        )

    def estimate_budget(
        self, target: Any, config: Mapping[str, Any]
    ) -> BackendBudget:
        resolution = int(config.get("resolution", 64))
        views = max(1, len(getattr(target, "constraints", ()) or ()))
        voxels = resolution**3 * views
        return BackendBudget(
            estimated_seconds=voxels / 15_000_000.0,
            estimated_memory_mb=(resolution**3) / (1024 * 1024),
            notes=("visual hull cost scales cubically with resolution",),
        )

    def reconstruct(self, request: CandidateRequest) -> CandidateResult:
        from blender_blocking.evaluation.cost_model import timed_call
        recorder = getattr(request.context, "cost_recorder", None)
        from blender_blocking.reconstruction.native_geometry import GeometryArrays
        geometry = None
        run_config = visual_hull_run_config(request.config)
        resolution = run_config.resolution
        requested_backend = run_config.requested_backend
        if not request.target.constraints:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="skipped",
                warnings=("visual hull requires at least one view constraint",),
            )
        try:
            from reconstruction.point_cloud import visual_hull_grid_from_target
            from volume import extract_mesh

            grid = timed_call(recorder, "construction", visual_hull_grid_from_target,
                request.target,
                resolution=resolution,
                chunk_size=request.config.get("chunk_size"),
                backend=requested_backend,
                adaptive=bool(request.config.get("adaptive_hull", False)),
                conservative=bool(request.config.get("preserve_thin_features", False)),
                boundary_refine=run_config.boundary_refine,
                boundary_dilate_px=run_config.boundary_dilate_px,
                cache_directory=run_config.cache_directory,
                cache_namespace=run_config.cache_namespace,
                cache_read=run_config.cache_read,
                cache_write=run_config.cache_write,
                cache_owned_writes=run_config.cache_owned_writes,
            )

            openvdb_status = getattr(grid, "openvdb_status", None)
            stats = grid.stats().to_dict()
        except Exception as exc:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="failed",
                errors=(str(exc),),
            )

        artifacts: dict[str, Path] = {}
        warnings: list[str] = []
        errors: list[str] = []
        if requested_backend == "openvdb":
            openvdb_payload = (
                openvdb_status.to_dict()
                if hasattr(openvdb_status, "to_dict")
                else dict(openvdb_status or {})
            )
            if bool(openvdb_payload.get("available")):
                warnings.append(
                    "openvdb backend requested; sparse chunks are used in memory and "
                    "direct OpenVDB artifact export will be attempted when artifacts "
                    "are enabled"
                )
            else:
                warnings.append(
                    "openvdb backend requested but direct OpenVDB export is unavailable; "
                    "using NPZ-interchange sparse backing"
                )
        volume_path = None
        mesh_path = None
        mesh_metrics: dict[str, Any] = {
            "optional_dependencies": _visual_hull_dependency_report(),
            "boundary_refinement": {
                "enabled": run_config.boundary_refine,
                "boundary_dilate_px": run_config.boundary_dilate_px,
            },
        }
        if requested_backend == "openvdb" and openvdb_status is not None:
            mesh_metrics["openvdb"] = (
                openvdb_status.to_dict()
                if hasattr(openvdb_status, "to_dict")
                else dict(openvdb_status)
            )

        direct_sparse_builder = direct_sparse_builder_requested(requested_backend)
        volume_backend_metadata = {
            "requested": requested_backend,
            "storage": getattr(grid, "backend", type(grid).__name__),
            "direct_sparse_builder": direct_sparse_builder,
        }
        mesh_metrics["volume_backend"] = volume_backend_metadata
        mesh_metrics["adaptive_hull"] = getattr(grid, "adaptive_report", {})

        volume_metadata_extra = {
            "backend": self.name,
            "config": dict(request.config),
            "volume_backend": volume_backend_metadata,
        }
        try:
            from evaluation.uncertainty import uncertainty_report_from_target

            uncertainty_report = uncertainty_report_from_target(request.target).to_dict()
            mesh_metrics["uncertainty_report"] = uncertainty_report
            volume_metadata_extra["uncertainty_report"] = uncertainty_report
        except Exception as exc:
            uncertainty_report = {"status": "unavailable", "message": str(exc)}
            mesh_metrics["uncertainty_report"] = uncertainty_report
        if openvdb_status is not None:
            volume_metadata_extra["openvdb"] = openvdb_status.to_dict()
        cache_status = getattr(grid, "chunk_cache_status", None)
        if cache_status is not None:
            cache_payload = (
                cache_status.to_dict()
                if hasattr(cache_status, "to_dict")
                else dict(cache_status)
            )
            if run_config.cache_owned_writes:
                receipt = getattr(grid, "chunk_cache_last_ownership_receipt", None)
                cache_payload.update(
                    owned_writes=True,
                    ownership_status="retained" if receipt is not None else "unrun",
                    last_ownership_receipt=str(receipt) if receipt is not None else None,
                    ownership_scope=(
                        "last fresh store during this grid construction; "
                        "cache hits create no ownership"
                    ),
                )
            mesh_metrics["chunk_cache"] = cache_payload
            volume_metadata_extra["chunk_cache"] = cache_payload

        root = request.candidate_artifact_root()
        sdf_result = None
        mesh_source_grid = grid
        if _sdf_projection_enabled(request.config):
            try:
                from volume import signed_distance_grid_from_volume

                narrow_band_voxels = request.config.get("sdf_narrow_band_voxels")
                sdf_result = signed_distance_grid_from_volume(
                    grid,
                    occupancy_threshold=float(
                        request.config.get(
                            "sdf_occupancy_threshold",
                            request.config.get("occupancy_threshold", 0.5),
                        )
                    ),
                    prefer_scipy=bool(request.config.get("sdf_prefer_scipy", True)),
                    narrow_band_voxels=(
                        None
                        if narrow_band_voxels is None
                        else int(narrow_band_voxels)
                    ),
                    output_backend=str(request.config.get("sdf_backend", "dense")),
                    numpy_max_voxels=int(
                        request.config.get("sdf_numpy_max_voxels", 250_000)
                    ),
                )
                mesh_metrics["sdf_projection"] = sdf_result.report.to_dict()
                if sdf_result.report.warnings:
                    warnings.extend(
                        f"SDF projection warning: {warning}"
                        for warning in sdf_result.report.warnings
                    )
                if _mesh_uses_sdf(request.config):
                    mesh_source_grid = sdf_result.grid
                    mesh_metrics["mesh_source"] = {
                        "kind": "signed_distance",
                        "sign_convention": sdf_result.report.sign_convention,
                        "source_backend": sdf_result.report.source_backend,
                        "output_backend": sdf_result.report.output_backend,
                    }
                else:
                    mesh_metrics["mesh_source"] = {"kind": "occupancy"}
            except Exception as exc:
                message = f"SDF projection failed: {exc}"
                warnings.append(message)
                mesh_metrics["sdf_projection"] = {
                    "status": "failed",
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                    "required": _sdf_required(request.config),
                }
                if _sdf_required(request.config):
                    errors.append(message)
        else:
            mesh_metrics["mesh_source"] = {"kind": "occupancy"}

        source_views = tuple(
            str(constraint.view) for constraint in request.target.constraints
        )
        if root is not None:
            volume_path = timed_call(recorder, "serialization", save_volume_artifacts,
                grid=grid,
                root=root,
                candidate_id=request.candidate_id,
                source_views=source_views,
                extra=volume_metadata_extra,
                artifacts=artifacts,
                metrics=mesh_metrics,
                warnings=warnings,
            )
            if sdf_result is not None:
                save_sdf_volume_artifacts(
                    sdf_grid=sdf_result.grid,
                    root=root,
                    candidate_id=request.candidate_id,
                    source_views=source_views,
                    sdf_report=sdf_result.report.to_dict(),
                    artifacts=artifacts,
                    metrics=mesh_metrics,
                    warnings=warnings,
                )

            if requested_backend == "openvdb" or bool(
                request.config.get("export_openvdb")
            ):
                export_openvdb_artifact(
                    grid=grid,
                    root=root,
                    artifacts=artifacts,
                    metrics=mesh_metrics,
                    warnings=warnings,
                )

        mesh_result = None
        postprocess_status: dict[str, Any] = _evaluate_postprocess(
            None,
            str(request.config.get("postprocess", "none")),
            config=request.config,
        )
        mesh_metrics["mesh_postprocess"] = postprocess_status
        try:
            mesh_result = timed_call(recorder, "extraction", extract_mesh,
                mesh_source_grid,
                recorder=recorder,
                method=str(request.config.get("mesh_method", "marching_cubes")),
                allow_point_cloud_fallback=bool(
                    request.config.get("allow_point_cloud_fallback", False)
                ),
            )
            postprocess = str(request.config.get("postprocess", "none"))
            final_mesh_result, postprocess_status = _postprocess_mesh(
                mesh_result,
                postprocess,
                config={**request.config, "postprocess_artifact_root": str(root / "poisson")} if root else request.config,
            )
            mesh_metrics["mesh_postprocess"] = postprocess_status
            if postprocess_status["status"] == "failed":
                warnings.append(f"mesh postprocess failed: {postprocess_status['message']}")
            elif (
                postprocess_status["status"] == "skipped"
                and postprocess_status["method"] != "none"
            ):
                warnings.append(
                    f"mesh postprocess skipped: {postprocess_status['message']}"
                )
            mesh_metrics["mesh_extraction"] = mesh_result.to_dict()
            if final_mesh_result is not mesh_result:
                mesh_metrics["mesh_postprocess_result"] = final_mesh_result.to_dict()
            mesh_metrics["mesh"] = _mesh_metadata(final_mesh_result)
            if final_mesh_result.available:
                geometry = GeometryArrays.capture(final_mesh_result.vertices, final_mesh_result.faces)
                if request.config.get('adaptive_hull', False):
                    from ...adaptive_geometry import refine_hull_boundary
                    from dataclasses import replace
                    before_refinement = geometry.content_hash
                    geometry, mesh_metrics['boundary_mesh_refinement'] = refine_hull_boundary(request.target, geometry, resolution)
                    changed = geometry.content_hash != before_refinement
                    final_mesh_result = replace(
                        final_mesh_result, vertices=geometry.vertices, faces=geometry.faces,
                        normals=None if changed else final_mesh_result.normals,
                        values=None if changed else final_mesh_result.values)
                    mesh_metrics["boundary_mesh_result"] = final_mesh_result.to_dict()
                from blender_blocking.metrics.topology_receipt import topology_for_geometry

                receipt = getattr(final_mesh_result, "topology_receipt", None)
                cache = getattr(request.context, "geometry_cache", None)
                if cache is not None:
                    before_reuses = getattr(cache, "receipt_reuses", 0)
                    mesh_metrics["topology"] = timed_call(
                        recorder, "topology", cache.topology_report, geometry, receipt=receipt)
                    reused = getattr(cache, "receipt_reuses", 0) > before_reuses
                else:
                    topology, reused = timed_call(
                        recorder, "topology", topology_for_geometry, geometry, receipt)
                    mesh_metrics["topology"] = topology
                mesh_metrics["topology_reuse"] = {
                    "extraction_receipt_reused": reused,
                    "evaluator": "index_connectivity_v1",
                    "scope": "index connectivity only; no geometric-solid qualification",
                }
                mesh_metrics["mesh"] = _mesh_metadata(final_mesh_result)
                if root is not None:
                    try:
                        mesh_path = timed_call(recorder, "serialization", write_visual_hull_mesh_artifact,
                            root=root,
                            candidate_id=request.candidate_id,
                            backend_name=self.name,
                            vertices=final_mesh_result.vertices,
                            faces=final_mesh_result.faces,
                            artifacts=artifacts,
                        )
                        mesh_metrics["mesh_artifact_export"] = {
                            "status": "exported",
                            "path": str(mesh_path),
                            "required_for_render": True,
                        }
                    except Exception as exc:
                        message = f"mesh artifact export failed: {exc}"
                        warnings.append(message)
                        mesh_metrics["mesh_artifact_export"] = {
                            "status": "failed",
                            "message": str(exc),
                            "error_type": type(exc).__name__,
                            "required_for_render": True,
                        }
                else:
                    mesh_metrics["mesh_artifact_export"] = {
                        "status": "skipped",
                        "reason": "no_artifact_root",
                        "required_for_render": False,
                    }
            elif getattr(mesh_result, "topology", None):
                mesh_metrics["topology"] = dict(mesh_result.topology)
                mesh_metrics["mesh_artifact_export"] = {
                    "status": "skipped",
                    "reason": "mesh_unavailable",
                    "required_for_render": False,
                }
                if str(getattr(mesh_result, "method", "")) != "points":
                    warnings.append(
                        "mesh extraction did not produce a mesh: "
                        f"{getattr(mesh_result, 'message', 'unknown reason')}"
                    )
        except Exception as exc:
            message = f"mesh extraction failed: {exc}"
            warnings.append(message)
            mesh_metrics["mesh_extraction"] = {
                "status": "failed",
                "method": str(request.config.get("mesh_method", "marching_cubes")),
                "message": str(exc),
                "error_type": type(exc).__name__,
            }
            mesh_metrics["mesh_artifact_export"] = {
                "status": "skipped",
                "reason": "mesh_extraction_failed",
                "required_for_render": False,
            }

        occupied_voxels = int(stats.get("active_voxels", 0))
        topology_score = float(
            mesh_metrics.get("topology", {}).get("topology_score", 0.0)
        )
        per_view_metrics = timed_call(recorder, "projection", collect_visual_hull_projection_metrics,
            target=request.target,
            grid=grid,
            max_metric_voxels=run_config.projection_metric_max_voxels,
            boundary_refine=run_config.boundary_refine,
            boundary_dilate_px=run_config.boundary_dilate_px,
            metrics=mesh_metrics,
            warnings=warnings,
        )

        retopology_decision = _record_retopology_policy(
            mesh_metrics=mesh_metrics,
            per_view_metrics=per_view_metrics,
            editability_score=0.15,
            config=request.config,
            warnings=warnings,
        )
        effective_editability_score = 0.15
        editable_proxy = timed_call(recorder, "editable_proxy", _maybe_emit_editable_proxy,
            target=request.target,
            candidate_id=request.candidate_id,
            root=root,
            artifacts=artifacts,
            retopology_decision=retopology_decision,
            mesh_metrics=mesh_metrics,
            config=request.config,
            warnings=warnings,
        )
        if editable_proxy is not None:
            mesh_metrics["editable_proxy"] = editable_proxy
            try:
                effective_editability_score = max(
                    effective_editability_score,
                    float(editable_proxy.get("editability_score", 0.0)),
                )
            except (TypeError, ValueError):
                pass

        status, degraded = visual_hull_result_status(
            config=request.config,
            requested_backend=requested_backend,
            mesh_metrics=mesh_metrics,
            mesh_result=mesh_result,
            postprocess_status=postprocess_status,
            retopology_decision=retopology_decision,
            errors=errors,
        )

        metrics = CandidateMetrics(
            per_view=per_view_metrics,
            editability_score=effective_editability_score,
            topology_score=topology_score,
            uncertainty_consistency=float(
                mesh_metrics.get("uncertainty_report", {}).get(
                    "consistency_score",
                    0.0,
                )
                if isinstance(mesh_metrics.get("uncertainty_report"), Mapping)
                else 0.0
            ),
            complexity_penalty=min(1.0, resolution / 256.0),
            extras={
                "visual_hull_stats": stats,
                "occupied_voxels": occupied_voxels,
                **mesh_metrics,
            },
        )
        return CandidateResult(
            candidate_id=request.candidate_id,
            backend_name=self.name,
            status=status,
            metric_result=metrics,
            volume_path=volume_path,
            mesh_path=mesh_path,
            artifacts=artifacts,
            warnings=tuple(warnings),
            errors=tuple(errors),
            degraded=degraded,
            payload=grid,
            geometry=geometry,
        )


def _maybe_emit_editable_proxy(
    *,
    target: Any,
    candidate_id: str,
    root: Path | None,
    artifacts: dict[str, Path],
    retopology_decision: Any,
    mesh_metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    warnings: list[str],
) -> dict[str, Any] | None:
    should_emit = bool(config.get("emit_editable_proxy", True))
    if retopology_decision is not None and getattr(
        retopology_decision,
        "accepted_for_editing",
        False,
    ):
        should_emit = bool(config.get("always_emit_editable_proxy", False))
    if not should_emit:
        return None

    proxy_config = {
        "root_strategy": str(
            config.get("editable_proxy_root_strategy", "hybrid_profile_bounds")
        ),
        "residual_policy": str(
            config.get("editable_proxy_residual_policy", "suggest_patches")
        ),
        "max_nodes": int(config.get("editable_proxy_max_nodes", 64)),
        "program_search_candidates": int(
            config.get("editable_proxy_program_search_candidates", 4)
        ),
        "program_search_objective": str(
            config.get("editable_proxy_program_search_objective", "editable_balanced")
        ),
    }
    try:
        from ..shape_program.builder import build_shape_program_from_target
        from ..shape_program.editability import _complexity_penalty, _editability_score

        try:
            from blender_blocking.primitives.shape_program import validate_shape_program
        except Exception:  # pragma: no cover - legacy script import path
            from primitives.shape_program import validate_shape_program

        program, diagnostics = build_shape_program_from_target(
            target,
            config=proxy_config,
            program_id=str(
                config.get(
                    "editable_proxy_program_id",
                    f"{candidate_id}_editable_proxy",
                )
            ),
        )
        validation_errors = list(validate_shape_program(program))
        diagnostics = {
            **dict(diagnostics),
            "validation_errors": validation_errors,
            "source": "visual_hull_retopology_policy",
            "dense_mesh": dict(mesh_metrics.get("mesh", {})),
            "retopology_policy": (
                retopology_decision.to_dict()
                if hasattr(retopology_decision, "to_dict")
                else None
            ),
            "proxy_config": proxy_config,
        }
        editability_score = float(
            _editability_score(program, proxy_config, compiled=False)
        )
        payload = {
            "available": not validation_errors,
            "status": "ok" if not validation_errors else "failed",
            "kind": "shape_program",
            "editability_score": editability_score,
            "complexity_penalty": float(_complexity_penalty(program, proxy_config)),
            "program": program.to_dict(),
            "diagnostics": diagnostics,
            "accepted_for_editing": not validation_errors,
            "dense_mesh_retained": True,
            "message": (
                "editable shape-program proxy emitted for dense visual hull"
                if not validation_errors
                else "editable proxy validation failed"
            ),
        }
        if root is not None:
            write_visual_hull_editable_proxy_artifacts(
                root=root,
                program_payload=payload["program"],
                diagnostics=diagnostics,
                artifacts=artifacts,
            )
        if validation_errors:
            warnings.append(
                "editable proxy emitted with validation errors: "
                + "; ".join(str(error) for error in validation_errors)
            )
        return payload
    except Exception as exc:
        warnings.append(f"editable proxy generation failed: {exc}")
        return {
            "available": False,
            "status": "failed",
            "kind": "shape_program",
            "message": str(exc),
            "error_type": type(exc).__name__,
            "dense_mesh_retained": True,
        }
