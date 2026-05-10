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
from .dependency_policy import _visual_hull_dependency_report
from .metrics import _mesh_metadata, _openvdb_required, _record_retopology_policy, _retopology_required
from .postprocess import _evaluate_postprocess, _mesh_required, _postprocess_mesh, _postprocess_required
from .volume_flow import _mesh_uses_sdf, _sdf_projection_enabled, _sdf_required, _visual_hull_cache_directory


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
        resolution = int(request.config.get("resolution", 64))
        requested_backend = str(request.config.get("backend", "dense")).strip().lower()
        if not request.target.constraints:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="skipped",
                warnings=("visual hull requires at least one view constraint",),
            )
        try:
            from reconstruction.point_cloud import visual_hull_grid_from_target
            from volume import extract_mesh, save_volume

            boundary_refine = bool(request.config.get("boundary_refine", True))
            boundary_dilate_px = request.config.get("boundary_dilate_px")
            grid = visual_hull_grid_from_target(
                request.target,
                resolution=resolution,
                chunk_size=request.config.get("chunk_size"),
                backend=requested_backend,
                boundary_refine=boundary_refine,
                boundary_dilate_px=(
                    None if boundary_dilate_px is None else int(boundary_dilate_px)
                ),
                cache_directory=_visual_hull_cache_directory(request.config),
                cache_namespace=str(
                    request.config.get("cache_namespace", "visual_hull")
                ),
                cache_read=bool(
                    request.config.get(
                        "cache_read",
                        not bool(request.config.get("cache_write_only", False)),
                    )
                ),
                cache_write=bool(
                    request.config.get(
                        "cache_write",
                        not bool(request.config.get("cache_read_only", False)),
                    )
                ),
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
                "enabled": bool(request.config.get("boundary_refine", True)),
                "boundary_dilate_px": request.config.get("boundary_dilate_px"),
            },
        }
        if requested_backend == "openvdb" and openvdb_status is not None:
            mesh_metrics["openvdb"] = (
                openvdb_status.to_dict()
                if hasattr(openvdb_status, "to_dict")
                else dict(openvdb_status)
            )

        direct_sparse_builder = requested_backend in {
            "chunked",
            "sparse_hash",
            "openvdb",
        }
        volume_backend_metadata = {
            "requested": requested_backend,
            "storage": getattr(grid, "backend", type(grid).__name__),
            "direct_sparse_builder": direct_sparse_builder,
        }
        mesh_metrics["volume_backend"] = volume_backend_metadata

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

        if root is not None:
            try:
                metadata = save_volume(
                    grid,
                    root / "volume",
                    source_candidate_id=request.candidate_id,
                    source_views=tuple(
                        constraint.view for constraint in request.target.constraints
                    ),
                    extra=volume_metadata_extra,
                )
                volume_path = root / "volume"
                artifacts["volume_metadata"] = root / "volume" / "volume.json"
                artifacts["volume_npz"] = root / "volume" / "volume.npz"
                mesh_metrics["volume_metadata"] = metadata.to_dict()
            except Exception as exc:
                warnings.append(f"failed to save volume artifact: {exc}")

            if sdf_result is not None:
                try:
                    sdf_metadata = save_volume(
                        sdf_result.grid,
                        root / "sdf_volume",
                        source_candidate_id=request.candidate_id,
                        source_views=tuple(
                            constraint.view
                            for constraint in request.target.constraints
                        ),
                        extra={
                            "backend": self.name,
                            "source": "visual_hull_occupancy",
                            "sdf_projection": sdf_result.report.to_dict(),
                        },
                    )
                    artifacts["sdf_volume_metadata"] = (
                        root / "sdf_volume" / "volume.json"
                    )
                    artifacts["sdf_volume_npz"] = root / "sdf_volume" / "volume.npz"
                    mesh_metrics["sdf_volume_metadata"] = sdf_metadata.to_dict()
                except Exception as exc:
                    warnings.append(f"failed to save SDF volume artifact: {exc}")

            if requested_backend == "openvdb" or bool(
                request.config.get("export_openvdb")
            ):
                try:
                    from volume import export_to_openvdb

                    openvdb_export_path = root / "volume" / "volume.vdb"
                    openvdb_export_status = export_to_openvdb(
                        grid,
                        openvdb_export_path,
                    )
                    mesh_metrics["openvdb_export"] = openvdb_export_status.to_dict()
                    if openvdb_export_status.status == "exported":
                        artifacts["volume_openvdb"] = openvdb_export_path
                    else:
                        warnings.append(
                            "OpenVDB artifact export did not complete: "
                            f"{openvdb_export_status.message}"
                        )
                except Exception as exc:
                    warnings.append(f"failed to export OpenVDB artifact: {exc}")
                    mesh_metrics["openvdb_export"] = {
                        "available": False,
                        "status": "failed",
                        "message": str(exc),
                        "error_type": type(exc).__name__,
                    }

        mesh_result = None
        postprocess_status: dict[str, Any] = _evaluate_postprocess(
            None,
            str(request.config.get("postprocess", "none")),
            config=request.config,
        )
        mesh_metrics["mesh_postprocess"] = postprocess_status
        try:
            mesh_result = extract_mesh(
                mesh_source_grid,
                method=str(request.config.get("mesh_method", "marching_cubes")),
                allow_point_cloud_fallback=bool(
                    request.config.get("allow_point_cloud_fallback", False)
                ),
            )
            postprocess = str(request.config.get("postprocess", "none"))
            final_mesh_result, postprocess_status = _postprocess_mesh(
                mesh_result,
                postprocess,
                config=request.config,
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
                from metrics.topology import mesh_topology_report

                topology = mesh_topology_report(
                    final_mesh_result.vertices,
                    final_mesh_result.faces,
                )
                mesh_metrics["topology"] = topology.to_dict()
                if root is not None:
                    from reconstruction.mesh_io import write_obj

                    mesh_path = write_obj(
                        root / "mesh" / "visual_hull.obj",
                        {
                            "vertices": final_mesh_result.vertices,
                            "faces": final_mesh_result.faces,
                        },
                        header=(f"candidate {request.candidate_id}", self.name),
                    )
                    artifacts["mesh_obj"] = mesh_path
            elif getattr(mesh_result, "topology", None):
                mesh_metrics["topology"] = dict(mesh_result.topology)
                if str(getattr(mesh_result, "method", "")) != "points":
                    warnings.append(
                        "mesh extraction did not produce a mesh: "
                        f"{getattr(mesh_result, 'message', 'unknown reason')}"
                    )
        except Exception as exc:
            warnings.append(f"mesh extraction failed: {exc}")

        occupied_voxels = int(stats.get("active_voxels", 0))
        topology_score = float(
            mesh_metrics.get("topology", {}).get("topology_score", 0.0)
        )
        per_view_metrics: dict[str, Any] = {}
        try:
            from reconstruction.point_cloud import (
                visual_hull_view_diagnostics_from_target,
                visual_hull_projection_metrics_from_target,
            )

            per_view_metrics = visual_hull_projection_metrics_from_target(
                request.target,
                grid,
                max_metric_voxels=int(
                    request.config.get("projection_metric_max_voxels", 4_000_000)
                ),
            )
            skipped_metric = per_view_metrics.pop("_skipped", None)
            if skipped_metric:
                mesh_metrics["projection_metrics_skipped"] = skipped_metric
                warnings.append(str(skipped_metric.get("reason", "projection metrics skipped")))
            else:
                diagnostics = visual_hull_view_diagnostics_from_target(
                    request.target,
                    grid,
                    per_view_metrics=per_view_metrics,
                    boundary_refine=bool(request.config.get("boundary_refine", True)),
                    boundary_dilate_px=(
                        None
                        if request.config.get("boundary_dilate_px") is None
                        else int(request.config.get("boundary_dilate_px"))
                    ),
                )
                mesh_metrics["visual_hull_view_diagnostics"] = diagnostics
                if diagnostics.get("axis_or_transform_suspect"):
                    warnings.append("visual hull view diagnostics flagged axis_or_transform_suspect")
                if diagnostics.get("catastrophic_view_failure"):
                    warnings.append("visual hull view diagnostics flagged catastrophic_view_failure")
        except Exception as exc:
            warnings.append(f"projection metrics failed: {exc}")

        retopology_decision = _record_retopology_policy(
            mesh_metrics=mesh_metrics,
            per_view_metrics=per_view_metrics,
            editability_score=0.15,
            config=request.config,
            warnings=warnings,
        )

        status = "failed" if errors else "success"
        if _openvdb_required(request.config):
            openvdb_payload = mesh_metrics.get("openvdb")
            openvdb_export_payload = mesh_metrics.get("openvdb_export")
            if requested_backend == "openvdb" and isinstance(openvdb_payload, Mapping):
                if not bool(openvdb_payload.get("available")):
                    status = "failed"
                    errors.append(
                        str(
                            openvdb_payload.get(
                                "message",
                                "OpenVDB backend was required but bindings were unavailable",
                            )
                        )
                    )
            if bool(request.config.get("export_openvdb")):
                if (
                    not isinstance(openvdb_export_payload, Mapping)
                    or openvdb_export_payload.get("status") != "exported"
                ):
                    status = "failed"
                    if isinstance(openvdb_export_payload, Mapping):
                        errors.append(
                            str(
                                openvdb_export_payload.get(
                                    "message",
                                    "OpenVDB export was required but did not complete",
                                )
                            )
                        )
                    else:
                        errors.append(
                            "OpenVDB export was required but no export status was produced"
                        )
        if postprocess_status.get("status") == "failed" and _postprocess_required(
            request.config
        ):
            status = "failed"
            errors.append(str(postprocess_status.get("message", "mesh postprocess failed")))
        if (
            mesh_result is not None
            and not mesh_result.available
            and str(getattr(mesh_result, "method", "")) != "points"
        ):
            message = str(
                getattr(mesh_result, "message", "mesh extraction did not produce a mesh")
            )
            if _mesh_required(request.config):
                status = "failed"
                errors.append(message)
            elif status == "success":
                status = "degraded"
        if (
            retopology_decision is not None
            and not retopology_decision.accepted_for_editing
            and _retopology_required(request.config)
        ):
            status = "failed"
            errors.append(str(retopology_decision.reason))

        metrics = CandidateMetrics(
            per_view=per_view_metrics,
            editability_score=0.15,
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
            degraded=status == "degraded",
            payload=grid,
        )
