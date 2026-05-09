"""Visual-hull reconstruction backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..backend import BackendCapabilities, BaseBackend, BackendBudget
from ..types import CandidateMetrics, CandidateRequest, CandidateResult


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
                optional_dependencies=("skimage",),
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

            grid = visual_hull_grid_from_target(
                request.target,
                resolution=resolution,
                chunk_size=request.config.get("chunk_size"),
                backend=requested_backend,
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
        if requested_backend == "openvdb":
            openvdb_payload = (
                openvdb_status.to_dict()
                if hasattr(openvdb_status, "to_dict")
                else dict(openvdb_status or {})
            )
            if bool(openvdb_payload.get("available")):
                warnings.append(
                    "openvdb backend requested; using NPZ-interchange sparse backing "
                    "because direct OpenVDB export is intentionally not implemented"
                )
            else:
                warnings.append(
                    "openvdb backend requested but direct OpenVDB export is unavailable; "
                    "using NPZ-interchange sparse backing"
                )
        volume_path = None
        mesh_path = None
        mesh_metrics: dict[str, Any] = {}
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
        if openvdb_status is not None:
            volume_metadata_extra["openvdb"] = openvdb_status.to_dict()

        root = request.candidate_artifact_root()
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

        mesh_result = None
        postprocess_status: dict[str, Any] = _evaluate_postprocess(
            None,
            str(request.config.get("postprocess", "none")),
            config=request.config,
        )
        mesh_metrics["mesh_postprocess"] = postprocess_status
        try:
            mesh_result = extract_mesh(
                grid,
                method=str(request.config.get("mesh_method", "marching_cubes")),
                allow_point_cloud_fallback=True,
            )
            postprocess = str(request.config.get("postprocess", "none"))
            postprocess_status = _evaluate_postprocess(
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
            mesh_metrics["mesh"] = _mesh_metadata(mesh_result)
            if mesh_result.available:
                from metrics.topology import mesh_topology_report

                topology = mesh_topology_report(mesh_result.vertices, mesh_result.faces)
                mesh_metrics["topology"] = topology.to_dict()
                if root is not None:
                    from reconstruction.mesh_io import write_obj

                    mesh_path = write_obj(
                        root / "mesh" / "visual_hull.obj",
                        {"vertices": mesh_result.vertices, "faces": mesh_result.faces},
                        header=(f"candidate {request.candidate_id}", self.name),
                    )
                    artifacts["mesh_obj"] = mesh_path
            elif getattr(mesh_result, "topology", None):
                mesh_metrics["topology"] = dict(mesh_result.topology)
        except Exception as exc:
            warnings.append(f"mesh extraction failed: {exc}")

        occupied_voxels = int(stats.get("active_voxels", 0))
        topology_score = float(
            mesh_metrics.get("topology", {}).get("topology_score", 0.0)
        )
        errors: list[str] = []
        status = "success"
        if postprocess_status.get("status") == "failed" and _postprocess_required(
            request.config
        ):
            status = "failed"
            errors.append(str(postprocess_status.get("message", "mesh postprocess failed")))

        metrics = CandidateMetrics(
            editability_score=0.15,
            topology_score=topology_score,
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
            payload=grid,
        )


def _evaluate_postprocess(
    mesh_result: Any,
    postprocess: str,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    method = str(postprocess).strip().lower()
    required = _postprocess_required(config)
    if method in {"", "none"}:
        return {
            "method": "none",
            "status": "skipped",
            "required": required,
            "message": "mesh postprocess disabled",
        }
    if method not in {"poisson", "screened_poisson"}:
        return {
            "method": method,
            "status": "failed",
            "required": required,
            "message": f"unsupported mesh postprocess mode: {method!r}",
        }
    if mesh_result is None:
        return {
            "method": method,
            "status": "skipped",
            "required": required,
            "message": "mesh extraction has not completed",
        }
    if not mesh_result.available:
        status = "failed" if required else "skipped"
        return {
            "method": method,
            "status": status,
            "required": required,
            "mesh_status": mesh_result.status,
            "message": (
                f"postprocess {method!r} requires a mesh, but mesh extraction "
                f"status was {mesh_result.status!r}"
            ),
        }

    dependency = _optional_dependency_status("open3d")
    if not dependency["available"]:
        status = "failed" if required else "skipped"
        return {
            "method": method,
            "status": status,
            "required": required,
            "dependency": dependency,
            "message": (
                f"postprocess {method!r} requires optional dependency "
                f"{dependency['module_name']!r}"
            ),
        }
    return {
        "method": method,
        "status": "skipped",
        "required": required,
        "dependency": dependency,
        "message": (
            f"postprocess {method!r} dependency is available, but backend-neutral "
            "Poisson mesh reconstruction is not implemented"
        ),
    }


def _postprocess_required(config: Mapping[str, Any]) -> bool:
    return bool(
        config.get("postprocess_required")
        or config.get("require_postprocess")
        or config.get("fail_on_postprocess_skip")
    )


def _optional_dependency_status(module_name: str) -> dict[str, Any]:
    try:
        module = __import__(module_name)
    except Exception as exc:
        return {
            "module_name": module_name,
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "module_name": module_name,
        "available": True,
        "module_version": getattr(module, "__version__", None),
    }


def _mesh_metadata(mesh_result: Any) -> dict[str, Any]:
    return {
        "available": bool(mesh_result.available),
        "status": mesh_result.status,
        "method": mesh_result.method,
        "requested_method": getattr(mesh_result, "requested_method", mesh_result.method),
        "vertex_count": int(len(mesh_result.vertices)),
        "face_count": int(len(mesh_result.faces)),
        "has_faces": bool(len(mesh_result.faces)),
        "has_normals": mesh_result.normals is not None,
    }
