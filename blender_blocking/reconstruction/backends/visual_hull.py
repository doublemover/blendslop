"""Visual-hull reconstruction backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from utils.optional_deps import dependency_report, probe_dependency

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
            "optional_dependencies": _visual_hull_dependency_report()
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

            if requested_backend == "openvdb":
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
        except Exception as exc:
            warnings.append(f"mesh extraction failed: {exc}")

        occupied_voxels = int(stats.get("active_voxels", 0))
        topology_score = float(
            mesh_metrics.get("topology", {}).get("topology_score", 0.0)
        )
        per_view_metrics: dict[str, Any] = {}
        try:
            from reconstruction.point_cloud import (
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
        except Exception as exc:
            warnings.append(f"projection metrics failed: {exc}")

        errors: list[str] = []
        status = "success"
        if postprocess_status.get("status") == "failed" and _postprocess_required(
            request.config
        ):
            status = "failed"
            errors.append(str(postprocess_status.get("message", "mesh postprocess failed")))

        metrics = CandidateMetrics(
            per_view=per_view_metrics,
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
    _mesh, status = _postprocess_mesh(mesh_result, postprocess, config=config)
    return status


def _postprocess_mesh(
    mesh_result: Any,
    postprocess: str,
    *,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    method = str(postprocess).strip().lower()
    required = _postprocess_required(config)
    if method in {"", "none"}:
        return mesh_result, {
            "method": "none",
            "status": "skipped",
            "required": required,
            "message": "mesh postprocess disabled",
        }
    if method not in {"poisson", "screened_poisson"}:
        return mesh_result, {
            "method": method,
            "status": "failed",
            "required": required,
            "message": f"unsupported mesh postprocess mode: {method!r}",
        }
    if mesh_result is None:
        return mesh_result, {
            "method": method,
            "status": "skipped",
            "required": required,
            "message": "mesh extraction has not completed",
        }
    if not mesh_result.available:
        status = "failed" if required else "skipped"
        return mesh_result, {
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
        return mesh_result, {
            "method": method,
            "status": status,
            "required": required,
            "dependency": dependency,
            "message": (
                f"postprocess {method!r} requires optional dependency "
                f"{dependency['module_name']!r}"
            ),
        }
    try:
        processed = _run_open3d_poisson(mesh_result, method, config)
    except Exception as exc:
        status = "failed" if required else "skipped"
        return mesh_result, {
            "method": method,
            "status": status,
            "required": required,
            "dependency": dependency,
            "message": f"postprocess {method!r} failed: {exc}",
            "error_type": type(exc).__name__,
        }
    return processed, {
        "method": method,
        "status": "ok",
        "required": required,
        "dependency": dependency,
        "message": f"postprocess {method!r} completed with Open3D Poisson",
        "input_vertices": int(len(mesh_result.vertices)),
        "input_faces": int(len(mesh_result.faces)),
        "output_vertices": int(len(processed.vertices)),
        "output_faces": int(len(processed.faces)),
        "implementation": "open3d.geometry.TriangleMesh.create_from_point_cloud_poisson",
    }


def _run_open3d_poisson(
    mesh_result: Any,
    method: str,
    config: Mapping[str, Any],
) -> Any:
    import open3d as o3d
    from metrics.topology import mesh_topology_report
    from volume import MeshExtractionResult

    vertices = np.asarray(mesh_result.vertices, dtype=np.float64)
    faces = _triangulated_faces(np.asarray(mesh_result.faces, dtype=np.int64))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise RuntimeError("Poisson postprocess requires at least four 3D vertices")

    source_mesh = o3d.geometry.TriangleMesh()
    source_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    if len(faces):
        source_mesh.triangles = o3d.utility.Vector3iVector(faces)
        source_mesh.remove_duplicated_vertices()
        source_mesh.remove_degenerate_triangles()
        source_mesh.remove_duplicated_triangles()
        source_mesh.remove_non_manifold_edges()
        source_mesh.compute_vertex_normals()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = source_mesh.vertices
    normals = np.asarray(getattr(source_mesh, "vertex_normals", ()), dtype=np.float64)
    if normals.shape == vertices.shape and np.linalg.norm(normals, axis=1).sum() > 0.0:
        point_cloud.normals = source_mesh.vertex_normals
    else:
        point_cloud.estimate_normals()
        try:
            point_cloud.orient_normals_consistent_tangent_plane(
                int(config.get("poisson_normal_neighbors", 16))
            )
        except Exception:
            pass

    depth = int(config.get("poisson_depth", config.get("postprocess_depth", 8)))
    scale = float(config.get("poisson_scale", 1.1))
    linear_fit = bool(config.get("poisson_linear_fit", False))
    kwargs = {"depth": depth, "scale": scale, "linear_fit": linear_fit}
    try:
        processed_mesh, densities = (
            o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                point_cloud,
                **kwargs,
            )
        )
    except TypeError:
        kwargs.pop("linear_fit", None)
        processed_mesh, densities = (
            o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                point_cloud,
                **kwargs,
            )
        )

    if bool(config.get("poisson_crop_to_input_bounds", True)) and len(faces):
        bbox = source_mesh.get_axis_aligned_bounding_box()
        crop_scale = float(config.get("poisson_crop_scale", 1.05))
        if crop_scale > 0.0:
            bbox = bbox.scale(crop_scale, bbox.get_center())
        processed_mesh = processed_mesh.crop(bbox)

    densities_array = np.asarray(densities, dtype=np.float64)
    density_quantile = config.get("poisson_density_quantile")
    if density_quantile is not None and densities_array.size:
        threshold = float(np.quantile(densities_array, float(density_quantile)))
        processed_mesh.remove_vertices_by_mask(densities_array < threshold)

    processed_mesh.remove_duplicated_vertices()
    processed_mesh.remove_degenerate_triangles()
    processed_mesh.remove_duplicated_triangles()
    processed_mesh.compute_vertex_normals()

    output_vertices = np.asarray(processed_mesh.vertices, dtype=np.float64)
    output_faces = np.asarray(processed_mesh.triangles, dtype=np.int64)
    output_normals = np.asarray(processed_mesh.vertex_normals, dtype=np.float64)
    if len(output_vertices) == 0 or len(output_faces) == 0:
        raise RuntimeError("Open3D Poisson produced an empty mesh")

    topology = mesh_topology_report(output_vertices, output_faces).to_dict()
    metrics = {
        **dict(getattr(mesh_result, "metrics", {})),
        "postprocess": method,
        "postprocess_backend": "open3d",
        "poisson_depth": depth,
        "poisson_scale": scale,
        "poisson_linear_fit": linear_fit,
        "input_vertex_count": int(len(vertices)),
        "input_face_count": int(len(faces)),
        "output_vertex_count": int(len(output_vertices)),
        "output_face_count": int(len(output_faces)),
    }
    if densities_array.size:
        metrics.update(
            {
                "density_min": float(np.min(densities_array)),
                "density_max": float(np.max(densities_array)),
                "density_mean": float(np.mean(densities_array)),
            }
        )
    return MeshExtractionResult(
        status="ok",
        method=f"{method}_open3d",
        requested_method=getattr(mesh_result, "requested_method", mesh_result.method),
        method_aliases=tuple(getattr(mesh_result, "method_aliases", ())),
        vertices=output_vertices,
        faces=output_faces,
        normals=output_normals if output_normals.shape == output_vertices.shape else None,
        message=f"{method} Open3D postprocess completed",
        metrics=metrics,
        topology=topology,
    )


def _triangulated_faces(faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    triangles: list[tuple[int, int, int]] = []
    for face in np.asarray(faces, dtype=np.int64):
        if len(face) < 3:
            continue
        first = int(face[0])
        for index in range(1, len(face) - 1):
            tri = (first, int(face[index]), int(face[index + 1]))
            if len(set(tri)) == 3:
                triangles.append(tri)
    if not triangles:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(triangles, dtype=np.int64)


def _postprocess_required(config: Mapping[str, Any]) -> bool:
    return bool(
        config.get("postprocess_required")
        or config.get("require_postprocess")
        or config.get("fail_on_postprocess_skip")
    )


def _optional_dependency_status(module_name: str) -> dict[str, Any]:
    return probe_dependency(module_name).to_dict()


def _visual_hull_dependency_report() -> dict[str, Any]:
    report: dict[str, Any] = dependency_report(("skimage", "open3d"))
    try:
        from volume import detect_openvdb

        report["openvdb"] = detect_openvdb().to_dict()
    except Exception as exc:
        report["openvdb"] = {
            "module_name": "openvdb",
            "available": False,
            "status": "probe_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return report


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
