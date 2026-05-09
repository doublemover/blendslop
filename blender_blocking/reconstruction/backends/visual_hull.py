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
        requested_backend = str(request.config.get("backend", "dense"))
        if not request.target.constraints:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="skipped",
                warnings=("visual hull requires at least one view constraint",),
            )
        try:
            from reconstruction.point_cloud import visual_hull_grid_from_target
            from volume import (
                ChunkedVolumeGrid,
                SparseHashVolumeGrid,
                extract_mesh,
                save_volume,
            )

            grid = visual_hull_grid_from_target(
                request.target,
                resolution=resolution,
                chunk_size=request.config.get("chunk_size"),
            )
            if requested_backend == "chunked":
                grid = ChunkedVolumeGrid.from_dense(
                    grid.to_dense(),
                    grid.bounds,
                    transform=grid.transform,
                    chunk_size=int(request.config.get("chunk_size") or grid.chunk_size),
                )
            elif requested_backend in {"sparse_hash", "openvdb"}:
                grid = SparseHashVolumeGrid.from_dense(
                    grid.to_dense(),
                    grid.bounds,
                    transform=grid.transform,
                    chunk_size=int(request.config.get("chunk_size") or grid.chunk_size),
                )
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
            warnings.append(
                "openvdb backend requested; using sparse_hash interchange because "
                "Python OpenVDB bindings are optional"
            )
        volume_path = None
        mesh_path = None
        mesh_metrics: dict[str, Any] = {}
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
                    extra={"backend": self.name, "config": dict(request.config)},
                )
                volume_path = root / "volume"
                artifacts["volume_metadata"] = root / "volume" / "volume.json"
                artifacts["volume_npz"] = root / "volume" / "volume.npz"
                mesh_metrics["volume_metadata"] = metadata.to_dict()
            except Exception as exc:
                warnings.append(f"failed to save volume artifact: {exc}")

        mesh_result = None
        try:
            mesh_result = extract_mesh(
                grid,
                method=str(request.config.get("mesh_method", "marching_cubes")),
                allow_point_cloud_fallback=True,
            )
            postprocess = str(request.config.get("postprocess", "none"))
            if postprocess != "none":
                warnings.append(
                    f"postprocess {postprocess!r} requested but optional surface "
                    "postprocessors are not available in this environment"
                )
            mesh_metrics["mesh_extraction"] = mesh_result.to_dict()
            if mesh_result.available and root is not None:
                from reconstruction.mesh_io import write_obj
                from metrics.topology import mesh_topology_report

                mesh_path = write_obj(
                    root / "mesh" / "visual_hull.obj",
                    {"vertices": mesh_result.vertices, "faces": mesh_result.faces},
                    header=(f"candidate {request.candidate_id}", self.name),
                )
                artifacts["mesh_obj"] = mesh_path
                topology = mesh_topology_report(mesh_result.vertices, mesh_result.faces)
                mesh_metrics["topology"] = topology.to_dict()
        except Exception as exc:
            warnings.append(f"mesh extraction failed: {exc}")

        occupied_voxels = int(stats.get("active_voxels", 0))
        topology_score = float(
            mesh_metrics.get("topology", {}).get("topology_score", 0.0)
        )
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
            status="success",
            metric_result=metrics,
            volume_path=volume_path,
            mesh_path=mesh_path,
            artifacts=artifacts,
            warnings=tuple(warnings),
            payload=grid,
        )
