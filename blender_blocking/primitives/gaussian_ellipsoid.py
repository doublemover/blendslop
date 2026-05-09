"""Gaussian/ellipsoid proxy reconstruction backend helpers."""

from __future__ import annotations

import time
from typing import Mapping

import numpy as np

from placement.resfit_initialization import (
    PrimitiveInitializationConfig,
    initialize_ellipsoids_from_points,
    initialize_gaussians_from_points,
)
from reconstruction.mesh_io import combine_primitive_meshes, write_obj, write_primitive_set
from reconstruction.point_cloud import target_surface_points
from reconstruction.types import CandidateMetrics, CandidateResult


def run_gaussian_ellipsoid_proxy(request: object) -> CandidateResult:
    """Build editable Gaussian/ellipsoid proxies from silhouette-derived points."""
    start = time.perf_counter()
    config = dict(getattr(request, "config", {}) or {})
    candidate_id = getattr(request, "candidate_id")
    backend_name = getattr(request, "backend_name", "gaussian_ellipsoid_proxy")
    target = getattr(request, "target")

    primitive_count = int(config.get("primitive_count", config.get("max_primitives", 24)))
    target_point_count = int(config.get("target_point_count", 4096))
    resolution = int(config.get("visual_hull_resolution", config.get("resolution", 48)))
    family = str(config.get("family", config.get("proxy_family", "gaussian")))

    try:
        points, point_meta = target_surface_points(
            target,
            resolution=resolution,
            max_points=target_point_count,
            chunk_size=config.get("chunk_size"),
        )
        init_config = PrimitiveInitializationConfig(
            primitive_count=max(1, primitive_count),
            target_point_count=target_point_count,
            min_radius=float(config.get("min_radius", 0.03)),
            covariance_floor=float(config.get("covariance_floor", 1e-4)),
            kmeans_iterations=int(config.get("kmeans_iterations", 8)),
        )
        if family in {"ellipsoid", "ellipsoids"}:
            primitives = tuple(initialize_ellipsoids_from_points(points, init_config))
        else:
            primitives = tuple(initialize_gaussians_from_points(points, init_config))
    except Exception as exc:
        return CandidateResult(
            candidate_id=candidate_id,
            backend_name=backend_name,
            status="failed",
            errors=(str(exc),),
        )

    mesh_proxy = combine_primitive_meshes(primitives, resolution=20)
    root = request.candidate_artifact_root()
    primitive_path = None
    mesh_path = None
    artifacts = {}
    if root is not None:
        primitive_path = write_primitive_set(
            root / "primitives" / "gaussian-ellipsoid.json",
            primitives,
            metadata={
                "family": family,
                "point_meta": point_meta,
                "primitive_count": len(primitives),
            },
        )
        artifacts["primitive_json"] = primitive_path
        if bool(config.get("export_mesh_proxy", True)):
            mesh_path = write_obj(
                root / "mesh" / "gaussian-ellipsoid-proxy.obj",
                mesh_proxy,
                header=(f"candidate {candidate_id}", backend_name),
            )
            artifacts["mesh_obj"] = mesh_path

    elapsed = time.perf_counter() - start
    coverage = _coverage_score(points, primitives)
    metrics = CandidateMetrics(
        topology_score=0.65,
        editability_score=0.75,
        complexity_penalty=min(1.0, len(primitives) / 128.0),
        elapsed_s=elapsed,
        extras={
            "family": family,
            "primitive_count": len(primitives),
            "surface_points": point_meta,
            "coverage_score": coverage,
        },
    )
    return CandidateResult(
        candidate_id=candidate_id,
        backend_name=backend_name,
        status="success" if primitives else "skipped",
        primitive_path=primitive_path,
        mesh_path=mesh_path,
        metric_result=metrics,
        artifacts=artifacts,
        warnings=() if primitives else ("no proxy primitives were initialized",),
        payload=primitives,
    )


def _coverage_score(points: np.ndarray, primitives: tuple[object, ...]) -> float:
    if len(points) == 0 or not primitives:
        return 0.0
    sdf_rows = []
    for primitive in primitives:
        if hasattr(primitive, "sdf_batch"):
            sdf_rows.append(np.asarray(primitive.sdf_batch(points), dtype=float))
    if not sdf_rows:
        return 0.0
    nearest = np.min(np.abs(np.vstack(sdf_rows)), axis=0)
    scale = max(1e-6, float(np.percentile(np.linalg.norm(points, axis=1), 95)))
    return float(1.0 / (1.0 + np.mean(nearest) / scale))


def primitive_payload_summary(primitives: tuple[object, ...]) -> Mapping[str, object]:
    """Return a compact manifest-safe primitive summary."""
    return {
        "primitive_count": len(primitives),
        "types": [type(primitive).__name__ for primitive in primitives],
    }
