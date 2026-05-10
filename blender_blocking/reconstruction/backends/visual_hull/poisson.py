from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from blender_blocking.utils.optional_deps import dependency_report, probe_dependency
except Exception:  # pragma: no cover - script-style imports
    from utils.optional_deps import dependency_report, probe_dependency


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
