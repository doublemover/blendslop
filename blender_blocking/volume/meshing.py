"""Volume surface extraction and mesh extraction helpers."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable, Optional, Tuple

import numpy as np

from .contracts import (
    MeshExtractionResult,
    VolumeGrid,
    normalize_mesh_extraction_method,
)


_MESH_METHOD_LEWINER_ALIASES = {
    "marching_cubes",
    "lewiner",
    "marching_cubes_lewiner",
}


def extract_surface_voxels(
    occupied: np.ndarray, *, prefer_scipy: bool = True
) -> np.ndarray:
    """Return occupied voxels with at least one empty 6-connected neighbor."""
    occupied = np.asarray(occupied, dtype=bool)
    if occupied.ndim != 3:
        raise ValueError("occupied must be a 3D array")
    if not occupied.any():
        return np.zeros_like(occupied, dtype=bool)

    if prefer_scipy:
        try:
            from scipy import ndimage

            structure = ndimage.generate_binary_structure(3, 1)
            eroded = ndimage.binary_erosion(
                occupied, structure=structure, border_value=0
            )
            return np.logical_and(occupied, np.logical_not(eroded))
        except Exception:
            pass

    padded = np.pad(occupied, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    all_neighbors = (
        padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return center & ~all_neighbors


def surface_points(grid: VolumeGrid, *, max_voxels: Optional[int] = None) -> np.ndarray:
    dense = grid.to_dense(max_voxels=max_voxels)
    occupied = _occupancy_for_surface(dense, grid.value_type, grid.default_value)
    surface = extract_surface_voxels(occupied, prefer_scipy=False)
    indices = np.argwhere(surface)
    if len(indices) == 0:
        return np.empty((0, 3), dtype=float)
    return grid.transform.index_to_world(indices)


def extract_mesh(
    grid: VolumeGrid,
    *,
    method: str = "marching_cubes_lewiner",
    level: Optional[float] = None,
    max_voxels: Optional[int] = None,
    allow_point_cloud_fallback: bool = False,
) -> MeshExtractionResult:
    """Extract a mesh from a volume grid when optional dependencies are available."""
    requested_method = method
    normalized_method = str(requested_method).strip().lower().replace("-", "_").replace(
        " ", "_"
    )
    canonical_method = normalize_mesh_extraction_method(normalized_method)
    method_aliases: Tuple[str, ...] = (
        (normalized_method,) if canonical_method != normalized_method else ()
    )

    if canonical_method not in {"marching_cubes", "dual_contouring", "points"}:
        method_aliases = (
            method_aliases + (canonical_method,)
            if canonical_method not in method_aliases
            else method_aliases
        )
        canonical_method = "marching_cubes"

    if canonical_method == "points":
        points = surface_points(grid, max_voxels=max_voxels)
        return MeshExtractionResult(
            status="skipped",
            method="points",
            requested_method=requested_method,
            method_aliases=method_aliases,
            vertices=points,
            faces=np.empty((0, 3), dtype=np.int64),
            topology=_mesh_topology_summary(points, np.empty((0, 3), dtype=np.int64)),
            message="point cloud diagnostic output; no mesh faces generated",
            metrics={"surface_points": int(len(points))},
        )

    if canonical_method == "dual_contouring":
        return MeshExtractionResult.unavailable(
            canonical_method,
            "dual contouring requires an optional meshing backend that is not available",
            requested_method=requested_method,
            method_aliases=method_aliases,
            topology=_mesh_topology_summary(
                np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
            ),
        )

    try:
        from skimage import measure
    except Exception as exc:
        points = surface_points(grid, max_voxels=max_voxels)
        if allow_point_cloud_fallback:
            return MeshExtractionResult(
                status="skipped",
                method=canonical_method,
                requested_method=requested_method,
                method_aliases=method_aliases,
                vertices=points,
                faces=np.empty((0, 3), dtype=np.int64),
                topology=_mesh_topology_summary(
                    points, np.empty((0, 3), dtype=np.int64)
                ),
                message=f"skimage unavailable; returned diagnostic points: {exc}",
                metrics={"surface_points": int(len(points))},
            )
        return MeshExtractionResult.unavailable(
            canonical_method,
            f"skimage.measure.marching_cubes unavailable: {exc}",
            requested_method=requested_method,
            method_aliases=method_aliases,
            topology=_mesh_topology_summary(
                np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
            ),
        )

    dense = grid.to_dense(max_voxels=max_voxels)
    field = _field_for_marching_cubes(dense, grid.value_type, grid.default_value)
    if level is None:
        level = 0.0 if grid.value_type == "signed_distance" else 0.5

    if field.size == 0 or not np.isfinite(field).all():
        return MeshExtractionResult.skipped(
            canonical_method,
            "field is empty or contains non-finite values",
            requested_method=requested_method,
            method_aliases=method_aliases,
            topology=_mesh_topology_summary(
                np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
            ),
        )

    min_value = float(np.min(field))
    max_value = float(np.max(field))
    if not (min_value <= level <= max_value) or min_value == max_value:
        return MeshExtractionResult.skipped(
            canonical_method,
            f"level {level} outside scalar range [{min_value}, {max_value}]",
            requested_method=requested_method,
            method_aliases=method_aliases,
            topology=_mesh_topology_summary(
                np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
            ),
        )

    skimage_method = (
        "lewiner"
        if canonical_method == "marching_cubes"
        and normalized_method in _MESH_METHOD_LEWINER_ALIASES
        else "lorensen"
    )

    try:
        vertices, faces, normals, values = measure.marching_cubes(
            field,
            level=level,
            spacing=(1.0, 1.0, 1.0),
            method=skimage_method,
        )
    except TypeError:
        vertices, faces, normals, values = measure.marching_cubes(
            field,
            level=level,
            spacing=(1.0, 1.0, 1.0),
        )
    except Exception as exc:
        return MeshExtractionResult.unavailable(
            canonical_method,
            f"marching cubes failed: {exc}",
            requested_method=requested_method,
            method_aliases=method_aliases,
            topology=_mesh_topology_summary(
                np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
            ),
        )

    world_vertices = grid.transform.index_to_world(vertices)
    topology = _mesh_topology_summary(world_vertices.astype(float), faces.astype(np.int64))
    return MeshExtractionResult(
        status="ok",
        method=canonical_method,
        requested_method=requested_method,
        method_aliases=method_aliases,
        vertices=world_vertices.astype(float, copy=False),
        faces=faces.astype(np.int64, copy=False),
        normals=normals.astype(float, copy=False),
        values=values,
        topology=topology,
        metrics={
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "level": float(level),
            "field_min": min_value,
            "field_max": max_value,
        },
    )


def _occupancy_for_surface(
    dense: np.ndarray, value_type: str, default_value: object
) -> np.ndarray:
    if value_type == "signed_distance":
        return np.asarray(dense) <= 0.0
    if np.asarray(dense).dtype == np.dtype(bool):
        return np.asarray(dense, dtype=bool)
    if value_type in {"occupancy_prob", "confidence", "view_agreement"}:
        return np.asarray(dense, dtype=float) > 0.5
    return np.asarray(dense) != default_value


def _field_for_marching_cubes(
    dense: np.ndarray, value_type: str, default_value: object
) -> np.ndarray:
    if value_type == "signed_distance":
        return np.asarray(dense, dtype=np.float32)
    if np.asarray(dense).dtype == np.dtype(bool):
        return np.asarray(dense, dtype=np.float32)
    if value_type in {"occupancy_prob", "confidence", "view_agreement"}:
        return np.asarray(dense, dtype=np.float32)
    return (np.asarray(dense) != default_value).astype(np.float32)


def _mesh_topology_summary(
    vertices: np.ndarray, faces: np.ndarray
) -> dict[str, Any]:
    vertex_array = np.asarray(vertices)
    face_array = np.asarray(faces, dtype=np.int64)

    vertex_count = int(vertex_array.shape[0]) if vertex_array.size else 0
    face_count = int(len(face_array))

    if face_count == 0:
        return _empty_mesh_topology(vertex_count)

    edge_to_faces = defaultdict(list)
    used_vertices: set[int] = set()
    degenerate_faces = 0
    for face_index, face in enumerate(face_array):
        if len(face) < 3:
            degenerate_faces += 1
            continue
        if len(set(int(v) for v in face)) < 3:
            degenerate_faces += 1
            continue
        if any(int(v) < 0 or int(v) >= vertex_count for v in face):
            degenerate_faces += 1
            continue
        used_vertices.update(int(v) for v in face)
        for index, start in enumerate(face):
            end = int(face[(index + 1) % len(face)])
            if int(start) == end:
                continue
            edge = (int(start), end) if int(start) < end else (end, int(start))
            edge_to_faces[edge].append(face_index)

    edge_owners = (len(owners) for owners in edge_to_faces.values())
    boundary_edges = sum(1 for owners in edge_owners if owners == 1)
    non_manifold_edges = sum(1 for owners in edge_to_faces.values() if len(owners) > 2)
    loose_vertices = vertex_count - len(used_vertices)
    connected_components = _connected_components(used_vertices, edge_to_faces.keys())

    watertight = (
        boundary_edges == 0
        and non_manifold_edges == 0
        and degenerate_faces == 0
        and face_count > 0
    )
    topology_score = _topology_score(
        boundary_edges=boundary_edges,
        non_manifold_edges=non_manifold_edges,
        degenerate_faces=degenerate_faces,
        loose_vertices=loose_vertices,
        connected_components=connected_components,
    )
    return {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "edge_count": len(edge_to_faces),
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "degenerate_faces": degenerate_faces,
        "loose_vertices": loose_vertices,
        "connected_components": connected_components,
        "euler_characteristic": vertex_count - len(edge_to_faces) + face_count,
        "watertight": watertight,
        "topology_score": topology_score,
        "topology_style": "mesh",
        "has_faces": True,
    }


def _empty_mesh_topology(vertex_count: int) -> dict[str, Any]:
    return {
        "vertex_count": vertex_count,
        "face_count": 0,
        "edge_count": 0,
        "boundary_edges": 0,
        "non_manifold_edges": 0,
        "degenerate_faces": 0,
        "loose_vertices": vertex_count,
        "connected_components": 0,
        "euler_characteristic": vertex_count,
        "watertight": False,
        "topology_score": 0.0,
        "topology_style": "none",
        "has_faces": False,
    }


def _topology_score(
    *,
    boundary_edges: int,
    non_manifold_edges: int,
    degenerate_faces: int,
    loose_vertices: int,
    connected_components: int,
) -> float:
    score = 1.0
    score -= min(0.35, non_manifold_edges * 0.02)
    score -= min(0.25, boundary_edges * 0.01)
    score -= min(0.20, degenerate_faces * 0.03)
    score -= min(0.10, loose_vertices * 0.01)
    score -= min(0.20, max(0, connected_components - 1) * 0.05)
    return max(0.0, score)


def _connected_components(
    used_vertices: set[int], edges: Iterable[tuple[int, int]]
) -> int:
    if not used_vertices:
        return 0

    graph: dict[int, set[int]] = {vertex: set() for vertex in used_vertices}
    for start, end in edges:
        if start in graph and end in graph:
            graph[start].add(end)
            graph[end].add(start)

    remaining = set(graph)
    components = 0
    while remaining:
        components += 1
        queue: deque[int] = deque([remaining.pop()])
        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)

    return components
