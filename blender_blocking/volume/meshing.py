"""Volume surface extraction and mesh extraction helpers."""

from __future__ import annotations

from typing import Any, Optional, Tuple

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
    selective = getattr(grid, "surface_indices", None)
    if callable(selective):
        return grid.transform.index_to_world(selective(max_voxels=max_voxels))
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
    recorder: Any = None,
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

    selective = getattr(grid, "extract_selective_mesh", None)
    if callable(selective) and canonical_method == "marching_cubes":
        if level not in (None, .5):
            return MeshExtractionResult.unavailable(
                canonical_method, "Boolean hierarchy extraction requires level 0.5",
                requested_method=requested_method, method_aliases=method_aliases)
        try:
            from dataclasses import replace
            skimage_method = "lorensen" if "lorensen" in normalized_method else "lewiner"
            result = selective(method=canonical_method, skimage_method=skimage_method,
                               recorder=recorder)
            # No global dense fallback disguised as sparse/adaptive execution.
            return replace(result, requested_method=requested_method,
                           method_aliases=method_aliases)
        except Exception as exc:
            return MeshExtractionResult.unavailable(
                canonical_method, "selective hierarchy extraction failed: " + str(exc),
                requested_method=requested_method, method_aliases=method_aliases)

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
    index_offset = np.zeros(3, dtype=float)
    if _should_pad_marching_field(field, grid.value_type):
        field = np.pad(field, 1, mode="constant", constant_values=0.0)
        index_offset[:] = -1.0
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

    # scikit-image documents face orientation using the left-hand convention.
    # Blendslop exports right-handed outward material boundaries. Occupancy is
    # high inside, whereas signed distance is negative inside: their winding
    # choices must therefore be opposite instead of sharing the default.
    gradient_direction = "descent" if grid.value_type == "signed_distance" else "ascent"
    try:
        try:
            vertices, faces, normals, values = measure.marching_cubes(
                field, level=level, spacing=(1.0,1.0,1.0), method=skimage_method,
                gradient_direction=gradient_direction, allow_degenerate=False)
        except TypeError:
            # Older supported versions may omit the method selector, but cannot
            # silently omit the geometric winding/degeneracy contract.
            vertices, faces, normals, values = measure.marching_cubes(
                field, level=level, spacing=(1.0,1.0,1.0),
                gradient_direction=gradient_direction, allow_degenerate=False)
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

    world_vertices = grid.transform.index_to_world(vertices + index_offset)
    # The returned field normals use high-to-low direction independently of
    # triangle order. Apply the field sign and inverse-transpose voxel scale.
    world_normals = np.asarray(normals,float) / grid.transform.voxel_size_array
    if grid.value_type == "signed_distance":
        world_normals = -world_normals
    lengths = np.linalg.norm(world_normals,axis=1)
    world_normals = np.divide(world_normals,lengths[:,None],out=np.zeros_like(world_normals),where=lengths[:,None]>0.)
    from blender_blocking.evaluation.cost_model import timed_call
    from blender_blocking.metrics.topology_receipt import ConnectivityTopologyReceipt
    topology_receipt = timed_call(
        recorder, "extraction_topology", ConnectivityTopologyReceipt.capture,
        world_vertices, faces)
    topology = topology_receipt.summary()
    return MeshExtractionResult(
        status="ok",
        method=canonical_method,
        requested_method=requested_method,
        method_aliases=method_aliases,
        vertices=world_vertices.astype(float, copy=False),
        faces=faces.astype(np.int64, copy=False),
        normals=world_normals,
        values=values,
        topology=topology,
        topology_receipt=topology_receipt,
        metrics={
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "gradient_direction": gradient_direction,
            "winding_contract": "right-handed outward material boundary",
            "normals_space": "world; field sign and inverse voxel scale applied",
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


def _should_pad_marching_field(field: np.ndarray, value_type: str) -> bool:
    """Pad occupancy-like fields so boundary/full volumes expose an isosurface."""
    if value_type == "signed_distance":
        return False
    return np.asarray(field).ndim == 3 and np.asarray(field).size > 0


def _mesh_topology_summary(
    vertices: np.ndarray, faces: np.ndarray
) -> dict[str, Any]:
    vertex_array = np.asarray(vertices)
    face_array = np.asarray(faces, dtype=np.int64)

    vertex_count = int(vertex_array.shape[0]) if vertex_array.size else 0
    face_count = int(len(face_array))

    if face_count == 0:
        return _empty_mesh_topology(vertex_count)

    from blender_blocking.metrics.topology_receipt import ConnectivityTopologyReceipt
    return ConnectivityTopologyReceipt.capture(vertex_array, face_array).summary()


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
