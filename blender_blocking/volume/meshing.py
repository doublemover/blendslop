"""Volume surface extraction and mesh extraction helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .contracts import MeshExtractionResult, VolumeGrid


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
    normalized_method = method.lower()
    if normalized_method in {"points", "point_cloud", "point_cloud_only"}:
        points = surface_points(grid, max_voxels=max_voxels)
        return MeshExtractionResult(
            status="skipped",
            method="point_cloud_only",
            vertices=points,
            faces=np.empty((0, 3), dtype=np.int64),
            message="point cloud diagnostic output; no mesh faces generated",
            metrics={"surface_points": int(len(points))},
        )
    if normalized_method == "dual_contouring":
        return MeshExtractionResult.unavailable(
            normalized_method,
            "dual contouring requires an optional meshing backend that is not available",
        )

    try:
        from skimage import measure
    except Exception as exc:
        if allow_point_cloud_fallback:
            points = surface_points(grid, max_voxels=max_voxels)
            return MeshExtractionResult(
                status="skipped",
                method=normalized_method,
                vertices=points,
                faces=np.empty((0, 3), dtype=np.int64),
                message=f"skimage unavailable; returned diagnostic points: {exc}",
                metrics={"surface_points": int(len(points))},
            )
        return MeshExtractionResult.unavailable(
            normalized_method, f"skimage.measure.marching_cubes unavailable: {exc}"
        )

    dense = grid.to_dense(max_voxels=max_voxels)
    field = _field_for_marching_cubes(dense, grid.value_type, grid.default_value)
    if level is None:
        level = 0.0 if grid.value_type == "signed_distance" else 0.5

    if field.size == 0 or not np.isfinite(field).all():
        return MeshExtractionResult.skipped(
            normalized_method, "field is empty or contains non-finite values"
        )

    min_value = float(np.min(field))
    max_value = float(np.max(field))
    if not (min_value <= level <= max_value) or min_value == max_value:
        return MeshExtractionResult.skipped(
            normalized_method,
            f"level {level} outside scalar range [{min_value}, {max_value}]",
        )

    lewiner_methods = {"marching_cubes", "marching_cubes_lewiner", "lewiner"}
    skimage_method = "lewiner" if normalized_method in lewiner_methods else "lorensen"

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
            normalized_method, f"marching cubes failed: {exc}"
        )

    world_vertices = grid.transform.index_to_world(vertices)
    return MeshExtractionResult(
        status="ok",
        method=normalized_method,
        vertices=world_vertices.astype(float, copy=False),
        faces=faces.astype(np.int64, copy=False),
        normals=normals.astype(float, copy=False),
        values=values,
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
