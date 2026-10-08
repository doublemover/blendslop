from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

try:
    from volume import (
        Bounds3D as VolumeBounds3D,
        ChunkKey,
        ChunkedVolumeGrid,
        DenseVolumeGrid,
        OpenVDBVolumeGrid,
        SparseHashVolumeGrid,
        VolumeChunkCache,
        chunk_cache_key,
        detect_openvdb,
        extract_surface_voxels,
        surface_points,
    )
except ImportError:  # pragma: no cover - package import path
    from ...volume import (
        Bounds3D as VolumeBounds3D,
        ChunkKey,
        ChunkedVolumeGrid,
        DenseVolumeGrid,
        OpenVDBVolumeGrid,
        SparseHashVolumeGrid,
        VolumeChunkCache,
        chunk_cache_key,
        detect_openvdb,
        extract_surface_voxels,
        surface_points,
    )

from ..types import Bounds3D, ReconstructionTarget

from .bounds import target_bounds
from .hull import visual_hull_grid_from_target


def target_surface_points(
    target: ReconstructionTarget,
    *,
    resolution: int = 48,
    max_points: int = 4096,
    chunk_size: Optional[int] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return deterministic surface points for primitive/research backends."""
    if "surface_points" in target.extras:
        points = np.asarray(target.extras["surface_points"], dtype=float)
        return _bounded(points, max_points), {
            "source": "target.extras.surface_points",
        }
    grid = visual_hull_grid_from_target(
        target,
        resolution=resolution,
        chunk_size=chunk_size,
        use_vectorized=True,
        backend="dense",
    )
    points = surface_points(grid, max_voxels=max(1, resolution**3))
    if len(points) == 0:
        points = _fallback_bounds_points(target_bounds(target), max_points)
        return points, {
            "source": "bounds_fallback",
            "warning": "visual hull produced no surface points",
        }
    return _bounded(points, max_points), {
        "source": "visual_hull_surface",
        "resolution": int(resolution),
        "stats": grid.stats().to_dict(),
    }


def target_occupied_points(
    target: ReconstructionTarget,
    *,
    resolution: int = 40,
    max_points: int = 4096,
    chunk_size: Optional[int] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return bounded occupied voxel centers for volume occupancy losses."""
    grid = visual_hull_grid_from_target(
        target,
        resolution=resolution,
        chunk_size=chunk_size,
        use_vectorized=True,
        backend="dense",
    )
    dense = grid.to_dense(max_voxels=max(1, resolution**3))
    indices = np.argwhere(dense.astype(bool, copy=False))
    if len(indices) == 0:
        return np.empty((0, 3), dtype=float), {"source": "empty_visual_hull"}
    points = grid.transform.index_to_world(indices)
    return _bounded(points, max_points), {
        "source": "visual_hull_occupied",
        "resolution": int(resolution),
        "occupied_voxels": int(len(indices)),
    }


def surface_mask_from_grid(
    grid: (
        DenseVolumeGrid
        | ChunkedVolumeGrid
        | SparseHashVolumeGrid
        | OpenVDBVolumeGrid
    ),
) -> np.ndarray:
    """Expose the vectorized surface mask for tests and diagnostics."""
    dense = grid.to_dense()
    return extract_surface_voxels(dense.astype(bool, copy=False), prefer_scipy=True)


def _bounded(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if max_points <= 0 or len(points) <= max_points:
        return points.copy()
    indices = np.linspace(0, len(points) - 1, int(max_points)).round().astype(np.int64)
    return points[indices]


def _fallback_bounds_points(bounds: Bounds3D, max_points: int) -> np.ndarray:
    count = max(8, int(max_points))
    side = max(2, int(round(count ** (1.0 / 3.0))))
    xs = np.linspace(bounds.min_x, bounds.max_x, side)
    ys = np.linspace(bounds.min_y, bounds.max_y, side)
    zs = np.linspace(bounds.min_z, bounds.max_z, side)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    shell = (
        (np.isclose(xx, bounds.min_x))
        | (np.isclose(xx, bounds.max_x))
        | (np.isclose(yy, bounds.min_y))
        | (np.isclose(yy, bounds.max_y))
        | (np.isclose(zz, bounds.min_z))
        | (np.isclose(zz, bounds.max_z))
    )
    points = np.stack([xx[shell], yy[shell], zz[shell]], axis=1)
    return _bounded(points, max_points)
