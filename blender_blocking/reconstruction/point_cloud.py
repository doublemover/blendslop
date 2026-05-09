"""Point-cloud construction from silhouette reconstruction targets."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from volume import Bounds3D as VolumeBounds3D
from volume import DenseVolumeGrid, extract_surface_voxels, surface_points

from .types import Bounds3D, ReconstructionTarget


def default_bounds() -> Bounds3D:
    """Conservative default bounds when inputs lack scale metadata."""
    return Bounds3D(-1.0, 1.0, -1.0, 1.0, 0.0, 2.0)


def target_bounds(target: ReconstructionTarget) -> Bounds3D:
    """Return explicit target bounds or a stable fallback."""
    return target.bounds or default_bounds()


def volume_bounds_from_target(target: ReconstructionTarget) -> VolumeBounds3D:
    bounds = target_bounds(target)
    return VolumeBounds3D(
        bounds.min_x,
        bounds.max_x,
        bounds.min_y,
        bounds.max_y,
        bounds.min_z,
        bounds.max_z,
    )


def visual_hull_grid_from_target(
    target: ReconstructionTarget,
    *,
    resolution: int = 64,
    chunk_size: Optional[int] = None,
    use_vectorized: bool = True,
) -> DenseVolumeGrid:
    """Build a DenseVolumeGrid by intersecting target silhouette cones."""
    from integration.multi_view.visual_hull import MultiViewVisualHull

    bounds = target_bounds(target)
    hull = MultiViewVisualHull(
        resolution=int(resolution),
        bounds_min=np.asarray((bounds.min_x, bounds.min_y, bounds.min_z), dtype=float),
        bounds_max=np.asarray((bounds.max_x, bounds.max_y, bounds.max_z), dtype=float),
        chunk_size=chunk_size,
    )
    for constraint in target.constraints:
        mask = np.asarray(getattr(constraint.mask, "mask", constraint.mask)).astype(bool)
        if mask.ndim != 2:
            continue
        hull.add_view_from_silhouette(
            mask,
            angle=float(getattr(constraint.camera, "azimuth_deg", 0.0)),
            view_type="top" if constraint.view == "top" else "lateral",
        )
    if not hull.views:
        raise ValueError("target does not contain any 2D silhouette constraints")
    voxels = hull.reconstruct(
        verbose=False,
        use_vectorized=use_vectorized,
        chunk_size=chunk_size,
    )
    return DenseVolumeGrid(
        voxels.astype(bool, copy=False),
        volume_bounds_from_target(target),
        value_type="occupancy_bool",
        default_value=False,
        chunk_size=chunk_size or 32,
    )


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
        return _bounded(points, max_points), {"source": "target.extras.surface_points"}
    grid = visual_hull_grid_from_target(
        target,
        resolution=resolution,
        chunk_size=chunk_size,
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


def surface_mask_from_grid(grid: DenseVolumeGrid) -> np.ndarray:
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
