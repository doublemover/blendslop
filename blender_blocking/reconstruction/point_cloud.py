"""Point-cloud construction from silhouette reconstruction targets."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from volume import (
    Bounds3D as VolumeBounds3D,
    ChunkKey,
    ChunkedVolumeGrid,
    DenseVolumeGrid,
    OpenVDBVolumeGrid,
    SparseHashVolumeGrid,
    detect_openvdb,
    extract_surface_voxels,
    surface_points,
)

from .types import Bounds3D, ReconstructionTarget


_SUPPORTED_VISUAL_HULL_BACKENDS = {
    "dense",
    "chunked",
    "sparse",
    "sparse_hash",
    "openvdb",
}


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
    backend: str = "dense",
) -> (
    DenseVolumeGrid
    | ChunkedVolumeGrid
    | SparseHashVolumeGrid
    | OpenVDBVolumeGrid
):
    """Build a visual-hull volume grid by intersecting target silhouette cones."""
    requested_backend = _normalize_visual_hull_backend(backend)
    if requested_backend == "dense":
        return visual_hull_dense_grid_from_target(
            target,
            resolution=resolution,
            chunk_size=chunk_size,
            use_vectorized=use_vectorized,
        )
    if requested_backend == "chunked":
        return visual_hull_chunked_grid_from_target(
            target,
            resolution=resolution,
            chunk_size=chunk_size,
            use_vectorized=use_vectorized,
        )
    if requested_backend == "sparse_hash":
        return visual_hull_sparse_hash_grid_from_target(
            target,
            resolution=resolution,
            chunk_size=chunk_size,
            use_vectorized=use_vectorized,
        )
    if requested_backend == "openvdb":
        return visual_hull_openvdb_grid_from_target(
            target,
            resolution=resolution,
            chunk_size=chunk_size,
            use_vectorized=use_vectorized,
        )
    raise ValueError(f"unsupported visual hull backend: {requested_backend!r}")


def visual_hull_dense_grid_from_target(
    target: ReconstructionTarget,
    *,
    resolution: int = 64,
    chunk_size: Optional[int] = None,
    use_vectorized: bool = True,
) -> DenseVolumeGrid:
    """Build a DenseVolumeGrid by intersecting target silhouette cones."""
    hull = _build_visual_hull_from_target(
        target,
        resolution=resolution,
        chunk_size=chunk_size,
    )
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


def visual_hull_chunked_grid_from_target(
    target: ReconstructionTarget,
    *,
    resolution: int = 64,
    chunk_size: Optional[int] = None,
    use_vectorized: bool = True,
) -> ChunkedVolumeGrid:
    """Build a ChunkedVolumeGrid by intersecting target silhouette cones."""
    return _visual_hull_chunked_grid_from_target(
        target,
        grid_type=ChunkedVolumeGrid,
        resolution=resolution,
        chunk_size=chunk_size,
        use_vectorized=use_vectorized,
    )


def visual_hull_sparse_hash_grid_from_target(
    target: ReconstructionTarget,
    *,
    resolution: int = 64,
    chunk_size: Optional[int] = None,
    use_vectorized: bool = True,
) -> SparseHashVolumeGrid:
    """Build a SparseHashVolumeGrid by intersecting target silhouette cones."""
    return _visual_hull_chunked_grid_from_target(
        target,
        grid_type=SparseHashVolumeGrid,
        resolution=resolution,
        chunk_size=chunk_size,
        use_vectorized=use_vectorized,
    )


def visual_hull_openvdb_grid_from_target(
    target: ReconstructionTarget,
    *,
    resolution: int = 64,
    chunk_size: Optional[int] = None,
    use_vectorized: bool = True,
) -> OpenVDBVolumeGrid:
    """Build an OpenVDB-labeled sparse interchange grid directly from views."""
    return _visual_hull_chunked_grid_from_target(
        target,
        grid_type=OpenVDBVolumeGrid,
        resolution=resolution,
        chunk_size=chunk_size,
        use_vectorized=use_vectorized,
        openvdb_status=detect_openvdb(),
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


def _normalize_visual_hull_backend(backend: str) -> str:
    value = str(backend).strip().lower()
    if value not in _SUPPORTED_VISUAL_HULL_BACKENDS:
        raise ValueError(f"backend must be one of {_SUPPORTED_VISUAL_HULL_BACKENDS}")
    if value == "sparse":
        return "sparse_hash"
    return value


def _build_visual_hull_from_target(
    target: ReconstructionTarget,
    *,
    resolution: int,
    chunk_size: Optional[int] = None,
):
    from integration.multi_view.visual_hull import MultiViewVisualHull

    bounds = target_bounds(target)
    hull = MultiViewVisualHull(
        resolution=int(resolution),
        bounds_min=np.asarray(
            (bounds.min_x, bounds.min_y, bounds.min_z), dtype=float
        ),
        bounds_max=np.asarray(
            (bounds.max_x, bounds.max_y, bounds.max_z), dtype=float
        ),
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
    return hull


def _visual_hull_chunked_grid_from_target(
    target: ReconstructionTarget,
    *,
    grid_type: type[ChunkedVolumeGrid],
    resolution: int = 64,
    chunk_size: Optional[int] = None,
    use_vectorized: bool = True,
    openvdb_status: Any = None,
) -> ChunkedVolumeGrid:
    # Non-vectorized behavior remains available for compatibility, but the direct
    # chunk path intentionally preserves projection semantics from the existing
    # vectorized kernel.
    # Keep the parameter for API compatibility, but avoid dense-first reconstruction.
    _ = bool(use_vectorized)

    sparse_hash_mode = issubclass(grid_type, SparseHashVolumeGrid)

    hull = _build_visual_hull_from_target(
        target,
        resolution=resolution,
        chunk_size=chunk_size,
    )
    bounds_min = hull.bounds_min
    bounds_max = hull.bounds_max
    center = (bounds_min + bounds_max) / 2.0

    resolution_i = int(resolution)
    slab_size = int(chunk_size) if chunk_size is not None else 32
    if slab_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if slab_size > resolution_i:
        slab_size = resolution_i

    x_coords = np.linspace(bounds_min[0], bounds_max[0], resolution_i)
    y_coords = np.linspace(bounds_min[1], bounds_max[1], resolution_i)
    z_coords = np.linspace(bounds_min[2], bounds_max[2], resolution_i)

    x_indices = range(0, resolution_i, slab_size)
    y_indices = range(0, resolution_i, slab_size)

    grid_kwargs: dict[str, Any] = {}
    if issubclass(grid_type, OpenVDBVolumeGrid):
        grid_kwargs["openvdb_status"] = openvdb_status

    grid = grid_type(
        chunks={},
        shape=(resolution_i, resolution_i, resolution_i),
        bounds=volume_bounds_from_target(target),
        value_type="occupancy_bool",
        dtype=bool,
        default_value=False,
        chunk_size=slab_size,
        **grid_kwargs,
    )

    for z_start in range(0, resolution_i, slab_size):
        z_end = min(resolution_i, z_start + slab_size)
        zz = z_coords[z_start:z_end][None, None, :]

        for cx in x_indices:
            x_end = min(resolution_i, cx + slab_size)
            xx = x_coords[cx:x_end][:, None, None]
            for cy in y_indices:
                y_end = min(resolution_i, cy + slab_size)
                yy = y_coords[cy:y_end][None, :, None]
                chunk_slice = np.ones(
                    (x_end - cx, y_end - cy, z_end - z_start),
                    dtype=bool,
                )
                for view in hull.views:
                    view_mask = hull._project_view_mask(
                        view=view,
                        xx=xx,
                        yy=yy,
                        zz=zz,
                        bounds_min=bounds_min,
                        bounds_max=bounds_max,
                        center=center,
                    )
                    chunk_slice &= view_mask
                    if not chunk_slice.any():
                        break

                if sparse_hash_mode and not chunk_slice.any():
                    continue

                chunk = np.full(
                    (slab_size, slab_size, slab_size),
                    grid.default_value,
                    dtype=bool,
                )
                chunk[: x_end - cx, : y_end - cy, : z_end - z_start] = chunk_slice
                grid.set_chunk(
                    ChunkKey(cx // slab_size, cy // slab_size, z_start // slab_size),
                    chunk,
                )

    return grid


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
