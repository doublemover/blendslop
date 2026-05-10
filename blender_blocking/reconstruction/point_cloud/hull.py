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

from .bounds import target_bounds, volume_bounds_from_target
from .cache import _visual_hull_cache_payload

_SUPPORTED_VISUAL_HULL_BACKENDS = {
    "dense",
    "chunked",
    "sparse",
    "sparse_hash",
    "openvdb",
}


def _auto_boundary_radius(
    mask: np.ndarray,
    resolution: int,
    configured: Optional[int],
) -> int:
    if configured is not None:
        return max(0, int(configured))
    height, width = mask.shape
    if resolution <= 0:
        return 1
    return max(1, int(round(min(height, width) / max(1, int(resolution)) * 0.35)))


def _dilate_rect(mask: np.ndarray, *, radius_x: int, radius_y: int) -> np.ndarray:
    if radius_x <= 0 and radius_y <= 0:
        return mask
    height, width = mask.shape
    padded = np.pad(
        mask,
        ((int(radius_y), int(radius_y)), (int(radius_x), int(radius_x))),
        mode="constant",
        constant_values=False,
    )
    result = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * int(radius_y) + 1):
        for dx in range(2 * int(radius_x) + 1):
            result |= padded[dy : dy + height, dx : dx + width]
    return result


def visual_hull_grid_from_target(
    target: ReconstructionTarget,
    *,
    resolution: int = 64,
    chunk_size: Optional[int] = None,
    use_vectorized: bool = True,
    backend: str = "dense",
    boundary_refine: bool = False,
    boundary_dilate_px: Optional[int] = None,
    cache_directory: Optional[str | Path] = None,
    cache_namespace: str = "visual_hull",
    cache_read: bool = True,
    cache_write: bool = True,
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
            boundary_refine=boundary_refine,
            boundary_dilate_px=boundary_dilate_px,
        )
    if requested_backend == "chunked":
        return visual_hull_chunked_grid_from_target(
            target,
            resolution=resolution,
            chunk_size=chunk_size,
            use_vectorized=use_vectorized,
            boundary_refine=boundary_refine,
            boundary_dilate_px=boundary_dilate_px,
            cache_directory=cache_directory,
            cache_namespace=cache_namespace,
            cache_read=cache_read,
            cache_write=cache_write,
        )
    if requested_backend == "sparse_hash":
        return visual_hull_sparse_hash_grid_from_target(
            target,
            resolution=resolution,
            chunk_size=chunk_size,
            use_vectorized=use_vectorized,
            boundary_refine=boundary_refine,
            boundary_dilate_px=boundary_dilate_px,
            cache_directory=cache_directory,
            cache_namespace=cache_namespace,
            cache_read=cache_read,
            cache_write=cache_write,
        )
    if requested_backend == "openvdb":
        return visual_hull_openvdb_grid_from_target(
            target,
            resolution=resolution,
            chunk_size=chunk_size,
            use_vectorized=use_vectorized,
            boundary_refine=boundary_refine,
            boundary_dilate_px=boundary_dilate_px,
            cache_directory=cache_directory,
            cache_namespace=cache_namespace,
            cache_read=cache_read,
            cache_write=cache_write,
        )
    raise ValueError(f"unsupported visual hull backend: {requested_backend!r}")


def visual_hull_dense_grid_from_target(
    target: ReconstructionTarget,
    *,
    resolution: int = 64,
    chunk_size: Optional[int] = None,
    use_vectorized: bool = True,
    boundary_refine: bool = False,
    boundary_dilate_px: Optional[int] = None,
) -> DenseVolumeGrid:
    """Build a DenseVolumeGrid by intersecting target silhouette cones."""
    hull = _build_visual_hull_from_target(
        target,
        resolution=resolution,
        chunk_size=chunk_size,
        boundary_refine=boundary_refine,
        boundary_dilate_px=boundary_dilate_px,
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
    boundary_refine: bool = False,
    boundary_dilate_px: Optional[int] = None,
    cache_directory: Optional[str | Path] = None,
    cache_namespace: str = "visual_hull",
    cache_read: bool = True,
    cache_write: bool = True,
) -> ChunkedVolumeGrid:
    """Build a ChunkedVolumeGrid by intersecting target silhouette cones."""
    return _visual_hull_chunked_grid_from_target(
        target,
        grid_type=ChunkedVolumeGrid,
        resolution=resolution,
        chunk_size=chunk_size,
        use_vectorized=use_vectorized,
        boundary_refine=boundary_refine,
        boundary_dilate_px=boundary_dilate_px,
        cache_directory=cache_directory,
        cache_namespace=cache_namespace,
        cache_read=cache_read,
        cache_write=cache_write,
    )


def visual_hull_sparse_hash_grid_from_target(
    target: ReconstructionTarget,
    *,
    resolution: int = 64,
    chunk_size: Optional[int] = None,
    use_vectorized: bool = True,
    boundary_refine: bool = False,
    boundary_dilate_px: Optional[int] = None,
    cache_directory: Optional[str | Path] = None,
    cache_namespace: str = "visual_hull",
    cache_read: bool = True,
    cache_write: bool = True,
) -> SparseHashVolumeGrid:
    """Build a SparseHashVolumeGrid by intersecting target silhouette cones."""
    return _visual_hull_chunked_grid_from_target(
        target,
        grid_type=SparseHashVolumeGrid,
        resolution=resolution,
        chunk_size=chunk_size,
        use_vectorized=use_vectorized,
        boundary_refine=boundary_refine,
        boundary_dilate_px=boundary_dilate_px,
        cache_directory=cache_directory,
        cache_namespace=cache_namespace,
        cache_read=cache_read,
        cache_write=cache_write,
    )


def visual_hull_openvdb_grid_from_target(
    target: ReconstructionTarget,
    *,
    resolution: int = 64,
    chunk_size: Optional[int] = None,
    use_vectorized: bool = True,
    boundary_refine: bool = False,
    boundary_dilate_px: Optional[int] = None,
    cache_directory: Optional[str | Path] = None,
    cache_namespace: str = "visual_hull",
    cache_read: bool = True,
    cache_write: bool = True,
) -> OpenVDBVolumeGrid:
    """Build an OpenVDB-labeled sparse interchange grid directly from views."""
    return _visual_hull_chunked_grid_from_target(
        target,
        grid_type=OpenVDBVolumeGrid,
        resolution=resolution,
        chunk_size=chunk_size,
        use_vectorized=use_vectorized,
        boundary_refine=boundary_refine,
        boundary_dilate_px=boundary_dilate_px,
        openvdb_status=detect_openvdb(),
        cache_directory=cache_directory,
        cache_namespace=cache_namespace,
        cache_read=cache_read,
        cache_write=cache_write,
    )


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
    boundary_refine: bool = False,
    boundary_dilate_px: Optional[int] = None,
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
        original_mask = mask
        if boundary_refine:
            radius = _auto_boundary_radius(mask, resolution, boundary_dilate_px)
            mask = _dilate_rect(mask, radius_x=radius, radius_y=radius)
        image_bounds = None
        bbox = getattr(constraint, "bbox", None)
        if bbox is not None and hasattr(bbox, "to_xyxy"):
            image_bounds = bbox.to_xyxy()
        hull.add_view_from_silhouette(
            mask,
            angle=float(getattr(constraint.camera, "azimuth_deg", 0.0)),
            view_type="top" if constraint.view == "top" else "lateral",
            image_bounds=image_bounds,
        )
        hull.views[-1].source_mask_area = int(original_mask.sum())
        hull.views[-1].refined_mask_area = int(mask.sum())
        hull.views[-1].boundary_refine = bool(boundary_refine)
        hull.views[-1].boundary_dilate_px = (
            _auto_boundary_radius(original_mask, resolution, boundary_dilate_px)
            if boundary_refine
            else 0
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
    boundary_refine: bool = False,
    boundary_dilate_px: Optional[int] = None,
    openvdb_status: Any = None,
    cache_directory: Optional[str | Path] = None,
    cache_namespace: str = "visual_hull",
    cache_read: bool = True,
    cache_write: bool = True,
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
        boundary_refine=boundary_refine,
        boundary_dilate_px=boundary_dilate_px,
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
    cache = (
        VolumeChunkCache(
            cache_directory,
            namespace=cache_namespace,
            read=cache_read,
            write=cache_write,
        )
        if cache_directory is not None
        else None
    )
    cache_base = _visual_hull_cache_payload(
        hull,
        target=target,
        resolution=resolution_i,
        slab_size=slab_size,
        boundary_refine=boundary_refine,
        boundary_dilate_px=boundary_dilate_px,
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
                cache_key = None
                chunk = None
                key = ChunkKey(cx // slab_size, cy // slab_size, z_start // slab_size)
                if cache is not None:
                    cache_key = chunk_cache_key(
                        cache.namespace,
                        {
                            **cache_base,
                            "chunk_key": key.to_tuple(),
                            "origin_index": (cx, cy, z_start),
                            "valid_shape": (
                                x_end - cx,
                                y_end - cy,
                                z_end - z_start,
                            ),
                        },
                    )
                    chunk = cache.load(
                        cache_key,
                        expected_shape=(slab_size, slab_size, slab_size),
                        expected_dtype=bool,
                    )
                if chunk is None:
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

                    chunk = np.full(
                        (slab_size, slab_size, slab_size),
                        grid.default_value,
                        dtype=bool,
                    )
                    chunk[: x_end - cx, : y_end - cy, : z_end - z_start] = chunk_slice
                    if cache is not None and cache_key is not None:
                        cache.store(
                            cache_key,
                            chunk,
                            metadata={
                                "origin_index": [cx, cy, z_start],
                                "valid_shape": [x_end - cx, y_end - cy, z_end - z_start],
                                "resolution": resolution_i,
                                "chunk_size": slab_size,
                            },
                        )

                if sparse_hash_mode and not chunk[: x_end - cx, : y_end - cy, : z_end - z_start].any():
                    continue

                grid.set_chunk(key, chunk)

    if cache is not None:
        setattr(grid, "chunk_cache_status", cache.stats())
    return grid
