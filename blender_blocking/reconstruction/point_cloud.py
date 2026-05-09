"""Point-cloud construction from silhouette reconstruction targets."""

from __future__ import annotations

import math
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


def visual_hull_projection_metrics_from_target(
    target: ReconstructionTarget,
    grid: (
        DenseVolumeGrid
        | ChunkedVolumeGrid
        | SparseHashVolumeGrid
        | OpenVDBVolumeGrid
    ),
    *,
    max_metric_voxels: int = 4_000_000,
) -> dict[str, dict[str, Any]]:
    """Compare occupied-volume projections against the target silhouettes."""
    from metrics.silhouette import silhouette_metric_result

    shape = tuple(int(v) for v in getattr(grid, "shape", ()))
    if len(shape) != 3:
        return {}
    total_voxels = int(np.prod(np.asarray(shape, dtype=np.int64)))
    if total_voxels > int(max_metric_voxels):
        return {
            "_skipped": {
                "reason": "projection metric voxel budget exceeded",
                "total_voxels": total_voxels,
                "max_metric_voxels": int(max_metric_voxels),
            }
        }

    dense = grid.to_dense(max_voxels=max(1, total_voxels)).astype(bool, copy=False)
    active_indices = np.argwhere(dense)
    hull = _build_visual_hull_from_target(
        target,
        resolution=shape[0],
        chunk_size=getattr(grid, "chunk_size", None),
    )
    view_names = _target_view_names_with_masks(target)
    if len(view_names) != len(hull.views):
        view_names = [f"view_{idx}" for idx in range(len(hull.views))]

    if len(active_indices) == 0:
        points = np.empty((0, 3), dtype=float)
    else:
        points = grid.transform.index_to_world(active_indices)

    per_view: dict[str, dict[str, Any]] = {}
    for name, view in zip(view_names, hull.views):
        projected = _project_points_to_view_mask(
            view,
            points,
            bounds_min=hull.bounds_min,
            bounds_max=hull.bounds_max,
            grid_shape=shape,
        )
        metric = silhouette_metric_result(
            view.silhouette,
            projected,
            view=name,
            required=True,
        ).to_dict()
        metric["candidate_projection_source"] = "occupied_volume_voxel_centers"
        per_view[name] = metric
    return per_view


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
    if not hull.views:
        raise ValueError("target does not contain any 2D silhouette constraints")
    return hull


def _target_view_names_with_masks(target: ReconstructionTarget) -> list[str]:
    names: list[str] = []
    for constraint in target.constraints:
        mask = np.asarray(getattr(constraint.mask, "mask", constraint.mask))
        if mask.ndim == 2:
            names.append(str(constraint.view))
    return names


def _project_points_to_view_mask(
    view: Any,
    points: np.ndarray,
    *,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    mask = np.zeros(view.silhouette.shape, dtype=bool)
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return mask

    center = (bounds_min + bounds_max) / 2.0
    if view.view_type == "top":
        world_x = points[:, 0]
        world_y = points[:, 1]
        y_axis = 1
    else:
        angle_rad = math.radians(float(view.angle))
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        local_x = points[:, 0] - center[0]
        local_y = points[:, 1] - center[1]
        world_x = (local_x * cos_a - local_y * sin_a) + center[0]
        world_y = points[:, 2]
        y_axis = 2

    x_min, x_max = _projected_horizontal_bounds(view, bounds_min, bounds_max)
    x_range = max(float(x_max - x_min), 1e-12)
    y_range = max(float(bounds_max[y_axis] - bounds_min[y_axis]), 1e-12)
    x_norm = np.clip((world_x - x_min) / x_range, 0.0, 1.0)
    y_norm = np.clip((world_y - bounds_min[y_axis]) / y_range, 0.0, 1.0)

    x0, y0, x1, y1 = view.image_bounds
    u_span = max(1.0, (x1 - x0) - 1.0)
    v_span = max(1.0, (y1 - y0) - 1.0)
    u = np.floor(x0 + x_norm * u_span + 1e-9).astype(np.int64)
    v = np.floor(y0 + (1.0 - y_norm) * v_span + 1e-9).astype(np.int64)
    valid = (
        np.isfinite(x_norm)
        & np.isfinite(y_norm)
        & (x_norm >= 0.0)
        & (x_norm <= 1.0)
        & (y_norm >= 0.0)
        & (y_norm <= 1.0)
        & (u >= 0)
        & (u < view.width)
        & (v >= 0)
        & (v < view.height)
    )
    if np.any(valid):
        mask[v[valid], u[valid]] = True

    radius_x, radius_y = _projection_mask_radius(view, grid_shape, y_axis)
    return _dilate_rect(mask, radius_x=radius_x, radius_y=radius_y)


def _projected_horizontal_bounds(
    view: Any,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> tuple[float, float]:
    if getattr(view, "view_type", "") == "top":
        return float(bounds_min[0]), float(bounds_max[0])
    if hasattr(view, "projected_horizontal_bounds"):
        return view.projected_horizontal_bounds(bounds_min, bounds_max)

    center = (bounds_min + bounds_max) / 2.0
    angle_rad = math.radians(float(getattr(view, "angle", 0.0)))
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    corners = (
        (bounds_min[0], bounds_min[1]),
        (bounds_min[0], bounds_max[1]),
        (bounds_max[0], bounds_min[1]),
        (bounds_max[0], bounds_max[1]),
    )
    projected = []
    for x, y in corners:
        local_x = float(x - center[0])
        local_y = float(y - center[1])
        projected.append(local_x * cos_a - local_y * sin_a + float(center[0]))
    return min(projected), max(projected)


def _projection_mask_radius(
    view: Any,
    grid_shape: tuple[int, int, int],
    y_axis: int,
) -> tuple[int, int]:
    x0, y0, x1, y1 = view.image_bounds
    x_samples = max(1, int(grid_shape[0]))
    y_samples = max(1, int(grid_shape[y_axis]))
    radius_x = int(math.ceil(((x1 - x0) / x_samples) * 0.55))
    radius_y = int(math.ceil(((y1 - y0) / y_samples) * 0.55))
    return max(0, radius_x), max(0, radius_y)


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
