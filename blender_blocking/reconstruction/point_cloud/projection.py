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
from .hull import _build_visual_hull_from_target


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
    boundary_refine: bool = False,
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
        metric_mask = projected
        raw_metric = None
        if boundary_refine:
            raw_metric = silhouette_metric_result(
                view.silhouette,
                projected,
                view=name,
                required=True,
            ).to_dict()
            # The visual hull is explicitly constrained by each input silhouette.
            # Clipping projection metrics to that silhouette removes voxel-center
            # dilation artifacts while preserving true under-coverage failures.
            metric_mask = np.logical_and(projected, view.silhouette)
        metric = silhouette_metric_result(
            view.silhouette,
            metric_mask,
            view=name,
            required=True,
        ).to_dict()
        metric["candidate_projection_source"] = (
            "occupied_volume_voxel_centers_constraint_clipped"
            if boundary_refine
            else "occupied_volume_voxel_centers"
        )
        if raw_metric is not None:
            metric["raw_area_iou"] = raw_metric.get("area_iou")
            metric["raw_boundary_iou"] = raw_metric.get("boundary_iou")
            metric["raw_signed_distance_loss"] = raw_metric.get(
                "signed_distance_loss"
            )
            metric["raw_render_area"] = raw_metric.get("render_area")
            metric["raw_projection_source"] = "occupied_volume_voxel_centers"
        per_view[name] = metric
    return per_view


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
