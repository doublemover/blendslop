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

from .hull import _build_visual_hull_from_target
from .projection import _target_view_names_with_masks


def visual_hull_view_diagnostics_from_target(
    target: ReconstructionTarget,
    grid: (
        DenseVolumeGrid
        | ChunkedVolumeGrid
        | SparseHashVolumeGrid
        | OpenVDBVolumeGrid
    ),
    *,
    per_view_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    boundary_refine: bool = False,
    boundary_dilate_px: Optional[int] = None,
) -> dict[str, Any]:
    """Return view-level transform and projection diagnostics for visual hulls."""
    shape = tuple(int(v) for v in getattr(grid, "shape", ()))
    resolution = shape[0] if len(shape) == 3 else 0
    hull = _build_visual_hull_from_target(
        target,
        resolution=max(1, resolution),
        chunk_size=getattr(grid, "chunk_size", None),
        boundary_refine=boundary_refine,
        boundary_dilate_px=boundary_dilate_px,
    )
    view_names = _target_view_names_with_masks(target)
    if len(view_names) != len(hull.views):
        view_names = [f"view_{idx}" for idx in range(len(hull.views))]
    metric_map = per_view_metrics or {}
    diagnostics: dict[str, Any] = {
        "view_count": len(hull.views),
        "required_view_count": sum(bool(metric_map.get(name, {}).get("required", True)) for name in view_names),
        "views": {},
    }
    ious: list[float] = []
    failed: list[str] = []
    top_like_failures: list[str] = []
    centroid_deltas: list[float] = []
    for name, view in zip(view_names, hull.views):
        payload = dict(metric_map.get(name, {}))
        area_iou = _optional_float(payload.get("area_iou"))
        boundary_iou = _optional_float(payload.get("boundary_iou"))
        signed_distance_loss = _optional_float(payload.get("signed_distance_loss"))
        if area_iou is not None:
            ious.append(area_iou)
        if area_iou is not None and area_iou < 0.35:
            failed.append(name)
            if name == "top" or getattr(view, "view_type", "") == "top":
                top_like_failures.append(name)
        source_bbox = _mask_bbox(view.silhouette)
        source_centroid = _mask_centroid(view.silhouette)
        centroid_delta = None
        if "reference_centroid" in payload and "candidate_centroid" in payload:
            try:
                ref = np.asarray(payload["reference_centroid"], dtype=float)
                cand = np.asarray(payload["candidate_centroid"], dtype=float)
                centroid_delta = float(np.linalg.norm(ref - cand))
                centroid_deltas.append(centroid_delta)
            except Exception:
                centroid_delta = None
        diagnostics["views"][name] = {
            "view_type": getattr(view, "view_type", ""),
            "angle_deg": float(getattr(view, "angle", 0.0)),
            "mask_shape": [int(view.height), int(view.width)],
            "image_bounds": [float(v) for v in getattr(view, "image_bounds", ())],
            "source_mask_area": int(getattr(view, "source_mask_area", view.silhouette.sum())),
            "refined_mask_area": int(getattr(view, "refined_mask_area", view.silhouette.sum())),
            "boundary_refine": bool(getattr(view, "boundary_refine", False)),
            "boundary_dilate_px": int(getattr(view, "boundary_dilate_px", 0)),
            "source_bbox": source_bbox,
            "source_centroid": source_centroid,
            "area_iou": area_iou,
            "boundary_iou": boundary_iou,
            "signed_distance_loss": signed_distance_loss,
            "raw_area_iou": _optional_float(payload.get("raw_area_iou")),
            "raw_boundary_iou": _optional_float(payload.get("raw_boundary_iou")),
            "raw_signed_distance_loss": _optional_float(
                payload.get("raw_signed_distance_loss")
            ),
            "raw_render_area": _optional_float(payload.get("raw_render_area")),
            "projection_source": str(payload.get("candidate_projection_source", "")),
            "centroid_delta_px": centroid_delta,
            "passed": bool(payload.get("passed", payload.get("pass", False))),
            "reason": str(payload.get("reason", "")),
        }
    min_iou = min(ious) if ious else None
    diagnostics["min_area_iou"] = min_iou
    diagnostics["average_iou"] = (sum(ious) / len(ious)) if ious else None
    diagnostics["failed_views"] = failed
    diagnostics["top_like_failures"] = top_like_failures
    diagnostics["mean_centroid_delta_px"] = (
        sum(centroid_deltas) / len(centroid_deltas) if centroid_deltas else None
    )
    diagnostics["axis_or_transform_suspect"] = bool(
        (min_iou is not None and min_iou < 0.35 and len(ious) >= 2)
        or top_like_failures
        or (
            diagnostics["mean_centroid_delta_px"] is not None
            and diagnostics["mean_centroid_delta_px"] > 0.2 * max(1, max(view.height for view in hull.views))
        )
    )
    diagnostics["catastrophic_view_failure"] = bool(
        min_iou is not None and min_iou < 0.2
    )
    return diagnostics


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    indices = np.argwhere(mask)
    if indices.size == 0:
        return None
    y0, x0 = indices.min(axis=0)
    y1, x1 = indices.max(axis=0)
    return [int(x0), int(y0), int(x1) + 1, int(y1) + 1]


def _mask_centroid(mask: np.ndarray) -> list[float] | None:
    indices = np.argwhere(mask)
    if indices.size == 0:
        return None
    yx = indices.mean(axis=0)
    return [float(yx[1]), float(yx[0])]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
