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


def _visual_hull_cache_payload(
    hull: Any,
    *,
    target: ReconstructionTarget,
    resolution: int,
    slab_size: int,
    boundary_refine: bool,
    boundary_dilate_px: Optional[int],
) -> dict[str, Any]:
    bounds = target_bounds(target)
    return {
        "kind": "visual_hull_chunk",
        "resolution": int(resolution),
        "chunk_size": int(slab_size),
        "bounds": {
            "min_x": bounds.min_x,
            "max_x": bounds.max_x,
            "min_y": bounds.min_y,
            "max_y": bounds.max_y,
            "min_z": bounds.min_z,
            "max_z": bounds.max_z,
        },
        "boundary_refine": bool(boundary_refine),
        "boundary_dilate_px": boundary_dilate_px,
        "views": [
            {
                "view_type": str(getattr(view, "view_type", "")),
                "angle": float(getattr(view, "angle", 0.0)),
                "image_bounds": [float(v) for v in getattr(view, "image_bounds", ())],
                "height": int(getattr(view, "height", 0)),
                "width": int(getattr(view, "width", 0)),
                "mask_sha256": _mask_sha256(getattr(view, "silhouette", None)),
                "source_mask_area": int(
                    getattr(view, "source_mask_area", np.asarray(view.silhouette).sum())
                ),
                "refined_mask_area": int(
                    getattr(view, "refined_mask_area", np.asarray(view.silhouette).sum())
                ),
            }
            for view in hull.views
        ],
    }


def _mask_sha256(mask: Any) -> str:
    array = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(int(v) for v in array.shape)).encode("utf-8"))
    digest.update(np.packbits(array.reshape(-1)).tobytes())
    return digest.hexdigest()
