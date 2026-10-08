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
