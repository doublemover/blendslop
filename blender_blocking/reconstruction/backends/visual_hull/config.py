from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .volume_flow import _visual_hull_cache_directory


@dataclass(frozen=True)
class VisualHullRunConfig:
    resolution: int
    requested_backend: str
    boundary_refine: bool
    boundary_dilate_px: int | None
    cache_directory: Path | None
    cache_namespace: str
    cache_read: bool
    cache_write: bool
    projection_metric_max_voxels: int


def visual_hull_run_config(config: Mapping[str, Any]) -> VisualHullRunConfig:
    boundary_dilate_px = config.get("boundary_dilate_px")
    return VisualHullRunConfig(
        resolution=int(config.get("resolution", 64)),
        requested_backend=str(config.get("backend", "dense")).strip().lower(),
        boundary_refine=bool(config.get("boundary_refine", True)),
        boundary_dilate_px=(
            None if boundary_dilate_px is None else int(boundary_dilate_px)
        ),
        cache_directory=_visual_hull_cache_directory(config),
        cache_namespace=str(config.get("cache_namespace", "visual_hull")),
        cache_read=bool(
            config.get("cache_read", not bool(config.get("cache_write_only", False)))
        ),
        cache_write=bool(
            config.get("cache_write", not bool(config.get("cache_read_only", False)))
        ),
        projection_metric_max_voxels=int(
            config.get("projection_metric_max_voxels", 4_000_000)
        ),
    )


def direct_sparse_builder_requested(requested_backend: str) -> bool:
    return requested_backend in {"chunked", "sparse_hash", "openvdb"}
