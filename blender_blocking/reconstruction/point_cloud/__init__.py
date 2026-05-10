"""Point-cloud, volume, and projection helpers for reconstruction targets."""

from __future__ import annotations

from .build import (
    default_bounds,
    surface_mask_from_grid,
    target_bounds,
    target_occupied_points,
    target_surface_points,
    visual_hull_chunked_grid_from_target,
    visual_hull_dense_grid_from_target,
    visual_hull_grid_from_target,
    visual_hull_openvdb_grid_from_target,
    visual_hull_projection_metrics_from_target,
    visual_hull_sparse_hash_grid_from_target,
    visual_hull_view_diagnostics_from_target,
    volume_bounds_from_target,
)

__all__ = [
    "default_bounds",
    "surface_mask_from_grid",
    "target_bounds",
    "target_occupied_points",
    "target_surface_points",
    "visual_hull_chunked_grid_from_target",
    "visual_hull_dense_grid_from_target",
    "visual_hull_grid_from_target",
    "visual_hull_openvdb_grid_from_target",
    "visual_hull_projection_metrics_from_target",
    "visual_hull_sparse_hash_grid_from_target",
    "visual_hull_view_diagnostics_from_target",
    "volume_bounds_from_target",
]
