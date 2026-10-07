from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from metrics.topology import mesh_topology_report
from placement.resfit_initialization import (
    PrimitiveInitializationConfig,
    initialize_ellipsoids_from_points,
    initialize_gaussians_from_points,
)
from reconstruction.artifacts import write_json
from reconstruction.mesh_io import (
    combine_primitive_meshes,
    mesh_arrays_from_object,
    write_obj,
    write_primitive_set,
)
from reconstruction.point_cloud import target_surface_points
from reconstruction.types import CandidateMetrics, CandidateResult

try:
    from primitives.shape_program import ShapeNode, ShapeProgram, validate_shape_program
    from primitives.proxy_distillation import distill_proxy_field, write_proxy_field_npz
except ImportError:  # pragma: no cover - package import path
    from ..shape_program import ShapeNode, ShapeProgram, validate_shape_program
    from ..proxy_distillation import distill_proxy_field, write_proxy_field_npz


def _adaptive_point_count(
    base_count: int,
    *,
    profile_signal: Mapping[str, Any],
    uncertainty_signal: Mapping[str, Any],
    constraint_signal: Mapping[str, Any],
    topology_signal: Mapping[str, Any],
    surface_signal: Mapping[str, Any],
) -> int:
    complexity = float(profile_signal.get("complexity", 0.0))
    constraint_penalty = min(1.0, 0.15 * int(constraint_signal.get("constraint_count", 0)))
    topology_headroom = 0.5 + 0.5 * float(topology_signal.get("score", 0.4))
    uncertainty_mean = float(uncertainty_signal.get("overall_confidence_mean", 0.75))
    surface_density = float(surface_signal.get("density_hint", 0.75))
    adapted = int(
        round(
            base_count
            * (0.7 + 0.5 * complexity)
            * (0.5 + 0.5 * uncertainty_mean)
            * topology_headroom
            * (1.0 - constraint_penalty * 0.35)
            * surface_density
        )
    )
    min_points = 64 if base_count >= 64 else max(16, base_count)
    max_points = max(base_count * 2, min_points)
    return int(np.clip(adapted, min_points, max_points))


def _adaptive_primitive_count(
    base_count: int,
    *,
    profile_signal: Mapping[str, Any],
    constraint_signal: Mapping[str, Any],
    topology_signal: Mapping[str, Any],
    uncertainty_signal: Mapping[str, Any],
) -> int:
    complexity = float(profile_signal.get("complexity", 0.0))
    uncertainty = float(uncertainty_signal.get("overall_confidence_mean", 0.75))
    topology = float(topology_signal.get("score", 0.4))
    constraint_count = int(constraint_signal.get("constraint_count", 0))
    adapted = int(
        round(
            base_count
            * (0.9 + 0.4 * complexity + 0.15 * (1.0 - topology))
            * (1.0 - 0.06 * constraint_count)
            * (0.8 + 0.2 * uncertainty)
        )
    )
    adapted = max(1, adapted)
    return int(np.clip(adapted, 1, max(1, base_count * 2)))


def _prepare_initialization_points(
    points: np.ndarray,
    *,
    method: str,
    primitive_count: int,
    target_point_count: int,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points
    desired = max(int(primitive_count * 6), int(target_point_count))
    if method == "grid":
        return _sample_grid(points, desired)
    if method == "farthest_point":
        return _sample_farthest_points(points, desired)
    return points


def _sample_grid(points: np.ndarray, target_count: int) -> np.ndarray:
    if len(points) <= target_count:
        return points.copy()
    grid = int(np.ceil(len(points) / max(1, target_count)))
    return points[::grid]


def _sample_farthest_points(points: np.ndarray, target_count: int) -> np.ndarray:
    if len(points) <= target_count:
        return points.copy()
    if target_count <= 1:
        # Preserve the historical zero/single-sample behavior.
        return points[:target_count].copy()
    # Maintain the nearest selected point distance: O(N*K), O(N) memory.
    centroid = points.mean(axis=0)
    next_idx = int(np.argmax(np.linalg.norm(points - centroid, axis=1)))
    indices = []
    nearest = np.full(len(points), np.inf)
    for _ in range(target_count):
        indices.append(next_idx)
        distances = np.linalg.norm(points - points[next_idx], axis=1)
        np.minimum(nearest, distances, out=nearest)
        next_idx = int(np.argmax(nearest))
    return points[np.asarray(indices)]
