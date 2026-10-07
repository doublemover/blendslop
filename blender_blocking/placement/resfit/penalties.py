from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..resfit_initialization import (
    PrimitiveInitializationConfig,
    initialize_ellipsoids_from_points,
    initialize_from_profile_bands,
    initialize_gaussians_from_points,
    initialize_superfrusta_from_points,
)
from ..resfit_objective import (
    PenaltyHook,
    ResFitLossWeights,
    ResFitObjectiveResult,
    SilhouetteHook,
    evaluate_resfit_objective,
)
from ..resfit_optimizer import (
    CoordinateDescentConfig,
    OptimizationRecord,
    coordinate_descent_optimize,
)

from .config import by_view_total
from .profiles import _bounds_z_range


def _build_profile_silhouette_hook(
    *,
    profile_rows: Sequence[Mapping[str, Any]],
    uncertainty_by_view: Mapping[str, Mapping[str, Any]],
) -> Callable[[Sequence[object]], Mapping[str, float]]:
    from collections import OrderedDict
    from .profile_intervals import union_intervals, intersect_intervals, interval_disagreement, projected_row_intervals
    from ..resfit_objective import _geometry_key
    meshes = OrderedDict()
    def hook(primitives: Sequence[object]) -> Mapping[str, float]:
        by_view = {}
        for row in profile_rows:
            view = str(row.get("view", "generic"))
            axes = tuple(row.get("axes", {"front":(0,2), "side":(1,2), "top":(0,1)}.get(view,(0,2))))
            vertical = float(row.get("vertical_world", row.get("z_world", 0)))
            width, center = float(row.get("width_world", 0)), float(row.get("center_x_world", 0))
            target = row.get("intervals_world", [(center-width*.5, center+width*.5)] if width else [])
            predicted = []
            for part in primitives:
                mesh = None
                if type(part).__name__ not in {"EllipsoidPrimitive", "AnisotropicGaussianPrimitive"}:
                    from blender_blocking.reconstruction.mesh_io import mesh_arrays_from_object
                    key = _geometry_key(part)
                    if key is not None and key in meshes:
                        mesh = meshes[key]
                        meshes.move_to_end(key)
                    else:
                        mesh = mesh_arrays_from_object(part.to_mesh_data(16))
                        if key is not None:
                            meshes[key] = mesh
                            while len(meshes) > 32:
                                meshes.popitem(last=False)
                predicted.extend(projected_row_intervals(part, axes, vertical, mesh))
            predicted = union_intervals(predicted)
            known = row.get("known_intervals_world")
            if known is not None:
                predicted, target = intersect_intervals(predicted, known), intersect_intervals(target, known)
            viewport = row.get("viewport_world", (0, max(width,1e-3),0,1))
            scale = max(float(viewport[1]-viewport[0]), float(row.get("pixel_world", 1e-3)))
            weight = max(0., float(row.get("confidence",1))) * max(0., float(uncertainty_by_view.get(view,{}).get("mean_confidence",1)))
            by_view.setdefault(view, []).append((weight, interval_disagreement(target,predicted,scale)))
        # Return one normalized residual per view; never count an aggregate twice.
        return {f"profile_{view}": float(sum(w*v for w,v in values) / sum(w for w,v in values))
                for view, values in by_view.items() if sum(w for w,v in values) > 0}
    return hook


def _build_topology_penalty_hook(topology_signal: Mapping[str, Any]) -> PenaltyHook:
    topology_target = float(topology_signal.get("score", 1.0))
    complexity = float(topology_signal.get("complexity", 0.0))

    def hook(primitives: Sequence[object]) -> float:
        if topology_target <= 0.0 and not complexity:
            return 0.0
        count = float(len(primitives))
        return float((1.0 - topology_target) + 0.01 * complexity * count)

    return hook


def _build_constraint_penalty_hook(
    *,
    constraint_signal: Mapping[str, Any],
    bounds: Any | None,
    geometry_dependent: bool = False,
) -> PenaltyHook:
    score = float(constraint_signal.get("score", 1.0))
    count = int(constraint_signal.get("constraint_count", 0))
    has_bounds = bounds is not None
    bounds_min = None
    bounds_max = None
    if has_bounds:
        bounds_min = np.array([bounds.min_x, bounds.min_y, bounds.min_z], dtype=float)
        bounds_max = np.array([bounds.max_x, bounds.max_y, bounds.max_z], dtype=float)
        extents = bounds_max - bounds_min
        extents = np.where(extents > 0.0, extents, 1.0)

    from collections import OrderedDict
    from ..resfit_objective import _geometry_key
    extent_cache = OrderedDict()
    def hook(primitives: Sequence[object]) -> float:
        base = 0.0
        for primitive in primitives:
            center_attr = "position" if hasattr(primitive, "position") else "center"
            if not hasattr(primitive, center_attr):
                continue
            if has_bounds:
                position = np.asarray(getattr(primitive, center_attr), dtype=float)
                if position.size == 3:
                    if geometry_dependent and hasattr(primitive, "to_mesh_data"):
                        key = _geometry_key(primitive)
                        if key is not None and key in extent_cache:
                            candidate_min, candidate_max = extent_cache[key]
                            extent_cache.move_to_end(key)
                        else:
                            vertices = np.asarray(primitive.to_mesh_data(12).vertices, float)
                            candidate_min, candidate_max = vertices.min(axis=0), vertices.max(axis=0)
                            if key is not None:
                                extent_cache[key] = (candidate_min, candidate_max)
                                while len(extent_cache) > 64:
                                    extent_cache.popitem(last=False)
                    else:
                        candidate_min = candidate_max = position
                    below = np.maximum(bounds_min - candidate_min, 0.0)
                    above = np.maximum(candidate_max - bounds_max, 0.0)
                    normal = (below + above) / extents
                    base += float(np.linalg.norm(normal) ** 2)
        # Weakly penalize strong constraint budgets or dense explicit constraint payloads.
        constraint_pressure = float((1.0 - score) * max(1, count) * 0.1)
        return base if geometry_dependent else base + constraint_pressure

    return hook


def _build_uncertainty_penalty_hook(
    uncertainty_signal: Mapping[str, Any],
) -> PenaltyHook:
    consistency = float(uncertainty_signal.get("consistency", 0.75))
    overall = float(uncertainty_signal.get("overall_confidence_mean", 1.0))
    overall_std = float(uncertainty_signal.get("overall_confidence_std", 0.0))

    def hook(_: Sequence[object]) -> float:
        return float(max(0.0, 1.0 - overall) + 0.25 * overall_std + 0.15 * (1.0 - consistency))

    return hook
