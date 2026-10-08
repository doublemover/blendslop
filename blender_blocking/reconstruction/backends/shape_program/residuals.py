from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ...artifacts import write_json
from ...backend import BackendBudget, BackendCapabilities, BaseBackend
from ...types import CandidateMetrics, CandidateRequest, CandidateResult, ProfileBand, ReconstructionTarget

try:  # Package import when called as blender_blocking.*
    from blender_blocking.primitives.shape_program import (
        ResidualPatch,
        ShapeConstraint,
        ShapeNode,
        ShapeProgram,
        validate_shape_program,
    )
    from blender_blocking.primitives.grammar import default_shape_program_grammar
    from blender_blocking.primitives.program_search import search_shape_program_candidates
    from blender_blocking.primitives.shape_dsl import program_to_dsl
except Exception:  # pragma: no cover - legacy script import path
    from primitives.shape_program import (  # type: ignore
        ResidualPatch,
        ShapeConstraint,
        ShapeNode,
        ShapeProgram,
        validate_shape_program,
    )
    from primitives.grammar import default_shape_program_grammar  # type: ignore
    from primitives.program_search import search_shape_program_candidates  # type: ignore
    from primitives.shape_dsl import program_to_dsl  # type: ignore

_ROOT_STRATEGIES = {"profile_lathe", "bounds_box", "hybrid_profile_bounds"}
_RESIDUAL_POLICIES = {"ignore", "report", "suggest_patches"}
from .editability import _mean


def _profile_residual_hints(
    view: str,
    bands: Sequence[ProfileBand],
    max_nodes: int,
    *,
    size: tuple[float, float, float],
    include_suggested_nodes: bool,
) -> list[ResidualPatch]:
    if not bands:
        return []
    residuals: list[ResidualPatch] = []
    max_width_px = max(
        (
            float(getattr(band, "width_px", 0.0) or 0.0)
            for band in bands
        ),
        default=0.0,
    )
    mean_center_px = _mean(
        [
            float(band.center_x)
            for band in bands
            if getattr(band, "center_x", None) is not None
        ]
    )
    for index, band in enumerate(bands):
        intervals = getattr(band, "intervals", ()) or ()
        holes = getattr(band, "holes", ()) or ()
        if len(intervals) <= 1 and not holes:
            continue
        category = "multi_interval_profile" if len(intervals) > 1 else "profile_hole"
        confidence = min(1.0, max(0.0, float(getattr(band, "confidence", 1.0) or 0.0)))
        suggested_node = None
        if include_suggested_nodes:
            suggested_node = _suggested_residual_node(
                view=view,
                band=band,
                index=index,
                category=category,
                size=size,
                max_width_px=max_width_px,
                mean_center_px=mean_center_px,
                confidence=confidence,
            )
        residuals.append(
            ResidualPatch(
                patch_id=f"{category}_{index:03d}",
                category=category,
                source_view=view,
                confidence=confidence,
                suggested_node=suggested_node,
                notes=(
                    "profile row contains detail that a single lathe primitive cannot express",
                    "suggested node is editable and can be kept, resized, or converted to a boolean detail patch",
                ),
            )
        )
        if len(residuals) >= max(0, max_nodes - 1):
            break
    return residuals

def _suggested_residual_node(
    *,
    view: str,
    band: ProfileBand,
    index: int,
    category: str,
    size: tuple[float, float, float],
    max_width_px: float,
    mean_center_px: float,
    confidence: float,
) -> ShapeNode:
    width_world, depth_world, height_world = size
    intervals = tuple(getattr(band, "intervals", ()) or ())
    holes = tuple(getattr(band, "holes", ()) or ())
    focus_interval = _residual_focus_interval(
        category=category,
        intervals=intervals,
        holes=holes,
    )
    focus_width_px = max(
        1e-6,
        float(getattr(focus_interval, "width", 0.0) or 0.0)
        if focus_interval is not None
        else float(getattr(band, "width_px", 0.0) or 0.0),
    )
    focus_center_px = (
        float(getattr(focus_interval, "center", 0.0) or 0.0)
        if focus_interval is not None
        else float(getattr(band, "center_x", mean_center_px) or mean_center_px)
    )
    width_ratio = focus_width_px / max(max_width_px, 1e-6)
    t = max(0.0, min(1.0, float(getattr(band, "t", 0.0) or 0.0)))
    z_world = (t - 0.5) * height_world
    x_world = ((focus_center_px - mean_center_px) / max(max_width_px, 1e-6)) * width_world
    patch_height = max(height_world * 0.025, height_world / float(max(16, index + 8)))
    patch_width = max(width_world * width_ratio, width_world * 0.04, 1e-6)
    patch_depth = max(depth_world * (0.08 if category == "profile_hole" else 0.14), 1e-6)
    operation = "difference" if category == "profile_hole" else "attach"
    primitive_type = "plane_patch" if category == "profile_hole" else "residual_mesh_patch"
    return ShapeNode(
        node_id=f"residual_{category}_{index:03d}",
        operation=operation,
        primitive_type=primitive_type,
        name=f"{view or 'profile'} {category.replace('_', ' ')} {index:03d}",
        editable=True,
        parameters={
            "source_view": view or "",
            "category": category,
            "t": t,
            "location_x": x_world,
            "location_y": 0.0,
            "location_z": z_world,
            "width_world": patch_width,
            "depth_world": patch_depth,
            "height_world": patch_height,
            "confidence": confidence,
            "interval_count": len(intervals),
            "hole_count": len(holes),
            "focus_x0_px": 0.0 if focus_interval is None else float(focus_interval.x0),
            "focus_x1_px": 0.0 if focus_interval is None else float(focus_interval.x1),
            "focus_width_px": focus_width_px,
            "target_root_node": "root_profile_00",
            "suggested_boolean_role": "subtract" if operation == "difference" else "add",
        },
    )

def _residual_focus_interval(
    *,
    category: str,
    intervals: Sequence[Any],
    holes: Sequence[Any],
) -> Any:
    if category == "profile_hole" and holes:
        return max(holes, key=lambda interval: float(getattr(interval, "width", 0.0) or 0.0))
    if len(intervals) > 1:
        ordered = sorted(
            intervals,
            key=lambda interval: float(getattr(interval, "width", 0.0) or 0.0),
            reverse=True,
        )
        return ordered[1] if len(ordered) > 1 else ordered[0]
    return intervals[0] if intervals else None

def _has_uncertainty(target: ReconstructionTarget) -> bool:
    return any(constraint.uncertainty is not None for constraint in target.constraints)

def _uncompiled_per_view_metrics(
    target: ReconstructionTarget,
) -> dict[str, dict[str, Any]]:
    metrics = {}
    for constraint in target.constraints:
        metrics[constraint.view] = {
            "required": True,
            "passed": False,
            "area_iou": 0.0,
            "boundary_iou": None,
            "soft_iou": None,
            "signed_distance_loss": None,
            "reason": "shape program has not been compiled and rendered yet",
        }
    return metrics
