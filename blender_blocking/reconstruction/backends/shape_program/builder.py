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
from .residuals import _has_uncertainty, _profile_residual_hints


def build_shape_program_from_target(
    target: ReconstructionTarget,
    *,
    config: Mapping[str, Any],
    program_id: str,
) -> tuple[ShapeProgram, dict[str, Any]]:
    """Build a deterministic editable program from bounds/profile evidence."""
    root_strategy = str(config.get("root_strategy", "hybrid_profile_bounds"))
    residual_policy = str(config.get("residual_policy", "suggest_patches"))
    max_nodes = int(config.get("max_nodes", 64))
    dominant_view, dominant_bands = _dominant_profile(target)
    size = _target_size(target)
    nodes: list[ShapeNode] = []
    constraints: list[ShapeConstraint] = []
    residuals: list[ResidualPatch] = []

    if dominant_bands and root_strategy in {"profile_lathe", "hybrid_profile_bounds"}:
        stats = _profile_stats(dominant_bands)
        profile_curve = _profile_curve_payload(dominant_bands, size, stats)
        nodes.append(
            ShapeNode(
                node_id="root_profile_00",
                operation="add",
                primitive_type="lathe_profile",
                name=f"{dominant_view or 'front'} profile lathe",
                parameters={
                    "source_view": dominant_view or "",
                    "band_count": stats["band_count"],
                    "mean_width_px": stats["mean_width_px"],
                    "max_width_px": stats["max_width_px"],
                    "mean_center_px": stats["mean_center_px"],
                    "profile_curve": profile_curve,
                    "height_world": size[2],
                    "width_world": size[0],
                    "depth_world": size[1],
                    "confidence": stats["mean_confidence"],
                    "preserves_multiple_intervals": stats["multi_interval_rows"] > 0,
                    "preserves_hole_hints": stats["hole_rows"] > 0,
                },
            )
        )
    else:
        nodes.append(
            ShapeNode(
                node_id="root_bounds_00",
                operation="add",
                primitive_type="rounded_box",
                name="bounds primitive",
                parameters={
                    "width_world": size[0],
                    "depth_world": size[1],
                    "height_world": size[2],
                    "corner_radius_world": min(size) * 0.04,
                },
            )
        )

    if len(target.constraints) >= 2 and len(nodes) < max_nodes:
        constraints.append(
            ShapeConstraint(
                kind="orthographic_silhouette_alignment",
                target_nodes=(nodes[0].node_id,),
                parameters={
                    "views": [constraint.view for constraint in target.constraints],
                    "profile_view": dominant_view or "",
                    "hard": True,
                },
            )
        )
    if _has_uncertainty(target) and len(nodes) < max_nodes:
        constraints.append(
            ShapeConstraint(
                kind="boundary_uncertainty_weighted_fit",
                target_nodes=(nodes[0].node_id,),
                parameters={
                    "source": "UncertainMask",
                    "lower_weight_on_ambiguous_edges": True,
                },
            )
        )

    if residual_policy != "ignore":
        residuals.extend(
            _profile_residual_hints(
                dominant_view,
                dominant_bands,
                max_nodes,
                size=size,
                include_suggested_nodes=residual_policy == "suggest_patches",
            )
        )
    realized_residual_node_ids: list[str] = []
    if residual_policy == "suggest_patches":
        for residual in residuals:
            suggested = residual.suggested_node
            if suggested is None or len(nodes) >= max_nodes:
                continue
            nodes.append(suggested)
            realized_residual_node_ids.append(suggested.node_id)

    metadata = {
        "source": "silhouette_profile_bounds",
        "root_strategy": root_strategy,
        "residual_policy": residual_policy,
        "max_nodes": max_nodes,
        "target_views": list(target.views()),
        "bounds": None if target.bounds is None else target.bounds.to_dict(),
        "config_hash": target.config_hash,
        "constraint_hash": target.constraint_hash,
        "dominant_profile_view": dominant_view,
    }
    program = ShapeProgram(
        schema_version="shape-program-v1",
        program_id=program_id,
        root_nodes=tuple(nodes[:max_nodes]),
        constraints=tuple(constraints),
        residual_patches=tuple(residuals[:max_nodes]),
        metadata=metadata,
    )
    grammar = default_shape_program_grammar()
    search_result = search_shape_program_candidates(
        program,
        grammar=grammar,
        max_candidates=int(config.get("program_search_candidates", 4)),
        objective=str(config.get("program_search_objective", "editable_balanced")),
    )
    program = search_result.selected.program
    program = ShapeProgram(
        schema_version=program.schema_version,
        program_id=program.program_id,
        root_nodes=program.root_nodes[:max_nodes],
        constraints=program.constraints,
        residual_patches=program.residual_patches[:max_nodes],
        metadata={
            **dict(program.metadata),
            "grammar_id": search_result.grammar.grammar_id,
            "program_dsl": program_to_dsl(program),
        },
    )
    diagnostics = {
        "dominant_profile_view": dominant_view,
        "dominant_profile_band_count": len(dominant_bands),
        "profile_curve_rows": _dominant_profile_curve_row_count(program),
        "node_count": program.node_count(),
        "residual_patch_count": program.residual_patch_count(),
        "suggested_residual_node_count": sum(
            1 for residual in residuals if residual.suggested_node is not None
        ),
        "realized_residual_node_count": len(realized_residual_node_ids),
        "realized_residual_node_ids": realized_residual_node_ids,
        "has_uncertainty": _has_uncertainty(target),
        "grammar_search": search_result.to_dict(),
        "selected_grammar_candidate": search_result.selected.candidate_id,
        "program_validation_errors": list(validate_shape_program(program)),
    }
    return program, diagnostics

def _dominant_profile(target: ReconstructionTarget) -> tuple[str, tuple[ProfileBand, ...]]:
    best_view = ""
    best_bands: tuple[ProfileBand, ...] = ()
    best_score = -1.0
    for view, bands in target.profile_bands.items():
        score = 0.0
        for band in bands:
            score += max(0.0, float(getattr(band, "width_px", 0.0) or 0.0))
            score += len(getattr(band, "intervals", ()) or ()) * 0.01
        if score > best_score:
            best_view = str(view)
            best_bands = tuple(bands)
            best_score = score
    return best_view, best_bands

def _target_size(target: ReconstructionTarget) -> tuple[float, float, float]:
    if target.bounds is None:
        return (1.0, 1.0, 1.0)
    size = target.bounds.size
    return tuple(max(1e-6, float(value)) for value in size)

def _profile_stats(bands: Sequence[ProfileBand]) -> dict[str, Any]:
    widths = [float(getattr(band, "width_px", 0.0) or 0.0) for band in bands]
    centers = [
        float(band.center_x)
        for band in bands
        if getattr(band, "center_x", None) is not None
    ]
    confidences = [float(getattr(band, "confidence", 1.0) or 0.0) for band in bands]
    return {
        "band_count": len(bands),
        "mean_width_px": _mean(widths),
        "max_width_px": max(widths) if widths else 0.0,
        "mean_center_px": _mean(centers),
        "mean_confidence": _mean(confidences) if confidences else 1.0,
        "multi_interval_rows": sum(
            1 for band in bands if len(getattr(band, "intervals", ()) or ()) > 1
        ),
        "hole_rows": sum(1 for band in bands if getattr(band, "holes", ()) or ()),
    }

def _dominant_profile_curve_row_count(program: ShapeProgram) -> int:
    for node in program.root_nodes:
        curve = node.parameters.get("profile_curve")
        if isinstance(curve, Sequence) and not isinstance(curve, (str, bytes, bytearray)):
            return len(curve)
    return 0

def _profile_curve_payload(
    bands: Sequence[ProfileBand],
    size: tuple[float, float, float],
    stats: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    if not bands:
        return ()
    max_width_px = max(float(stats.get("max_width_px", 0.0) or 0.0), 1e-6)
    mean_center_px = float(stats.get("mean_center_px", 0.0) or 0.0)
    width_world, depth_world, height_world = size
    rows = []
    for band in sorted(bands, key=lambda item: float(getattr(item, "t", 0.0))):
        width_px = max(0.0, float(getattr(band, "width_px", 0.0) or 0.0))
        if width_px <= 0.0:
            continue
        t = max(0.0, min(1.0, float(getattr(band, "t", 0.0) or 0.0)))
        width_ratio = width_px / max_width_px
        center_x = getattr(band, "center_x", None)
        center_offset_world = 0.0
        if center_x is not None:
            center_offset_world = ((float(center_x) - mean_center_px) / max_width_px) * width_world
        intervals = getattr(band, "intervals", ()) or ()
        holes = getattr(band, "holes", ()) or ()
        rows.append(
            {
                "t": t,
                "z_world": (t - 0.5) * height_world,
                "width_px": width_px,
                "radius_x_world": max(width_world * 0.5 * width_ratio, 1e-6),
                "radius_y_world": max(depth_world * 0.5 * width_ratio, 1e-6),
                "center_offset_world": center_offset_world,
                "confidence": max(0.0, min(1.0, float(getattr(band, "confidence", 1.0) or 0.0))),
                "interval_count": len(intervals),
                "hole_count": len(holes),
            }
        )
    if len(rows) == 1:
        row = dict(rows[0])
        duplicate = dict(row)
        row["z_world"] = -height_world * 0.5
        row["t"] = 0.0
        duplicate["z_world"] = height_world * 0.5
        duplicate["t"] = 1.0
        rows = [row, duplicate]
    return tuple(rows)
