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


def _editability_score(
    program: ShapeProgram,
    config: Mapping[str, Any],
    *,
    compiled: bool = False,
) -> float:
    bias = _float(config.get("editability_bias", 1.0), 1.0)
    editable = sum(1 for node in program.root_nodes if node.editable)
    base = editable / float(max(1, program.node_count()))
    residual_penalty = min(0.2, program.residual_patch_count() * 0.01)
    compile_bonus = 0.08 if compiled else 0.0
    return max(0.0, min(1.0, (0.75 + 0.25 * base + compile_bonus - residual_penalty) * bias))

def _editability_report(
    program: ShapeProgram,
    config: Mapping[str, Any],
    *,
    compiled: bool = False,
    topology_score: float = 0.0,
    export_roundtrip_score: float = 0.0,
) -> dict[str, Any]:
    node_count = max(1, program.node_count())
    residual_count = program.residual_patch_count()
    return {
        "object_hierarchy_score": 1.0 if compiled else 0.6,
        "primitive_score": sum(1 for node in program.root_nodes if node.editable) / float(node_count),
        "modifier_score": 0.8 if compiled else 0.0,
        "mesh_density_score": 1.0,
        "semantic_part_score": 1.0 if all(node.name for node in program.root_nodes) else 0.65,
        "topology_score": topology_score,
        "export_roundtrip_score": export_roundtrip_score,
        "warnings": _editability_report_warnings(
            compiled=compiled,
            export_roundtrip_score=export_roundtrip_score,
        ),
        "metadata": {
            "node_count": node_count,
            "residual_patch_count": residual_count,
            "compiled": compiled,
            "max_nodes": int(config.get("max_nodes", 64)),
        },
    }

def _complexity_penalty(program: ShapeProgram, config: Mapping[str, Any]) -> float:
    max_nodes = max(1, int(config.get("max_nodes", 64)))
    node_pressure = program.node_count() / float(max_nodes)
    residual_pressure = program.residual_patch_count() / float(max_nodes)
    return max(0.0, min(1.0, node_pressure * 0.5 + residual_pressure * 0.5))

def _editability_report_warnings(
    *,
    compiled: bool,
    export_roundtrip_score: float,
) -> tuple[str, ...]:
    if not compiled:
        return ("Blender compilation has not run for shape program",)
    if export_roundtrip_score > 0.0:
        return ("render QA has not been run for compiled shape program",)
    return ("render/export round-trip QA has not been run for compiled shape program",)

def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0

def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
