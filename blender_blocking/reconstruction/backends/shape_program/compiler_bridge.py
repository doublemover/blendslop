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


def _should_compile_blender(request: CandidateRequest) -> bool:
    if not bool(request.config.get("compile_blender", True)):
        return False
    context = request.context
    if context is not None and not bool(getattr(context, "blender_available", False)):
        return False
    return True

def _compile_program(
    program: ShapeProgram, config: Mapping[str, Any], *, context=None, process_budget=None,
) -> Any:
    try:
        from blender_blocking.primitives.shape_program_compiler import (
            compile_shape_program,
        )
    except Exception:  # pragma: no cover - legacy script import path
        from primitives.shape_program_compiler import compile_shape_program  # type: ignore

    return compile_shape_program(
        program,
        lathe_segments=int(config.get("lathe_segments", 48)),
        bevel_modifier=bool(config.get("bevel_modifier", True)),
        weighted_normals=bool(config.get("weighted_normals", True)),
        csg_options=config,
        timeout_s=float(config.get("program_timeout_s", 45.)),
        context=context,
        process_budget=process_budget,
    )

def _run_shape_program_export_qa(
    compiled: Any,
    output_root: Path,
    *,
    targets: Sequence[str],
) -> tuple[Any, ...]:
    try:
        from blender_blocking.integration.blender_ops.export_qa import (
            run_export_roundtrip_qa,
        )
    except Exception:  # pragma: no cover - legacy script import path
        from integration.blender_ops.export_qa import run_export_roundtrip_qa  # type: ignore

    return run_export_roundtrip_qa(
        tuple(getattr(compiled, "objects", ()) or ()),
        output_root,
        targets=targets,
    )

def _export_qa_targets(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = config.get("export_qa_targets", ("obj", "glb"))
    if isinstance(raw, str):
        values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    elif isinstance(raw, Sequence):
        values = tuple(str(part).strip().lower() for part in raw if str(part).strip())
    else:
        values = ("obj", "glb")
    return values or ("obj", "glb")

def _export_qa_score(reports: Sequence[Any]) -> float:
    if not reports:
        return 0.0
    scores = [float(getattr(report, "qa_score", 0.0) or 0.0) for report in reports]
    return float(sum(scores) / len(scores)) if scores else 0.0
