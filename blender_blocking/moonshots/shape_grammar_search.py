"""Program-search moonshot for editable shape grammars."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from .contracts import (
    MoonshotExperiment,
    MoonshotRequest,
    MoonshotResult,
    bundle_result,
    error_result,
    skipped_result,
)
from .papers import SUPERQUADRICS
from .support import (
    bounded,
    candidate_rows,
    metric_extras,
    row_failures,
    row_metric,
    selected_candidate_payload,
    target_signals,
)


EXPERIMENT = MoonshotExperiment(
    experiment_id="shape_grammar_search",
    title="Shape grammar search over editable primitives",
    subsystem="shape_program",
    hypothesis=(
        "Search over constructive shape programs can recover cleaner editable "
        "structure than direct mesh extraction for blocky, furniture, and vehicle inputs."
    ),
    expected_wins={
        "editability": "clean named primitives, lower boolean/mesh cleanup load",
        "quality": "better thin-support and repeated-part reconstruction",
    },
    required_inputs=("multi_view_silhouettes", "profile_bands", "constraints"),
    validation_metrics=("min_view_iou", "boundary_iou", "editability_index", "component_sanity"),
    papers=(SUPERQUADRICS,),
)


def run(request: MoonshotRequest) -> MoonshotResult:
    try:
        seed = _seed_program(request)
        if seed is None:
            return skipped_result(
                request,
                reason="shape grammar search needs target profile/bounds signals or a candidate shape program",
                next_steps=("run after target building or a shape_program/primitive backend emits evidence",),
            )
        return _run_search(request, seed)
    except Exception as exc:
        return error_result(request, error=f"{type(exc).__name__}: {exc}")


def _run_search(request: MoonshotRequest, seed: Any) -> MoonshotResult:
    try:
        from blender_blocking.primitives.program_search import search_shape_program_candidates
    except Exception:  # pragma: no cover - script-style imports
        from primitives.program_search import search_shape_program_candidates  # type: ignore

    config = dict(request.config or {})
    max_candidates = max(1, int(config.get("max_candidates", config.get("beam_width", 6)) or 6))
    objective = str(config.get("shape_grammar_objective", "editable_balanced"))
    search = search_shape_program_candidates(
        seed,
        max_candidates=max_candidates,
        objective=objective,
    )
    signals = target_signals(request, include_profile_rows=True)
    rows = candidate_rows(request.candidate)
    candidate_payloads = []
    for index, candidate in enumerate(search.candidates):
        metrics = _score_candidate(
            candidate.to_dict(),
            signals=signals,
            rows=rows,
        )
        candidate_payload = candidate.to_dict()
        candidate_payload["beam_rank"] = index + 1
        candidate_payload["compile_plan"] = _compile_plan(candidate_payload)
        candidate_payload["fingerprint"] = _program_fingerprint(candidate_payload)
        candidate_payload["moonshot_metrics"] = metrics
        candidate_payload["moonshot_score"] = metrics["total"]
        candidate_payload["selection_reason"] = _selection_reason(metrics, candidate_payload)
        candidate_payloads.append(candidate_payload)
    candidate_payloads.sort(
        key=lambda item: float(item["moonshot_score"]),
        reverse=True,
    )
    selected = candidate_payloads[0]
    evidence = {
        "beam_width": max_candidates,
        "objective": objective,
        "seed_program_id": getattr(seed, "program_id", ""),
        "grammar_id": search.grammar.grammar_id,
        "family_hints": _family_hints(signals, rows),
        "beam_layers": _beam_layers(candidate_payloads),
        "validation_summary": _validation_summary(candidate_payloads),
        "signal_summary": {key: dict(value) for key, value in signals.items()},
        "selected_candidate_id": selected["candidate_id"],
        "selected_fingerprint": selected["fingerprint"],
        "selected_compile_plan": selected["compile_plan"],
        "candidates": candidate_payloads,
    }
    metrics = {
        "ran": 1.0,
        "candidate_count": float(len(candidate_payloads)),
        "selected_score": float(selected["moonshot_score"]),
        "selected_editability": float(
            selected["moonshot_metrics"].get("editability", 0.0)
        ),
        "selected_topology": float(selected["moonshot_metrics"].get("topology", 0.0)),
        "selected_silhouette_proxy": float(
            selected["moonshot_metrics"].get("silhouette_proxy", 0.0)
        ),
        "valid_candidate_count": float(
            sum(1 for item in candidate_payloads if not _validation_errors(item))
        ),
        "family_hint_count": float(len(evidence["family_hints"])),
    }
    return bundle_result(
        request,
        status="ran",
        metrics=metrics,
        evidence=evidence,
        artifact_name="shape-grammar-search.json",
        next_steps=(
            "compile the selected program in Blender when render/export QA is available",
            "promote only if render-IoU and editability gates pass",
        ),
    )


def _seed_program(request: MoonshotRequest) -> Any | None:
    program = _program_from_candidate(request)
    if program is not None:
        return program
    target = request.target
    if target is not None:
        try:
            from blender_blocking.reconstruction.backends.shape_program.builder import (
                build_shape_program_from_target,
            )
        except Exception:  # pragma: no cover - script-style imports
            from reconstruction.backends.shape_program.builder import (  # type: ignore
                build_shape_program_from_target,
            )

        program, _diagnostics = build_shape_program_from_target(
            target,
            config={"root_strategy": "hybrid_profile_bounds"},
            program_id="moonshot_shape_seed",
        )
        return program
    signals = target_signals(request, include_profile_rows=True)
    if not bool(signals["profile"].get("available")) and not bool(
        signals["surface"].get("available")
    ):
        return None
    return _fallback_program_from_signals(signals)


def _program_from_candidate(request: MoonshotRequest) -> Any | None:
    for row in candidate_rows(request.candidate):
        source = selected_candidate_payload(row)
        for payload in (
            source.get("payload"),
            source.get("shape_program"),
            metric_extras(row).get("shape_program"),
            metric_extras(row).get("diagnostics", {}).get("shape_program")
            if isinstance(metric_extras(row).get("diagnostics"), Mapping)
            else None,
        ):
            if isinstance(payload, Mapping):
                program = _program_from_dict(payload)
                if program is not None:
                    return program
    return None


def _program_from_dict(payload: Mapping[str, Any]) -> Any | None:
    if "root_nodes" not in payload:
        return None
    try:
        from blender_blocking.primitives.shape_program import (
            ResidualPatch,
            ShapeConstraint,
            ShapeNode,
            ShapeProgram,
        )
    except Exception:  # pragma: no cover
        from primitives.shape_program import (  # type: ignore
            ResidualPatch,
            ShapeConstraint,
            ShapeNode,
            ShapeProgram,
        )

    nodes = tuple(
        ShapeNode(
            node_id=str(item.get("node_id", f"node_{index:02d}")),
            operation=str(item.get("operation", "add")),
            primitive_type=item.get("primitive_type"),
            parameters=dict(item.get("parameters", {}) or {}),
            children=tuple(str(child) for child in item.get("children", ()) or ()),
            name=str(item.get("name", "")),
            editable=bool(item.get("editable", True)),
        )
        for index, item in enumerate(payload.get("root_nodes", ()) or ())
        if isinstance(item, Mapping)
    )
    constraints = tuple(
        ShapeConstraint(
            kind=str(item.get("kind", "")),
            target_nodes=tuple(str(node) for node in item.get("target_nodes", ()) or ()),
            parameters=dict(item.get("parameters", {}) or {}),
        )
        for item in payload.get("constraints", ()) or ()
        if isinstance(item, Mapping)
    )
    patches = tuple(
        ResidualPatch(
            patch_id=str(item.get("patch_id", "")),
            category=str(item.get("category", "")),
            source_view=str(item.get("source_view", "")),
            confidence=float(item.get("confidence", 0.0) or 0.0),
            notes=tuple(str(note) for note in item.get("notes", ()) or ()),
        )
        for item in payload.get("residual_patches", ()) or ()
        if isinstance(item, Mapping)
    )
    if not nodes:
        return None
    return ShapeProgram(
        schema_version=str(payload.get("schema_version", "shape_program_v1")),
        program_id=str(payload.get("program_id", "moonshot_shape_seed")),
        root_nodes=nodes,
        constraints=constraints,
        residual_patches=patches,
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def _fallback_program_from_signals(signals: Mapping[str, Mapping[str, Any]]) -> Any:
    try:
        from blender_blocking.primitives.shape_program import ShapeConstraint, ShapeNode, ShapeProgram
    except Exception:  # pragma: no cover
        from primitives.shape_program import ShapeConstraint, ShapeNode, ShapeProgram  # type: ignore

    profile = signals["profile"]
    topology = signals["topology"]
    width = max(1.0, float(profile.get("max_width", profile.get("mean_width", 1.0)) or 1.0))
    depth = max(1.0, float(profile.get("mean_width", width) or width))
    height = max(1.0, float(profile.get("band_samples", 1.0) or 1.0))
    primitive = "rounded_box" if float(topology.get("complexity", 0.0) or 0.0) < 0.35 else "superquadric"
    return ShapeProgram(
        schema_version="shape_program_v1",
        program_id="moonshot_shape_seed",
        root_nodes=(
            ShapeNode(
                node_id="moonshot_root",
                operation="add",
                primitive_type=primitive,
                name="moonshot signal root",
                parameters={
                    "width_world": width,
                    "depth_world": depth,
                    "height_world": height,
                },
            ),
        ),
        constraints=(
            ShapeConstraint(
                kind="signal_seed",
                target_nodes=("moonshot_root",),
                parameters={"source": "moonshot_shape_grammar_search"},
            ),
        ),
        metadata={
            "bounds": True,
            "profile_signal": dict(profile),
            "topology_signal": dict(topology),
        },
    )


def _score_candidate(
    payload: Mapping[str, Any],
    *,
    signals: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    base_metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), Mapping) else {}
    editability = float(base_metrics.get("editability", 0.0) or 0.0)
    grammar_gain = float(base_metrics.get("grammar_gain", 0.0) or 0.0)
    complexity = float(base_metrics.get("complexity", 0.0) or 0.0)
    validation_penalty = float(base_metrics.get("validation_penalty", 0.0) or 0.0)
    topology = float(signals["topology"].get("score", 0.75) or 0.75)
    profile = signals["profile"]
    profile_available = 1.0 if profile.get("available") else 0.0
    row_quality = _row_quality(rows)
    silhouette_proxy = bounded(
        0.45
        + 0.20 * profile_available
        + 0.20 * row_quality
        + 0.10 * (1.0 - float(profile.get("complexity", 0.0) or 0.0))
    )
    total = (
        silhouette_proxy * 0.42
        + editability * 0.28
        + topology * 0.20
        + grammar_gain * 0.20
        - complexity * 0.12
        - validation_penalty * 0.22
    )
    return {
        "total": float(total),
        "silhouette_proxy": float(silhouette_proxy),
        "editability": float(editability),
        "topology": float(topology),
        "grammar_gain": float(grammar_gain),
        "complexity": float(complexity),
        "validation_penalty": float(validation_penalty),
    }


def _row_quality(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    values = [
        row_metric(row, "render.min_view_iou", "min_view_iou", "area_iou_min", default=0.0)
        for row in rows
    ]
    return bounded(sum(values) / max(1, len(values)))


def _validation_errors(candidate_payload: Mapping[str, Any]) -> tuple[str, ...]:
    diagnostics = candidate_payload.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return ()
    errors = diagnostics.get("validation_errors")
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
        return tuple(str(item) for item in errors if item)
    return ()


def _compile_plan(candidate_payload: Mapping[str, Any]) -> dict[str, Any]:
    program = candidate_payload.get("program")
    program = program if isinstance(program, Mapping) else {}
    nodes = program.get("root_nodes") if isinstance(program, Mapping) else ()
    patches = program.get("residual_patches") if isinstance(program, Mapping) else ()
    constraints = program.get("constraints") if isinstance(program, Mapping) else ()
    validation_errors = _validation_errors(candidate_payload)
    node_count = len(nodes) if isinstance(nodes, Sequence) and not isinstance(nodes, (str, bytes)) else 0
    patch_count = len(patches) if isinstance(patches, Sequence) and not isinstance(patches, (str, bytes)) else 0
    constraint_count = len(constraints) if isinstance(constraints, Sequence) and not isinstance(constraints, (str, bytes)) else 0
    return {
        "schema_version": "shape_grammar_compile_plan_v1",
        "compile_ready": not validation_errors and node_count > 0,
        "node_count": node_count,
        "residual_patch_count": patch_count,
        "constraint_count": constraint_count,
        "validation_errors": list(validation_errors),
        "steps": [
            "compile primitives and named modifiers",
            "apply residual patch hints as editable annotations",
            "run render QA against required views",
            "run export round-trip editability QA",
        ],
        "guards": [
            "reject if render.min_view_iou regresses",
            "reject if topology.score falls below floor",
            "keep shape program artifact attached to row evidence",
        ],
    }


def _program_fingerprint(candidate_payload: Mapping[str, Any]) -> str:
    program = candidate_payload.get("program")
    digest = hashlib.sha1(repr(program).encode("utf-8")).hexdigest()[:16]
    return f"sp-{digest}"


def _selection_reason(
    metrics: Mapping[str, float],
    candidate_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "score": float(metrics.get("total", 0.0)),
        "silhouette_proxy": float(metrics.get("silhouette_proxy", 0.0)),
        "editability": float(metrics.get("editability", 0.0)),
        "topology": float(metrics.get("topology", 0.0)),
        "validation_error_count": float(len(_validation_errors(candidate_payload))),
    }


def _family_hints(
    signals: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    profile = signals["profile"]
    topology = signals["topology"]
    hints = []
    complexity = float(profile.get("complexity", 0.0) or 0.0)
    hole_count = int(profile.get("hole_count", 0) or 0)
    interval_count = int(profile.get("interval_count", 0) or 0)
    band_samples = max(1, int(profile.get("band_samples", 1) or 1))
    if complexity < 0.32:
        hints.append({"family": "rounded_box", "reason": "low profile complexity", "priority": 0.72})
    else:
        hints.append({"family": "superquadric", "reason": "curved or complex silhouette", "priority": 0.68})
    if hole_count > 0:
        hints.append({"family": "boolean_cutout", "reason": "profile holes present", "priority": 0.76})
    if interval_count / band_samples > 1.2:
        hints.append({"family": "part_decomposition", "reason": "multi-interval profile bands", "priority": 0.70})
    if float(topology.get("complexity", 0.0) or 0.0) > 0.45:
        hints.append({"family": "residual_patch_program", "reason": "topology complexity", "priority": 0.62})
    failures = {failure for row in rows for failure in row_failures(row)}
    if "axis_or_transform_suspect" in failures:
        hints.append({"family": "axis_calibrated_program", "reason": "autopsy transform evidence", "priority": 0.66})
    return hints


def _beam_layers(candidate_payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    layers: dict[int, list[str]] = {}
    for item in candidate_payloads:
        rank = int(item.get("beam_rank", 0) or 0)
        layer = 0 if not item.get("applied_rules") else 1
        if rank > 4:
            layer = max(layer, 2)
        layers.setdefault(layer, []).append(str(item.get("candidate_id", "")))
    return [
        {"layer": layer, "candidate_ids": ids, "count": len(ids)}
        for layer, ids in sorted(layers.items())
    ]


def _validation_summary(candidate_payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    invalid = []
    for item in candidate_payloads:
        errors = _validation_errors(item)
        if errors:
            invalid.append({"candidate_id": str(item.get("candidate_id", "")), "errors": list(errors)})
    return {
        "candidate_count": len(candidate_payloads),
        "valid_count": len(candidate_payloads) - len(invalid),
        "invalid_count": len(invalid),
        "invalid": invalid[:8],
    }
