"""Editable shape-program reconstruction backend.

This backend turns silhouette/profile evidence into a structured, editable
program artifact.  In pure Python it reports a research-only program; inside
Blender it can compile editable scene objects, but still reports degraded
status until render/topology/export round-trip QA is executed.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import write_json
from ..backend import BackendBudget, BackendCapabilities, BaseBackend
from ..types import (
    CandidateMetrics,
    CandidateRequest,
    CandidateResult,
    ProfileBand,
    ReconstructionTarget,
)

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


class ShapeProgramBackend(BaseBackend):
    """Emit a Blender-editable primitive program from current target evidence."""

    def __init__(self) -> None:
        super().__init__(
            name="shape_program",
            capabilities=BackendCapabilities(
                requires_blender=False,
                supports_pure_python=True,
                supports_multi_view=True,
                supports_top_view=True,
                supports_uncertainty=True,
                supports_constraints=True,
                outputs_primitive_set=True,
                editability_score=0.95,
            ),
        )

    def validate_config(self, config: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        max_nodes = config.get("max_nodes", 64)
        if not isinstance(max_nodes, int) or max_nodes < 1:
            errors.append("shape_program.max_nodes must be an integer >= 1")
        root_strategy = str(config.get("root_strategy", "hybrid_profile_bounds"))
        if root_strategy not in _ROOT_STRATEGIES:
            errors.append(f"shape_program.root_strategy must be one of {_ROOT_STRATEGIES}")
        residual_policy = str(config.get("residual_policy", "suggest_patches"))
        if residual_policy not in _RESIDUAL_POLICIES:
            errors.append(
                f"shape_program.residual_policy must be one of {_RESIDUAL_POLICIES}"
            )
        editability_bias = _float(config.get("editability_bias", 1.0), 1.0)
        if not (0.0 <= editability_bias <= 1.0):
            errors.append("shape_program.editability_bias must be in [0, 1]")
        lathe_segments = config.get("lathe_segments", 48)
        if not isinstance(lathe_segments, int) or lathe_segments < 8:
            errors.append("shape_program.lathe_segments must be an integer >= 8")
        if not isinstance(config.get("run_export_qa", False), bool):
            errors.append("shape_program.run_export_qa must be a boolean")
        search_candidates = config.get("program_search_candidates", 4)
        if not isinstance(search_candidates, int) or search_candidates < 1:
            errors.append("shape_program.program_search_candidates must be an integer >= 1")
        search_objective = str(config.get("program_search_objective", "editable_balanced"))
        if search_objective not in {"editable_balanced", "minimal", "part_aware"}:
            errors.append(
                "shape_program.program_search_objective must be editable_balanced, minimal, or part_aware"
            )
        targets = _export_qa_targets(config)
        invalid_targets = set(targets) - {"obj", "glb", "gltf"}
        if invalid_targets:
            errors.append("shape_program.export_qa_targets must contain only obj/glb/gltf")
        if not isinstance(config.get("evaluate_texture_materials", False), bool):
            errors.append("shape_program.evaluate_texture_materials must be a boolean")
        if not isinstance(config.get("uv_strict", False), bool):
            errors.append("shape_program.uv_strict must be a boolean")
        material_target = str(config.get("material_target", "pbr"))
        if material_target not in {"pbr", "simple", "none"}:
            errors.append("shape_program.material_target must be pbr/simple/none")
        max_texture_memory = config.get("max_texture_memory_mb")
        if max_texture_memory is not None and _float(max_texture_memory, -1.0) <= 0.0:
            errors.append("shape_program.max_texture_memory_mb must be > 0 when provided")
        texture_reference_dir = config.get("texture_reference_dir")
        if texture_reference_dir is not None and not str(texture_reference_dir).strip():
            errors.append("shape_program.texture_reference_dir must not be blank")
        return errors

    def estimate_budget(
        self, target: ReconstructionTarget, config: Mapping[str, Any]
    ) -> BackendBudget:
        band_count = sum(len(bands) for bands in target.profile_bands.values())
        return BackendBudget(
            estimated_seconds=0.01 + band_count * 0.0002,
            estimated_memory_mb=1.0,
            notes=(
                "pure mode emits editable JSON; Blender mode may compile editable scene objects",
            ),
        )

    def benchmark_cases(self) -> Sequence[Mapping[str, Any]]:
        return (
            {
                "name": "shape_program_profile_smoke",
                "suite": "synthetic_smoke",
                "root_strategy": "hybrid_profile_bounds",
                "max_nodes": 32,
            },
        )

    def reconstruct(self, request: CandidateRequest) -> CandidateResult:
        errors = self.validate_config(request.config)
        if errors:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="failed",
                errors=tuple(errors),
            )

        started = time.perf_counter()
        program, diagnostics = build_shape_program_from_target(
            request.target,
            config=request.config,
            program_id=request.candidate_id,
        )
        program_errors = validate_shape_program(program)
        if program_errors:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="failed",
                errors=tuple(program_errors),
                payload=program,
            )

        artifacts = {}
        primitive_path = None
        diagnostics_path = None
        compiled = None
        root = request.candidate_artifact_root()
        if root is not None:
            primitive_path = write_json(
                root / "shape-program" / "program.json",
                program.to_dict(),
            )
            artifacts["shape_program"] = primitive_path
            diagnostics_path = root / "shape-program" / "diagnostics.json"

        if _should_compile_blender(request):
            try:
                compiled = _compile_program(program, request.config)
                diagnostics["compiled_blender"] = compiled.to_dict()
            except Exception as exc:
                diagnostics["compile_error"] = str(exc)
        export_qa_reports: tuple[Any, ...] = ()
        if compiled is not None and bool(request.config.get("run_export_qa", False)):
            if root is None:
                diagnostics["export_qa_skipped"] = "artifact root is required for export QA"
            else:
                try:
                    export_qa_reports = _run_shape_program_export_qa(
                        compiled,
                        root / "shape-program" / "export-qa",
                        targets=_export_qa_targets(request.config),
                    )
                    diagnostics["export_qa"] = [
                        report.to_dict() for report in export_qa_reports
                    ]
                except Exception as exc:
                    diagnostics["export_qa_error"] = str(exc)
        elapsed_s = time.perf_counter() - started
        if diagnostics_path is not None:
            artifacts["shape_program_diagnostics"] = write_json(
                diagnostics_path,
                diagnostics,
            )

        compiled_scene_summary = _compiled_scene_summary(compiled)
        compiled_topology = _compiled_topology_summary(compiled)
        appearance_summary = _compiled_appearance_summary(compiled, request.config)
        if compiled is None:
            status = "research_only"
            warnings = (
                "shape_program is research_only: editable program emitted, but Blender compilation did not run",
            )
            degraded = False
        else:
            status = "degraded"
            warnings = (
                "shape_program compiled editable Blender objects, but render/export round-trip QA has not run yet",
            )
            degraded = True
        if compiled is not None and compiled.warnings:
            warnings = warnings + tuple(compiled.warnings)
        extras = {
            "shape_program": program.to_dict(),
            "diagnostics": diagnostics,
            "compiled_blender": None if compiled is None else compiled.to_dict(),
            "compiled_scene_summary": compiled_scene_summary,
            "primitive_editability": 1.0,
            "modifier_editability": 0.8 if compiled is not None else 0.0,
            "object_hierarchy_score": 1.0 if compiled is not None else 0.6,
            "editability": _editability_report(
                program,
                request.config,
                compiled=compiled is not None,
                topology_score=float(compiled_topology.get("topology_score", 0.0)),
                export_roundtrip_score=_export_qa_score(export_qa_reports),
            ),
            "topology": compiled_topology,
            "research_only": compiled is None,
            "editable_output": True,
        }
        if appearance_summary is not None:
            extras["appearance"] = appearance_summary
        if export_qa_reports:
            extras["export_qa"] = {
                "reports": [report.to_dict() for report in export_qa_reports],
            }
        metrics = CandidateMetrics(
            per_view=_uncompiled_per_view_metrics(request.target),
            editability_score=_editability_score(
                program,
                request.config,
                compiled=compiled is not None,
            ),
            complexity_penalty=_complexity_penalty(program, request.config),
            elapsed_s=elapsed_s,
            extras=extras,
        )
        return CandidateResult(
            candidate_id=request.candidate_id,
            backend_name=self.name,
            status=status,
            primitive_path=primitive_path,
            metric_result=metrics,
            artifacts=artifacts,
            warnings=warnings,
            degraded=degraded,
            payload=compiled.root_object if compiled is not None else program,
        )


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


def _should_compile_blender(request: CandidateRequest) -> bool:
    if not bool(request.config.get("compile_blender", True)):
        return False
    context = request.context
    if context is not None and not bool(getattr(context, "blender_available", False)):
        return False
    return True


def _compile_program(program: ShapeProgram, config: Mapping[str, Any]) -> Any:
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


def _compiled_scene_summary(compiled: Any) -> dict[str, Any]:
    if compiled is None:
        return {
            "compiled": False,
            "object_count": 0,
            "mesh_object_count": 0,
            "vertex_count": 0,
            "face_count": 0,
            "modifier_count": 0,
            "material_count": 0,
        }
    objects = tuple(getattr(compiled, "objects", ()) or ())
    mesh_objects = [
        obj
        for obj in objects
        if getattr(getattr(obj, "data", None), "vertices", None) is not None
    ]
    vertex_count = 0
    face_count = 0
    modifier_count = 0
    material_count = 0
    for obj in objects:
        data = getattr(obj, "data", None)
        vertices = getattr(data, "vertices", ()) or ()
        polygons = getattr(data, "polygons", ()) or ()
        materials = getattr(data, "materials", ()) or ()
        modifiers = getattr(obj, "modifiers", ()) or ()
        vertex_count += len(vertices)
        face_count += len(polygons)
        material_count += len(materials)
        modifier_count += len(modifiers)
    return {
        "compiled": True,
        "object_count": len(objects),
        "mesh_object_count": len(mesh_objects),
        "residual_marker_count": len(getattr(compiled, "residual_markers", ()) or ()),
        "vertex_count": vertex_count,
        "face_count": face_count,
        "modifier_count": modifier_count,
        "material_count": material_count,
        "object_names": list(compiled.object_names()) if hasattr(compiled, "object_names") else [],
        "marker_names": list(compiled.marker_names()) if hasattr(compiled, "marker_names") else [],
    }


def _compiled_topology_summary(compiled: Any) -> dict[str, Any]:
    if compiled is None:
        return {
            "status": "not_applicable",
            "reason": "shape program has not been compiled",
            "topology_score": 0.0,
            "penalty": 1.0,
            "source": "shape_program_compiler",
        }
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for obj in tuple(getattr(compiled, "objects", ()) or ()):
        data = getattr(obj, "data", None)
        obj_vertices = getattr(data, "vertices", None)
        obj_polygons = getattr(data, "polygons", None)
        if obj_vertices is None or obj_polygons is None:
            continue
        offset = len(vertices)
        matrix = getattr(obj, "matrix_world", None)
        for vertex in obj_vertices:
            co = getattr(vertex, "co", None)
            if co is None:
                continue
            if matrix is not None:
                try:
                    co = matrix @ co
                except Exception:
                    pass
            vertices.append((float(co[0]), float(co[1]), float(co[2])))
        for polygon in obj_polygons:
            indices = tuple(int(index) + offset for index in getattr(polygon, "vertices", ()) or ())
            if len(indices) >= 3:
                faces.append(indices)
    if not vertices or not faces:
        return {
            "status": "skipped",
            "reason": "compiled shape program produced no mesh faces",
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "topology_score": 0.0,
            "penalty": 1.0,
            "source": "shape_program_compiler",
        }
    try:
        from metrics.topology import mesh_topology_report

        report = mesh_topology_report(
            np.asarray(vertices, dtype=float),
            tuple(faces),
        ).to_dict()
        report["status"] = "measured"
        report["source"] = "shape_program_compiler"
        return report
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"compiled shape program topology report failed: {exc}",
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "topology_score": 0.0,
            "penalty": 1.0,
            "source": "shape_program_compiler",
        }


def _compiled_appearance_summary(
    compiled: Any,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    required = bool(config.get("evaluate_texture_materials", False))
    if compiled is None:
        if not required:
            return None
        return {
            "source": "shape_program_compiler",
            "required": True,
            "strict_uv": bool(config.get("uv_strict", False)),
            "material_target": str(config.get("material_target", "pbr")),
            "texture_reference_dir": config.get("texture_reference_dir"),
            "has_uv_map": False,
            "uv_valid": False,
            "warnings": ["appearance evaluation requested but program was not compiled"],
            "errors": ["compiled_blender_asset_missing"],
        }

    objects = tuple(getattr(compiled, "objects", ()) or ())
    mesh_objects = tuple(obj for obj in objects if _object_is_mesh(obj))
    uv_meshes = tuple(obj for obj in mesh_objects if _object_has_uv_map(obj))
    material_slots = _material_slots(objects)
    materials = tuple(_slot_material(slot) for slot in material_slots)
    material_names = tuple(
        str(getattr(material, "name", "") or "")
        for material in materials
        if material is not None
    )
    named_material_count = sum(
        1 for name in material_names if name and not name.lower().startswith("material")
    )
    unique_names = {name for name in material_names if name}
    texture_images = _texture_images(materials)
    texture_memory_mb = sum(_image_memory_mb(image) for image in texture_images)
    polygon_count = sum(_object_polygon_count(obj) for obj in mesh_objects)
    missing_uv_faces = 0
    for obj in mesh_objects:
        if not _object_has_uv_map(obj):
            missing_uv_faces += _object_polygon_count(obj)
    has_uv_map = bool(mesh_objects) and len(uv_meshes) == len(mesh_objects)
    uv_valid = has_uv_map and missing_uv_faces == 0
    pbr_channels = _pbr_channel_coverage(materials)
    material_slot_count = len(material_slots)
    named_ratio = (
        named_material_count / material_slot_count if material_slot_count else None
    )
    appearance = {
        "source": "shape_program_compiler",
        "required": required,
        "strict_uv": bool(config.get("uv_strict", False)),
        "material_target": str(config.get("material_target", "pbr")),
        "texture_reference_dir": config.get("texture_reference_dir"),
        "max_texture_memory_mb": _float(config.get("max_texture_memory_mb"), None),
        "uv": {
            "has_uv_map": has_uv_map,
            "uv_valid": uv_valid,
            "missing_uv_faces": missing_uv_faces,
        },
        "texture": {
            "texture_file_count": len(texture_images),
            "texture_memory_mb": texture_memory_mb,
        },
        "materials": {
            "material_slot_count": material_slot_count,
            "named_material_ratio": named_ratio,
            "pbr_channel_coverage": pbr_channels,
            "duplicate_material_count": max(0, len(material_names) - len(unique_names)),
            "orphan_texture_count": 0,
        },
        "appearance_attribution": {
            "boundary_geometry_fidelity": 0.75 if mesh_objects else 0.0,
            "geometry_detail_score": 0.8 if mesh_objects else 0.0,
            "texture_only_detail_score": 0.1 if texture_images else 0.0,
        },
    }
    warnings: list[str] = []
    errors: list[str] = []
    if required and not mesh_objects:
        errors.append("compiled_shape_program_has_no_mesh_objects")
    if required and not has_uv_map:
        errors.append("compiled_shape_program_uv_missing")
    elif mesh_objects and not has_uv_map:
        warnings.append("compiled_shape_program_uv_missing")
    if material_slot_count == 0:
        warnings.append("compiled_shape_program_materials_missing")
    if warnings:
        appearance["warnings"] = warnings
    if errors:
        appearance["errors"] = errors
    return appearance


def _object_is_mesh(obj: Any) -> bool:
    if getattr(obj, "type", None) == "MESH":
        return True
    data = getattr(obj, "data", None)
    return hasattr(data, "polygons") or hasattr(data, "vertices")


def _object_has_uv_map(obj: Any) -> bool:
    data = getattr(obj, "data", None)
    uv_layers = getattr(data, "uv_layers", None)
    if uv_layers is None:
        return False
    try:
        return len(uv_layers) > 0
    except TypeError:
        return bool(uv_layers)


def _object_polygon_count(obj: Any) -> int:
    polygons = getattr(getattr(obj, "data", None), "polygons", None)
    if polygons is None:
        return 0
    try:
        return len(polygons)
    except TypeError:
        return 0


def _material_slots(objects: Sequence[Any]) -> tuple[Any, ...]:
    slots: list[Any] = []
    for obj in objects:
        obj_slots = getattr(obj, "material_slots", ()) or ()
        try:
            slots.extend(list(obj_slots))
        except TypeError:
            continue
    return tuple(slots)


def _slot_material(slot: Any) -> Any:
    if slot is None:
        return None
    return getattr(slot, "material", slot)


def _texture_images(materials: Sequence[Any]) -> tuple[Any, ...]:
    images: list[Any] = []
    seen: set[int] = set()
    for material in materials:
        if material is None:
            continue
        node_tree = getattr(material, "node_tree", None)
        nodes = getattr(node_tree, "nodes", ()) if node_tree is not None else ()
        try:
            iterable = list(nodes)
        except TypeError:
            iterable = ()
        for node in iterable:
            image = getattr(node, "image", None)
            if image is None:
                continue
            identity = id(image)
            if identity not in seen:
                seen.add(identity)
                images.append(image)
    return tuple(images)


def _image_memory_mb(image: Any) -> float:
    size = getattr(image, "size", None)
    if not isinstance(size, Sequence) or len(size) < 2:
        return 0.0
    try:
        width = int(size[0])
        height = int(size[1])
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, width * height * 4 / (1024.0 * 1024.0))


def _pbr_channel_coverage(materials: Sequence[Any]) -> dict[str, bool]:
    if not materials:
        return {
            "base_color": False,
            "roughness": False,
            "metallic": False,
            "normal": False,
        }
    channel_hits = {
        "base_color": False,
        "roughness": False,
        "metallic": False,
        "normal": False,
    }
    for material in materials:
        if material is None:
            continue
        if getattr(material, "diffuse_color", None) is not None:
            channel_hits["base_color"] = True
        for attr, channel in (
            ("roughness", "roughness"),
            ("metallic", "metallic"),
        ):
            if getattr(material, attr, None) is not None:
                channel_hits[channel] = True
        node_tree = getattr(material, "node_tree", None)
        nodes = getattr(node_tree, "nodes", ()) if node_tree is not None else ()
        try:
            iterable = list(nodes)
        except TypeError:
            iterable = ()
        for node in iterable:
            node_text = f"{getattr(node, 'type', '')} {getattr(node, 'name', '')}".lower()
            if "normal" in node_text:
                channel_hits["normal"] = True
            if "roughness" in node_text:
                channel_hits["roughness"] = True
            if "metallic" in node_text:
                channel_hits["metallic"] = True
            if "base" in node_text or "color" in node_text or getattr(node, "image", None) is not None:
                channel_hits["base_color"] = True
    return channel_hits


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
