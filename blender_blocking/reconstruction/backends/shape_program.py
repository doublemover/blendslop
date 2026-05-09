"""Editable shape-program reconstruction backend.

This backend turns silhouette/profile evidence into a structured, editable
program artifact.  It is intentionally marked ``research_only`` until there is
a compiler that turns the program into Blender geometry and renders it back for
real silhouette metrics.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

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
except Exception:  # pragma: no cover - legacy script import path
    from primitives.shape_program import (  # type: ignore
        ResidualPatch,
        ShapeConstraint,
        ShapeNode,
        ShapeProgram,
        validate_shape_program,
    )


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
        return errors

    def estimate_budget(
        self, target: ReconstructionTarget, config: Mapping[str, Any]
    ) -> BackendBudget:
        band_count = sum(len(bands) for bands in target.profile_bands.values())
        return BackendBudget(
            estimated_seconds=0.01 + band_count * 0.0002,
            estimated_memory_mb=1.0,
            notes=(
                "shape_program emits editable JSON only; no mesh compiler cost included",
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
        elapsed_s = time.perf_counter() - started
        if diagnostics_path is not None:
            artifacts["shape_program_diagnostics"] = write_json(
                diagnostics_path,
                diagnostics,
            )

        warnings = (
            "shape_program is research_only: program emitted, but no Blender "
            "render validation is implemented yet",
        )
        if compiled is not None and compiled.warnings:
            warnings = warnings + tuple(compiled.warnings)
        metrics = CandidateMetrics(
            per_view=_uncompiled_per_view_metrics(request.target),
            editability_score=_editability_score(
                program,
                request.config,
                compiled=compiled is not None,
            ),
            complexity_penalty=_complexity_penalty(program, request.config),
            elapsed_s=elapsed_s,
            extras={
                "shape_program": program.to_dict(),
                "diagnostics": diagnostics,
                "compiled_blender": None if compiled is None else compiled.to_dict(),
                "primitive_editability": 1.0,
                "modifier_editability": 0.8 if compiled is not None else 0.0,
                "object_hierarchy_score": 1.0 if compiled is not None else 0.6,
                "editability": _editability_report(
                    program,
                    request.config,
                    compiled=compiled is not None,
                ),
                "topology": {
                    "status": "not_applicable",
                    "reason": "compiled object render/topology QA was not executed",
                },
                "research_only": True,
                "editable_output": True,
            },
        )
        return CandidateResult(
            candidate_id=request.candidate_id,
            backend_name=self.name,
            status="research_only",
            primitive_path=primitive_path,
            metric_result=metrics,
            artifacts=artifacts,
            warnings=warnings,
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
        residuals.extend(_profile_residual_hints(dominant_view, dominant_bands, max_nodes))

    metadata = {
        "source": "silhouette_profile_bounds",
        "root_strategy": root_strategy,
        "residual_policy": residual_policy,
        "max_nodes": max_nodes,
        "target_views": list(target.views()),
        "bounds": None if target.bounds is None else target.bounds.to_dict(),
        "config_hash": target.config_hash,
        "constraint_hash": target.constraint_hash,
    }
    program = ShapeProgram(
        schema_version="shape-program-v1",
        program_id=program_id,
        root_nodes=tuple(nodes[:max_nodes]),
        constraints=tuple(constraints),
        residual_patches=tuple(residuals[: max(0, max_nodes - len(nodes))]),
        metadata=metadata,
    )
    diagnostics = {
        "dominant_profile_view": dominant_view,
        "dominant_profile_band_count": len(dominant_bands),
        "node_count": program.node_count(),
        "residual_patch_count": program.residual_patch_count(),
        "has_uncertainty": _has_uncertainty(target),
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


def _profile_residual_hints(
    view: str,
    bands: Sequence[ProfileBand],
    max_nodes: int,
) -> list[ResidualPatch]:
    if not bands:
        return []
    residuals: list[ResidualPatch] = []
    for index, band in enumerate(bands):
        intervals = getattr(band, "intervals", ()) or ()
        holes = getattr(band, "holes", ()) or ()
        if len(intervals) <= 1 and not holes:
            continue
        category = "multi_interval_profile" if len(intervals) > 1 else "profile_hole"
        confidence = min(1.0, max(0.0, float(getattr(band, "confidence", 1.0) or 0.0)))
        residuals.append(
            ResidualPatch(
                patch_id=f"{category}_{index:03d}",
                category=category,
                source_view=view,
                confidence=confidence,
                notes=(
                    "profile row contains detail that a single lathe primitive cannot express",
                ),
            )
        )
        if len(residuals) >= max(0, max_nodes - 1):
            break
    return residuals


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
) -> dict[str, Any]:
    node_count = max(1, program.node_count())
    residual_count = program.residual_patch_count()
    return {
        "object_hierarchy_score": 1.0 if compiled else 0.6,
        "primitive_score": sum(1 for node in program.root_nodes if node.editable) / float(node_count),
        "modifier_score": 0.8 if compiled else 0.0,
        "mesh_density_score": 1.0,
        "semantic_part_score": 1.0 if all(node.name for node in program.root_nodes) else 0.65,
        "topology_score": 0.0,
        "export_roundtrip_score": 0.0,
        "warnings": (
            "render/topology/export QA has not been run for compiled shape program",
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


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
