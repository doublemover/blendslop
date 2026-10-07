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
from .appearance import _compiled_appearance_summary
from .builder import build_shape_program_from_target
from .compiled_scene import _compiled_scene_summary, _compiled_topology_summary
from .compiler_bridge import (
    _compile_program,
    _export_qa_score,
    _export_qa_targets,
    _run_shape_program_export_qa,
    _should_compile_blender,
)
from .editability import _complexity_penalty, _editability_report, _editability_score, _float


_DEFAULT_REQUIRED_RENDER_VIEWS = ("front", "side", "top")


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
        routed_program = request.config.get("routed_program")
        if routed_program is not None:
            from dataclasses import replace
            if not isinstance(routed_program, ShapeProgram):
                raise ValueError("routed_program must be a validated ShapeProgram proposal")
            program = replace(routed_program, program_id=request.candidate_id)
            diagnostics = {"initialization":"reused_screened_whole_program",
                           "proposal_evidence":request.config.get("routing_proposal_evidence",{})}
        else:
            program, diagnostics = build_shape_program_from_target(
                request.target,
                config=request.config,
                program_id=request.candidate_id,
            )
        geometric_result = None
        if _should_compile_blender(request):
            from .geometry_search import geometric_program_search
            program, geometric_result, geometric_search = geometric_program_search(request,program)
            diagnostics["geometric_search"] = geometric_search
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
                root / "sp" / "program.json",
                program.to_dict(),
            )
            artifacts["shape_program"] = primitive_path
            diagnostics_path = root / "sp" / "diag.json"

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
                        root / "sp" / "qa",
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
        render_per_view, render_qa = _shape_program_render_qa(
            request,
            compiled=compiled,
        )
        if geometric_result is not None:
            render_per_view = geometric_result.metric_result.per_view
            render_qa = {"status":"complete","missing_required_metrics":False,
                "failed_required_views":[view for view,row in render_per_view.items() if not row["passed"]],
                "source":"actual_compiled_fixed_camera_geometry"}
        if compiled is None:
            status = "research_only"
            warnings = (
                "shape_program is research_only: editable program emitted, but Blender compilation did not run",
            )
            degraded = False
        elif render_per_view and not render_qa.get("missing_required_metrics"):
            failed_views = tuple(render_qa.get("failed_required_views", ()) or ())
            if failed_views:
                status = "degraded"
                warnings = (
                    "shape_program render QA ran, but at least one required view failed",
                )
                degraded = True
            else:
                status = "success"
                warnings = (
                    "shape_program compiled editable Blender objects and render QA metrics are available",
                )
                degraded = False
        else:
            status = "degraded"
            warnings = (
                "degraded_no_render_qa: shape_program compiled editable Blender objects, but required render QA has not run yet",
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
            "render_qa": render_qa,
            "editable_output": True,
        }
        if appearance_summary is not None:
            extras["appearance"] = appearance_summary
        if export_qa_reports:
            extras["export_qa"] = {
                "reports": [report.to_dict() for report in export_qa_reports],
            }
        metrics = CandidateMetrics(
            per_view=render_per_view,
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


def _shape_program_render_qa(
    request: CandidateRequest,
    *,
    compiled: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    required_views = _required_render_views(request.target)
    if compiled is None:
        return {}, {
            "status": "not_applicable",
            "reason": "shape program has not been compiled",
            "required_views": list(required_views),
            "missing_required_metrics": bool(required_views),
        }
    raw = _render_qa_payload(request.config)
    per_view = _coerce_render_qa_per_view(
        raw,
        required_views=required_views,
        min_area_iou=_float(request.config.get("render_qa_min_area_iou", 0.7), 0.7),
    )
    if not per_view:
        return {}, {
            "status": "missing",
            "reason": "render QA has not been run for compiled shape program",
            "required_views": list(required_views),
            "missing_required_views": list(required_views),
            "missing_required_metrics": bool(required_views),
            "next_step": "render compiled shape-program objects against required views",
        }

    missing_views = [view for view in required_views if view not in per_view]
    missing_metrics: list[str] = []
    failed_views: list[str] = []
    for view in required_views:
        metrics = per_view.get(view)
        if not isinstance(metrics, Mapping):
            continue
        if metrics.get("area_iou") is None:
            missing_metrics.append(f"{view}.area_iou")
        if metrics.get("boundary_iou") is None:
            missing_metrics.append(f"{view}.boundary_iou")
        if metrics.get("signed_distance_loss") is None:
            missing_metrics.append(f"{view}.signed_distance_loss")
        if bool(metrics.get("required", True)) and not bool(metrics.get("passed", False)):
            failed_views.append(view)
    status = "complete"
    if missing_views or missing_metrics:
        status = "incomplete"
    elif failed_views:
        status = "failed"
    return per_view, {
        "status": status,
        "source": "shape_program_render_qa",
        "required_views": list(required_views),
        "missing_required_views": missing_views,
        "missing_required_metric_names": missing_metrics,
        "missing_required_metrics": bool(missing_views or missing_metrics),
        "failed_required_views": failed_views,
        "per_view": per_view,
    }


def _required_render_views(target: ReconstructionTarget) -> tuple[str, ...]:
    try:
        views = tuple(str(view) for view in target.views() if str(view))
    except Exception:
        views = ()
    return views or _DEFAULT_REQUIRED_RENDER_VIEWS


def _render_qa_payload(config: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("render_qa_metrics", "shape_program_render_qa", "render_qa"):
        value = config.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _coerce_render_qa_per_view(
    payload: Mapping[str, Any],
    *,
    required_views: Sequence[str],
    min_area_iou: float,
) -> dict[str, dict[str, Any]]:
    source = payload.get("per_view") if isinstance(payload.get("per_view"), Mapping) else payload
    if not isinstance(source, Mapping):
        return {}
    output: dict[str, dict[str, Any]] = {}
    for view, value in source.items():
        if not isinstance(value, Mapping):
            continue
        view_name = str(view)
        area = _optional_metric_float(value.get("area_iou", value.get("iou")))
        boundary = _optional_metric_float(value.get("boundary_iou"))
        signed_distance = _optional_metric_float(value.get("signed_distance_loss"))
        required = _truthy(value.get("required", view_name in required_views))
        passed_raw = value.get("passed", value.get("pass"))
        passed = (
            _truthy(passed_raw)
            if passed_raw is not None
            else bool(area is not None and area >= min_area_iou)
        )
        output[view_name] = {
            "required": required,
            "passed": passed,
            "pass": passed,
            "area_iou": area,
            "boundary_iou": boundary,
            "signed_distance_loss": signed_distance,
            "soft_iou": _optional_metric_float(value.get("soft_iou")),
            "reason": str(value.get("reason", "")),
        }
    return output


def _optional_metric_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(parsed):
        return parsed
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "passed"}
    return bool(value)
