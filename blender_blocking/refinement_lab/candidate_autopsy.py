"""Structured failure classification for refinement candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import ExperimentResult, json_safe


SUSPECTED_TRANSFORM_FILES = (
    "blender_blocking/reconstruction/point_cloud.py",
    "blender_blocking/volume/meshing.py",
    "blender_blocking/volume/contracts.py",
    "blender_blocking/reconstruction/target_builder.py",
    "blender_blocking/reconstruction/targets.py",
    "blender_blocking/integration/blender_ops/camera_framing.py",
    "blender_blocking/test_e2e_validation.py",
)


def autopsy_candidate(
    result: ExperimentResult,
    *,
    run_root: Path | None = None,
    bounds_debug: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    findings = []
    bounds = bounds_debug or result.bounds_debug or {}
    findings.extend(_backend_findings(result))
    findings.extend(_metric_findings(result))
    findings.extend(_mesh_findings(result))
    findings.extend(_bounds_findings(result, bounds))
    findings.extend(_artifact_findings(result, run_root))
    if not findings and result.status == "pass":
        findings.append(
            _finding(
                "pass",
                "info",
                "Candidate passed available validation checks.",
            )
        )
    if not findings:
        findings.append(
            _finding(
                "unknown_error",
                "medium",
                "Candidate did not match a specific autopsy category.",
            )
        )
    primary = _primary_findings(findings)[0]
    return {
        "schema_version": "candidate_autopsy_v1",
        "category": primary["category"],
        "severity": primary["severity"],
        "summary": primary["summary"],
        "findings": findings,
    }


def write_autopsy(
    result: ExperimentResult,
    path: Path,
    *,
    run_root: Path | None = None,
    bounds_debug: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    payload = autopsy_candidate(result, run_root=run_root, bounds_debug=bounds_debug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return payload


def _backend_findings(result: ExperimentResult) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    backend = result.backend_result if isinstance(result.backend_result, Mapping) else {}
    status = _backend_status(backend)
    errors = " ".join(str(error) for error in result.errors)
    if status in {"failed", "error"} or result.status == "error":
        findings.append(
            _finding(
                "backend_failed",
                "fatal",
                "Backend execution failed.",
                evidence={"backend_status": status, "errors": list(result.errors)},
            )
        )
    if status == "skipped":
        findings.append(
            _finding(
                "backend_skipped",
                "medium",
                "Backend skipped reconstruction.",
                evidence={"backend_status": status},
            )
        )
    if "ModuleNotFoundError" in errors or "No module named" in errors:
        findings.append(
            _finding(
                "missing_optional_dependency",
                "high",
                "Candidate failed because an optional dependency was unavailable.",
                evidence={"errors": list(result.errors)},
            )
        )
    dependency_reports = _find_dependency_reports(backend)
    for name, report in dependency_reports.items():
        if isinstance(report, Mapping) and report.get("available") is False:
            findings.append(
                _finding(
                    "missing_optional_dependency",
                    "medium",
                    f"Optional dependency {name} is unavailable.",
                    evidence={name: json_safe(report)},
                )
            )
    return findings


def _metric_findings(result: ExperimentResult) -> list[dict[str, object]]:
    findings = []
    has_view_metric = any(result.view_iou(view) is not None for view in ("front", "side", "top"))
    if not has_view_metric and result.status == "pass":
        findings.append(
            _finding(
                "missing_required_metrics",
                "high",
                "Candidate passed without required per-view render metrics.",
                evidence={"metrics": json_safe(result.metrics)},
            )
        )
    catastrophic = {
        view: result.view_iou(view)
        for view in ("front", "side", "top")
        if result.view_iou(view) is None or (result.view_iou(view) or 0.0) < 0.2
    }
    if catastrophic:
        findings.append(
            _finding(
                "catastrophic_view_failure",
                "high",
                "At least one required view has catastrophic IoU.",
                evidence=catastrophic,
            )
        )
    views = result.metrics.get("views", {})
    if isinstance(views, Mapping):
        for view, payload in views.items():
            if isinstance(payload, Mapping):
                if int(payload.get("ref_area", payload.get("reference_area", 1)) or 1) == 0:
                    findings.append(
                        _finding(
                            "empty_reference_mask",
                            "high",
                            f"{view} reference mask is empty.",
                            evidence={str(view): json_safe(payload)},
                        )
                    )
                if int(payload.get("render_area", payload.get("candidate_area", 1)) or 1) == 0:
                    findings.append(
                        _finding(
                            "empty_render_mask",
                            "high",
                            f"{view} render mask is empty.",
                            evidence={str(view): json_safe(payload)},
                        )
                    )
    if not has_view_metric and _backend_area_iou(result) > 0.9:
        findings.append(
            _finding(
                "quality_metric_only",
                "medium",
                "Candidate has strong backend metrics but no render-IoU evidence.",
                evidence={"area_iou_mean": _backend_area_iou(result)},
            )
        )
    return findings


def _mesh_findings(result: ExperimentResult) -> list[dict[str, object]]:
    findings = []
    mesh_path = _mesh_path(result)
    mesh_mode = result.mode in {
        "visual_hull_voxel",
        "hybrid_loft_hull",
        "primitive_fit_refine",
        "gaussian_ellipsoid_proxy",
        "differentiable_refine",
    }
    if mesh_mode and mesh_path is None:
        findings.append(
            _finding(
                "mesh_missing",
                "high",
                "Mesh-producing candidate did not report a mesh artifact.",
            )
        )
    elif mesh_path is not None and not mesh_path.exists():
        findings.append(
            _finding(
                "mesh_missing",
                "high",
                "Reported mesh artifact does not exist.",
                evidence={"mesh_path": mesh_path.as_posix()},
            )
        )
    topology = _topology(result)
    if topology:
        bad_counts = {
            key: int(topology.get(key, 0) or 0)
            for key in (
                "loose_vertices",
                "non_manifold_edges",
                "degenerate_faces",
                "zero_area_faces",
            )
        }
        if any(value > 0 for value in bad_counts.values()) or float(topology.get("topology_score", 1.0) or 1.0) < 0.75:
            findings.append(
                _finding(
                    "mesh_topology_problem",
                    "medium",
                    "Mesh topology metrics indicate cleanup problems.",
                    evidence=bad_counts | {"topology_score": topology.get("topology_score")},
                )
            )
    return findings


def _bounds_findings(
    result: ExperimentResult,
    bounds: Mapping[str, Any],
) -> list[dict[str, object]]:
    findings = []
    if not isinstance(bounds, Mapping) or not bounds:
        return findings
    comparisons = bounds.get("comparisons", {})
    if isinstance(comparisons, Mapping):
        mesh_target = comparisons.get("mesh_vs_target_bounds", {})
        if isinstance(mesh_target, Mapping):
            agreement = _float(mesh_target.get("agreement"), 1.0)
            if agreement < 0.25:
                findings.append(
                    _finding(
                        "bounds_mismatch",
                        "high",
                        "Mesh bounds disagree strongly with target bounds.",
                        evidence=json_safe(mesh_target),
                        suspected_files=SUSPECTED_TRANSFORM_FILES,
                    )
                )
    axis = bounds.get("axis_permutation_search", {})
    if isinstance(axis, Mapping):
        baseline = axis.get("baseline", {})
        best = axis.get("best", {})
        improvement = _float(best.get("average_iou")) - _float(baseline.get("average_iou"))
        if improvement > 0.25:
            findings.append(
                _finding(
                    "axis_or_transform_suspect",
                    "high",
                    "A non-baseline axis/scale/origin diagnostic transform improved projected IoU.",
                    evidence={"improvement": improvement, "best": json_safe(best), "baseline": json_safe(baseline)},
                    suspected_files=SUSPECTED_TRANSFORM_FILES,
                )
            )
    if result.mode == "visual_hull_voxel":
        topology_score = _float(_topology(result).get("topology_score") if _topology(result) else 0.0)
        if topology_score >= 0.9 and result.min_iou < 0.2:
            findings.append(
                _finding(
                    "axis_or_transform_suspect",
                    "high",
                    "Visual hull mesh topology is clean but render IoU is catastrophic.",
                    evidence={"topology_score": topology_score, "min_iou": result.min_iou},
                    suspected_files=SUSPECTED_TRANSFORM_FILES,
                )
            )
    diagnosis = bounds.get("diagnosis")
    if isinstance(diagnosis, Sequence) and "likely_blender_render_framing_bug" in diagnosis:
        findings.append(
            _finding(
                "render_framing_suspect",
                "high",
                "Projection diagnostics suggest render camera/framing mismatch.",
                evidence={"diagnosis": list(diagnosis)},
                suspected_files=("blender_blocking/integration/blender_ops/camera_framing.py",),
            )
        )
    return findings


def _artifact_findings(
    result: ExperimentResult,
    run_root: Path | None,
) -> list[dict[str, object]]:
    if run_root is None:
        return []
    root = Path(run_root).resolve(strict=False)
    escaped = []
    for key, path in result.artifacts.items():
        resolved = Path(path).resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError:
            escaped.append({"key": key, "path": resolved.as_posix()})
    if escaped:
        return [
            _finding(
                "artifact_escape",
                "medium",
                "Candidate reported generated artifacts outside the refinement run root.",
                evidence={"escaped": escaped},
            )
        ]
    return []


def _primary_findings(findings: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    order = {"fatal": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
    return sorted(findings, key=lambda item: order.get(str(item.get("severity")), 0), reverse=True)


def _finding(
    category: str,
    severity: str,
    summary: str,
    *,
    evidence: Mapping[str, Any] | None = None,
    suspected_files: Sequence[str] = (),
    recommended_next_actions: Sequence[str] = (),
) -> dict[str, object]:
    return {
        "category": category,
        "severity": severity,
        "summary": summary,
        "evidence": json_safe(evidence or {}),
        "suspected_files": list(suspected_files),
        "recommended_next_actions": list(recommended_next_actions),
    }


def _backend_status(backend: Mapping[str, Any]) -> str:
    selected = backend.get("selected")
    if isinstance(selected, Mapping):
        return str(selected.get("status", ""))
    return str(backend.get("status", ""))


def _find_dependency_reports(value: Any) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if isinstance(value, Mapping):
        if "optional_dependencies" in value and isinstance(value["optional_dependencies"], Mapping):
            found.update(value["optional_dependencies"])
        for nested in value.values():
            found.update(_find_dependency_reports(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            found.update(_find_dependency_reports(nested))
    return found


def _mesh_path(result: ExperimentResult) -> Path | None:
    for key in ("mesh", "mesh_obj"):
        path = result.artifacts.get(key)
        if path is not None:
            return Path(path)
    backend = result.backend_result if isinstance(result.backend_result, Mapping) else {}
    selected = backend.get("selected") if isinstance(backend, Mapping) else None
    source = selected if isinstance(selected, Mapping) else backend
    if isinstance(source, Mapping):
        value = source.get("mesh_path")
        if value:
            return Path(str(value))
        artifacts = source.get("artifacts", {})
        if isinstance(artifacts, Mapping) and artifacts.get("mesh_obj"):
            return Path(str(artifacts["mesh_obj"]))
    return None


def _topology(result: ExperimentResult) -> Mapping[str, Any]:
    backend = result.backend_result if isinstance(result.backend_result, Mapping) else {}
    selected = backend.get("selected") if isinstance(backend, Mapping) else None
    source = selected if isinstance(selected, Mapping) else backend
    metrics = source.get("metric_result", {}) if isinstance(source, Mapping) else {}
    extras = metrics.get("extras", {}) if isinstance(metrics, Mapping) else {}
    topology = extras.get("topology") if isinstance(extras, Mapping) else None
    if isinstance(topology, Mapping):
        return topology
    mesh_quality = metrics.get("mesh_quality") if isinstance(metrics, Mapping) else None
    return mesh_quality if isinstance(mesh_quality, Mapping) else {}


def _backend_area_iou(result: ExperimentResult) -> float:
    backend = result.backend_result if isinstance(result.backend_result, Mapping) else {}
    selected = backend.get("selected") if isinstance(backend, Mapping) else None
    source = selected if isinstance(selected, Mapping) else backend
    metrics = source.get("metric_result", {}) if isinstance(source, Mapping) else {}
    return _float(metrics.get("area_iou_mean") if isinstance(metrics, Mapping) else None)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default
