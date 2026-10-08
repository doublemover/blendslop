"""Structured failure classification for refinement candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import ExperimentResult, json_safe
try:
    from blender_blocking.metrics.values import float_or as _float
except ImportError:  # pragma: no cover - script-style imports
    from metrics.values import float_or as _float


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
    topology_plan = _topology_repair_plan_from_findings(findings)
    payload: dict[str, object] = {
        "schema_version": "candidate_autopsy_v1",
        "category": primary["category"],
        "severity": primary["severity"],
        "summary": primary["summary"],
        "findings": findings,
    }
    if topology_plan:
        payload["topology_repair_plan"] = topology_plan
    return payload


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
    backend_warnings = _string_items(backend.get("warnings", ()))
    backend_errors = _string_items(backend.get("errors", ()))
    degradation_text = " ".join(
        backend_warnings
        + backend_errors
        + [str(item) for item in result.warnings]
        + [str(item) for item in result.errors]
    )
    if (
        result.mode == "primitive_fit_refine"
        and status == "degraded"
        and (
            "min IoU" in degradation_text
            or "proxy" in degradation_text
            or "internal fit" in degradation_text
        )
    ):
        findings.append(
            _finding(
                "primitive_fit_proxy_floor_degraded",
                "medium",
                (
                    "Primitive-fit emitted renderable artifacts, but its internal "
                    "fit/proxy quality floor was weak."
                ),
                evidence={
                    "backend_status": status,
                    "warnings": backend_warnings,
                    "errors": backend_errors,
                },
                recommended_next_actions=(
                    "rerun primitive-fit with a larger objective budget",
                    "try silhouette/profile-weighted primitive family variants",
                    "probe target axis and bounds assumptions before promotion",
                ),
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
    if result.mode == "hybrid_loft_hull":
        weak_views = {
            view: result.view_iou(view)
            for view in ("front", "side", "top")
            if result.view_iou(view) is not None
            and (result.view_iou(view) or 0.0) < 0.5
        }
        if weak_views:
            findings.append(
                _finding(
                    "hybrid_loft_one_view_failure",
                    "medium",
                    "Hybrid loft has a one-view or thin-support silhouette collapse.",
                    evidence={
                        "weak_views": weak_views,
                        "average_iou": result.avg_iou,
                        "min_iou": result.min_iou,
                    },
                    recommended_next_actions=(
                        "run visual_hull_voxel as the reliability baseline",
                        "emit per-component support confidence for thin structures",
                        "compare top-view footprint deltas before accepting loft output",
                        "retry mesh extraction/postprocess instead of broad parameter fanout",
                    ),
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
                "boundary_edges",
                "non_manifold_edges",
                "degenerate_faces",
                "zero_area_faces",
            )
        }
        topology_score = float(topology.get("topology_score", 1.0) or 1.0)
        if any(value > 0 for value in bad_counts.values()) or topology_score < 0.75:
            repair_plan = _topology_repair_plan(topology, bad_counts)
            findings.append(
                _finding(
                    "mesh_topology_problem",
                    "medium",
                    "Mesh topology metrics indicate cleanup problems.",
                    evidence={
                        **bad_counts,
                        "topology_score": topology.get("topology_score"),
                        "repair_plan": repair_plan,
                    },
                    recommended_next_actions=tuple(repair_plan["actions"]),
                )
            )
    return findings


def _topology_repair_plan_from_findings(
    findings: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    for finding in findings:
        if finding.get("category") != "mesh_topology_problem":
            continue
        evidence = finding.get("evidence")
        if isinstance(evidence, Mapping) and isinstance(
            evidence.get("repair_plan"),
            Mapping,
        ):
            return evidence["repair_plan"]  # type: ignore[return-value]
    return {}


def _topology_repair_plan(
    topology: Mapping[str, Any],
    bad_counts: Mapping[str, int],
) -> Mapping[str, object]:
    actions: list[str] = []
    probes: list[Mapping[str, object]] = []

    if int(bad_counts.get("loose_vertices", 0)) > 0:
        actions.append("remove tiny shells and isolated loose vertices")
        probes.append(
            _topology_probe(
                "remove-tiny-shells",
                "remove isolated dust shells before evaluating editability",
                {"topology.loose_vertices": "decrease"},
            )
        )
    if int(bad_counts.get("boundary_edges", 0)) > 0:
        actions.append("fill boundary loops with guarded hole filling")
        probes.append(
            _topology_probe(
                "fill-boundary-loops",
                "fill only short boundary loops that preserve silhouette support",
                {"topology.boundary_edges": "decrease"},
            )
        )
    if int(bad_counts.get("non_manifold_edges", 0)) > 0:
        actions.append("weld close vertices and split non-manifold edges")
        probes.append(
            _topology_probe(
                "weld-close-vertices",
                "merge close duplicate vertices before retrying mesh extraction",
                {"topology.non_manifold_edges": "decrease"},
            )
        )
    if (
        int(bad_counts.get("degenerate_faces", 0)) > 0
        or int(bad_counts.get("zero_area_faces", 0)) > 0
    ):
        actions.append("dissolve degenerate and zero-area faces")
        probes.append(
            _topology_probe(
                "dissolve-degenerate-faces",
                "remove zero-area faces before export round-trip checks",
                {"topology.degenerate_faces": "decrease"},
            )
        )

    topology_score = _float(topology.get("topology_score"), 1.0)
    if topology_score < 0.75:
        actions.append("retry mesh extraction/postprocess with topology repair enabled")
        probes.append(
            _topology_probe(
                "retry-topology-postprocess",
                "rerun mesh extraction with guarded topology_repair postprocess",
                {"topology.score": "increase"},
            )
        )

    actions.append("avoid repairs that erase silhouette detail")
    if not probes:
        probes.append(
            _topology_probe(
                "topology-recheck",
                "rerun topology and export round-trip checks after repair",
                {"topology.score": "increase"},
            )
        )

    destructive_counts = (
        int(bad_counts.get("boundary_edges", 0))
        + int(bad_counts.get("non_manifold_edges", 0))
    )
    return {
        "schema_version": "topology_repair_plan_v1",
        "safe_automatic": destructive_counts <= 16,
        "actions": actions,
        "probes": probes,
        "source_topology": json_safe(topology),
    }


def _topology_probe(
    probe_id: str,
    action: str,
    expected_win: Mapping[str, str],
) -> Mapping[str, object]:
    return {
        "probe_id": probe_id,
        "action": action,
        "expected_win": dict(expected_win),
    }


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


def _string_items(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if item]
    return []


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
    metric_topology = _topology_from_metrics(result.metrics)
    if metric_topology:
        return metric_topology
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


def _topology_from_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    topology: dict[str, Any] = {}
    topology_group = metrics.get("topology")
    if isinstance(topology_group, Mapping):
        topology.update(topology_group)
    aliases = {
        "score": "topology_score",
        "topology_score": "topology_score",
        "watertight": "watertight",
        "connected_components": "connected_components",
        "boundary_edges": "boundary_edges",
        "non_manifold_edges": "non_manifold_edges",
        "loose_vertices": "loose_vertices",
        "degenerate_faces": "degenerate_faces",
        "zero_area_faces": "zero_area_faces",
    }
    for source_key, target_key in aliases.items():
        if source_key in topology and target_key not in topology:
            topology[target_key] = topology[source_key]
    for source_key in aliases.values():
        if source_key in metrics and source_key not in topology:
            topology[source_key] = metrics[source_key]
    return topology


def _backend_area_iou(result: ExperimentResult) -> float:
    backend = result.backend_result if isinstance(result.backend_result, Mapping) else {}
    selected = backend.get("selected") if isinstance(backend, Mapping) else None
    source = selected if isinstance(selected, Mapping) else backend
    metrics = source.get("metric_result", {}) if isinstance(source, Mapping) else {}
    return _float(metrics.get("area_iou_mean") if isinstance(metrics, Mapping) else None)
