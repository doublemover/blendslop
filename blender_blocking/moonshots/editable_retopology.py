"""Editable retopology moonshot for Blender output quality."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import (
    MoonshotExperiment,
    MoonshotRequest,
    MoonshotResult,
    bundle_result,
    error_result,
    skipped_result,
)
from .papers import LEWINER_MC, MARCHING_CUBES, SCREENED_POISSON
from .support import (
    bounded,
    candidate_rows,
    per_view_boundary_iou,
    row_failures,
    row_metric,
    topology_payload,
)


EXPERIMENT = MoonshotExperiment(
    experiment_id="editable_retopology",
    title="Topology-aware editable retopology pass",
    subsystem="mesh_generation",
    hypothesis=(
        "Mesh extraction should optimize for Blender editability, not only silhouette agreement; "
        "quad-friendly topology and component sanity can be scored as first-class outputs."
    ),
    expected_wins={
        "editability": "cleaner components, fewer non-manifold artifacts, better sculpt/readiness",
        "reliability": "explicit failure when visual quality is mesh-hostile",
    },
    required_inputs=("mesh_candidate", "topology_report", "shape_program_or_primitive_proxy"),
    validation_metrics=("manifoldness", "watertightness", "component_sanity", "editable_index"),
    papers=(MARCHING_CUBES, LEWINER_MC, SCREENED_POISSON),
)


def run(request: MoonshotRequest) -> MoonshotResult:
    try:
        rows = candidate_rows(request.candidate)
        if not rows:
            return skipped_result(
                request,
                reason="editable retopology needs a mesh candidate or topology report",
                next_steps=("run after a mesh-producing backend emits candidate metrics",),
            )
        return _retopology_report(request, rows[0])
    except Exception as exc:
        return error_result(request, error=f"{type(exc).__name__}: {exc}")


def _retopology_report(
    request: MoonshotRequest,
    row: Mapping[str, Any],
) -> MoonshotResult:
    topology = dict(topology_payload(row))
    defects = _defects(topology, row)
    before = _topology_score(topology, row)
    repair_plan = _repair_plan(defects)
    phase_plan = _phase_plan(defects, row)
    repair_gain = sum(float(item.get("estimated_score_gain", 0.0)) for item in repair_plan)
    silhouette_risk = _silhouette_risk(row, defects)
    after = bounded(before + repair_gain - silhouette_risk * 0.25)
    editability_before = row_metric(row, "editability.qa_score", "editability_score", default=0.0)
    if editability_before <= 0.0:
        editability_before = bounded(0.45 + before * 0.35 - len(defects) * 0.04)
    editability_after = bounded(editability_before + repair_gain * 0.8 - silhouette_risk * 0.15)
    evidence = {
        "topology": topology,
        "defects": defects,
        "repair_plan": repair_plan,
        "phase_plan": phase_plan,
        "acceptance_gates": _acceptance_gates(row, silhouette_risk),
        "qa": {
            "topology_score_before": before,
            "topology_score_after_estimate": after,
            "editability_score_before": editability_before,
            "editability_score_after_estimate": editability_after,
            "silhouette_risk": silhouette_risk,
            "safe_automatic": silhouette_risk < 0.4 and len(defects) <= 4,
            "human_review_required": silhouette_risk >= 0.4 or any(
                defect.get("severity") == "critical" for defect in defects
            ),
        },
    }
    metrics = {
        "ran": 1.0,
        "defect_count": float(len(defects)),
        "topology_score_before": before,
        "topology_score_after_estimate": after,
        "topology_score_delta": after - before,
        "editability_score_delta": editability_after - editability_before,
        "silhouette_risk": silhouette_risk,
        "phase_count": float(len(phase_plan)),
        "critical_defect_count": float(sum(1 for defect in defects if defect.get("severity") == "critical")),
    }
    return bundle_result(
        request,
        status="ran",
        metrics=metrics,
        evidence=evidence,
        artifact_name="editable-retopology.json",
        next_steps=(
            "run repair actions only when safe_automatic is true or a reviewer accepts the plan",
            "rerender required views after repair and reject if silhouette metrics regress",
        ),
    )


def _defects(topology: Mapping[str, Any], row: Mapping[str, Any]) -> list[dict[str, Any]]:
    defects: list[dict[str, Any]] = []
    checks = (
        ("loose_vertices", "remove isolated loose vertices", 0.04),
        ("boundary_edges", "fill guarded boundary loops", 0.08),
        ("non_manifold_edges", "split or weld non-manifold edges", 0.10),
        ("degenerate_faces", "dissolve degenerate faces", 0.05),
        ("zero_area_faces", "dissolve zero-area faces", 0.05),
        ("internal_disconnected_shells", "separate and classify internal shells", 0.08),
    )
    for key, action, gain in checks:
        count = int(float(topology.get(key, row_metric(row, f"topology.{key}", default=0.0)) or 0.0))
        if count <= 0:
            continue
        defects.append(
            {
                "defect": key,
                "count": count,
                "severity": (
                    "critical"
                    if key in {"non_manifold_edges", "internal_disconnected_shells"} and count > 8
                    else "high"
                    if key in {"non_manifold_edges", "boundary_edges"}
                    else "medium"
                ),
                "recommended_action": action,
                "estimated_score_gain": min(0.18, gain + min(count, 20) * 0.002),
            }
        )
    topology_score = _topology_score(topology, row)
    if topology_score < 0.75:
        defects.append(
            {
                "defect": "topology_below_floor",
                "count": 1,
                "severity": "high",
                "recommended_action": "rerun extraction with topology repair policy enabled",
                "estimated_score_gain": min(0.20, 0.75 - topology_score),
            }
        )
    face_count = int(float(topology.get("faces", topology.get("face_count", 0)) or 0.0))
    if face_count > 250_000:
        defects.append(
            {
                "defect": "mesh_too_dense_for_editing",
                "count": face_count,
                "severity": "medium",
                "recommended_action": "apply quad-friendly decimation before export QA",
                "estimated_score_gain": 0.06,
            }
        )
    component_count = int(float(topology.get("connected_components", 1) or 1.0))
    if component_count > 1:
        defects.append(
            {
                "defect": "multiple_components",
                "count": component_count,
                "severity": "high",
                "recommended_action": "classify components as semantic parts before joining or pruning",
                "estimated_score_gain": min(0.16, component_count * 0.025),
            }
        )
    failures = set(row_failures(row))
    if "proxy_render_disagreement" in failures or "axis_or_transform_suspect" in failures:
        defects.append(
            {
                "defect": "render_proxy_alignment_risk",
                "count": 1,
                "severity": "high",
                "recommended_action": "run alignment-preserving retopology and rerender before promotion",
                "estimated_score_gain": 0.04,
            }
        )
    return defects


def _repair_plan(defects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan = []
    for index, defect in enumerate(defects):
        plan.append(
            {
                "step": index + 1,
                "action": defect["recommended_action"],
                "targets": [defect["defect"]],
                "estimated_score_gain": defect["estimated_score_gain"],
                "guard": "preserve render.min_view_iou and boundary_iou_mean",
            }
        )
    if not plan:
        plan.append(
            {
                "step": 1,
                "action": "no repair needed; run export round-trip QA",
                "targets": [],
                "estimated_score_gain": 0.0,
                "guard": "do not change mesh topology",
            }
        )
    return plan


def _phase_plan(
    defects: list[dict[str, Any]],
    row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    boundary = per_view_boundary_iou(row)
    weakest_boundary = min(boundary.values()) if boundary else None
    phases = [
        {
            "phase": "preflight",
            "actions": [
                "snapshot mesh statistics",
                "record selected backend and source primitive/volume artifacts",
            ],
            "exit_gate": "all source artifacts remain addressable",
        },
        {
            "phase": "repair",
            "actions": [str(defect["recommended_action"]) for defect in defects]
            or ["no topology mutation; export round-trip QA only"],
            "exit_gate": "topology.score and watertightness do not regress",
        },
        {
            "phase": "silhouette_guard",
            "actions": [
                "rerender required views",
                "compare boundary IoU against pre-repair baseline",
            ],
            "weakest_boundary_iou": weakest_boundary,
            "exit_gate": "render.min_view_iou and boundary_iou_mean clear acceptance gates",
        },
    ]
    if any(defect.get("severity") == "critical" for defect in defects):
        phases.insert(
            2,
            {
                "phase": "manual_review",
                "actions": ["review critical topology edits before export"],
                "exit_gate": "reviewer accepts semantic component changes",
            },
        )
    return phases


def _acceptance_gates(
    row: Mapping[str, Any],
    silhouette_risk: float,
) -> list[dict[str, object]]:
    min_iou = row_metric(row, "render.min_view_iou", "min_view_iou", "area_iou_min", default=0.55)
    boundary_mean = row_metric(row, "render.boundary_iou_mean", "boundary_iou_mean", default=0.0)
    if boundary_mean <= 0.0:
        boundary = per_view_boundary_iou(row)
        boundary_mean = sum(boundary.values()) / len(boundary) if boundary else 0.45
    return [
        {
            "metric": "render.min_view_iou",
            "minimum": max(0.45, min_iou - 0.01),
            "hard": True,
        },
        {
            "metric": "render.boundary_iou_mean",
            "minimum": max(0.20, boundary_mean - 0.04),
            "hard": silhouette_risk > 0.2,
        },
        {
            "metric": "editability.export_roundtrip_score",
            "minimum": 0.01,
            "hard": False,
        },
    ]


def _topology_score(topology: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    value = topology.get("topology_score", topology.get("score"))
    if value is None:
        value = row_metric(row, "topology.score", "topology_score", default=0.75)
    return bounded(float(value or 0.0), 0.0, 1.0)


def _silhouette_risk(row: Mapping[str, Any], defects: list[dict[str, Any]]) -> float:
    min_iou = row_metric(row, "render.min_view_iou", "min_view_iou", "area_iou_min", default=0.0)
    if min_iou <= 0.0:
        min_iou = 0.55
    topology_pressure = min(1.0, len(defects) / 6.0)
    return bounded((1.0 - min_iou) * 0.5 + topology_pressure * 0.35)
