"""Autopsy-pack proposal builders for adaptive refinement."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import RefinementProposal
from .mutations import _proposal


def _autopsy_plan_proposals(payload: Mapping[str, Any]) -> tuple[RefinementProposal, ...]:
    proposals: list[RefinementProposal] = []
    for pack in _autopsy_packs(payload):
        if _mapping(pack.get("boundary_refinement_plan")):
            proposals.append(_autopsy_boundary_refinement(pack))
        if _mapping(pack.get("calibration_plan")):
            proposals.append(_autopsy_calibration_sweep(pack))
        if _mapping(pack.get("topology_repair_plan")):
            proposals.append(_autopsy_topology_repair(pack))
        if _mapping(pack.get("active_view_plan")):
            proposals.append(_autopsy_active_view(pack))
    return tuple(proposal for proposal in proposals if proposal is not None)


def _autopsy_packs(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    packs: list[Mapping[str, Any]] = []
    single = payload.get("autopsy_pack")
    if isinstance(single, Mapping):
        packs.append(single)
    many = payload.get("autopsy_packs")
    if isinstance(many, Sequence) and not isinstance(many, (str, bytes)):
        packs.extend(pack for pack in many if isinstance(pack, Mapping))
    return tuple(packs)


def _autopsy_boundary_refinement(pack: Mapping[str, Any]) -> RefinementProposal:
    plan = _mapping(pack.get("boundary_refinement_plan"))
    probes = _probe_ids(plan)
    return _proposal(
        "autopsy-boundary-refinement",
        "Autopsy boundary refinement sweep",
        "Autopsy evidence requested a boundary-first probe matrix; run mask, morphology, boundary-band, patch, and differentiable loss-weight sweeps.",
        mode="ensemble",
        cli_args=(
            "--ensemble-candidates",
            "visual_hull_voxel,primitive_fit_refine,shape_program,differentiable_refine",
            "--ensemble-policy",
            "research_fidelity",
            "--validation-mode",
            "backend-status",
            "--shape-residual-policy",
            "suggest_patches",
            "--ref-morph-close",
            "3",
            "--ref-morph-open",
            "1",
            "--vh-boundary-refine",
            "--vh-mesh-method",
            "lewiner",
            "--diff-loss-weights-json",
            '{"silhouette":1.0,"boundary":0.5,"sdf":0.45}',
            "--primitive-loss-weights-json",
            '{"silhouette":1.0,"boundary":0.45,"sdf":0.35,"surface":0.45}',
        ),
        expected_win={
            "silhouette.min_boundary_iou": "increase",
            "silhouette.mean_signed_distance_loss": "decrease",
        },
        tags=("autopsy-plan", "boundary", "content-adaptive-patches"),
        priority=8,
        risk="medium",
        source={"autopsy_pack": _pack_summary(pack), "probe_ids": probes},
    )


def _autopsy_calibration_sweep(pack: Mapping[str, Any]) -> RefinementProposal:
    plan = _mapping(pack.get("calibration_plan"))
    probes = _probe_ids(plan)
    return _proposal(
        "autopsy-calibration-sweep",
        "Autopsy calibration sweep",
        "Autopsy evidence indicates transform, view-role, bounds, or framing mismatch; run bounded visual-hull calibration probes before fitting finer geometry.",
        mode="visual_hull_voxel",
        cli_args=(
            "--validation-mode",
            "backend-status",
            "--vh-backend",
            "chunked",
            "--vh-resolution",
            "96",
            "--vh-chunk-size",
            "24",
            "--vh-mesh-method",
            "lewiner",
            "--vh-postprocess",
            "smooth_guarded",
            "--ref-morph-close",
            "3",
        ),
        expected_win={
            "diagnostics.visual_hull.axis_or_transform_suspect": "decrease",
            "silhouette.min_view_iou": "increase",
        },
        tags=("autopsy-plan", "calibration", "visual-hull-transform"),
        priority=9,
        risk="low",
        source={"autopsy_pack": _pack_summary(pack), "probe_ids": probes},
    )


def _autopsy_topology_repair(pack: Mapping[str, Any]) -> RefinementProposal:
    plan = _mapping(pack.get("topology_repair_plan"))
    return _proposal(
        "autopsy-topology-repair",
        "Autopsy topology repair pass",
        "Autopsy topology plan recommends conservative cleanup before higher-risk remeshing or Poisson postprocess.",
        mode="visual_hull_voxel",
        cli_args=(
            "--validation-mode",
            "backend-status",
            "--vh-backend",
            "chunked",
            "--vh-resolution",
            "96",
            "--vh-mesh-method",
            "lewiner",
            "--vh-postprocess",
            "topology_repair",
        ),
        expected_win={
            "topology.non_manifold_edges": "decrease",
            "topology.loose_vertices": "decrease",
            "topology.score": "increase",
        },
        tags=("autopsy-plan", "topology", "repair"),
        priority=18,
        risk="low" if bool(plan.get("safe_automatic")) else "medium",
        source={"autopsy_pack": _pack_summary(pack), "probe_ids": _probe_ids(plan)},
    )


def _autopsy_active_view(pack: Mapping[str, Any]) -> RefinementProposal:
    plan = _mapping(pack.get("active_view_plan"))
    requests = plan.get("requests", ())
    first_request = {}
    if (
        isinstance(requests, Sequence)
        and not isinstance(requests, (str, bytes))
        and requests
    ):
        if isinstance(requests[0], Mapping):
            first_request = dict(requests[0])
    return _proposal(
        "autopsy-active-view",
        "Autopsy active-view capture",
        "Autopsy evidence says current silhouettes are underconstrained; request the highest-information next view before overfitting parameters.",
        mode="ensemble",
        cli_args=(
            "--ensemble-candidates",
            "visual_hull_voxel,primitive_fit_refine,shape_program",
            "--ensemble-policy",
            "fidelity",
            "--validation-mode",
            "backend-status",
        ),
        expected_win={
            "geometry.ambiguity_gap": "decrease",
            "silhouette.min_view_iou": "increase",
        },
        tags=("autopsy-plan", "active-view", "human-input"),
        priority=7,
        risk="low",
        source={"autopsy_pack": _pack_summary(pack), "next_view": first_request},
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _probe_ids(plan: Mapping[str, Any]) -> tuple[str, ...]:
    probes = plan.get("probes", ())
    if not isinstance(probes, Sequence) or isinstance(probes, (str, bytes)):
        return ()
    return tuple(
        str(probe.get("probe_id", ""))
        for probe in probes
        if isinstance(probe, Mapping) and probe.get("probe_id")
    )


def _pack_summary(pack: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "candidate_id": pack.get("candidate_id", ""),
        "status": pack.get("status", ""),
        "failures": list(pack.get("failures", ()) or ()),
    }
