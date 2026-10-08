"""Proposal selection, deduplication, and variant conversion helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .autopsy import _autopsy_plan_proposals
from ..contracts import ExperimentVariant
from .contracts import RefinementProposal
from .mutations import (
    _active_view_capture,
    _appearance_asset_audit,
    _boundary_first,
    _compile_or_crosscheck,
    _content_adaptive_patches,
    _proxy_grounding,
    _primitive_fit_proxy_retry,
    _shape_program_editability,
    _shape_program_render_qa_retry,
    _topology_preserving_mesh,
    _uncertainty_sweep,
    _visual_hull_resolution,
)
from .signals import _failure_codes, _metric, _metric_index, _status, ambiguity_signal


def proposals_from_bundle(
    bundle: Any,
    *,
    max_proposals: int = 8,
) -> tuple[RefinementProposal, ...]:
    metrics = _metric_index(bundle)
    failures = _failure_codes(bundle)
    proposals: list[RefinementProposal] = []

    min_iou = _metric(metrics, "silhouette.min_view_iou")
    boundary = _metric(metrics, "silhouette.mean_boundary_iou")
    editable = _metric(metrics, "editability.editable_reconstruction_index")
    topology = _metric(metrics, "topology.score")
    penalty = _metric(metrics, "topology.penalty")
    sdf_loss = _metric(metrics, "silhouette.mean_signed_distance_loss")
    fscore = _metric(metrics, "geometry.fscore_tau", default=1.0)
    coverage = _metric(metrics, "geometry.surface_coverage", default=1.0)
    uv_valid = _metric(metrics, "appearance.uv_valid", default=1.0)
    pbr_coverage = _metric(
        metrics,
        "appearance.pbr_channel_coverage_ratio",
        default=1.0,
    )
    texture_only = _metric(
        metrics,
        "appearance.attribution_texture_only_detail_score",
        default=0.0,
    )
    geometry_detail = _metric(
        metrics,
        "appearance.attribution_geometry_detail_score",
        default=1.0,
    )

    if boundary < max(0.35, min_iou - 0.15) or any("boundary" in f for f in failures):
        proposals.append(_boundary_first(metrics, failures))
    if (
        boundary < 0.55
        or sdf_loss > 0.08
        or fscore < 0.65
        or coverage < 0.72
        or any("detail" in f or "surface" in f for f in failures)
    ):
        proposals.append(_content_adaptive_patches(metrics, failures))
    if min_iou < 0.7 or any("silhouette" in f for f in failures):
        proposals.append(_visual_hull_resolution(metrics, failures))
        proposals.append(_uncertainty_sweep(metrics, failures))
    if topology < 0.85 or penalty > 0.1 or any("topology" in f for f in failures):
        proposals.append(_topology_preserving_mesh(metrics, failures))
    if editable < 0.55 or any("editable" in f or "asset" in f for f in failures):
        proposals.append(_shape_program_editability(metrics, failures))
    if (
        uv_valid <= 0.0
        or pbr_coverage < 0.75
        or texture_only > geometry_detail + 0.25
        or any("appearance" in f or "uv_" in f or "pbr" in f for f in failures)
    ):
        proposals.append(_appearance_asset_audit(metrics, failures))
    if any("metric_only" in f or "proxy" in f for f in failures):
        proposals.append(_proxy_grounding(metrics, failures))
    if _status(bundle) in {"research_only", "degraded"}:
        proposals.append(_compile_or_crosscheck(metrics, failures, _status(bundle)))
    if ambiguity_signal(metrics, failures):
        proposals.append(_active_view_capture(metrics, failures))

    deduped = _dedupe(proposals)
    return tuple(sorted(deduped, key=lambda item: item.priority)[:max_proposals])


def variants_from_bundle(
    bundle: Any,
    *,
    parent_variant_id: str = "",
    max_proposals: int = 8,
) -> tuple[ExperimentVariant, ...]:
    return tuple(
        proposal.to_variant(parent_variant_id=parent_variant_id)
        for proposal in proposals_from_bundle(bundle, max_proposals=max_proposals)
    )


def merge_proposals(
    proposals: Sequence[RefinementProposal],
    *,
    max_proposals: int = 8,
) -> tuple[RefinementProposal, ...]:
    by_id: dict[str, RefinementProposal] = {}
    for proposal in proposals:
        current = by_id.get(proposal.proposal_id)
        if current is None or proposal.priority < current.priority:
            by_id[proposal.proposal_id] = proposal
    return tuple(
        sorted(by_id.values(), key=lambda proposal: proposal.priority)[:max_proposals]
    )


def proposals_from_result_payload(
    payload: Mapping[str, Any],
    *,
    max_proposals: int = 8,
) -> tuple[RefinementProposal, ...]:
    proposals: list[RefinementProposal] = list(_autopsy_plan_proposals(payload))
    bundles = payload.get("evaluation_bundles")
    if isinstance(bundles, Sequence) and not isinstance(bundles, (str, bytes)):
        for bundle in bundles:
            proposals.extend(proposals_from_bundle(bundle, max_proposals=max_proposals))
        return tuple(
            sorted(_dedupe(proposals), key=lambda item: item.priority)[:max_proposals]
        )
    bundle = payload.get("evaluation_bundle")
    if isinstance(bundle, Mapping):
        proposals.extend(proposals_from_bundle(bundle, max_proposals=max_proposals))
        return tuple(
            sorted(_dedupe(proposals), key=lambda item: item.priority)[:max_proposals]
        )
    backend = payload.get("backend_result")
    if isinstance(backend, Mapping):
        proposals.extend(
            _fallback_proposals_from_backend(backend, max_proposals=max_proposals)
        )
    return tuple(
        sorted(_dedupe(proposals), key=lambda item: item.priority)[:max_proposals]
    )


def _fallback_proposals_from_backend(
    backend: Mapping[str, Any],
    *,
    max_proposals: int,
) -> tuple[RefinementProposal, ...]:
    autopsy_proposals = list(_autopsy_plan_proposals(backend))
    status = str(backend.get("status", ""))
    metrics = backend.get("metric_result", {})
    metric_map = metrics if isinstance(metrics, Mapping) else {}
    backend_name = _backend_name(backend)
    backend_text = _backend_message_text(backend)
    if (
        backend_name == "primitive_fit_refine"
        and status == "degraded"
        and (
            _metric(metric_map, "area_iou_min", default=1.0) < 0.55
            or "proxy" in backend_text
            or "min iou" in backend_text
        )
    ):
        autopsy_proposals.append(
            _primitive_fit_proxy_retry(
                metric_map,
                ("primitive_fit_proxy_floor_degraded",),
            )
        )
    if (
        backend_name == "shape_program"
        and status in {"research_only", "degraded"}
        and _shape_program_render_qa_missing(backend)
    ):
        autopsy_proposals.append(
            _shape_program_render_qa_retry(
                metric_map,
                ("shape_program_missing_render_qa",),
            )
        )
    nested_bundles = backend.get("evaluation_bundles")
    if isinstance(nested_bundles, Sequence) and not isinstance(
        nested_bundles, (str, bytes)
    ):
        autopsy_proposals.extend(
            proposals_from_result_payload(
                {"evaluation_bundles": nested_bundles},
                max_proposals=max_proposals,
            )
        )
        return tuple(
            sorted(_dedupe(autopsy_proposals), key=lambda item: item.priority)[
                :max_proposals
            ]
        )
    nested_bundle = backend.get("evaluation_bundle")
    if isinstance(nested_bundle, Mapping):
        autopsy_proposals.extend(
            proposals_from_bundle(nested_bundle, max_proposals=max_proposals)
        )
        return tuple(
            sorted(_dedupe(autopsy_proposals), key=lambda item: item.priority)[
                :max_proposals
            ]
        )

    fake_bundle = {
        "status": status,
        "metric_groups": (),
        "failures": (),
        "metrics": metric_map,
    }
    autopsy_proposals.extend(
        proposals_from_bundle(fake_bundle, max_proposals=max_proposals)
    )
    return tuple(
        sorted(_dedupe(autopsy_proposals), key=lambda item: item.priority)[
            :max_proposals
        ]
    )


def _backend_name(backend: Mapping[str, Any]) -> str:
    selected = backend.get("selected")
    source = selected if isinstance(selected, Mapping) else backend
    for key in ("backend_name", "name", "mode"):
        value = source.get(key)
        if value:
            return str(value)
    return ""


def _backend_message_text(backend: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("warnings", "errors", "degradation_reasons", "degraded_reasons"):
        value = backend.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            parts.extend(str(item) for item in value if item)
    return " ".join(parts).lower()


def _shape_program_render_qa_missing(backend: Mapping[str, Any]) -> bool:
    metrics = backend.get("metric_result")
    extras = metrics.get("extras") if isinstance(metrics, Mapping) else None
    render_qa = extras.get("render_qa") if isinstance(extras, Mapping) else None
    if not isinstance(render_qa, Mapping):
        return True
    if render_qa.get("missing_required_metrics") is True:
        return True
    return str(render_qa.get("status", "")).lower() in {
        "",
        "missing",
        "not_applicable",
        "incomplete",
    }


def _dedupe(proposals: Sequence[RefinementProposal]) -> tuple[RefinementProposal, ...]:
    by_title: dict[str, RefinementProposal] = {}
    for proposal in proposals:
        current = by_title.get(proposal.title)
        if current is None or proposal.priority < current.priority:
            by_title[proposal.title] = proposal
    return tuple(by_title.values())
