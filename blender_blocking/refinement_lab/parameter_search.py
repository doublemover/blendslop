"""Objective scoring and ranking for refinement experiment results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

try:
    from blender_blocking.metrics.namespaces import (
        get_metric_path,
        has_complete_required_render_views,
        optional_float,
        render_view_iou,
        required_render_metrics_missing,
    )
except ImportError:  # pragma: no cover - script-style imports
    from metrics.namespaces import (
        get_metric_path,
        has_complete_required_render_views,
        optional_float,
        render_view_iou,
        required_render_metrics_missing,
    )

from .contracts import ExperimentResult, json_safe

try:
    from .preset_catalog import get_track_preset
except ImportError:  # pragma: no cover - script-style imports
    from preset_catalog import get_track_preset


OBJECTIVES = {
    "quality_win",
    "reliability_first",
    "min_view_iou",
    "mean_iou",
    "profile_editable",
    "visual_hull_alignment",
    "fast_preview",
    "human_adjusted",
}

_BACKEND_FAILURE_STATUSES = {"failed", "error", "skipped"}
_PROMOTION_TIER_RANK = {
    "blocked": 0,
    "metric_only": 1,
    "research_only": 2,
    "unverified": 3,
    "degraded": 4,
    "promotable": 5,
}


@dataclass(frozen=True)
class ScoreTerm:
    name: str
    value: float
    weight: float
    description: str = ""

    @property
    def weighted(self) -> float:
        return self.value * self.weight

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "weight": self.weight,
            "weighted": self.weighted,
            "description": self.description,
        }


@dataclass(frozen=True)
class PromotionDecision:
    tier: str
    promotable: bool
    requires_review: bool
    backend_status: str
    backend_degraded: bool
    state: str
    blockers: tuple[str, ...] = ()

    @property
    def rank(self) -> int:
        return _PROMOTION_TIER_RANK.get(self.tier, 0)

    def to_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "promotable": self.promotable,
            "requires_review": self.requires_review,
            "backend_status": self.backend_status,
            "backend_degraded": self.backend_degraded,
            "state": self.state,
            "blockers": list(self.blockers),
            "rank": self.rank,
        }


def score_result(
    result: ExperimentResult,
    *,
    objective: str = "quality_win",
    labels: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown refinement objective: {objective}")
    if objective == "reliability_first":
        terms = _reliability_first_terms(result)
    elif objective == "min_view_iou":
        terms = _min_view_terms(result)
    elif objective == "mean_iou":
        terms = _mean_iou_terms(result)
    elif objective == "profile_editable":
        terms = _profile_editable_terms(result)
    elif objective == "visual_hull_alignment":
        terms = _visual_hull_alignment_terms(result)
    elif objective == "fast_preview":
        terms = _fast_preview_terms(result)
    elif objective == "human_adjusted":
        terms = _quality_terms(result) + _human_label_terms(labels or ())
    else:
        terms = _quality_terms(result)
    terms = terms + _promotion_terms(result)
    total = sum(term.weighted for term in terms)
    if objective == "reliability_first" and not promotion_decision(result).promotable:
        total = 0.0
    return {
        "objective": objective,
        "total": total,
        "promotion": promotion_decision(result).to_dict(),
        "terms": [term.to_dict() for term in terms],
    }


def rank_results(
    results: Iterable[ExperimentResult],
    *,
    objective: str = "quality_win",
    labels_by_variant: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> list[tuple[ExperimentResult, dict[str, object]]]:
    scored = [
        (
            result,
            score_result(
                result,
                objective=objective,
                labels=(labels_by_variant or {}).get(result.variant_id, ()),
            ),
        )
        for result in results
    ]
    return sorted(
        scored,
        key=lambda pair: (
            float(pair[1]["total"]),
            promotion_decision(pair[0]).rank,
            pair[0].min_iou,
            pair[0].avg_iou,
            -pair[0].elapsed_s,
        ),
        reverse=True,
    )


def top_k(
    results: Iterable[ExperimentResult],
    *,
    k: int,
    objective: str = "quality_win",
) -> list[tuple[ExperimentResult, dict[str, object]]]:
    return rank_results(results, objective=objective)[: max(1, int(k))]


def promotion_decision(result: ExperimentResult) -> PromotionDecision:
    backend_status = _backend_status(result)
    backend_degraded = _backend_degraded(result)
    metric_only = metric_only_candidate(result) > 0.0
    blockers: list[str] = []

    if result.status != "pass":
        blockers.append(f"result_status:{result.status}")
    if backend_status in _BACKEND_FAILURE_STATUSES:
        blockers.append(f"backend_status:{backend_status}")
    if backend_status == "research_only":
        blockers.append("research_only_backend")
    if backend_degraded or backend_status == "degraded":
        blockers.append("degraded_backend")
    if metric_only:
        blockers.append("metric_only_candidate")
    if missing_required_metrics(result) > 0.0:
        blockers.append("missing_required_metrics")
    if catastrophic_view_failure(result) > 0.0:
        blockers.append("catastrophic_view_failure")
    if proxy_render_namespace_violation(result) > 0.0:
        blockers.append("proxy_render_namespace_violation")
    if proxy_render_disagreement(result) > 0.0:
        blockers.append("proxy_render_disagreement")
    topology_blocker = _topology_blocker(result)
    if topology_blocker:
        blockers.append(topology_blocker)
    editability_blocker = _editability_blocker(result)
    if editability_blocker:
        blockers.append(editability_blocker)
    if backend_status == "unreported":
        blockers.append("unreported_backend_status")

    hard_blocked = any(
        item.startswith("result_status:")
        or item.startswith("backend_status:")
        or item
        in {
            "missing_required_metrics",
            "catastrophic_view_failure",
            "topology_below_floor",
            "topology_not_watertight",
            "topology_boundary_edges",
            "topology_non_manifold_edges",
            "editability_roundtrip_missing",
            "proxy_render_namespace_violation",
            "proxy_render_disagreement",
        }
        for item in blockers
    )
    if hard_blocked:
        tier = "blocked"
    elif metric_only:
        tier = "metric_only"
    elif backend_status == "research_only":
        tier = "research_only"
    elif backend_degraded or backend_status == "degraded":
        tier = "degraded"
    elif backend_status == "unreported":
        tier = "unverified"
    else:
        tier = "promotable"
    state = _promotion_state(
        tier=tier,
        backend_status=backend_status,
        blockers=blockers,
        result=result,
    )

    return PromotionDecision(
        tier=tier,
        promotable=tier == "promotable",
        requires_review=tier != "promotable",
        backend_status=backend_status,
        backend_degraded=backend_degraded,
        state=state,
        blockers=tuple(blockers),
    )


def metric_value(result: ExperimentResult, key: str, default: float = 0.0) -> float:
    value = get_metric_path(result.metrics, key)
    if value is None:
        value = result.metrics.get(
            {
                "render.average_iou": "average_iou",
                "render.min_view_iou": "min_view_iou",
                "render.boundary_iou_mean": "boundary_iou_mean",
                "render.signed_distance_loss_mean": "signed_distance_loss_mean",
                "topology.score": "topology_score",
                "editability.qa_score": "editability_score",
                "editability.export_roundtrip_score": "export_roundtrip_score",
                "cost.total_wall_ms": "cost_total_wall_ms",
            }.get(key, "")
        )
    if value is None:
        backend_metrics = _selected_metric_result(result)
        value = (
            get_metric_path(backend_metrics, key)
            if isinstance(backend_metrics, Mapping)
            else None
        )
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default


def required_views_all_pass(result: ExperimentResult) -> float:
    if result.status != "pass":
        return 0.0
    if not has_complete_required_render_views(result.metrics):
        return 0.0
    for view in ("front", "side", "top"):
        value = render_view_iou(result.metrics, view)
        if value is None or value < 0.7:
            return 0.0
    return 1.0


def catastrophic_view_failure(result: ExperimentResult) -> float:
    for view in ("front", "side", "top"):
        value = render_view_iou(result.metrics, view)
        if value is None or value < 0.2:
            return 1.0
    return 0.0


def missing_required_metrics(result: ExperimentResult) -> float:
    if result.status == "pass" and required_render_metrics_missing(result.metrics):
        return 1.0
    return 0.0


def artifact_escape(result: ExperimentResult) -> float:
    autopsy = result.autopsy or {}
    category = str(autopsy.get("category", ""))
    if category == "artifact_escape":
        return 1.0
    findings = autopsy.get("findings", ())
    if isinstance(findings, Sequence):
        for finding in findings:
            if isinstance(finding, Mapping) and finding.get("category") == "artifact_escape":
                return 1.0
    return 0.0


def metric_only_candidate(result: ExperimentResult) -> float:
    if any(render_view_iou(result.metrics, view) is not None for view in ("front", "side", "top")):
        return 0.0
    backend_mean = get_metric_path(result.metrics, "backend.area_iou_mean")
    if backend_mean is None:
        backend_mean = metric_value(result, "area_iou_mean")
    if optional_float(backend_mean) and float(backend_mean) > 0.9:
        return 1.0
    return 0.0


def proxy_render_namespace_violation(result: ExperimentResult) -> float:
    backend_min = optional_float(get_metric_path(result.metrics, "backend.area_iou_min"))
    render_min = optional_float(get_metric_path(result.metrics, "render.min_view_iou"))
    if render_min is None:
        render_min = result.min_iou if result.min_iou > 0.0 else None
    if backend_min is None:
        return 0.0
    if backend_min >= 0.9 and (render_min is None or render_min <= 0.2):
        return 1.0
    return 0.0


def proxy_render_disagreement(result: ExperimentResult) -> float:
    backend_min = optional_float(get_metric_path(result.metrics, "backend.area_iou_min"))
    if backend_min is None or backend_min < 0.95:
        return 0.0
    render_min = optional_float(get_metric_path(result.metrics, "render.min_view_iou"))
    if render_min is None:
        render_min = result.min_iou if result.min_iou > 0.0 else None
    boundary_min = optional_float(get_metric_path(result.metrics, "render.boundary_iou_min"))
    if boundary_min is None:
        boundary_values = [
            optional_float(get_metric_path(result.metrics, f"render.per_view.{view}.boundary_iou"))
            for view in ("front", "side", "top")
        ]
        present = [value for value in boundary_values if value is not None]
        boundary_min = min(present) if present else None
    if render_min is not None and 0.2 < render_min < 0.85:
        return 1.0
    if boundary_min is not None and boundary_min < 0.25:
        return 1.0
    return 0.0


def _reliability_first_terms(result: ExperimentResult) -> list[ScoreTerm]:
    decision = promotion_decision(result)
    if not decision.promotable:
        return [
            ScoreTerm(
                "promotion_required",
                0.0,
                1.0,
                "blocked candidates receive a hard zero reliability score",
            ),
            ScoreTerm(
                "blocking_state",
                1.0,
                0.0,
                decision.state,
            ),
        ]
    return [
        ScoreTerm("promotion_ready", 1.0, 1000.0),
        ScoreTerm("min_view_iou", result.min_iou, 800.0),
        ScoreTerm("boundary_iou_mean", metric_value(result, "render.boundary_iou_mean"), 450.0),
        ScoreTerm("signed_distance_loss", metric_value(result, "render.signed_distance_loss_mean"), -250.0),
        ScoreTerm("topology_score", metric_value(result, "topology.score"), 220.0),
        ScoreTerm("editability_score", metric_value(result, "editability.qa_score"), 160.0),
        ScoreTerm("elapsed_s", min(result.elapsed_s, 60.0), -4.0),
    ]


def _quality_terms(result: ExperimentResult) -> list[ScoreTerm]:
    return [
        ScoreTerm("required_views_all_pass", required_views_all_pass(result), 1000.0),
        ScoreTerm("min_view_iou", result.min_iou, 400.0),
        ScoreTerm("average_iou", result.avg_iou, 250.0),
        ScoreTerm("boundary_iou_mean", metric_value(result, "render.boundary_iou_mean"), 150.0),
        ScoreTerm("topology_score", metric_value(result, "topology.score"), 120.0),
        ScoreTerm("editability_score", metric_value(result, "editability.qa_score"), 80.0),
        ScoreTerm("complexity_penalty", metric_value(result, "complexity_penalty"), -60.0),
        ScoreTerm("elapsed_s", min(result.elapsed_s, 30.0), -10.0),
        ScoreTerm("catastrophic_view_failure", catastrophic_view_failure(result), -250.0),
        ScoreTerm("missing_required_metrics", missing_required_metrics(result), -100.0),
        ScoreTerm("artifact_escape", artifact_escape(result), -100.0),
        ScoreTerm(
            "proxy_render_namespace_violation",
            proxy_render_namespace_violation(result),
            -2000.0,
        ),
        ScoreTerm(
            "proxy_render_disagreement",
            proxy_render_disagreement(result),
            -1600.0,
        ),
    ]


def _min_view_terms(result: ExperimentResult) -> list[ScoreTerm]:
    return [
        ScoreTerm("min_view_iou", result.min_iou, 1000.0),
        ScoreTerm("average_iou", result.avg_iou, 200.0),
        ScoreTerm("topology_score", metric_value(result, "topology.score"), 100.0),
        ScoreTerm("missing_required_metrics", missing_required_metrics(result), -500.0),
    ]


def _mean_iou_terms(result: ExperimentResult) -> list[ScoreTerm]:
    return [
        ScoreTerm("average_iou", result.avg_iou, 1000.0),
        ScoreTerm("min_view_iou", result.min_iou, 150.0),
        ScoreTerm("catastrophic_view_failure", catastrophic_view_failure(result), -300.0),
    ]


def _profile_editable_terms(result: ExperimentResult) -> list[ScoreTerm]:
    return [
        ScoreTerm("min_view_iou", result.min_iou, 500.0),
        ScoreTerm("average_iou", result.avg_iou, 350.0),
        ScoreTerm("topology_score", metric_value(result, "topology.score"), 200.0),
        ScoreTerm("editability_score", metric_value(result, "editability.qa_score"), 160.0),
        ScoreTerm("complexity_penalty", metric_value(result, "complexity_penalty"), -75.0),
        ScoreTerm("catastrophic_view_failure", catastrophic_view_failure(result), -500.0),
    ]


def _visual_hull_alignment_terms(result: ExperimentResult) -> list[ScoreTerm]:
    bounds = result.bounds_debug or {}
    axis = bounds.get("axis_permutation_search", {}) if isinstance(bounds, Mapping) else {}
    best = axis.get("best", {}) if isinstance(axis, Mapping) else {}
    comparisons = bounds.get("comparisons", {}) if isinstance(bounds, Mapping) else {}
    return [
        ScoreTerm("axis_search_best_min_iou", _float(best.get("min_iou")), 500.0),
        ScoreTerm("axis_search_best_average_iou", _float(best.get("average_iou")), 250.0),
        ScoreTerm("baseline_average_iou", result.avg_iou, 250.0),
        ScoreTerm("baseline_min_iou", result.min_iou, 200.0),
        ScoreTerm(
            "mesh_target_bounds_agreement",
            _float(_nested(comparisons, ("mesh_vs_target_bounds", "agreement")), 0.0),
            150.0,
        ),
        ScoreTerm("mesh_missing", 1.0 if "mesh" not in result.artifacts and "mesh_obj" not in result.artifacts else 0.0, -300.0),
        ScoreTerm("catastrophic_view_failure", catastrophic_view_failure(result), -100.0),
    ]


def _fast_preview_terms(result: ExperimentResult) -> list[ScoreTerm]:
    acceptable = 1.0 if result.min_iou >= 0.7 or result.avg_iou >= 0.85 else 0.0
    return [
        ScoreTerm("acceptable_quality", acceptable, 1000.0),
        ScoreTerm("elapsed_s", min(result.elapsed_s, 60.0), -100.0),
        ScoreTerm("average_iou", result.avg_iou, 150.0),
        ScoreTerm("missing_required_metrics", missing_required_metrics(result), -500.0),
    ]


def _promotion_terms(result: ExperimentResult) -> list[ScoreTerm]:
    decision = promotion_decision(result)
    return [
        ScoreTerm(
            "promotion_ready",
            1.0 if decision.promotable else 0.0,
            500.0,
            "candidate can be promoted without capability review",
        ),
        ScoreTerm(
            "result_not_pass",
            1.0 if result.status != "pass" else 0.0,
            -5000.0,
            "experiment runner did not produce a passing result",
        ),
        ScoreTerm(
            "backend_failed_or_skipped",
            1.0 if decision.backend_status in _BACKEND_FAILURE_STATUSES else 0.0,
            -5000.0,
            "backend result is failed, error, or skipped",
        ),
        ScoreTerm(
            "metric_only_candidate",
            metric_only_candidate(result),
            -1800.0,
            "candidate has aggregate proxy metrics but no required per-view render evidence",
        ),
        ScoreTerm(
            "research_only_candidate",
            1.0 if decision.backend_status == "research_only" else 0.0,
            -1600.0,
            "candidate produced a research artifact without validated editable reconstruction",
        ),
        ScoreTerm(
            "degraded_candidate",
            1.0 if decision.backend_degraded or decision.backend_status == "degraded" else 0.0,
            -900.0,
            "candidate explicitly reported degraded reconstruction quality or capability",
        ),
        ScoreTerm(
            "unverified_backend_status",
            1.0 if decision.backend_status == "unreported" else 0.0,
            -300.0,
            "candidate did not expose backend CandidateResult status",
        ),
    ]


def _human_label_terms(labels: Sequence[Mapping[str, Any]]) -> list[ScoreTerm]:
    weights = {
        "sculptable": 120.0,
        "useful_proxy": 80.0,
        "too_blobby": -60.0,
        "wrong_silhouette": -160.0,
        "bad_topology": -140.0,
        "overfit_silhouette": -50.0,
        "underfit_silhouette": -70.0,
        "reject": -300.0,
    }
    terms = []
    for label in labels:
        name = str(label.get("label", ""))
        score = _float(label.get("score"), 3.0)
        if name in weights:
            terms.append(
                ScoreTerm(
                    f"human_label_{name}",
                    score / 5.0,
                    weights[name],
                    json_safe(label).get("notes", "") if isinstance(json_safe(label), Mapping) else "",
                )
            )
    return terms


def _selected_metric_result(result: ExperimentResult) -> Mapping[str, Any]:
    backend = result.backend_result or {}
    selected = backend.get("selected") if isinstance(backend, Mapping) else None
    if isinstance(selected, Mapping):
        metric = selected.get("metric_result")
        return metric if isinstance(metric, Mapping) else {}
    metric = backend.get("metric_result") if isinstance(backend, Mapping) else None
    return metric if isinstance(metric, Mapping) else {}


def _backend_source(result: ExperimentResult) -> Mapping[str, Any]:
    backend = result.backend_result or {}
    if not isinstance(backend, Mapping):
        return {}
    selected = backend.get("selected")
    if isinstance(selected, Mapping):
        return selected
    return backend


def _backend_status(result: ExperimentResult) -> str:
    source = _backend_source(result)
    status = source.get("status") if isinstance(source, Mapping) else None
    if status is None and isinstance(result.backend_result, Mapping):
        status = result.backend_result.get("status")
    if status is None:
        return "unreported"
    return str(status).strip().lower() or "unreported"


def _backend_degraded(result: ExperimentResult) -> bool:
    sources: list[Mapping[str, Any]] = []
    source = _backend_source(result)
    if source:
        sources.append(source)
    if isinstance(result.backend_result, Mapping):
        sources.append(result.backend_result)
    for item in sources:
        if _truthy(item.get("degraded")):
            return True
        degradation = item.get("degradation")
        if isinstance(degradation, Mapping) and _truthy(degradation.get("degraded")):
            return True
    return False


def _topology_blocker(result: ExperimentResult) -> str:
    topology_score = optional_float(get_metric_path(result.metrics, "topology.score"))
    if topology_score is None:
        topology_score = optional_float(result.metrics.get("topology_score"))
    if topology_score is not None and topology_score < 0.75:
        return "topology_below_floor"
    watertight = get_metric_path(result.metrics, "topology.watertight")
    if watertight is False:
        return "topology_not_watertight"
    boundary_edges = optional_float(get_metric_path(result.metrics, "topology.boundary_edges"))
    if boundary_edges is not None and boundary_edges > 0:
        return "topology_boundary_edges"
    non_manifold = optional_float(get_metric_path(result.metrics, "topology.non_manifold_edges"))
    if non_manifold is not None and non_manifold > 0:
        return "topology_non_manifold_edges"
    return ""


def _editability_blocker(result: ExperimentResult) -> str:
    if not _requires_editability_gate(result):
        return ""
    roundtrip = get_metric_path(result.metrics, "editability.export_roundtrip_score")
    if roundtrip is None:
        return ""
    value = optional_float(roundtrip)
    if value is not None and value <= 0.0:
        return "editability_roundtrip_missing"
    return ""


def _requires_editability_gate(result: ExperimentResult) -> bool:
    raw = get_metric_path(result.metrics, "editability.export_roundtrip_required")
    if raw is not None:
        return _truthy(raw)
    try:
        track = get_track_preset(result.track)
    except Exception:
        return False
    tags = {str(tag).lower() for tag in track.tags}
    return bool(tags.intersection({"editable", "printable"}))


def _promotion_state(
    *,
    tier: str,
    backend_status: str,
    blockers: Sequence[str],
    result: ExperimentResult,
) -> str:
    blocker_set = set(blockers)
    if tier == "promotable":
        return "promotable"
    if "missing_required_metrics" in blocker_set:
        if metric_only_candidate(result) > 0.0:
            return "metric_only_candidate"
        return "blocked_missing_render_metrics"
    if "catastrophic_view_failure" in blocker_set:
        return "blocked_required_view_failure"
    if any(item.startswith("topology_") for item in blocker_set):
        return "blocked_topology"
    if any(item.startswith("editability_") for item in blocker_set):
        return "blocked_editability"
    if "proxy_render_namespace_violation" in blocker_set:
        return "diagnostic_only"
    if "proxy_render_disagreement" in blocker_set:
        return "blocked_proxy_render_disagreement"
    if backend_status == "research_only":
        return "research_only"
    if "metric_only_candidate" in blocker_set:
        return "metric_only_candidate"
    if backend_status in _BACKEND_FAILURE_STATUSES:
        return "diagnostic_only"
    return tier


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "passed", "degraded"}
    return bool(value)


def _nested(data: Mapping[str, Any], keys: Sequence[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default
