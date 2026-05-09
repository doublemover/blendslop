"""Objective scoring and ranking for refinement experiment results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .contracts import ExperimentResult, json_safe


OBJECTIVES = {
    "quality_win",
    "min_view_iou",
    "mean_iou",
    "profile_editable",
    "visual_hull_alignment",
    "fast_preview",
    "human_adjusted",
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


def score_result(
    result: ExperimentResult,
    *,
    objective: str = "quality_win",
    labels: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown refinement objective: {objective}")
    if objective == "min_view_iou":
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
    total = sum(term.weighted for term in terms)
    return {
        "objective": objective,
        "total": total,
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


def metric_value(result: ExperimentResult, key: str, default: float = 0.0) -> float:
    value = result.metrics.get(key)
    if value is None:
        backend_metrics = _selected_metric_result(result)
        value = backend_metrics.get(key) if isinstance(backend_metrics, Mapping) else None
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default


def required_views_all_pass(result: ExperimentResult) -> float:
    if result.status != "pass":
        return 0.0
    for view in ("front", "side", "top"):
        value = result.view_iou(view)
        if value is None or value < 0.7:
            return 0.0
    return 1.0


def catastrophic_view_failure(result: ExperimentResult) -> float:
    for view in ("front", "side", "top"):
        value = result.view_iou(view)
        if value is None or value < 0.2:
            return 1.0
    return 0.0


def missing_required_metrics(result: ExperimentResult) -> float:
    render_iou_mode = result.metrics.get("validation_mode") == "render-iou"
    has_any_view = any(result.view_iou(view) is not None for view in ("front", "side", "top"))
    if render_iou_mode and not has_any_view:
        return 1.0
    if not has_any_view and result.status == "pass":
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
    if any(result.view_iou(view) is not None for view in ("front", "side", "top")):
        return 0.0
    if metric_value(result, "area_iou_mean") > 0.9:
        return 1.0
    return 0.0


def _quality_terms(result: ExperimentResult) -> list[ScoreTerm]:
    return [
        ScoreTerm("required_views_all_pass", required_views_all_pass(result), 1000.0),
        ScoreTerm("min_view_iou", result.min_iou, 400.0),
        ScoreTerm("average_iou", result.avg_iou, 250.0),
        ScoreTerm("boundary_iou_mean", metric_value(result, "boundary_iou_mean"), 150.0),
        ScoreTerm("topology_score", metric_value(result, "topology_score"), 120.0),
        ScoreTerm("editability_score", metric_value(result, "editability_score"), 80.0),
        ScoreTerm("complexity_penalty", metric_value(result, "complexity_penalty"), -60.0),
        ScoreTerm("elapsed_s", min(result.elapsed_s, 30.0), -10.0),
        ScoreTerm("catastrophic_view_failure", catastrophic_view_failure(result), -250.0),
        ScoreTerm("missing_required_metrics", missing_required_metrics(result), -100.0),
        ScoreTerm("artifact_escape", artifact_escape(result), -100.0),
        ScoreTerm("metric_only_candidate", metric_only_candidate(result), -75.0),
    ]


def _min_view_terms(result: ExperimentResult) -> list[ScoreTerm]:
    return [
        ScoreTerm("min_view_iou", result.min_iou, 1000.0),
        ScoreTerm("average_iou", result.avg_iou, 200.0),
        ScoreTerm("topology_score", metric_value(result, "topology_score"), 100.0),
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
        ScoreTerm("topology_score", metric_value(result, "topology_score"), 200.0),
        ScoreTerm("editability_score", metric_value(result, "editability_score"), 160.0),
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
