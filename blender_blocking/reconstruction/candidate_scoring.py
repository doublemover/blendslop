"""Candidate scoring policies for ensemble reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .types import CandidateResult, CandidateScore, CandidateScoreTerm


@dataclass(frozen=True)
class CandidateScoreWeights:
    status_success: float = 150.0
    status_degraded_penalty: float = -200.0
    required_view_all_pass: float = 1000.0
    failed_required_view_penalty: float = -1200.0
    missing_required_metric_penalty: float = -900.0
    min_area_iou: float = 200.0
    average_iou: float = 90.0
    min_boundary_iou: float = 180.0
    mean_boundary_iou: float = 150.0
    signed_distance_loss: float = -120.0
    true_geometry_fscore: float = 180.0
    recoverable_geometry_fscore: float = 220.0
    volumetric_iou: float = 120.0
    recoverable_volumetric_iou: float = 150.0
    ambiguity_gap_penalty: float = -80.0
    topology: float = 100.0
    topology_penalty: float = -100.0
    non_manifold_penalty: float = -250.0
    boundary_edge_penalty: float = -20.0
    uncertainty: float = 80.0
    editability: float = 60.0
    export_qa: float = 90.0
    constraint: float = 60.0
    constraint_penalty: float = -120.0
    research_only_penalty: float = -250.0
    complexity_penalty: float = -30.0
    time_penalty: float = -20.0
    warning_penalty: float = -10.0
    failure_warn_penalty: float = -100.0
    failure_fail_penalty: float = -2000.0
    quality_floor_penalty: float = -850.0
    failure_penalty: float = -10000.0


POLICY_WEIGHT_PRESETS: Mapping[str, CandidateScoreWeights] = {
    "best_score": CandidateScoreWeights(),
    "quality_first": CandidateScoreWeights(
        min_area_iou=320.0,
        average_iou=120.0,
        min_boundary_iou=320.0,
        mean_boundary_iou=240.0,
        signed_distance_loss=-220.0,
        true_geometry_fscore=260.0,
        recoverable_geometry_fscore=320.0,
        volumetric_iou=180.0,
        recoverable_volumetric_iou=220.0,
        editability=70.0,
        time_penalty=-8.0,
    ),
    "editability_first": CandidateScoreWeights(
        editability=320.0,
        export_qa=220.0,
        topology=220.0,
        topology_penalty=-220.0,
        non_manifold_penalty=-500.0,
        complexity_penalty=-80.0,
        min_area_iou=160.0,
        min_boundary_iou=160.0,
    ),
    "fast_preview": CandidateScoreWeights(
        time_penalty=-90.0,
        min_area_iou=140.0,
        average_iou=70.0,
        min_boundary_iou=90.0,
        mean_boundary_iou=80.0,
        editability=30.0,
        true_geometry_fscore=40.0,
        recoverable_geometry_fscore=60.0,
        status_degraded_penalty=-120.0,
    ),
    "research_explore": CandidateScoreWeights(
        research_only_penalty=-25.0,
        true_geometry_fscore=220.0,
        recoverable_geometry_fscore=260.0,
        min_boundary_iou=240.0,
        signed_distance_loss=-180.0,
        editability=140.0,
        warning_penalty=-4.0,
        failure_warn_penalty=-40.0,
    ),
}
POLICY_WEIGHT_PRESETS = {
    **POLICY_WEIGHT_PRESETS,
    "fidelity": POLICY_WEIGHT_PRESETS["quality_first"],
    "research_fidelity": POLICY_WEIGHT_PRESETS["research_explore"],
}


def _metric_bool(metrics: Mapping[str, Any], key: str, default: bool) -> bool:
    value = metrics.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "passed"}
    return bool(value)


def _required_views_pass(result: CandidateResult) -> float:
    per_view = result.metric_result.per_view
    if not per_view:
        return 0.0
    for metrics in per_view.values():
        if not isinstance(metrics, Mapping):
            return 0.0
        required = _metric_bool(metrics, "required", True)
        passed = _metric_bool(metrics, "passed", _metric_bool(metrics, "pass", False))
        if required and not passed:
            return 0.0
    return 1.0


def score_candidate(
    result: CandidateResult,
    *,
    policy: str = "best_score",
    weights: CandidateScoreWeights | None = None,
) -> CandidateScore:
    """Score a candidate with transparent weighted terms."""
    policy = _normalize_policy(policy)
    weights = _weights_for_policy(policy, weights)
    terms: list[CandidateScoreTerm] = []
    bundle = _evaluation_bundle(result)
    bundle_metrics = bundle.metric_index() if bundle is not None else {}
    failure_counts = _failure_counts(bundle)
    hard_failure_count = failure_counts.get("fail", 0)
    warn_failure_count = failure_counts.get("warn", 0)
    status_success = 1.0 if result.status == "success" else 0.0
    terms.append(
        CandidateScoreTerm(
            "status_success",
            status_success,
            weights.status_success,
            "candidate reported a successful reconstruction contract",
        )
    )
    if result.status in {"failed", "skipped", "error"}:
        terms.append(
            CandidateScoreTerm(
                "failure",
                1.0,
                weights.failure_penalty,
                f"candidate status is {result.status}",
            )
        )
    if result.status == "research_only":
        terms.append(
            CandidateScoreTerm(
                "research_only",
                1.0,
                weights.research_only_penalty,
                "candidate emitted research artifact without validated reconstruction",
            )
        )
    metrics = result.metric_result
    quality_floor_failures = _quality_floor_failures(metrics)
    failed_required_views = _metric_value(
        bundle_metrics,
        "silhouette.failed_required_view_count",
        0.0,
    )
    missing_required_metrics = _metric_value(
        bundle_metrics,
        "silhouette.missing_required_metric_count",
        0.0,
    )
    terms.extend(
        [
            CandidateScoreTerm(
                "required_view_all_pass",
                _required_views_pass(result),
                weights.required_view_all_pass,
            ),
            CandidateScoreTerm(
                "failed_required_views",
                failed_required_views,
                weights.failed_required_view_penalty,
                "required-view failures cannot be hidden by average IoU",
            ),
            CandidateScoreTerm(
                "missing_required_metrics",
                missing_required_metrics,
                weights.missing_required_metric_penalty,
                "success requires Boundary IoU and signed-distance loss for required views",
            ),
            CandidateScoreTerm(
                "quality_floor_failures",
                quality_floor_failures,
                weights.quality_floor_penalty,
                "candidate fell below hard quality floor for IoU, boundary, or topology",
            ),
            CandidateScoreTerm("min_area_iou", metrics.area_iou_min, weights.min_area_iou),
            CandidateScoreTerm(
                "average_iou",
                _metric_value(bundle_metrics, "silhouette.average_iou", metrics.area_iou_mean),
                weights.average_iou,
            ),
            CandidateScoreTerm(
                "min_boundary_iou",
                _metric_value(bundle_metrics, "silhouette.min_boundary_iou", 0.0),
                weights.min_boundary_iou,
            ),
            CandidateScoreTerm(
                "mean_boundary_iou",
                metrics.boundary_iou_mean,
                weights.mean_boundary_iou,
            ),
            CandidateScoreTerm(
                "signed_distance_loss",
                _metric_value(bundle_metrics, "silhouette.mean_signed_distance_loss", 0.0),
                weights.signed_distance_loss,
            ),
            CandidateScoreTerm(
                "true_geometry_fscore",
                _metric_value(bundle_metrics, "geometry.true.fscore_tau", 0.0),
                weights.true_geometry_fscore,
            ),
            CandidateScoreTerm(
                "recoverable_geometry_fscore",
                _metric_value(bundle_metrics, "geometry.recoverable.fscore_tau", 0.0),
                weights.recoverable_geometry_fscore,
            ),
            CandidateScoreTerm(
                "volumetric_iou",
                _metric_value(bundle_metrics, "geometry.volumetric_iou", 0.0),
                weights.volumetric_iou,
            ),
            CandidateScoreTerm(
                "recoverable_volumetric_iou",
                _metric_value(bundle_metrics, "geometry.recoverable.volumetric_iou", 0.0),
                weights.recoverable_volumetric_iou,
            ),
            CandidateScoreTerm(
                "ambiguity_gap_chamfer_l2",
                _metric_value(bundle_metrics, "geometry.ambiguity_gap_chamfer_l2", 0.0),
                weights.ambiguity_gap_penalty,
            ),
            CandidateScoreTerm("topology", metrics.topology_score, weights.topology),
            CandidateScoreTerm(
                "topology_penalty",
                metrics.topology_penalty,
                weights.topology_penalty,
            ),
            CandidateScoreTerm(
                "non_manifold_edges",
                _metric_value(bundle_metrics, "topology.non_manifold_edges", 0.0),
                weights.non_manifold_penalty,
            ),
            CandidateScoreTerm(
                "boundary_edges",
                _metric_value(bundle_metrics, "topology.boundary_edges", 0.0),
                weights.boundary_edge_penalty,
            ),
            CandidateScoreTerm(
                "uncertainty_consistency",
                metrics.uncertainty_consistency,
                weights.uncertainty,
            ),
            CandidateScoreTerm(
                "constraint_score", metrics.constraint_score, weights.constraint
            ),
            CandidateScoreTerm(
                "constraint_penalty",
                metrics.constraint_penalty,
                weights.constraint_penalty,
            ),
            CandidateScoreTerm(
                "editability", metrics.editability_score, weights.editability
            ),
            CandidateScoreTerm(
                "export_qa",
                _metric_value(bundle_metrics, "export.qa_score", 0.0),
                weights.export_qa,
            ),
            CandidateScoreTerm(
                "degraded",
                1.0 if result.degraded or result.status == "degraded" else 0.0,
                weights.status_degraded_penalty,
            ),
            CandidateScoreTerm(
                "complexity_penalty",
                metrics.complexity_penalty,
                weights.complexity_penalty,
            ),
            CandidateScoreTerm(
                "time_penalty",
                metrics.elapsed_s,
                weights.time_penalty,
            ),
            CandidateScoreTerm(
                "candidate_warnings",
                float(len(result.warnings)),
                weights.warning_penalty,
            ),
            CandidateScoreTerm(
                "classified_warning_failures",
                float(warn_failure_count),
                weights.failure_warn_penalty,
            ),
            CandidateScoreTerm(
                "classified_hard_failures",
                float(hard_failure_count),
                weights.failure_fail_penalty,
            ),
        ]
    )
    total = sum(term.weighted for term in terms)
    return CandidateScore(
        candidate_id=result.candidate_id,
        total=total,
        terms=tuple(terms),
        policy=policy,
    )


def _quality_floor_failures(metrics: Any) -> float:
    failures = 0
    if float(getattr(metrics, "area_iou_min", 0.0) or 0.0) < 0.55:
        failures += 1
    if float(getattr(metrics, "boundary_iou_mean", 0.0) or 0.0) < 0.25:
        failures += 1
    if float(getattr(metrics, "topology_score", 0.0) or 0.0) < 0.45:
        failures += 1
    return float(failures)


def rank_candidates(
    results: Iterable[CandidateResult],
    *,
    policy: str = "best_score",
    weights: CandidateScoreWeights | None = None,
) -> list[tuple[CandidateResult, CandidateScore]]:
    policy = _normalize_policy(policy)
    weights = _weights_for_policy(policy, weights)
    scored = [
        (result, score_candidate(result, policy=policy, weights=weights))
        for result in results
    ]
    if policy == "fast_preview":
        return sorted(
            scored,
            key=lambda pair: (
                -pair[1].total,
                pair[0].metric_result.elapsed_s,
            ),
        )
    if policy == "editability_first":
        return sorted(
            scored,
            key=lambda pair: (
                -pair[1].total,
                -pair[0].metric_result.editability_score,
                pair[0].metric_result.complexity_penalty,
            ),
        )
    if policy in {"quality_first", "fidelity"}:
        return sorted(
            scored,
            key=lambda pair: (
                -pair[1].total,
                -pair[0].metric_result.area_iou_min,
                -pair[0].metric_result.area_iou_mean,
                -pair[0].metric_result.boundary_iou_mean,
                -pair[0].metric_result.topology_score,
                pair[0].metric_result.topology_penalty,
            ),
        )
    if policy == "printable":
        return sorted(
            scored,
            key=lambda pair: (
                -pair[1].total,
                -pair[0].metric_result.topology_score,
                pair[0].metric_result.topology_penalty,
                -pair[0].metric_result.area_iou_min,
                -pair[0].metric_result.editability_score,
            ),
        )
    if policy in {"research_fidelity", "research_explore"}:
        return sorted(
            scored,
            key=lambda pair: (
                -pair[1].total,
                -pair[0].metric_result.area_iou_min,
                -pair[0].metric_result.boundary_iou_mean,
            ),
        )
    if policy == "pareto":
        return sorted(
            scored,
            key=lambda pair: (
                -pair[1].total,
                -_pareto_proxy(pair[0]),
            ),
        )
    return sorted(scored, key=lambda pair: pair[1].total, reverse=True)


def select_best(
    results: Sequence[CandidateResult],
    *,
    policy: str = "best_score",
    weights: CandidateScoreWeights | None = None,
) -> tuple[CandidateResult | None, list[tuple[CandidateResult, CandidateScore]]]:
    if not results:
        return None, []
    policy = _normalize_policy(policy)
    ranked = rank_candidates(results, policy=policy, weights=weights)
    return ranked[0][0], ranked


def _normalize_policy(policy: str) -> str:
    aliases = {
        "best_score": "best_score",
        "balanced": "best_score",
        "editable": "editability_first",
        "max_editability": "editability_first",
        "quality": "quality_first",
        "quality_first": "quality_first",
        "editable_first": "editability_first",
        "fast": "fast_preview",
        "preview": "fast_preview",
        "research": "research_explore",
    }
    return aliases.get(policy, policy)


def _weights_for_policy(
    policy: str,
    weights: CandidateScoreWeights | None,
) -> CandidateScoreWeights:
    if weights is not None:
        return weights
    return POLICY_WEIGHT_PRESETS.get(policy, POLICY_WEIGHT_PRESETS["best_score"])


def _evaluation_bundle(result: CandidateResult) -> Any:
    try:
        return result.to_evaluation_bundle(repo="")
    except Exception:
        return None


def _metric_value(
    metrics: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    metric = metrics.get(key)
    if metric is None:
        return default
    value = getattr(metric, "value", metric)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _failure_counts(bundle: Any) -> Mapping[str, int]:
    counts: dict[str, int] = {"fail": 0, "warn": 0}
    if bundle is None:
        return counts
    for failure in getattr(bundle, "failures", ()) or ():
        severity = str(getattr(failure, "severity", "")).lower()
        if severity in counts:
            counts[severity] += 1
    return counts


def _status_rank(result: CandidateResult) -> int:
    if result.status == "success":
        return 0
    if result.status == "degraded":
        return 1
    if result.status == "research_only":
        return 2
    if result.status == "skipped":
        return 3
    if result.status in {"failed", "error"}:
        return 4
    return 5


def _research_status_rank(result: CandidateResult) -> int:
    if result.status == "success":
        return 0
    if result.status == "degraded":
        return 1
    if result.status == "research_only":
        return 2
    if result.status == "skipped":
        return 4
    if result.status in {"failed", "error"}:
        return 5
    return 6


def _pareto_proxy(result: CandidateResult) -> float:
    metrics = result.metric_result
    return (
        metrics.area_iou_min
        + metrics.boundary_iou_mean
        + metrics.topology_score
        + metrics.editability_score
        - metrics.topology_penalty
        - metrics.complexity_penalty
    )
