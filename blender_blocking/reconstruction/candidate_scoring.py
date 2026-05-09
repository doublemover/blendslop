"""Candidate scoring policies for ensemble reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .types import CandidateResult, CandidateScore, CandidateScoreTerm


@dataclass(frozen=True)
class CandidateScoreWeights:
    required_view_all_pass: float = 1000.0
    min_area_iou: float = 200.0
    mean_boundary_iou: float = 150.0
    topology: float = 100.0
    topology_penalty: float = -100.0
    uncertainty: float = 80.0
    editability: float = 60.0
    constraint: float = 60.0
    constraint_penalty: float = -120.0
    degraded_penalty: float = -50.0
    complexity_penalty: float = -30.0
    time_penalty: float = -20.0
    failure_penalty: float = -10000.0


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
    weights: CandidateScoreWeights = CandidateScoreWeights(),
) -> CandidateScore:
    """Score a candidate with transparent weighted terms."""
    terms: list[CandidateScoreTerm] = []
    if result.status in {"failed", "skipped"}:
        terms.append(
            CandidateScoreTerm(
                "failure",
                1.0,
                weights.failure_penalty,
                f"candidate status is {result.status}",
            )
        )
    metrics = result.metric_result
    terms.extend(
        [
            CandidateScoreTerm(
                "required_view_all_pass",
                _required_views_pass(result),
                weights.required_view_all_pass,
            ),
            CandidateScoreTerm("min_area_iou", metrics.area_iou_min, weights.min_area_iou),
            CandidateScoreTerm(
                "mean_boundary_iou",
                metrics.boundary_iou_mean,
                weights.mean_boundary_iou,
            ),
            CandidateScoreTerm("topology", metrics.topology_score, weights.topology),
            CandidateScoreTerm(
                "topology_penalty",
                metrics.topology_penalty,
                weights.topology_penalty,
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
                "degraded", 1.0 if result.degraded else 0.0, weights.degraded_penalty
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
        ]
    )
    total = sum(term.weighted for term in terms)
    return CandidateScore(
        candidate_id=result.candidate_id,
        total=total,
        terms=tuple(terms),
        policy=policy,
    )


def rank_candidates(
    results: Iterable[CandidateResult],
    *,
    policy: str = "best_score",
    weights: CandidateScoreWeights = CandidateScoreWeights(),
) -> list[tuple[CandidateResult, CandidateScore]]:
    scored = [
        (result, score_candidate(result, policy=policy, weights=weights))
        for result in results
    ]
    if policy == "fast_preview":
        return sorted(
            scored,
            key=lambda pair: (
                pair[0].status not in {"success", "degraded"},
                pair[0].metric_result.elapsed_s,
                -pair[1].total,
            ),
        )
    if policy == "editability_first":
        return sorted(
            scored,
            key=lambda pair: (
                pair[0].status not in {"success", "degraded"},
                -pair[0].metric_result.editability_score,
                -pair[1].total,
            ),
        )
    return sorted(scored, key=lambda pair: pair[1].total, reverse=True)


def select_best(
    results: Sequence[CandidateResult],
    *,
    policy: str = "best_score",
    weights: CandidateScoreWeights = CandidateScoreWeights(),
) -> tuple[CandidateResult | None, list[tuple[CandidateResult, CandidateScore]]]:
    if not results:
        return None, []
    ranked = rank_candidates(results, policy=policy, weights=weights)
    return ranked[0][0], ranked
