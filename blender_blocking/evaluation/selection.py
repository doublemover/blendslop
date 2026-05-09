"""Selection evidence for evaluation bundles.

Candidate scoring decides which reconstruction wins.  This module records that
decision in evaluation artifacts so later reports can explain why a candidate
won or lost instead of only showing final metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .schemas import EvaluationBundle, MetricValue


DEFAULT_PARETO_METRICS = (
    "silhouette.min_view_iou",
    "silhouette.average_iou",
    "silhouette.mean_boundary_iou",
    "geometry.fscore_tau",
    "geometry.volumetric_iou",
    "novel_view.psnr",
    "novel_view.ssim",
    "editability.editable_reconstruction_index",
    "export.qa_score",
    "topology.score",
)


@dataclass(frozen=True)
class SelectionEvidence:
    policy: str
    rank: int
    selected: bool
    score_total: float | None = None
    terms: tuple[Mapping[str, Any], ...] = ()
    dominated_by: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "rank": self.rank,
            "selected": self.selected,
            "score_total": self.score_total,
            "terms": [dict(term) for term in self.terms],
            "dominated_by": list(self.dominated_by),
            "notes": list(self.notes),
        }


def attach_selection(
    bundle: EvaluationBundle,
    *,
    policy: str,
    rank: int,
    selected: bool,
    score: Any = None,
    dominated_by: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> EvaluationBundle:
    evidence = SelectionEvidence(
        policy=policy,
        rank=rank,
        selected=selected,
        score_total=_score_total(score),
        terms=_score_terms(score),
        dominated_by=tuple(dominated_by),
        notes=tuple(str(note) for note in notes),
    )
    return EvaluationBundle(
        schema_version=bundle.schema_version,
        run_id=bundle.run_id,
        created_at_utc=bundle.created_at_utc,
        repo_revision=bundle.repo_revision,
        mode=bundle.mode,
        suite=bundle.suite,
        candidate_id=bundle.candidate_id,
        target_id=bundle.target_id,
        status=bundle.status,
        metric_groups=bundle.metric_groups,
        artifacts=bundle.artifacts,
        dependency_state=bundle.dependency_state,
        degradation_state=bundle.degradation_state,
        timings_ms=bundle.timings_ms,
        peak_memory_mb=bundle.peak_memory_mb,
        failures=bundle.failures,
        errors=bundle.errors,
        warnings=bundle.warnings,
        selection=evidence.to_dict(),
        lineage_path=bundle.lineage_path,
    )


def pareto_dominators(
    bundles: Sequence[EvaluationBundle],
    *,
    metric_names: Sequence[str] = DEFAULT_PARETO_METRICS,
) -> dict[str, tuple[str, ...]]:
    """Return candidate IDs that dominate each candidate across metric_names."""
    vectors = {
        bundle.candidate_id: _metric_vector(bundle, metric_names) for bundle in bundles
    }
    dominators: dict[str, list[str]] = {bundle.candidate_id: [] for bundle in bundles}
    for candidate_id, vector in vectors.items():
        for other_id, other_vector in vectors.items():
            if candidate_id == other_id:
                continue
            if _dominates(other_vector, vector):
                dominators[candidate_id].append(other_id)
    return {key: tuple(value) for key, value in dominators.items()}


def pareto_front(
    bundles: Iterable[EvaluationBundle],
    *,
    metric_names: Sequence[str] = DEFAULT_PARETO_METRICS,
) -> tuple[EvaluationBundle, ...]:
    bundle_tuple = tuple(bundles)
    dominators = pareto_dominators(bundle_tuple, metric_names=metric_names)
    return tuple(bundle for bundle in bundle_tuple if not dominators[bundle.candidate_id])


def _metric_vector(
    bundle: EvaluationBundle,
    metric_names: Sequence[str],
) -> tuple[float, ...]:
    index = bundle.metric_index()
    return tuple(_metric_value(index.get(name)) for name in metric_names)


def _metric_value(metric: MetricValue | None) -> float:
    if metric is None or metric.value is None:
        return 0.0
    try:
        value = float(metric.value)
    except (TypeError, ValueError):
        return 0.0
    if metric.higher_is_better is False:
        return -value
    return value


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    if len(left) != len(right):
        return False
    greater_or_equal = all(a >= b for a, b in zip(left, right))
    strictly_greater = any(a > b for a, b in zip(left, right))
    return greater_or_equal and strictly_greater


def _score_total(score: Any) -> float | None:
    if score is None:
        return None
    value = getattr(score, "total", None)
    if value is None and isinstance(score, Mapping):
        value = score.get("total")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _score_terms(score: Any) -> tuple[Mapping[str, Any], ...]:
    if score is None:
        return ()
    terms = getattr(score, "terms", None)
    if terms is None and isinstance(score, Mapping):
        terms = score.get("terms")
    payloads = []
    for term in terms or ():
        if hasattr(term, "to_dict"):
            payloads.append(term.to_dict())
        elif isinstance(term, Mapping):
            payloads.append(dict(term))
    return tuple(payloads)
