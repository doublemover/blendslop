"""Pareto frontier reporting for reconstruction candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from .types import CandidateResult


@dataclass(frozen=True)
class ParetoObjective:
    name: str
    metric: str
    higher_is_better: bool = True
    missing_value: float = 0.0
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "missing_value": self.missing_value,
            "description": self.description,
        }


@dataclass(frozen=True)
class ParetoCandidate:
    candidate_id: str
    backend_name: str
    status: str
    vector: Mapping[str, float]
    comparable_vector: Mapping[str, float]
    normalized_vector: Mapping[str, float]
    frontier: bool
    selected: bool = False
    dominated_by: tuple[str, ...] = ()
    dominates: tuple[str, ...] = ()
    tradeoff_label: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "backend_name": self.backend_name,
            "status": self.status,
            "vector": dict(self.vector),
            "comparable_vector": dict(self.comparable_vector),
            "normalized_vector": dict(self.normalized_vector),
            "frontier": self.frontier,
            "selected": self.selected,
            "dominated_by": list(self.dominated_by),
            "dominates": list(self.dominates),
            "tradeoff_label": self.tradeoff_label,
        }


@dataclass(frozen=True)
class ParetoReport:
    policy: str
    selected_id: str
    objectives: tuple[ParetoObjective, ...]
    candidates: tuple[ParetoCandidate, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def frontier_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates if candidate.frontier)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "candidate_pareto_report_v1",
            "policy": self.policy,
            "selected_id": self.selected_id,
            "frontier_ids": list(self.frontier_ids),
            "objectives": [objective.to_dict() for objective in self.objectives],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "notes": list(self.notes),
        }


DEFAULT_PARETO_OBJECTIVES = (
    ParetoObjective(
        "silhouette_min",
        "area_iou_min",
        True,
        0.0,
        "Worst required-view silhouette fidelity.",
    ),
    ParetoObjective(
        "boundary",
        "boundary_iou_mean",
        True,
        0.0,
        "Mean Boundary IoU; catches contour errors missed by area IoU.",
    ),
    ParetoObjective(
        "recoverable_geometry",
        "geometry.recoverable.fscore_tau",
        True,
        0.0,
        "Recoverable geometry F-score where synthetic/recoverability evidence exists.",
    ),
    ParetoObjective(
        "topology",
        "topology_score",
        True,
        0.0,
        "Mesh topology/editing health score.",
    ),
    ParetoObjective(
        "editability",
        "editability_score",
        True,
        0.0,
        "Editable Blender output score.",
    ),
    ParetoObjective(
        "runtime",
        "elapsed_s",
        False,
        1e9,
        "Candidate runtime; lower is better.",
    ),
    ParetoObjective(
        "complexity",
        "complexity_penalty",
        False,
        1e9,
        "Candidate complexity penalty; lower is better.",
    ),
)


def pareto_report_from_candidates(
    candidates: Sequence[CandidateResult],
    *,
    selected_id: str = "",
    policy: str = "best_score",
    objectives: Sequence[ParetoObjective] = DEFAULT_PARETO_OBJECTIVES,
) -> ParetoReport:
    """Build a Pareto frontier report from backend candidate results."""

    objective_tuple = tuple(objectives)
    raw_vectors = {
        candidate.candidate_id: _candidate_vector(candidate, objective_tuple)
        for candidate in candidates
    }
    comparable_vectors = {
        candidate_id: _comparable_vector(vector, objective_tuple)
        for candidate_id, vector in raw_vectors.items()
    }
    dominators: dict[str, list[str]] = {candidate.candidate_id: [] for candidate in candidates}
    dominates: dict[str, list[str]] = {candidate.candidate_id: [] for candidate in candidates}
    for left in candidates:
        for right in candidates:
            if left.candidate_id == right.candidate_id:
                continue
            if _dominates(
                comparable_vectors[left.candidate_id],
                comparable_vectors[right.candidate_id],
            ):
                dominates[left.candidate_id].append(right.candidate_id)
                dominators[right.candidate_id].append(left.candidate_id)

    normalized = _normalized_vectors(raw_vectors, objective_tuple)
    pareto_candidates = []
    for candidate in candidates:
        candidate_id = candidate.candidate_id
        frontier = not dominators[candidate_id]
        pareto_candidates.append(
            ParetoCandidate(
                candidate_id=candidate_id,
                backend_name=candidate.backend_name,
                status=candidate.status,
                vector=raw_vectors[candidate_id],
                comparable_vector=comparable_vectors[candidate_id],
                normalized_vector=normalized[candidate_id],
                frontier=frontier,
                selected=candidate_id == selected_id,
                dominated_by=tuple(sorted(dominators[candidate_id])),
                dominates=tuple(sorted(dominates[candidate_id])),
                tradeoff_label=_tradeoff_label(normalized[candidate_id], objective_tuple),
            )
        )
    return ParetoReport(
        policy=policy,
        selected_id=selected_id,
        objectives=objective_tuple,
        candidates=tuple(
            sorted(
                pareto_candidates,
                key=lambda item: (
                    not item.frontier,
                    not item.selected,
                    item.candidate_id,
                ),
            )
        ),
        notes=_report_notes(pareto_candidates, selected_id=selected_id, policy=policy),
    )


def _candidate_vector(
    candidate: CandidateResult,
    objectives: Sequence[ParetoObjective],
) -> dict[str, float]:
    return {
        objective.name: _metric_value(candidate, objective.metric, objective.missing_value)
        for objective in objectives
    }


def _comparable_vector(
    vector: Mapping[str, float],
    objectives: Sequence[ParetoObjective],
) -> dict[str, float]:
    return {
        objective.name: (
            float(vector.get(objective.name, objective.missing_value))
            if objective.higher_is_better
            else -float(vector.get(objective.name, objective.missing_value))
        )
        for objective in objectives
    }


def _normalized_vectors(
    vectors: Mapping[str, Mapping[str, float]],
    objectives: Sequence[ParetoObjective],
) -> dict[str, dict[str, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for objective in objectives:
        values = [
            float(vector.get(objective.name, objective.missing_value))
            for vector in vectors.values()
        ]
        finite = [value for value in values if math.isfinite(value)]
        if not finite:
            ranges[objective.name] = (0.0, 1.0)
        else:
            ranges[objective.name] = (min(finite), max(finite))
    normalized: dict[str, dict[str, float]] = {}
    for candidate_id, vector in vectors.items():
        values = {}
        for objective in objectives:
            lo, hi = ranges[objective.name]
            raw = float(vector.get(objective.name, objective.missing_value))
            if not math.isfinite(raw) or abs(hi - lo) <= 1e-12:
                score = 0.5
            elif objective.higher_is_better:
                score = (raw - lo) / (hi - lo)
            else:
                score = (hi - raw) / (hi - lo)
            values[objective.name] = max(0.0, min(1.0, float(score)))
        normalized[candidate_id] = values
    return normalized


def _dominates(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    keys = tuple(left)
    if not keys:
        return False
    greater_or_equal = all(float(left[key]) >= float(right.get(key, -math.inf)) for key in keys)
    strictly_greater = any(float(left[key]) > float(right.get(key, -math.inf)) for key in keys)
    return greater_or_equal and strictly_greater


def _tradeoff_label(
    normalized: Mapping[str, float],
    objectives: Sequence[ParetoObjective],
) -> str:
    if not normalized:
        return "unscored"
    best = max(
        objectives,
        key=lambda objective: float(normalized.get(objective.name, 0.0)),
    )
    if best.name in {"silhouette_min", "boundary", "recoverable_geometry"}:
        return "fidelity"
    if best.name in {"editability", "topology"}:
        return "editable"
    if best.name == "runtime":
        return "fast"
    if best.name == "complexity":
        return "simple"
    return best.name


def _report_notes(
    candidates: Sequence[ParetoCandidate],
    *,
    selected_id: str,
    policy: str,
) -> tuple[str, ...]:
    notes = []
    frontier_ids = {candidate.candidate_id for candidate in candidates if candidate.frontier}
    if selected_id and selected_id not in frontier_ids:
        notes.append(
            f"selected candidate {selected_id!r} is dominated under Pareto objectives"
        )
    if policy in {"best_score", "balanced"} and len(frontier_ids) > 1:
        notes.append(
            "multiple non-dominated candidates remain; report preserves tradeoffs for intent-specific review"
        )
    return tuple(notes)


def _metric_value(candidate: CandidateResult, metric: str, default: float) -> float:
    metrics = candidate.metric_result
    direct = getattr(metrics, metric, None)
    if isinstance(direct, (int, float, bool)):
        return _float(direct, default)
    extras = metrics.extras if isinstance(metrics.extras, Mapping) else {}
    value = _nested(extras, metric.split("."))
    if value is None and metric.startswith("geometry."):
        value = _nested(extras, ("evaluation", *metric.split(".")))
    if value is None:
        per_view = metrics.per_view if isinstance(metrics.per_view, Mapping) else {}
        if metric == "area_iou_min" and per_view:
            values = [
                _float(view.get("area_iou", view.get("iou")), math.nan)
                for view in per_view.values()
                if isinstance(view, Mapping)
            ]
            finite = [item for item in values if math.isfinite(item)]
            value = min(finite) if finite else None
    return _float(value, default)


def _nested(data: Mapping[str, Any], keys: Sequence[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)
