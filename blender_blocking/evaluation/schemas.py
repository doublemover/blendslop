"""Versioned evaluation bundle schemas.

The evaluation package keeps raw measurements, gate decisions, and selection
scores separate.  These dataclasses are intentionally Blender-free and JSON
safe so every backend can report the same shape of evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


STATUS_VALUES = {
    "pass",
    "warn",
    "fail",
    "skip",
    "not_applicable",
    "degraded",
    "research_only",
    "contract_error",
    "metric_error",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_status(value: object, *, default: str = "not_applicable") -> str:
    status = str(value or default).strip().lower()
    return status if status in STATUS_VALUES else default


def json_safe(value: Any) -> Any:
    """Convert common Python values into JSON-compatible structures."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict())
    if is_dataclass(value):
        return json_safe({name: getattr(value, name) for name in value.__dataclass_fields__})
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(json_safe(item) for item in value)
    if isinstance(value, float):
        if value != value:
            return None
        if value == float("inf"):
            return None
        if value == float("-inf"):
            return None
    return value


@dataclass(frozen=True)
class MetricValue:
    """One raw metric plus threshold/status metadata."""

    name: str
    value: float | int | bool | str | None
    unit: str | None = None
    higher_is_better: bool | None = None
    required: bool = False
    status: str = "not_applicable"
    threshold: float | int | bool | str | None = None
    threshold_source: str | None = None
    source: str = "computed"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", normalize_status(self.status))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": json_safe(self.value),
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "required": self.required,
            "status": self.status,
            "threshold": json_safe(self.threshold),
            "threshold_source": self.threshold_source,
            "source": self.source,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class MetricGroup:
    """Named group of related metrics."""

    name: str
    status: str
    metrics: tuple[MetricValue, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", normalize_status(self.status))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class FailureObservation:
    code: str
    severity: str
    subsystem: str
    evidence_metrics: Mapping[str, Any] = field(default_factory=dict)
    evidence_artifacts: tuple[str, ...] = ()
    likely_causes: tuple[str, ...] = ()
    recommended_actions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "subsystem": self.subsystem,
            "evidence_metrics": json_safe(self.evidence_metrics),
            "evidence_artifacts": list(self.evidence_artifacts),
            "likely_causes": list(self.likely_causes),
            "recommended_actions": list(self.recommended_actions),
        }


@dataclass(frozen=True)
class EvaluationBundle:
    """Full multi-metric evidence packet for one candidate."""

    schema_version: str
    run_id: str
    created_at_utc: str
    repo_revision: str | None
    mode: str
    suite: str
    candidate_id: str
    target_id: str
    status: str
    metric_groups: tuple[MetricGroup, ...]
    artifacts: Mapping[str, str] = field(default_factory=dict)
    dependency_state: Mapping[str, Any] = field(default_factory=dict)
    degradation_state: Mapping[str, Any] = field(default_factory=dict)
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    peak_memory_mb: float | None = None
    failures: tuple[FailureObservation, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    selection: Mapping[str, Any] = field(default_factory=dict)
    lineage_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", normalize_status(self.status, default="fail"))

    def metric_index(self) -> dict[str, MetricValue]:
        index: dict[str, MetricValue] = {}
        for group in self.metric_groups:
            for metric in group.metrics:
                index[metric.name] = metric
        return index

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "repo_revision": self.repo_revision,
            "mode": self.mode,
            "suite": self.suite,
            "candidate_id": self.candidate_id,
            "target_id": self.target_id,
            "status": self.status,
            "metric_groups": [group.to_dict() for group in self.metric_groups],
            "artifacts": dict(self.artifacts),
            "dependency_state": json_safe(self.dependency_state),
            "degradation_state": json_safe(self.degradation_state),
            "timings_ms": json_safe(self.timings_ms),
            "peak_memory_mb": self.peak_memory_mb,
            "failures": [failure.to_dict() for failure in self.failures],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "selection": json_safe(self.selection),
            "lineage_path": self.lineage_path,
        }


def worst_status(statuses: Sequence[str]) -> str:
    order = {
        "contract_error": 0,
        "metric_error": 1,
        "fail": 2,
        "degraded": 3,
        "warn": 4,
        "research_only": 5,
        "skip": 6,
        "not_applicable": 7,
        "pass": 8,
    }
    if not statuses:
        return "not_applicable"
    return min((normalize_status(status) for status in statuses), key=lambda item: order[item])

