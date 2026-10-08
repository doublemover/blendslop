"""Versioned evaluation bundle schemas.

The evaluation package keeps raw measurements, gate decisions, and selection
scores separate.  These dataclasses are intentionally Blender-free and JSON
safe so every backend can report the same shape of evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

try:
    from blender_blocking.utils.json_io import json_safe
except ImportError:  # pragma: no cover - script-style imports
    from utils.json_io import json_safe


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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricValue":
        return cls(
            name=str(payload.get("name", "")),
            value=payload.get("value"),
            unit=_optional_str(payload.get("unit")),
            higher_is_better=_optional_bool(payload.get("higher_is_better")),
            required=bool(payload.get("required", False)),
            status=str(payload.get("status", "not_applicable")),
            threshold=payload.get("threshold"),
            threshold_source=_optional_str(payload.get("threshold_source")),
            source=str(payload.get("source", "computed")),
            notes=_string_tuple(payload.get("notes", ())),
        )


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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricGroup":
        metrics = payload.get("metrics", ())
        return cls(
            name=str(payload.get("name", "")),
            status=str(payload.get("status", "not_applicable")),
            metrics=tuple(
                MetricValue.from_dict(metric)
                for metric in _mapping_sequence(metrics)
            ),
            warnings=_string_tuple(payload.get("warnings", ())),
            errors=_string_tuple(payload.get("errors", ())),
            metadata=_mapping_or_empty(payload.get("metadata")),
        )


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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FailureObservation":
        return cls(
            code=str(payload.get("code", "")),
            severity=str(payload.get("severity", "")),
            subsystem=str(payload.get("subsystem", "")),
            evidence_metrics=_mapping_or_empty(payload.get("evidence_metrics")),
            evidence_artifacts=_string_tuple(payload.get("evidence_artifacts", ())),
            likely_causes=_string_tuple(payload.get("likely_causes", ())),
            recommended_actions=_string_tuple(payload.get("recommended_actions", ())),
        )


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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationBundle":
        metric_groups = payload.get("metric_groups", ())
        failures = payload.get("failures", ())
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            run_id=str(payload.get("run_id", "")),
            created_at_utc=str(payload.get("created_at_utc", "")),
            repo_revision=_optional_str(payload.get("repo_revision")),
            mode=str(payload.get("mode", "")),
            suite=str(payload.get("suite", "")),
            candidate_id=str(payload.get("candidate_id", "")),
            target_id=str(payload.get("target_id", "")),
            status=str(payload.get("status", "fail")),
            metric_groups=tuple(
                MetricGroup.from_dict(group)
                for group in _mapping_sequence(metric_groups)
            ),
            artifacts={
                str(key): str(value)
                for key, value in _mapping_or_empty(payload.get("artifacts")).items()
            },
            dependency_state=_mapping_or_empty(payload.get("dependency_state")),
            degradation_state=_mapping_or_empty(payload.get("degradation_state")),
            timings_ms={
                str(key): _float(value)
                for key, value in _mapping_or_empty(payload.get("timings_ms")).items()
            },
            peak_memory_mb=_optional_float(payload.get("peak_memory_mb")),
            failures=tuple(
                FailureObservation.from_dict(failure)
                for failure in _mapping_sequence(failures)
            ),
            errors=_string_tuple(payload.get("errors", ())),
            warnings=_string_tuple(payload.get("warnings", ())),
            selection=_mapping_or_empty(payload.get("selection")),
            lineage_path=_optional_str(payload.get("lineage_path")),
        )


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


def _mapping_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_bool(value: object) -> bool | None:
    return None if value is None else bool(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _float(value)


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
