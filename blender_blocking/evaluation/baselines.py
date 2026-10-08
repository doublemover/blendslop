"""Baseline distributions and threshold derivation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .gates import Threshold
from .schemas import EvaluationBundle, MetricValue, json_safe, utc_now_iso


_BASELINE_FIELDS = (
    "suite",
    "mode",
    "shape_family",
    "view_count",
    "mask_noise_profile",
    "resolution",
    "dependency_profile",
)
_FALLBACK_TIERS = (
    (
        "mode",
        "suite",
        "shape_family",
        "mask_noise_profile",
        "view_count",
        "resolution",
        "dependency_profile",
    ),
    ("mode", "suite", "shape_family", "mask_noise_profile"),
    ("mode", "suite", "shape_family"),
    ("mode", "suite"),
    ("suite",),
    (),
)


@dataclass(frozen=True)
class MetricDistribution:
    metric: str
    sample_count: int
    min_value: float
    max_value: float
    mean: float
    median: float
    p05: float
    p25: float
    p75: float
    p95: float
    stddev: float
    higher_is_better: bool
    unit: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "sample_count": self.sample_count,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean": self.mean,
            "median": self.median,
            "p05": self.p05,
            "p25": self.p25,
            "p75": self.p75,
            "p95": self.p95,
            "stddev": self.stddev,
            "higher_is_better": self.higher_is_better,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class BaselineSlice:
    suite: str
    mode: str
    shape_family: str
    view_count: int
    mask_noise_profile: str
    resolution: int
    dependency_profile: str
    distributions: tuple[MetricDistribution, ...]
    case_ids: tuple[str, ...] = ()
    generated_from_runs: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "mode": self.mode,
            "shape_family": self.shape_family,
            "view_count": self.view_count,
            "mask_noise_profile": self.mask_noise_profile,
            "resolution": self.resolution,
            "dependency_profile": self.dependency_profile,
            "distributions": [item.to_dict() for item in self.distributions],
            "case_ids": list(self.case_ids),
            "generated_from_runs": list(self.generated_from_runs),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class QualityBaseline:
    schema_version: str
    created_at_utc: str
    repo_revision: str | None
    environment_hash: str
    slices: tuple[BaselineSlice, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "repo_revision": self.repo_revision,
            "environment_hash": self.environment_hash,
            "slices": [item.to_dict() for item in self.slices],
        }


@dataclass(frozen=True)
class BaselineQuery:
    suite: str = ""
    mode: str = ""
    shape_family: str = ""
    view_count: int = 0
    mask_noise_profile: str = ""
    resolution: int = 0
    dependency_profile: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "mode": self.mode,
            "shape_family": self.shape_family,
            "view_count": self.view_count,
            "mask_noise_profile": self.mask_noise_profile,
            "resolution": self.resolution,
            "dependency_profile": self.dependency_profile,
        }


@dataclass(frozen=True)
class BaselineMatch:
    slice: BaselineSlice | None
    query: BaselineQuery
    fallback_level: str
    fallback_path: tuple[str, ...]
    exact: bool
    matched_fields: Mapping[str, object] = field(default_factory=dict)
    mismatched_fields: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "slice": self.slice.to_dict() if self.slice is not None else None,
            "query": self.query.to_dict(),
            "fallback_level": self.fallback_level,
            "fallback_path": list(self.fallback_path),
            "exact": self.exact,
            "matched_fields": json_safe(dict(self.matched_fields)),
            "mismatched_fields": json_safe(dict(self.mismatched_fields)),
            "warnings": list(self.warnings),
        }


def distribution_from_values(
    metric: str,
    values: Iterable[float],
    *,
    higher_is_better: bool,
    unit: str | None = None,
) -> MetricDistribution:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        raise ValueError("cannot build a distribution from no values")
    mean = sum(sorted_values) / len(sorted_values)
    variance = sum((value - mean) ** 2 for value in sorted_values) / len(sorted_values)
    return MetricDistribution(
        metric=metric,
        sample_count=len(sorted_values),
        min_value=sorted_values[0],
        max_value=sorted_values[-1],
        mean=mean,
        median=_percentile(sorted_values, 0.50),
        p05=_percentile(sorted_values, 0.05),
        p25=_percentile(sorted_values, 0.25),
        p75=_percentile(sorted_values, 0.75),
        p95=_percentile(sorted_values, 0.95),
        stddev=variance**0.5,
        higher_is_better=higher_is_better,
        unit=unit,
    )


def threshold_from_distribution(
    distribution: MetricDistribution,
    *,
    severity: str = "fail",
    source: str = "derived_baseline",
) -> Threshold:
    if distribution.higher_is_better:
        return Threshold(
            metric=distribution.metric,
            min_value=distribution.p05 - 0.5 * distribution.stddev,
            severity=severity,
            source=source,
        )
    return Threshold(
        metric=distribution.metric,
        max_value=distribution.p95 + 0.5 * distribution.stddev,
        severity=severity,
        source=source,
    )


def baseline_payload(baseline: QualityBaseline) -> Mapping[str, Any]:
    return json_safe(baseline.to_dict())


def baseline_from_bundles(
    bundles: Sequence[EvaluationBundle],
    *,
    suite: str = "",
    mode: str = "",
    shape_family: str = "",
    view_count: int = 0,
    mask_noise_profile: str = "",
    resolution: int = 0,
    dependency_profile: str = "",
    repo_revision: str | None = None,
    environment_hash: str = "",
    metric_names: Sequence[str] | None = None,
) -> QualityBaseline:
    """Create a baseline slice from already-built EvaluationBundle records."""

    if not bundles:
        raise ValueError("baseline_from_bundles requires at least one bundle")
    metrics_by_name: dict[str, list[tuple[float, MetricValue]]] = {}
    for bundle in bundles:
        for metric in bundle.metric_index().values():
            if metric_names is not None and metric.name not in metric_names:
                continue
            value = _numeric_or_none(metric.value)
            if value is None:
                continue
            metrics_by_name.setdefault(metric.name, []).append((value, metric))
    distributions = []
    for metric_name, values in sorted(metrics_by_name.items()):
        first_metric = values[0][1]
        higher = first_metric.higher_is_better
        distributions.append(
            distribution_from_values(
                metric_name,
                (value for value, _metric in values),
                higher_is_better=True if higher is None else bool(higher),
                unit=first_metric.unit,
            )
        )
    slice_ = BaselineSlice(
        suite=suite or _common_value(bundle.suite for bundle in bundles),
        mode=mode or _common_value(bundle.mode for bundle in bundles),
        shape_family=shape_family,
        view_count=view_count,
        mask_noise_profile=mask_noise_profile,
        resolution=resolution,
        dependency_profile=dependency_profile,
        distributions=tuple(distributions),
        case_ids=tuple(bundle.target_id for bundle in bundles),
        generated_from_runs=tuple(bundle.run_id for bundle in bundles),
    )
    return QualityBaseline(
        schema_version="quality-baseline-v1",
        created_at_utc=utc_now_iso(),
        repo_revision=repo_revision or bundles[0].repo_revision,
        environment_hash=environment_hash,
        slices=(slice_,),
    )


def thresholds_from_baseline(
    baseline: QualityBaseline,
    *,
    severity: str = "fail",
    source: str = "derived_baseline",
) -> tuple[Threshold, ...]:
    thresholds = []
    for slice_ in baseline.slices:
        for distribution in slice_.distributions:
            thresholds.append(
                threshold_from_distribution(
                    distribution,
                    severity=severity,
                    source=source,
                )
            )
    return tuple(thresholds)


def select_baseline_slice(
    baseline: QualityBaseline,
    query: BaselineQuery | Mapping[str, object],
) -> BaselineMatch:
    """Select the best baseline slice and record the fallback path used."""

    normalized = coerce_baseline_query(query)
    populated_fields = tuple(
        field_name
        for field_name in _BASELINE_FIELDS
        if _field_is_populated(getattr(normalized, field_name))
    )
    fallback_path: list[str] = []
    for tier in _FALLBACK_TIERS:
        active_tier = tuple(field for field in tier if field in populated_fields)
        tier_name = "+".join(active_tier) if active_tier else "global"
        fallback_path.append(tier_name)
        candidates = [
            slice_
            for slice_ in baseline.slices
            if _slice_matches_fields(slice_, normalized, active_tier)
        ]
        if not candidates:
            continue
        selected = sorted(
            candidates,
            key=lambda item: _slice_preference_key(item, normalized, populated_fields),
            reverse=True,
        )[0]
        matched, mismatched = _field_match_maps(selected, normalized, populated_fields)
        return BaselineMatch(
            slice=selected,
            query=normalized,
            fallback_level=tier_name,
            fallback_path=tuple(fallback_path),
            exact=not mismatched,
            matched_fields=matched,
            mismatched_fields=mismatched,
            warnings=()
            if not mismatched
            else (f"baseline selected via {tier_name} fallback",),
        )
    return BaselineMatch(
        slice=None,
        query=normalized,
        fallback_level="none",
        fallback_path=tuple(fallback_path),
        exact=False,
        warnings=("no compatible baseline slice found",),
    )


def thresholds_for_query(
    baseline: QualityBaseline,
    query: BaselineQuery | Mapping[str, object],
    *,
    severity: str = "fail",
) -> tuple[tuple[Threshold, ...], BaselineMatch]:
    """Derive thresholds from the selected baseline slice for one run context."""

    match = select_baseline_slice(baseline, query)
    if match.slice is None:
        return (), match
    source = f"derived_baseline:{match.fallback_level}"
    thresholds = tuple(
        threshold_from_distribution(
            distribution,
            severity=severity,
            source=source,
        )
        for distribution in match.slice.distributions
    )
    return thresholds, match


def baseline_metric_map(slice_: BaselineSlice, statistic: str = "median") -> dict[str, float]:
    return {
        distribution.metric: float(getattr(distribution, statistic))
        for distribution in slice_.distributions
        if hasattr(distribution, statistic)
    }


def coerce_baseline_query(query: BaselineQuery | Mapping[str, object]) -> BaselineQuery:
    if isinstance(query, BaselineQuery):
        return query
    return BaselineQuery(
        suite=str(query.get("suite", "") or ""),
        mode=str(query.get("mode", "") or ""),
        shape_family=str(query.get("shape_family", "") or ""),
        view_count=int(query.get("view_count", 0) or 0),
        mask_noise_profile=str(query.get("mask_noise_profile", "") or ""),
        resolution=int(query.get("resolution", 0) or 0),
        dependency_profile=str(query.get("dependency_profile", "") or ""),
    )


def metric_distribution_from_mapping(payload: Mapping[str, Any]) -> MetricDistribution:
    return MetricDistribution(
        metric=str(payload.get("metric", "")),
        sample_count=int(payload.get("sample_count", 0) or 0),
        min_value=float(payload.get("min_value", 0.0) or 0.0),
        max_value=float(payload.get("max_value", 0.0) or 0.0),
        mean=float(payload.get("mean", 0.0) or 0.0),
        median=float(payload.get("median", 0.0) or 0.0),
        p05=float(payload.get("p05", 0.0) or 0.0),
        p25=float(payload.get("p25", 0.0) or 0.0),
        p75=float(payload.get("p75", 0.0) or 0.0),
        p95=float(payload.get("p95", 0.0) or 0.0),
        stddev=float(payload.get("stddev", 0.0) or 0.0),
        higher_is_better=bool(payload.get("higher_is_better", True)),
        unit=str(payload.get("unit")) if payload.get("unit") is not None else None,
    )


def baseline_slice_from_mapping(payload: Mapping[str, Any]) -> BaselineSlice:
    distributions_payload = payload.get("distributions", ()) or ()
    return BaselineSlice(
        suite=str(payload.get("suite", "") or ""),
        mode=str(payload.get("mode", "") or ""),
        shape_family=str(payload.get("shape_family", "") or ""),
        view_count=int(payload.get("view_count", 0) or 0),
        mask_noise_profile=str(payload.get("mask_noise_profile", "") or ""),
        resolution=int(payload.get("resolution", 0) or 0),
        dependency_profile=str(payload.get("dependency_profile", "") or ""),
        distributions=tuple(
            metric_distribution_from_mapping(item)
            for item in distributions_payload
            if isinstance(item, Mapping)
        ),
        case_ids=tuple(str(item) for item in payload.get("case_ids", ()) or ()),
        generated_from_runs=tuple(
            str(item) for item in payload.get("generated_from_runs", ()) or ()
        ),
        notes=tuple(str(item) for item in payload.get("notes", ()) or ()),
    )


def quality_baseline_from_mapping(payload: Mapping[str, Any]) -> QualityBaseline:
    slices_payload = payload.get("slices", ()) or ()
    return QualityBaseline(
        schema_version=str(payload.get("schema_version", "quality-baseline-v1")),
        created_at_utc=str(payload.get("created_at_utc", "")),
        repo_revision=(
            str(payload.get("repo_revision"))
            if payload.get("repo_revision") is not None
            else None
        ),
        environment_hash=str(payload.get("environment_hash", "") or ""),
        slices=tuple(
            baseline_slice_from_mapping(item)
            for item in slices_payload
            if isinstance(item, Mapping)
        ),
    )


def _slice_matches_fields(
    slice_: BaselineSlice,
    query: BaselineQuery,
    fields: Sequence[str],
) -> bool:
    for field_name in fields:
        query_value = getattr(query, field_name)
        if _field_is_populated(query_value) and getattr(slice_, field_name) != query_value:
            return False
    return True


def _slice_preference_key(
    slice_: BaselineSlice,
    query: BaselineQuery,
    populated_fields: Sequence[str],
) -> tuple[int, int, int]:
    matched = sum(
        1
        for field_name in populated_fields
        if getattr(slice_, field_name) == getattr(query, field_name)
    )
    populated_slice_fields = sum(
        1
        for field_name in _BASELINE_FIELDS
        if _field_is_populated(getattr(slice_, field_name))
    )
    distribution_count = len(slice_.distributions)
    return matched, populated_slice_fields, distribution_count


def _field_match_maps(
    slice_: BaselineSlice,
    query: BaselineQuery,
    populated_fields: Sequence[str],
) -> tuple[dict[str, object], dict[str, Mapping[str, object]]]:
    matched: dict[str, object] = {}
    mismatched: dict[str, Mapping[str, object]] = {}
    for field_name in populated_fields:
        query_value = getattr(query, field_name)
        slice_value = getattr(slice_, field_name)
        if slice_value == query_value:
            matched[field_name] = query_value
        else:
            mismatched[field_name] = {
                "query": query_value,
                "baseline": slice_value,
            }
    return matched, mismatched


def _field_is_populated(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (int, float)):
        return value != 0
    return True


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    pos = max(0.0, min(1.0, q)) * (len(values) - 1)
    lower = int(pos)
    upper = min(len(values) - 1, lower + 1)
    frac = pos - lower
    return values[lower] * (1.0 - frac) + values[upper] * frac


def _numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _common_value(values: Iterable[str]) -> str:
    unique = tuple(dict.fromkeys(str(value) for value in values))
    return unique[0] if len(unique) == 1 else ""
