"""Baseline distributions and threshold derivation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .gates import Threshold
from .schemas import EvaluationBundle, MetricValue, json_safe, utc_now_iso


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


def baseline_metric_map(slice_: BaselineSlice, statistic: str = "median") -> dict[str, float]:
    return {
        distribution.metric: float(getattr(distribution, statistic))
        for distribution in slice_.distributions
        if hasattr(distribution, statistic)
    }


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
