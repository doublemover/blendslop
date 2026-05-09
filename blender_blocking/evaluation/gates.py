"""Quality and performance gate comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .schemas import EvaluationBundle, json_safe


@dataclass(frozen=True)
class Threshold:
    metric: str
    min_value: float | None = None
    max_value: float | None = None
    required: bool = True
    applies_to: tuple[str, ...] = ()
    severity: str = "fail"
    source: str = "manual"

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "required": self.required,
            "applies_to": list(self.applies_to),
            "severity": self.severity,
            "source": self.source,
        }


@dataclass(frozen=True)
class RegressionBudget:
    metric: str
    max_relative_regression: float | None = None
    max_absolute_regression: float | None = None
    baseline_selector: str = "latest-main"


@dataclass(frozen=True)
class EvaluationBudget:
    name: str
    suite: str
    thresholds: tuple[Threshold, ...]
    regressions: tuple[RegressionBudget, ...] = ()


@dataclass(frozen=True)
class GateResult:
    metric: str
    status: str
    value: object
    threshold: Threshold
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "status": self.status,
            "value": json_safe(self.value),
            "threshold": self.threshold.to_dict(),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GateDecision:
    status: str
    budget_name: str
    results: tuple[GateResult, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "budget_name": self.budget_name,
            "results": [result.to_dict() for result in self.results],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metadata": json_safe(self.metadata),
        }


def evaluate_budget(bundle: EvaluationBundle, budget: EvaluationBudget) -> GateDecision:
    index = bundle.metric_index()
    results: list[GateResult] = []
    errors: list[str] = []
    warnings: list[str] = []
    for threshold in budget.thresholds:
        metric = index.get(threshold.metric)
        if metric is None:
            status = "fail" if threshold.required else "warn"
            reason = "metric missing"
            results.append(GateResult(threshold.metric, status, None, threshold, reason))
            (errors if status == "fail" else warnings).append(f"{threshold.metric}: {reason}")
            continue
        status = "pass"
        reason = ""
        value = metric.value
        try:
            numeric = float(value) if value is not None else None
        except (TypeError, ValueError):
            numeric = None
        if numeric is None:
            status = "fail" if threshold.required else "warn"
            reason = "metric value is not numeric"
        elif threshold.min_value is not None and numeric < threshold.min_value:
            status = threshold.severity
            reason = f"{numeric:.6g} below {threshold.min_value:.6g}"
        elif threshold.max_value is not None and numeric > threshold.max_value:
            status = threshold.severity
            reason = f"{numeric:.6g} above {threshold.max_value:.6g}"
        results.append(GateResult(threshold.metric, status, value, threshold, reason))
        if status == "fail":
            errors.append(f"{threshold.metric}: {reason}")
        elif status == "warn":
            warnings.append(f"{threshold.metric}: {reason}")
    decision_status = "fail" if errors else "warn" if warnings else "pass"
    return GateDecision(
        status=decision_status,
        budget_name=budget.name,
        results=tuple(results),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )

