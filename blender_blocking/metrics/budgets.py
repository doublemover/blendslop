"""Quality and performance budget evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class BudgetThreshold:
    """One metric threshold with direction semantics."""

    metric: str
    threshold: float
    mode: str = "min"
    required: bool = True
    description: str = ""

    def evaluate(self, value: Optional[float]) -> "BudgetCheck":
        if value is None:
            return BudgetCheck(
                metric=self.metric,
                value=None,
                threshold=self.threshold,
                mode=self.mode,
                passed=not self.required,
                required=self.required,
                message="missing metric",
            )
        if self.mode == "min":
            passed = value >= self.threshold
        elif self.mode == "max":
            passed = value <= self.threshold
        else:
            raise ValueError(f"unknown budget mode: {self.mode}")
        return BudgetCheck(
            metric=self.metric,
            value=float(value),
            threshold=float(self.threshold),
            mode=self.mode,
            passed=bool(passed),
            required=self.required,
            message="" if passed else f"{value} violates {self.mode} {self.threshold}",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "threshold": self.threshold,
            "mode": self.mode,
            "required": self.required,
            "description": self.description,
        }


@dataclass(frozen=True)
class BudgetCheck:
    """Result for one budgeted metric."""

    metric: str
    value: Optional[float]
    threshold: float
    mode: str
    passed: bool
    required: bool
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "mode": self.mode,
            "passed": self.passed,
            "required": self.required,
            "message": self.message,
        }


@dataclass(frozen=True)
class BudgetReport:
    """Aggregate budget evaluation."""

    checks: tuple[BudgetCheck, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(check.passed or not check.required for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "metadata": dict(self.metadata),
        }


def evaluate_budgets(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any] | list[Mapping[str, Any]],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> BudgetReport:
    """Evaluate metric values against JSON-style thresholds."""
    normalized = _normalize_thresholds(thresholds)
    checks = []
    for threshold in normalized:
        value = _lookup_metric(metrics, threshold.metric)
        numeric_value = None if value is None else float(value)
        checks.append(threshold.evaluate(numeric_value))
    return BudgetReport(tuple(checks), dict(metadata or {}))


def load_budget_file(path: str | Path) -> tuple[BudgetThreshold, ...]:
    """Load thresholds from a JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _normalize_thresholds(payload.get("thresholds", payload))


def compare_metric_delta(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    tolerance: Mapping[str, float],
) -> BudgetReport:
    """Fail metrics that regress by more than configured absolute tolerance."""
    checks = []
    for metric, allowed_regression in tolerance.items():
        before = _lookup_metric(baseline, metric)
        after = _lookup_metric(current, metric)
        if before is None or after is None:
            checks.append(
                BudgetCheck(
                    metric=metric,
                    value=None,
                    threshold=float(allowed_regression),
                    mode="max",
                    passed=False,
                    required=True,
                    message="missing baseline or current metric",
                )
            )
            continue
        delta = float(before) - float(after)
        checks.append(
            BudgetCheck(
                metric=metric,
                value=delta,
                threshold=float(allowed_regression),
                mode="max",
                passed=delta <= float(allowed_regression),
                required=True,
                message="" if delta <= float(allowed_regression) else "regression",
            )
        )
    return BudgetReport(tuple(checks), {"comparison": "absolute_regression"})


def _normalize_thresholds(
    thresholds: Mapping[str, Any] | list[Mapping[str, Any]] | tuple[BudgetThreshold, ...],
) -> tuple[BudgetThreshold, ...]:
    if isinstance(thresholds, tuple) and all(
        isinstance(item, BudgetThreshold) for item in thresholds
    ):
        return thresholds
    if isinstance(thresholds, Mapping):
        items = []
        for metric, value in thresholds.items():
            if isinstance(value, Mapping):
                items.append({"metric": metric, **value})
            else:
                items.append({"metric": metric, "threshold": value, "mode": "min"})
    else:
        items = list(thresholds)
    return tuple(
        BudgetThreshold(
            metric=str(item["metric"]),
            threshold=float(item["threshold"]),
            mode=str(item.get("mode", "min")),
            required=bool(item.get("required", True)),
            description=str(item.get("description", "")),
        )
        for item in items
    )


def _lookup_metric(metrics: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = metrics
    for part in dotted_key.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current
