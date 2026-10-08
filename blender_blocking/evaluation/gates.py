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
    required: bool = False
    severity: str = "warn"

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "max_relative_regression": self.max_relative_regression,
            "max_absolute_regression": self.max_absolute_regression,
            "baseline_selector": self.baseline_selector,
            "required": self.required,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class EvaluationBudget:
    name: str
    suite: str
    thresholds: tuple[Threshold, ...]
    regressions: tuple[RegressionBudget, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "suite": self.suite,
            "thresholds": [threshold.to_dict() for threshold in self.thresholds],
            "regressions": [regression.to_dict() for regression in self.regressions],
        }


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
class RegressionGateResult:
    metric: str
    status: str
    current: object
    baseline: object
    regression: RegressionBudget
    absolute_delta: float | None = None
    relative_delta: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "status": self.status,
            "current": json_safe(self.current),
            "baseline": json_safe(self.baseline),
            "regression": self.regression.to_dict(),
            "absolute_delta": self.absolute_delta,
            "relative_delta": self.relative_delta,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GateDecision:
    status: str
    budget_name: str
    results: tuple[GateResult, ...] = ()
    regression_results: tuple[RegressionGateResult, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "budget_name": self.budget_name,
            "results": [result.to_dict() for result in self.results],
            "regression_results": [
                result.to_dict() for result in self.regression_results
            ],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metadata": json_safe(self.metadata),
        }


def evaluate_budget(
    bundle: EvaluationBundle,
    budget: EvaluationBudget,
    *,
    baseline_bundle: EvaluationBundle | None = None,
    baseline_metrics: Mapping[str, object] | None = None,
) -> GateDecision:
    index = bundle.metric_index()
    baseline_index = {}
    if baseline_bundle is not None:
        baseline_index = baseline_bundle.metric_index()
    if baseline_metrics:
        baseline_index = {**baseline_index, **dict(baseline_metrics)}
    results: list[GateResult] = []
    regression_results: list[RegressionGateResult] = []
    errors: list[str] = []
    warnings: list[str] = []
    for threshold in budget.thresholds:
        if not _threshold_applies(bundle, threshold):
            continue
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
    for regression in budget.regressions:
        result = _evaluate_regression(index, baseline_index, regression)
        regression_results.append(result)
        if result.status == "fail":
            errors.append(f"{regression.metric}: {result.reason}")
        elif result.status == "warn":
            warnings.append(f"{regression.metric}: {result.reason}")
    decision_status = "fail" if errors else "warn" if warnings else "pass"
    return GateDecision(
        status=decision_status,
        budget_name=budget.name,
        results=tuple(results),
        regression_results=tuple(regression_results),
        warnings=tuple(warnings),
        errors=tuple(errors),
        metadata={
            "suite": budget.suite,
            "bundle_suite": bundle.suite,
            "mode": bundle.mode,
            "baseline_attached": bool(baseline_index),
        },
    )


def sota_silhouette_budget(
    *,
    name: str = "sota-silhouette",
    suite: str = "synthetic",
    strict: bool = False,
) -> EvaluationBudget:
    """Recommended SOTA-style multi-metric budget for silhouette-only runs."""

    min_iou = 0.70 if strict else 0.55
    avg_iou = 0.78 if strict else 0.62
    boundary = 0.58 if strict else 0.35
    sdf = 0.18 if strict else 0.35
    editability = 0.62 if strict else 0.45
    return EvaluationBudget(
        name=name,
        suite=suite,
        thresholds=(
            Threshold("silhouette.min_view_iou", min_value=min_iou, source="sota_silhouette"),
            Threshold("silhouette.average_iou", min_value=avg_iou, source="sota_silhouette"),
            Threshold("silhouette.mean_boundary_iou", min_value=boundary, source="sota_silhouette"),
            Threshold("silhouette.mean_signed_distance_loss", max_value=sdf, source="sota_silhouette"),
            Threshold("silhouette.missing_required_metric_count", max_value=0.0, source="sota_silhouette"),
            Threshold("silhouette.failed_required_view_count", max_value=0.0, source="sota_silhouette"),
            Threshold("geometry.true.fscore_tau", min_value=0.35 if strict else 0.20, required=False, source="sota_synthetic"),
            Threshold("geometry.recoverable.fscore_tau", min_value=0.70 if strict else 0.55, required=False, source="sota_recoverability"),
            Threshold("geometry.true.volumetric_iou", min_value=0.40 if strict else 0.25, required=False, source="sota_synthetic"),
            Threshold("topology.non_manifold_edges", max_value=0.0, required=False, source="editable_mesh"),
            Threshold("editability.editable_reconstruction_index", min_value=editability, required=False, source="editable_blender"),
            Threshold("export.qa_score", min_value=0.75 if strict else 0.60, required=False, source="asset_delivery"),
        ),
        regressions=(
            RegressionBudget("silhouette.min_view_iou", max_absolute_regression=0.02, severity="fail"),
            RegressionBudget("silhouette.mean_boundary_iou", max_absolute_regression=0.03, severity="fail"),
            RegressionBudget("geometry.recoverable.fscore_tau", max_absolute_regression=0.03, severity="warn"),
            RegressionBudget("editability.editable_reconstruction_index", max_absolute_regression=0.04, severity="warn"),
            RegressionBudget("cost.total_wall_ms", max_relative_regression=0.20, severity="warn"),
        ),
    )


def _threshold_applies(bundle: EvaluationBundle, threshold: Threshold) -> bool:
    if not threshold.applies_to:
        return True
    applies_to = {str(value) for value in threshold.applies_to}
    return bool(
        {
            bundle.mode,
            bundle.suite,
            bundle.candidate_id,
            bundle.target_id,
        }
        & applies_to
    )


def _evaluate_regression(
    index: Mapping[str, object],
    baseline_index: Mapping[str, object],
    regression: RegressionBudget,
) -> RegressionGateResult:
    current = _metric_numeric(index.get(regression.metric))
    baseline = _metric_numeric(baseline_index.get(regression.metric))
    if baseline is None or current is None:
        status = "fail" if regression.required else "skip"
        reason = "missing baseline or current metric"
        return RegressionGateResult(
            regression.metric,
            status,
            current,
            baseline,
            regression,
            reason=reason,
        )
    absolute = current - baseline
    relative = absolute / max(abs(baseline), 1e-12)
    failures = []
    higher_is_better = _metric_higher_is_better(index.get(regression.metric))
    if regression.max_absolute_regression is not None:
        magnitude = _regression_magnitude(absolute, higher_is_better)
        if magnitude > regression.max_absolute_regression:
            failures.append(
                f"absolute regression {magnitude:.6g} > {regression.max_absolute_regression:.6g}"
            )
    if regression.max_relative_regression is not None:
        magnitude = _regression_magnitude(relative, higher_is_better)
        if magnitude > regression.max_relative_regression:
            failures.append(
                f"relative regression {magnitude:.6g} > {regression.max_relative_regression:.6g}"
            )
    status = regression.severity if failures else "pass"
    return RegressionGateResult(
        regression.metric,
        status,
        current,
        baseline,
        regression,
        absolute_delta=absolute,
        relative_delta=relative,
        reason="; ".join(failures),
    )


def _metric_numeric(metric: object) -> float | None:
    value = getattr(metric, "value", metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_higher_is_better(metric: object) -> bool:
    value = getattr(metric, "higher_is_better", True)
    if value is None:
        return True
    return bool(value)


def _regression_magnitude(delta: float, higher_is_better: bool) -> float:
    if higher_is_better:
        return max(0.0, -float(delta))
    return max(0.0, float(delta))
