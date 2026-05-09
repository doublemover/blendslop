"""Evaluation contracts for reconstruction quality, cost, and artifacts."""

from __future__ import annotations

from .bundle import bundle_from_candidate
from .gates import EvaluationBudget, Threshold, evaluate_budget
from .schemas import (
    EvaluationBundle,
    MetricGroup,
    MetricValue,
    STATUS_VALUES,
    json_safe,
)

__all__ = [
    "EvaluationBundle",
    "MetricGroup",
    "MetricValue",
    "STATUS_VALUES",
    "EvaluationBudget",
    "Threshold",
    "bundle_from_candidate",
    "evaluate_budget",
    "json_safe",
]
