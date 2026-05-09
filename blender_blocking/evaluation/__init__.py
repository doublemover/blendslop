"""Evaluation contracts for reconstruction quality, cost, and artifacts."""

from __future__ import annotations

from .bundle import bundle_from_candidate
from .geometry import (
    GeometryMetricReport,
    chamfer_distance,
    fscore_at_tolerance,
    normal_consistency,
    surface_distance_report,
    volumetric_iou,
)
from .gates import EvaluationBudget, Threshold, evaluate_budget
from .selection import SelectionEvidence, attach_selection, pareto_front
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
    "GeometryMetricReport",
    "Threshold",
    "SelectionEvidence",
    "attach_selection",
    "bundle_from_candidate",
    "chamfer_distance",
    "evaluate_budget",
    "fscore_at_tolerance",
    "normal_consistency",
    "pareto_front",
    "surface_distance_report",
    "volumetric_iou",
    "json_safe",
]
