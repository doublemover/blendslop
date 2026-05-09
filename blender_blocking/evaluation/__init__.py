"""Evaluation contracts for reconstruction quality, cost, and artifacts."""

from __future__ import annotations

from .bundle import bundle_from_candidate
from .export_qa import (
    ExportQAReport,
    aggregate_score as export_qa_aggregate_score,
    report_from_mapping as export_qa_report_from_mapping,
)
from .geometry import (
    GeometryMetricReport,
    chamfer_distance,
    fscore_at_tolerance,
    normal_consistency,
    surface_distance_report,
    volumetric_iou,
)
from .gates import EvaluationBudget, Threshold, evaluate_budget
from .novel_view import (
    NovelViewMetricReport,
    image_pair_report,
    image_set_report,
    psnr_from_mse,
)
from .selection import SelectionEvidence, attach_selection, pareto_front
from .schemas import (
    EvaluationBundle,
    MetricGroup,
    MetricValue,
    STATUS_VALUES,
    json_safe,
)
from .view_planning import ViewRequest, active_view_plan_payload, suggest_next_views

__all__ = [
    "EvaluationBundle",
    "MetricGroup",
    "MetricValue",
    "STATUS_VALUES",
    "EvaluationBudget",
    "ExportQAReport",
    "GeometryMetricReport",
    "NovelViewMetricReport",
    "Threshold",
    "ViewRequest",
    "SelectionEvidence",
    "active_view_plan_payload",
    "attach_selection",
    "bundle_from_candidate",
    "chamfer_distance",
    "evaluate_budget",
    "export_qa_aggregate_score",
    "export_qa_report_from_mapping",
    "fscore_at_tolerance",
    "image_pair_report",
    "image_set_report",
    "normal_consistency",
    "pareto_front",
    "psnr_from_mse",
    "suggest_next_views",
    "surface_distance_report",
    "volumetric_iou",
    "json_safe",
]
