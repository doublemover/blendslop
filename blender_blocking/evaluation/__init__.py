"""Evaluation contracts for reconstruction quality, cost, and artifacts."""

from __future__ import annotations

from .bundle import bundle_from_candidate
from .boundary_refinement import boundary_refinement_plan_payload
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
from .gates import (
    EvaluationBudget,
    RegressionBudget,
    Threshold,
    evaluate_budget,
    sota_silhouette_budget,
)
from .lineage import (
    ArtifactRecord,
    RunLineage,
    capture_environment,
    dirty_worktree,
    hash_file,
    hash_json_payload,
    repo_revision,
    write_reproduce_script,
    write_run_lineage,
)
from .novel_view import (
    NovelViewMetricReport,
    image_pair_report,
    image_set_report,
    psnr_from_mse,
)
from .selection import SelectionEvidence, attach_selection, pareto_front
from .recoverability import RecoverabilityReport
from .schemas import (
    EvaluationBundle,
    MetricGroup,
    MetricValue,
    STATUS_VALUES,
    json_safe,
)
from .silhouette_eval import (
    DEFAULT_REQUIRED_VIEWS,
    SilhouetteGateConfig,
    evaluate_silhouette_pair,
    missing_silhouette_view,
    summarize_silhouette_views,
)
from .view_planning import ViewRequest, active_view_plan_payload, suggest_next_views

__all__ = [
    "EvaluationBundle",
    "MetricGroup",
    "MetricValue",
    "STATUS_VALUES",
    "DEFAULT_REQUIRED_VIEWS",
    "EvaluationBudget",
    "ExportQAReport",
    "GeometryMetricReport",
    "NovelViewMetricReport",
    "RecoverabilityReport",
    "RegressionBudget",
    "ArtifactRecord",
    "RunLineage",
    "Threshold",
    "ViewRequest",
    "SilhouetteGateConfig",
    "SelectionEvidence",
    "active_view_plan_payload",
    "attach_selection",
    "boundary_refinement_plan_payload",
    "bundle_from_candidate",
    "capture_environment",
    "chamfer_distance",
    "dirty_worktree",
    "evaluate_budget",
    "evaluate_silhouette_pair",
    "export_qa_aggregate_score",
    "export_qa_report_from_mapping",
    "fscore_at_tolerance",
    "hash_file",
    "hash_json_payload",
    "image_pair_report",
    "image_set_report",
    "missing_silhouette_view",
    "normal_consistency",
    "pareto_front",
    "psnr_from_mse",
    "repo_revision",
    "sota_silhouette_budget",
    "summarize_silhouette_views",
    "suggest_next_views",
    "surface_distance_report",
    "volumetric_iou",
    "write_reproduce_script",
    "write_run_lineage",
    "json_safe",
]
