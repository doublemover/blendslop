from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from ..appearance import reports_from_payload
from ..capabilities import backend_dependency_state
from ..cost_model import cost_report_from_candidate
from ..editability import report_from_candidate_metrics
from ..export_qa import aggregate_score as export_qa_aggregate_score
from ..export_qa import reports_from_payload as export_qa_reports_from_payload
from ..failure_taxonomy import classify_bundle_failures
from ..geometry import GeometryMetricReport, report_from_mapping
from ..lineage import repo_revision
from ..novel_view import report_from_mapping as novel_view_report_from_mapping
from ..recoverability import report_from_mapping as recoverability_report_from_mapping
from ..schemas import EvaluationBundle, MetricGroup, MetricValue, utc_now_iso, worst_status
from .helpers import _float_or_none, _metric_key_fragment, _optional_bool_status, _pass_fail, _ratio_status, _score_status, _texture_memory_status


def _editability_group(metrics: Any) -> MetricGroup:
    if metrics is None:
        return MetricGroup("editability", "not_applicable")
    report = report_from_candidate_metrics(metrics)
    index = max(
        report.editable_reconstruction_index,
        float(getattr(metrics, "editability_score", 0.0) or 0.0),
    )
    status = _score_status(index, warn_floor=0.35, fail_floor=0.05)
    return MetricGroup(
        "editability",
        status,
        (
            MetricValue(
                "editability.editable_reconstruction_index",
                index,
                higher_is_better=True,
                required=True,
                status=status,
            ),
            MetricValue(
                "editability.object_hierarchy_score",
                report.object_hierarchy_score,
                higher_is_better=True,
                status="pass",
            ),
            MetricValue(
                "editability.primitive_score",
                report.primitive_score,
                higher_is_better=True,
                status="pass",
            ),
            MetricValue(
                "editability.modifier_score",
                report.modifier_score,
                higher_is_better=True,
                status="pass",
            ),
            MetricValue(
                "editability.mesh_density_score",
                report.mesh_density_score,
                higher_is_better=True,
                status="pass",
            ),
            MetricValue(
                "editability.semantic_part_score",
                report.semantic_part_score,
                higher_is_better=True,
                status="pass",
            ),
            MetricValue(
                "editability.export_roundtrip_score",
                report.export_roundtrip_score,
                higher_is_better=True,
                status="pass",
            ),
            MetricValue(
                "editability.complexity_penalty",
                float(getattr(metrics, "complexity_penalty", 0.0)),
                higher_is_better=False,
                status="pass",
            ),
        ),
        warnings=report.warnings,
        metadata=report.to_dict(),
    )
