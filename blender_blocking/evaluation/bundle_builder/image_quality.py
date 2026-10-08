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


def _novel_view_group(metrics: Any) -> MetricGroup:
    if metrics is None:
        return MetricGroup("novel_view", "not_applicable")
    extras = getattr(metrics, "extras", {}) or {}
    payload = extras.get("novel_view") if isinstance(extras, Mapping) else None
    if not isinstance(payload, Mapping):
        payload = extras.get("image_metrics") if isinstance(extras, Mapping) else None
    if not isinstance(payload, Mapping):
        return MetricGroup("novel_view", "not_applicable")
    report = novel_view_report_from_mapping(payload)
    values = []
    for name, value, higher, unit in (
        ("novel_view.psnr", report.psnr, True, "dB"),
        ("novel_view.ssim", report.ssim, True, None),
        ("novel_view.lpips", report.lpips, False, None),
        ("novel_view.mse", report.mse, False, None),
    ):
        if value is not None:
            values.append(
                MetricValue(
                    name,
                    value,
                    unit=unit,
                    higher_is_better=higher,
                    status="pass",
                    source=str(payload.get("source", "computed")),
                )
            )
    if report.image_count:
        values.append(
            MetricValue(
                "novel_view.image_count",
                report.image_count,
                unit="image",
                higher_is_better=None,
                status="not_applicable",
            )
        )
    return MetricGroup(
        "novel_view",
        "pass" if values else "not_applicable",
        tuple(values),
        warnings=report.warnings,
        metadata=report.to_dict(),
    )
