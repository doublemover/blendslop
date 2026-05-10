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


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _pass_fail(value: bool) -> str:
    return "pass" if value else "fail"

def _score_status(value: float, *, warn_floor: float, fail_floor: float) -> str:
    if value < fail_floor:
        return "fail"
    if value < warn_floor:
        return "warn"
    return "pass"

def _ratio_status(value: float | int | None, *, warn: float, fail: float) -> str:
    if value is None:
        return "not_applicable"
    parsed = float(value)
    if parsed > fail:
        return "fail"
    if parsed > warn:
        return "warn"
    return "pass"

def _texture_memory_status(value: float | None, maximum: float | None) -> str:
    if value is None:
        return "not_applicable"
    if maximum is None:
        return "pass"
    return _pass_fail(float(value) <= float(maximum))

def _optional_bool_status(value: bool | None, *, fail_when_false: bool) -> str:
    if value is None:
        return "not_applicable"
    if value:
        return "pass"
    return "fail" if fail_when_false else "warn"

def _metric_key_fragment(value: object) -> str:
    text = str(value or "asset").strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    collapsed = "_".join(part for part in "".join(chars).split("_") if part)
    return collapsed or "asset"
