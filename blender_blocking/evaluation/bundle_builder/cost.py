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


def _cost_group(result: Any) -> MetricGroup:
    report = cost_report_from_candidate(result)
    values = [
        MetricValue(
            "cost.total_wall_ms",
            report.total_wall_ms,
            unit="ms",
            higher_is_better=False,
            status="pass",
        ),
    ]
    if report.peak_memory_mb is not None:
        values.append(
            MetricValue(
                "cost.peak_memory_mb",
                report.peak_memory_mb,
                unit="MiB",
                higher_is_better=False,
                status="pass",
            )
        )
    for name, value in report.throughput.items():
        values.append(
            MetricValue(
                f"cost.throughput.{name}",
                float(value),
                higher_is_better=True,
                status="pass",
            )
        )
    for name, value in report.cache.items():
        values.append(
            MetricValue(
                f"cost.cache.{name}",
                _float_or_none(value),
                higher_is_better=False if name.endswith(("misses", "bytes_stored")) else True,
                status="pass",
            )
        )
    for stage in report.stages:
        safe_stage = str(stage.stage).replace(" ", "_")
        values.append(
            MetricValue(
                f"cost.stage.{safe_stage}.wall_ms",
                float(stage.wall_ms),
                unit="ms",
                higher_is_better=False,
                status=stage.status,
            )
        )
        if stage.peak_memory_mb is not None:
            values.append(
                MetricValue(
                    f"cost.stage.{safe_stage}.peak_memory_mb",
                    float(stage.peak_memory_mb),
                    unit="MiB",
                    higher_is_better=False,
                    status=stage.status,
                )
            )
        if stage.artifact_bytes:
            values.append(
                MetricValue(
                    f"cost.stage.{safe_stage}.artifact_bytes",
                    float(stage.artifact_bytes),
                    unit="byte",
                    higher_is_better=False,
                    status=stage.status,
                )
            )
        for unit_name, unit_value in stage.work_units.items():
            values.append(
                MetricValue(
                    f"cost.stage.{safe_stage}.work_units.{unit_name}",
                    _float_or_none(unit_value),
                    higher_is_better=None,
                    status=stage.status,
                )
            )
    return MetricGroup(
        "cost",
        "pass",
        tuple(values),
        metadata=report.to_dict(),
    )
