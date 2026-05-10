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


def _diagnostic_group(metrics: Any) -> MetricGroup:
    if metrics is None:
        return MetricGroup("diagnostics", "not_applicable")
    extras = getattr(metrics, "extras", {}) or {}
    diagnostics = (
        extras.get("visual_hull_view_diagnostics")
        if isinstance(extras, Mapping)
        else None
    )
    if not isinstance(diagnostics, Mapping):
        return MetricGroup("diagnostics", "not_applicable")
    failed_views = diagnostics.get("failed_views", ())
    if not isinstance(failed_views, (list, tuple)):
        failed_views = ()
    top_failures = diagnostics.get("top_like_failures", ())
    if not isinstance(top_failures, (list, tuple)):
        top_failures = ()
    axis_suspect = bool(diagnostics.get("axis_or_transform_suspect"))
    catastrophic = bool(diagnostics.get("catastrophic_view_failure"))
    values = (
        MetricValue(
            "diagnostics.visual_hull.axis_or_transform_suspect",
            axis_suspect,
            higher_is_better=False,
            status=_pass_fail(not axis_suspect),
            source="visual_hull_view_diagnostics",
        ),
        MetricValue(
            "diagnostics.visual_hull.catastrophic_view_failure",
            catastrophic,
            higher_is_better=False,
            status=_pass_fail(not catastrophic),
            source="visual_hull_view_diagnostics",
        ),
        MetricValue(
            "diagnostics.visual_hull.failed_view_count",
            len(failed_views),
            unit="view",
            higher_is_better=False,
            status=_pass_fail(len(failed_views) == 0),
            source="visual_hull_view_diagnostics",
        ),
        MetricValue(
            "diagnostics.visual_hull.top_like_failure_count",
            len(top_failures),
            unit="view",
            higher_is_better=False,
            status=_pass_fail(len(top_failures) == 0),
            source="visual_hull_view_diagnostics",
        ),
    )
    status = worst_status(tuple(value.status for value in values))
    return MetricGroup(
        "diagnostics",
        status,
        values,
        warnings=(
            ("axis_or_transform_suspect",) if axis_suspect else ()
        )
        + (("catastrophic_view_failure",) if catastrophic else ()),
        metadata={"visual_hull": diagnostics},
    )

def _export_qa_group(metrics: Any) -> MetricGroup:
    if metrics is None:
        return MetricGroup("export_qa", "not_applicable")
    extras = getattr(metrics, "extras", {}) or {}
    payload = None
    if isinstance(extras, Mapping):
        for key in ("export_qa", "asset_export", "export"):
            candidate = extras.get(key)
            if candidate is not None:
                payload = candidate
                break
    reports = export_qa_reports_from_payload(payload)
    if not reports:
        return MetricGroup("export_qa", "not_applicable")
    statuses = []
    values: list[MetricValue] = [
        MetricValue(
            "export.target_count",
            len(reports),
            unit="target",
            higher_is_better=None,
            status="not_applicable",
        )
    ]
    aggregate = export_qa_aggregate_score(reports)
    if aggregate is not None:
        values.append(
            MetricValue(
                "export.qa_score",
                aggregate,
                higher_is_better=True,
                status=_pass_fail(aggregate > 0.0),
            )
        )
    total_objects = 0
    total_vertices = 0
    total_faces = 0
    total_materials = 0
    count_present = {"object": False, "vertex": False, "face": False, "material": False}
    warnings: list[str] = []
    errors: list[str] = []
    any_reimport = False
    aggregate_status_ok = True
    aggregate_reimport_ok = True
    for report in reports:
        status_ok = report.status_ok
        reimport_ok = report.reimport_ok
        aggregate_status_ok = aggregate_status_ok and status_ok
        if reimport_ok is not None:
            any_reimport = True
            aggregate_reimport_ok = aggregate_reimport_ok and reimport_ok
        statuses.append("pass" if status_ok and reimport_ok is not False else "fail")
        target_key = _metric_key_fragment(report.target)
        values.append(
            MetricValue(
                f"export.per_target.{target_key}.qa_score",
                report.qa_score,
                higher_is_better=True,
                status=_pass_fail(report.qa_score > 0.0),
                source="export_qa",
            )
        )
        values.append(
            MetricValue(
                f"export.per_target.{target_key}.status_ok",
                status_ok,
                higher_is_better=True,
                status=_pass_fail(status_ok),
                source="export_qa",
            )
        )
        if reimport_ok is not None:
            values.append(
                MetricValue(
                    f"export.per_target.{target_key}.reimport_ok",
                    reimport_ok,
                    higher_is_better=True,
                    status=_pass_fail(reimport_ok),
                    source="export_qa",
                )
            )
        if report.object_count is not None:
            count_present["object"] = True
            total_objects += report.object_count
        if report.vertex_count is not None:
            count_present["vertex"] = True
            total_vertices += report.vertex_count
        if report.face_count is not None:
            count_present["face"] = True
            total_faces += report.face_count
        if report.material_count is not None:
            count_present["material"] = True
            total_materials += report.material_count
        warnings.extend(report.warnings)
        errors.extend(report.errors)
    values.append(
        MetricValue(
            "export.status_ok",
            aggregate_status_ok,
            higher_is_better=True,
            status=_pass_fail(aggregate_status_ok),
        )
    )
    if any_reimport:
        values.append(
            MetricValue(
                "export.reimport_ok",
                aggregate_reimport_ok,
                higher_is_better=True,
                status=_pass_fail(aggregate_reimport_ok),
            )
        )
    if count_present["object"]:
        values.append(
            MetricValue(
                "export.object_count",
                total_objects,
                unit="object",
                higher_is_better=None,
                status="pass",
            )
        )
    if count_present["vertex"]:
        values.append(
            MetricValue(
                "export.vertex_count",
                total_vertices,
                unit="vertex",
                higher_is_better=None,
                status="pass",
            )
        )
    if count_present["face"]:
        values.append(
            MetricValue(
                "export.face_count",
                total_faces,
                unit="face",
                higher_is_better=None,
                status="pass",
            )
        )
    if count_present["material"]:
        values.append(
            MetricValue(
                "export.material_count",
                total_materials,
                unit="material",
                higher_is_better=None,
                status="pass",
            )
        )
    if errors:
        group_status = "fail"
    elif statuses and all(status == "pass" for status in statuses):
        group_status = "pass" if not warnings else "warn"
    else:
        group_status = "fail"
    return MetricGroup(
        "export_qa",
        group_status,
        tuple(values),
        warnings=tuple(warnings),
        errors=tuple(errors),
        metadata={"reports": [report.to_dict() for report in reports]},
    )
