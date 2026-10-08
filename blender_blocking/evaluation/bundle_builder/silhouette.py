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


def _silhouette_group(metrics: Any) -> MetricGroup:
    if metrics is None:
        return MetricGroup(
            "silhouette",
            "fail",
            (
                MetricValue(
                    "silhouette.min_view_iou",
                    None,
                    higher_is_better=True,
                    required=True,
                    status="fail",
                    notes=("candidate emitted no CandidateMetrics",),
                ),
            ),
            warnings=("missing_candidate_metrics",),
        )
    per_view = getattr(metrics, "per_view", {}) or {}
    min_iou = float(getattr(metrics, "area_iou_min", 0.0))
    avg_iou = float(getattr(metrics, "area_iou_mean", 0.0))
    boundary_iou = float(getattr(metrics, "boundary_iou_mean", 0.0))
    min_status = _pass_fail(min_iou > 0.0)
    required_view_count = 0
    failed_required_view_count = 0
    missing_required_metric_count = 0
    warnings: list[str] = []
    values: list[MetricValue] = [
        MetricValue(
            "silhouette.min_view_iou",
            min_iou,
            higher_is_better=True,
            required=True,
            status=min_status,
            notes=() if min_iou > 0.0 else ("missing or zero required-view IoU",),
        ),
        MetricValue(
            "silhouette.average_iou",
            avg_iou,
            higher_is_better=True,
            status=_pass_fail(avg_iou > 0.0),
        ),
        MetricValue(
            "silhouette.mean_boundary_iou",
            boundary_iou,
            higher_is_better=True,
            required=True,
            status=_pass_fail(boundary_iou > 0.0),
            notes=()
            if boundary_iou > 0.0
            else ("missing or zero required-view Boundary IoU",),
        ),
    ]
    boundary_values = []
    sdf_values = []
    view_statuses = [min_status]
    if not per_view and (min_iou > 0.0 or avg_iou > 0.0 or boundary_iou > 0.0):
        warnings.append("per_view_metrics_missing_aggregate_only")
    for view, payload in per_view.items():
        if not isinstance(payload, Mapping):
            continue
        prefix = f"silhouette.per_view.{view}"
        required = bool(payload.get("required", True))
        if required:
            required_view_count += 1
        passed = bool(payload.get("passed", payload.get("pass", not required)))
        if required and not passed:
            failed_required_view_count += 1
        view_statuses.append("pass" if passed else "fail")
        values.append(
            MetricValue(
                f"{prefix}.area_iou",
                _float_or_none(payload.get("area_iou")),
                higher_is_better=True,
                required=required,
                status="pass" if passed else "fail",
                notes=(str(payload.get("reason", "")),)
                if payload.get("reason")
                else (),
            )
        )
        boundary = _float_or_none(payload.get("boundary_iou"))
        if boundary is not None:
            boundary_values.append(boundary)
            values.append(
                MetricValue(
                    f"{prefix}.boundary_iou",
                    boundary,
                    higher_is_better=True,
                    required=required,
                    status="pass" if passed else "fail",
                )
            )
        elif required:
            missing_required_metric_count += 1
            values.append(
                MetricValue(
                    f"{prefix}.boundary_iou",
                    None,
                    higher_is_better=True,
                    required=True,
                    status="fail",
                    notes=("required Boundary IoU missing",),
                )
            )
        sdf = _float_or_none(payload.get("signed_distance_loss"))
        if sdf is not None:
            sdf_values.append(sdf)
            values.append(
                MetricValue(
                    f"{prefix}.signed_distance_loss",
                    sdf,
                    higher_is_better=False,
                    required=required,
                    status="pass" if passed else "fail",
                )
            )
        elif required:
            missing_required_metric_count += 1
            values.append(
                MetricValue(
                    f"{prefix}.signed_distance_loss",
                    None,
                    higher_is_better=False,
                    required=True,
                    status="fail",
                    notes=("required signed-distance loss missing",),
                )
            )
    if boundary_values:
        values.append(
            MetricValue(
                "silhouette.min_boundary_iou",
                min(boundary_values),
                higher_is_better=True,
                required=True,
                status=_pass_fail(min(boundary_values) > 0.0),
            )
        )
    elif per_view:
        values.append(
            MetricValue(
                "silhouette.min_boundary_iou",
                None,
                higher_is_better=True,
                required=True,
                status="fail",
                notes=("no per-view Boundary IoU values were emitted",),
            )
        )
    if sdf_values:
        values.append(
            MetricValue(
                "silhouette.mean_signed_distance_loss",
                sum(sdf_values) / len(sdf_values),
                higher_is_better=False,
                required=True,
                status="pass",
            )
        )
    elif per_view:
        values.append(
            MetricValue(
                "silhouette.mean_signed_distance_loss",
                None,
                higher_is_better=False,
                required=True,
                status="fail",
                notes=("no per-view signed-distance losses were emitted",),
            )
        )
    values.extend(
        (
            MetricValue(
                "silhouette.required_view_count",
                required_view_count,
                unit="view",
                higher_is_better=None,
                status="not_applicable",
            ),
            MetricValue(
                "silhouette.failed_required_view_count",
                failed_required_view_count,
                unit="view",
                higher_is_better=False,
                status=_pass_fail(failed_required_view_count == 0),
            ),
            MetricValue(
                "silhouette.missing_required_metric_count",
                missing_required_metric_count,
                unit="metric",
                higher_is_better=False,
                status=_pass_fail(missing_required_metric_count == 0),
            ),
        )
    )
    if missing_required_metric_count:
        view_statuses.append("fail")
    if failed_required_view_count:
        view_statuses.append("fail")
    return MetricGroup(
        "silhouette",
        worst_status(view_statuses or ("pass",)),
        tuple(values),
        warnings=tuple(warnings),
    )
