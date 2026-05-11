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


def _geometry_group(metrics: Any) -> MetricGroup:
    if metrics is None:
        return MetricGroup("geometry", "not_applicable")
    extras = getattr(metrics, "extras", {}) or {}
    geometry = extras.get("geometry") if isinstance(extras, Mapping) else None
    if not isinstance(geometry, Mapping):
        geometry = (
            extras.get("geometry_metrics") if isinstance(extras, Mapping) else None
        )
    recoverability = (
        extras.get("recoverability") if isinstance(extras, Mapping) else None
    )
    true_geometry = (
        extras.get("geometry_true", extras.get("true_geometry"))
        if isinstance(extras, Mapping)
        else None
    )
    recoverable_geometry = (
        extras.get("geometry_recoverable", extras.get("recoverable_geometry"))
        if isinstance(extras, Mapping)
        else None
    )
    if isinstance(recoverability, Mapping):
        true_geometry = true_geometry or recoverability.get("true_geometry") or recoverability.get("true")
        recoverable_geometry = (
            recoverable_geometry
            or recoverability.get("recoverable_geometry")
            or recoverability.get("recoverable")
            or recoverability.get("visual_hull_envelope")
        )
    if (
        not isinstance(geometry, Mapping)
        and not isinstance(true_geometry, Mapping)
        and not isinstance(recoverable_geometry, Mapping)
    ):
        return MetricGroup("geometry", "not_applicable")
    values: list[MetricValue] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}
    if isinstance(geometry, Mapping):
        report = report_from_mapping(geometry)
        values.extend(
            _geometry_metric_values(
                "geometry",
                report,
                source=str(geometry.get("source", "computed")),
            )
        )
        warnings.extend(report.warnings)
        metadata["geometry"] = report.to_dict()
    if isinstance(true_geometry, Mapping):
        report = report_from_mapping(true_geometry)
        values.extend(
            _geometry_metric_values(
                "geometry.true",
                report,
                source=str(true_geometry.get("source", "synthetic_ground_truth")),
            )
        )
        warnings.extend(report.warnings)
        metadata["true_geometry"] = report.to_dict()
    if isinstance(recoverable_geometry, Mapping):
        report = report_from_mapping(recoverable_geometry)
        values.extend(
            _geometry_metric_values(
                "geometry.recoverable",
                report,
                source=str(recoverable_geometry.get("source", "recoverable_envelope")),
            )
        )
        warnings.extend(report.warnings)
        metadata["recoverable_geometry"] = report.to_dict()
    if isinstance(recoverability, Mapping):
        report = recoverability_report_from_mapping(recoverability)
        metadata["recoverability"] = report.to_dict()
        for name, value in (
            ("geometry.ambiguity_gap_chamfer_l1", report.ambiguity_gap_chamfer_l1),
            ("geometry.ambiguity_gap_chamfer_l2", report.ambiguity_gap_chamfer_l2),
            ("geometry.ambiguity_gap_volume_iou", report.ambiguity_gap_volume_iou),
        ):
            if value is not None:
                values.append(
                    MetricValue(
                        name,
                        value,
                        higher_is_better=False,
                        status="pass",
                        source=report.source,
                    )
                )
        warnings.extend(report.warnings)
    status = "pass" if values else "not_applicable"
    return MetricGroup(
        "geometry",
        status,
        tuple(values),
        warnings=tuple(dict.fromkeys(warnings)),
        metadata=metadata,
    )

def _recoverability_group(metrics: Any) -> MetricGroup:
    if metrics is None:
        return MetricGroup("recoverability", "not_applicable")
    extras = getattr(metrics, "extras", {}) or {}
    payload = extras.get("recoverability") if isinstance(extras, Mapping) else None
    if not isinstance(payload, Mapping):
        return MetricGroup("recoverability", "not_applicable")
    report = recoverability_report_from_mapping(payload)
    values: list[MetricValue] = []
    for name, value, higher in (
        ("recoverability.ambiguity_gap_chamfer_l1", report.ambiguity_gap_chamfer_l1, False),
        ("recoverability.ambiguity_gap_chamfer_l2", report.ambiguity_gap_chamfer_l2, False),
        ("recoverability.ambiguity_gap_volume_iou", report.ambiguity_gap_volume_iou, False),
    ):
        if value is not None:
            values.append(
                MetricValue(
                    name,
                    value,
                    higher_is_better=higher,
                    status="pass",
                    source=report.source,
                )
            )
    if report.true_geometry is not None:
        values.extend(
            _geometry_metric_values(
                "recoverability.true",
                report.true_geometry,
                source=report.source,
            )
        )
    if report.recoverable_geometry is not None:
        values.extend(
            _geometry_metric_values(
                "recoverability.recoverable",
                report.recoverable_geometry,
                source=report.source,
            )
        )
    return MetricGroup(
        "recoverability",
        "pass" if values else "not_applicable",
        tuple(values),
        warnings=report.warnings,
        metadata=report.to_dict(),
    )

def _geometry_metric_values(
    prefix: str,
    report: GeometryMetricReport,
    *,
    source: str,
) -> tuple[MetricValue, ...]:
    values: list[MetricValue] = []
    for name, value, higher in (
        (f"{prefix}.chamfer_l1", report.chamfer_l1, False),
        (f"{prefix}.chamfer_l1_normalized", report.chamfer_l1, False),
        (f"{prefix}.chamfer_l2", report.chamfer_l2, False),
        (f"{prefix}.chamfer_l2_normalized", report.chamfer_l2, False),
        (f"{prefix}.fscore_tau", report.fscore_tau, True),
        (f"{prefix}.volumetric_iou", report.volumetric_iou, True),
        (f"{prefix}.normal_consistency", report.normal_consistency, True),
        (f"{prefix}.surface_coverage", report.surface_coverage, True),
        (f"{prefix}.ambiguity_gap", report.ambiguity_gap, False),
    ):
        if value is not None:
            values.append(
                MetricValue(
                    name,
                    value,
                    higher_is_better=higher,
                    status="pass",
                    source=source,
                )
            )
    if report.fscore_tolerance is not None:
        values.append(
            MetricValue(
                f"{prefix}.fscore_tolerance",
                report.fscore_tolerance,
                higher_is_better=None,
                status="not_applicable",
                source=source,
            )
        )
    if report.sample_count_ref:
        values.append(
            MetricValue(
                f"{prefix}.sample_count_ref",
                report.sample_count_ref,
                unit="points",
                higher_is_better=None,
                status="not_applicable",
                source=source,
            )
        )
    if report.sample_count_candidate:
        values.append(
            MetricValue(
                f"{prefix}.sample_count_candidate",
                report.sample_count_candidate,
                unit="points",
                higher_is_better=None,
                status="not_applicable",
                source=source,
            )
        )
    return tuple(values)
