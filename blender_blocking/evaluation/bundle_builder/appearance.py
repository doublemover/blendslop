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


def _appearance_group(metrics: Any) -> MetricGroup:
    if metrics is None:
        return MetricGroup("appearance", "not_applicable")
    extras = getattr(metrics, "extras", {}) or {}
    payload = None
    if isinstance(extras, Mapping):
        for key in (
            "appearance",
            "texture_material",
            "texture_materials",
            "texture",
            "materials",
        ):
            candidate = extras.get(key)
            if candidate is not None:
                payload = candidate
                break
    reports = reports_from_payload(payload)
    if not reports:
        return MetricGroup("appearance", "not_applicable")

    values: list[MetricValue] = [
        MetricValue(
            "appearance.report_count",
            len(reports),
            unit="report",
            higher_is_better=None,
            status="not_applicable",
        )
    ]
    statuses: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    for index, report in enumerate(reports):
        prefix = "appearance" if len(reports) == 1 else f"appearance.report_{index}"
        warnings.extend(report.warnings)
        errors.extend(report.errors)
        source = report.source
        uv_status = _optional_bool_status(
            report.uv_valid,
            fail_when_false=report.required or report.strict_uv,
        )
        statuses.append(uv_status)
        report_values = _appearance_metric_values(prefix, report, source=source)
        statuses.extend(
            metric.status
            for metric in report_values
            if metric.status != "not_applicable"
        )
        values.extend(report_values)
    if errors:
        statuses.append("fail")
    elif warnings:
        statuses.append("warn")
    group_status = worst_status(tuple(statuses) or ("pass",))
    return MetricGroup(
        "appearance",
        group_status,
        tuple(values),
        warnings=tuple(dict.fromkeys(warnings)),
        errors=tuple(dict.fromkeys(errors)),
        metadata={"reports": [report.to_dict() for report in reports]},
    )

def _appearance_metric_values(prefix: str, report: Any, *, source: str) -> list[MetricValue]:
    values: list[MetricValue] = []
    if report.has_uv_map is not None:
        values.append(
            MetricValue(
                f"{prefix}.has_uv_map",
                bool(report.has_uv_map),
                higher_is_better=True,
                required=bool(report.required),
                status=_optional_bool_status(
                    report.has_uv_map,
                    fail_when_false=bool(report.required),
                ),
                source=source,
            )
        )
    if report.uv_valid is not None:
        values.append(
            MetricValue(
                f"{prefix}.uv_valid",
                bool(report.uv_valid),
                higher_is_better=True,
                required=bool(report.required or report.strict_uv),
                status=_optional_bool_status(
                    report.uv_valid,
                    fail_when_false=bool(report.required or report.strict_uv),
                ),
                source=source,
            )
        )
    for name, value, unit, higher, status in (
        (
            "uv_island_count",
            report.uv_island_count,
            "island",
            None,
            "not_applicable",
        ),
        (
            "uv_overlap_ratio",
            report.uv_overlap_ratio,
            None,
            False,
            _ratio_status(report.uv_overlap_ratio, warn=0.05, fail=0.10),
        ),
        (
            "uv_out_of_bounds_ratio",
            report.uv_out_of_bounds_ratio,
            None,
            False,
            _ratio_status(report.uv_out_of_bounds_ratio, warn=0.01, fail=0.05),
        ),
        (
            "uv_stretch_mean",
            report.uv_stretch_mean,
            None,
            False,
            _ratio_status(report.uv_stretch_mean, warn=2.0, fail=4.0),
        ),
        (
            "texel_density_cv",
            report.texel_density_cv,
            None,
            False,
            _ratio_status(report.texel_density_cv, warn=0.5, fail=1.0),
        ),
        (
            "missing_uv_faces",
            report.missing_uv_faces,
            "face",
            False,
            _pass_fail((report.missing_uv_faces or 0) == 0),
        ),
        (
            "texture_resolution",
            report.texture_resolution,
            "pixel",
            None,
            "not_applicable",
        ),
        (
            "texture_file_count",
            report.texture_file_count,
            "file",
            None,
            "not_applicable",
        ),
        (
            "texture_memory_mb",
            report.texture_memory_mb,
            "MiB",
            False,
            _texture_memory_status(report.texture_memory_mb, report.max_texture_memory_mb),
        ),
        (
            "seam_visibility_score",
            report.seam_visibility_score,
            None,
            False,
            _ratio_status(report.seam_visibility_score, warn=0.2, fail=0.45),
        ),
        (
            "material_slot_count",
            report.material_slot_count,
            "material",
            None,
            "not_applicable",
        ),
        (
            "named_material_ratio",
            report.named_material_ratio,
            None,
            True,
            _score_status(report.named_material_ratio or 0.0, warn_floor=0.5, fail_floor=0.2)
            if report.named_material_ratio is not None
            else "not_applicable",
        ),
        (
            "procedural_material_ratio",
            report.procedural_material_ratio,
            None,
            None,
            "not_applicable",
        ),
        (
            "bitmap_material_ratio",
            report.bitmap_material_ratio,
            None,
            None,
            "not_applicable",
        ),
        (
            "pbr_channel_coverage_ratio",
            report.pbr_channel_coverage_ratio,
            None,
            True,
            _score_status(report.pbr_channel_coverage_ratio or 0.0, warn_floor=0.75, fail_floor=0.25)
            if report.pbr_channel_coverage_ratio is not None
            else "not_applicable",
        ),
        (
            "duplicate_material_count",
            report.duplicate_material_count,
            "material",
            False,
            _pass_fail((report.duplicate_material_count or 0) == 0),
        ),
        (
            "orphan_texture_count",
            report.orphan_texture_count,
            "texture",
            False,
            _pass_fail((report.orphan_texture_count or 0) == 0),
        ),
    ):
        if value is None:
            continue
        values.append(
            MetricValue(
                f"{prefix}.{name}",
                value,
                unit=unit,
                higher_is_better=higher,
                status=status,
                source=source,
            )
        )
    for channel, present in sorted(report.pbr_channel_coverage.items()):
        values.append(
            MetricValue(
                f"{prefix}.pbr_channel.{_metric_key_fragment(channel)}",
                bool(present),
                higher_is_better=True,
                status=_pass_fail(bool(present)),
                source=source,
            )
        )
    for name, value in sorted(report.reprojection_metrics.items()):
        metric_name = _metric_key_fragment(name)
        higher = metric_name not in {"lpips", "mse", "mae", "error"}
        values.append(
            MetricValue(
                f"{prefix}.reprojection_{metric_name}",
                value,
                higher_is_better=higher,
                status="pass",
                source=source,
            )
        )
    for name, value in sorted(report.appearance_attribution.items()):
        metric_name = _metric_key_fragment(name)
        higher: bool | None = None
        if metric_name in {"boundary_geometry_fidelity", "geometry_detail_score"}:
            higher = True
        elif metric_name in {"texture_only_detail_score", "normal_displacement_contribution"}:
            higher = False
        status = "pass"
        if metric_name == "texture_only_detail_score" and _float_or_none(value) is not None:
            status = _ratio_status(_float_or_none(value), warn=0.55, fail=0.8)
        values.append(
            MetricValue(
                f"{prefix}.attribution_{metric_name}",
                value,
                higher_is_better=higher,
                status=status,
                source=source,
            )
        )
    values.append(
        MetricValue(
            f"{prefix}.image_space_hallucination_warning",
            bool(report.image_space_hallucination_warning),
            higher_is_better=False,
            status=_pass_fail(not report.image_space_hallucination_warning),
            source=source,
        )
    )
    return values
