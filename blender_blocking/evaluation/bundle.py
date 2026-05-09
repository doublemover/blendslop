"""EvaluationBundle adapters for reconstruction candidates."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .capabilities import backend_dependency_state
from .cost_model import cost_report_from_candidate
from .editability import report_from_candidate_metrics
from .export_qa import aggregate_score as export_qa_aggregate_score
from .export_qa import reports_from_payload as export_qa_reports_from_payload
from .failure_taxonomy import classify_bundle_failures
from .geometry import GeometryMetricReport, report_from_mapping
from .lineage import repo_revision
from .novel_view import report_from_mapping as novel_view_report_from_mapping
from .recoverability import report_from_mapping as recoverability_report_from_mapping
from .schemas import (
    EvaluationBundle,
    MetricGroup,
    MetricValue,
    utc_now_iso,
    worst_status,
)


def bundle_from_candidate(
    *,
    result: Any,
    target: Any = None,
    suite: str = "",
    run_id: str = "",
    repo: str | None = None,
) -> EvaluationBundle:
    """Create a multi-metric evaluation bundle from a CandidateResult-like object."""
    metrics = getattr(result, "metric_result", None)
    metric_groups = (
        _silhouette_group(metrics),
        _geometry_group(metrics),
        _recoverability_group(metrics),
        _novel_view_group(metrics),
        _topology_group(metrics),
        _editability_group(metrics),
        _export_qa_group(metrics),
        _diagnostic_group(metrics),
        _cost_group(result),
        _artifact_group(result),
    )
    errors = tuple(str(item) for item in getattr(result, "errors", ()) or ())
    warnings = tuple(str(item) for item in getattr(result, "warnings", ()) or ())
    status = _bundle_status(result, metric_groups)
    extras = getattr(metrics, "extras", {}) if metrics is not None else {}
    dependency_state = _dependency_state(
        result, extras if isinstance(extras, Mapping) else {}
    )
    degradation_state = {
        "degraded": bool(getattr(result, "degraded", False)),
        "candidate_status": str(getattr(result, "status", "")),
    }
    bundle = EvaluationBundle(
        schema_version="evaluation-bundle-v1",
        run_id=run_id or str(getattr(result, "candidate_id", "")),
        created_at_utc=utc_now_iso(),
        repo_revision=repo if repo is not None else repo_revision(),
        mode=str(getattr(result, "backend_name", "")),
        suite=suite,
        candidate_id=str(getattr(result, "candidate_id", "")),
        target_id=_target_id(target),
        status=status,
        metric_groups=metric_groups,
        artifacts=_artifact_paths(result),
        dependency_state=dependency_state,
        degradation_state=degradation_state,
        timings_ms=_timings(metrics),
        failures=(),
        errors=errors,
        warnings=warnings,
    )
    failures = classify_bundle_failures(bundle)
    return replace(
        bundle,
        failures=failures,
        status=_status_with_failures(bundle.status, failures),
    )


def _target_id(target: Any) -> str:
    if target is None:
        return ""
    for name in ("constraint_hash", "config_hash"):
        value = getattr(target, name, "")
        if value:
            return str(value)
    return str(id(target))


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


def _topology_group(metrics: Any) -> MetricGroup:
    if metrics is None:
        return MetricGroup("topology", "not_applicable")
    extras = getattr(metrics, "extras", {}) or {}
    topology = extras.get("topology") if isinstance(extras, Mapping) else None
    values = [
        MetricValue(
            "topology.score",
            float(getattr(metrics, "topology_score", 0.0)),
            higher_is_better=True,
            status="pass",
        ),
        MetricValue(
            "topology.penalty",
            float(getattr(metrics, "topology_penalty", 0.0)),
            higher_is_better=False,
            status="pass",
        ),
    ]
    if isinstance(topology, Mapping):
        for key in (
            "connected_components",
            "boundary_edges",
            "non_manifold_edges",
            "degenerate_faces",
        ):
            if key in topology:
                values.append(
                    MetricValue(
                        f"topology.{key}",
                        _float_or_none(topology.get(key)),
                        higher_is_better=False,
                        status="pass",
                    )
                )
        if "watertight" in topology:
            values.append(
                MetricValue(
                    "topology.watertight",
                    bool(topology.get("watertight")),
                    higher_is_better=True,
                    status=_pass_fail(bool(topology.get("watertight"))),
                )
            )
    return MetricGroup("topology", "pass", tuple(values))


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
        (f"{prefix}.chamfer_l2", report.chamfer_l2, False),
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
    return MetricGroup(
        "cost",
        "pass",
        tuple(values),
        metadata=report.to_dict(),
    )


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


def _artifact_group(result: Any) -> MetricGroup:
    artifacts = _artifact_paths(result)
    values = [
        MetricValue(
            "artifact.count",
            len(artifacts),
            unit="count",
            higher_is_better=None,
            status="pass",
        )
    ]
    return MetricGroup(
        "artifacts", "pass", tuple(values), metadata={"artifacts": artifacts}
    )


def _bundle_status(result: Any, groups: tuple[MetricGroup, ...]) -> str:
    raw_status = str(getattr(result, "status", "")).lower()
    if raw_status in {"failed", "error"} or getattr(result, "errors", ()):
        return "fail"
    if raw_status == "skipped":
        return "skip"
    if raw_status == "research_only":
        return "research_only"
    if bool(getattr(result, "degraded", False)) or raw_status == "degraded":
        return "degraded"
    group_status = worst_status(tuple(group.status for group in groups))
    return "pass" if group_status in {"pass", "not_applicable"} else group_status


def _status_with_failures(status: str, failures: tuple[Any, ...]) -> str:
    if not failures:
        return status
    severities = tuple(str(getattr(failure, "severity", "")) for failure in failures)
    return worst_status((status, *severities))


def _artifact_paths(result: Any) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for attr in ("mesh_path", "primitive_path", "volume_path"):
        value = getattr(result, attr, None)
        if value:
            artifacts[attr.removesuffix("_path")] = str(value)
    for key, value in (getattr(result, "artifacts", {}) or {}).items():
        artifacts[str(key)] = str(value)
    for key, value in (getattr(result, "render_paths", {}) or {}).items():
        artifacts[f"render.{key}"] = str(value)
    return artifacts


def _dependency_state(result: Any, extras: Mapping[str, Any]) -> dict[str, Any]:
    dependency_state: dict[str, Any] = {}
    optional = extras.get("optional_dependencies")
    if isinstance(optional, Mapping):
        dependency_state.update(optional)
    capabilities = getattr(getattr(result, "backend", None), "capabilities", None)
    if capabilities is not None:
        dependency_state.update(
            backend_dependency_state(getattr(capabilities, "optional_dependencies", ()))
        )
    return dependency_state


def _timings(metrics: Any) -> dict[str, float]:
    if metrics is None:
        return {}
    elapsed = float(getattr(metrics, "elapsed_s", 0.0) or 0.0)
    return {"candidate_elapsed_ms": elapsed * 1000.0} if elapsed else {}


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


def _metric_key_fragment(value: object) -> str:
    text = str(value or "asset").strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    collapsed = "_".join(part for part in "".join(chars).split("_") if part)
    return collapsed or "asset"
