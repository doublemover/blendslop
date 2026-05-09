"""EvaluationBundle adapters for reconstruction candidates."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .capabilities import backend_dependency_state
from .cost_model import cost_report_from_candidate
from .failure_taxonomy import classify_bundle_failures
from .lineage import repo_revision
from .schemas import EvaluationBundle, MetricGroup, MetricValue, utc_now_iso, worst_status


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
        _topology_group(metrics),
        _editability_group(metrics),
        _cost_group(result),
        _artifact_group(result),
    )
    errors = tuple(str(item) for item in getattr(result, "errors", ()) or ())
    warnings = tuple(str(item) for item in getattr(result, "warnings", ()) or ())
    status = _bundle_status(result, metric_groups)
    extras = getattr(metrics, "extras", {}) if metrics is not None else {}
    dependency_state = _dependency_state(result, extras if isinstance(extras, Mapping) else {})
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
    return replace(bundle, failures=failures)


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
        return MetricGroup("silhouette", "not_applicable")
    per_view = getattr(metrics, "per_view", {}) or {}
    values: list[MetricValue] = [
        MetricValue("silhouette.min_view_iou", float(getattr(metrics, "area_iou_min", 0.0)), higher_is_better=True, required=True, status=_pass_fail(float(getattr(metrics, "area_iou_min", 0.0)) > 0.0)),
        MetricValue("silhouette.average_iou", float(getattr(metrics, "area_iou_mean", 0.0)), higher_is_better=True, status="pass"),
        MetricValue("silhouette.mean_boundary_iou", float(getattr(metrics, "boundary_iou_mean", 0.0)), higher_is_better=True, status="pass"),
    ]
    boundary_values = []
    sdf_values = []
    view_statuses = []
    for view, payload in per_view.items():
        if not isinstance(payload, Mapping):
            continue
        prefix = f"silhouette.per_view.{view}"
        required = bool(payload.get("required", True))
        passed = bool(payload.get("passed", payload.get("pass", not required)))
        view_statuses.append("pass" if passed else "fail")
        values.append(
            MetricValue(
                f"{prefix}.area_iou",
                _float_or_none(payload.get("area_iou")),
                higher_is_better=True,
                required=required,
                status="pass" if passed else "fail",
                notes=(str(payload.get("reason", "")),) if payload.get("reason") else (),
            )
        )
        boundary = _float_or_none(payload.get("boundary_iou"))
        if boundary is not None:
            boundary_values.append(boundary)
            values.append(MetricValue(f"{prefix}.boundary_iou", boundary, higher_is_better=True, required=required, status="pass" if passed else "fail"))
        sdf = _float_or_none(payload.get("signed_distance_loss"))
        if sdf is not None:
            sdf_values.append(sdf)
            values.append(MetricValue(f"{prefix}.signed_distance_loss", sdf, higher_is_better=False, required=required, status="pass" if passed else "fail"))
    if boundary_values:
        values.append(MetricValue("silhouette.min_boundary_iou", min(boundary_values), higher_is_better=True, status="pass"))
    if sdf_values:
        values.append(MetricValue("silhouette.mean_signed_distance_loss", sum(sdf_values) / len(sdf_values), higher_is_better=False, status="pass"))
    return MetricGroup("silhouette", worst_status(view_statuses or ("pass",)), tuple(values))


def _topology_group(metrics: Any) -> MetricGroup:
    if metrics is None:
        return MetricGroup("topology", "not_applicable")
    extras = getattr(metrics, "extras", {}) or {}
    topology = extras.get("topology") if isinstance(extras, Mapping) else None
    values = [
        MetricValue("topology.score", float(getattr(metrics, "topology_score", 0.0)), higher_is_better=True, status="pass"),
        MetricValue("topology.penalty", float(getattr(metrics, "topology_penalty", 0.0)), higher_is_better=False, status="pass"),
    ]
    if isinstance(topology, Mapping):
        for key in ("connected_components", "boundary_edges", "non_manifold_edges", "degenerate_faces"):
            if key in topology:
                values.append(MetricValue(f"topology.{key}", _float_or_none(topology.get(key)), higher_is_better=False, status="pass"))
        if "watertight" in topology:
            values.append(MetricValue("topology.watertight", bool(topology.get("watertight")), higher_is_better=True, status=_pass_fail(bool(topology.get("watertight")))))
    return MetricGroup("topology", "pass", tuple(values))


def _editability_group(metrics: Any) -> MetricGroup:
    if metrics is None:
        return MetricGroup("editability", "not_applicable")
    return MetricGroup(
        "editability",
        "pass",
        (
            MetricValue(
                "editability.editable_reconstruction_index",
                float(getattr(metrics, "editability_score", 0.0)),
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
    )


def _cost_group(result: Any) -> MetricGroup:
    report = cost_report_from_candidate(result)
    return MetricGroup(
        "cost",
        "pass",
        (MetricValue("cost.total_wall_ms", report.total_wall_ms, unit="ms", higher_is_better=False, status="pass"),),
        metadata=report.to_dict(),
    )


def _artifact_group(result: Any) -> MetricGroup:
    artifacts = _artifact_paths(result)
    values = [
        MetricValue("artifact.count", len(artifacts), unit="count", higher_is_better=None, status="pass")
    ]
    return MetricGroup("artifacts", "pass", tuple(values), metadata={"artifacts": artifacts})


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
        dependency_state.update(backend_dependency_state(getattr(capabilities, "optional_dependencies", ())))
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
