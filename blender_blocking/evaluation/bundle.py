"""EvaluationBundle adapters for reconstruction candidates."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .bundle_builder import (
    _appearance_group,
    _artifact_group,
    _artifact_paths,
    _bundle_status,
    _cost_group,
    _dependency_state,
    _diagnostic_group,
    _editability_group,
    _export_qa_group,
    _geometry_group,
    _novel_view_group,
    _recoverability_group,
    _silhouette_group,
    _status_with_failures,
    _timings,
    _topology_group,
)
from .failure_taxonomy import classify_bundle_failures
from .lineage import repo_revision
from .schemas import EvaluationBundle, utc_now_iso

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
        _appearance_group(metrics),
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
