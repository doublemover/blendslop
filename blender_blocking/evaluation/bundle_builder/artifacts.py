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
