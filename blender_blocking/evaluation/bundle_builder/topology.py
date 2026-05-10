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
            "loose_vertices",
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
