"""Small text and HTML report renderers for evaluation bundles."""

from __future__ import annotations

from html import escape
from typing import Iterable

from .schemas import EvaluationBundle


def compact_console_summary(bundle: EvaluationBundle) -> str:
    metrics = bundle.metric_index()
    parts = [
        f"Candidate {bundle.candidate_id} [{bundle.mode}] status={bundle.status}",
        f"  min_iou={_metric(metrics, 'silhouette.min_view_iou')}",
        f"  avg_iou={_metric(metrics, 'silhouette.average_iou')}",
        f"  boundary={_metric(metrics, 'silhouette.mean_boundary_iou')}",
        f"  editable={_metric(metrics, 'editability.editable_reconstruction_index')}",
        f"  failures={len(bundle.failures)}",
    ]
    return "\n".join(parts)


def markdown_report(bundles: Iterable[EvaluationBundle], *, title: str = "Evaluation Report") -> str:
    rows = []
    for bundle in bundles:
        metrics = bundle.metric_index()
        rows.append(
            "| {candidate} | {mode} | {status} | {min_iou} | {avg_iou} | {boundary} | {editable} | {failures} |".format(
                candidate=bundle.candidate_id,
                mode=bundle.mode,
                status=bundle.status,
                min_iou=_metric(metrics, "silhouette.min_view_iou"),
                avg_iou=_metric(metrics, "silhouette.average_iou"),
                boundary=_metric(metrics, "silhouette.mean_boundary_iou"),
                editable=_metric(metrics, "editability.editable_reconstruction_index"),
                failures=len(bundle.failures),
            )
        )
    return "\n".join(
        [
            f"# {title}",
            "",
            "| Candidate | Mode | Status | Min IoU | Avg IoU | Boundary | Editable | Failures |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
        ]
    )


def static_html_report(bundles: Iterable[EvaluationBundle], *, title: str = "Evaluation Report") -> str:
    body = escape(markdown_report(bundles, title=title))
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{escape(title)}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;white-space:pre-wrap}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.35rem}</style>"
        "</head><body>"
        f"<pre>{body}</pre>"
        "</body></html>"
    )


def _metric(metrics: dict[str, object], name: str) -> str:
    metric = metrics.get(name)
    value = getattr(metric, "value", None)
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)

