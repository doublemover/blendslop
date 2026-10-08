"""Small text and HTML report renderers for evaluation bundles."""

from __future__ import annotations

from html import escape
from typing import Iterable

from .schemas import EvaluationBundle


def compact_console_summary(bundle: EvaluationBundle) -> str:
    metrics = bundle.metric_index()
    parts = [
        f"Candidate {bundle.candidate_id} [{bundle.mode}] status={bundle.status}",
        f"  rank={_selection(bundle, 'rank')} selected={_selection(bundle, 'selected')} score={_selection(bundle, 'score_total')}",
        f"  min_iou={_metric(metrics, 'silhouette.min_view_iou')}",
        f"  avg_iou={_metric(metrics, 'silhouette.average_iou')}",
        f"  boundary={_metric(metrics, 'silhouette.mean_boundary_iou')}",
        f"  sdf={_metric(metrics, 'silhouette.mean_signed_distance_loss')} missing_required={_metric(metrics, 'silhouette.missing_required_metric_count')}",
        f"  chamfer={_metric(metrics, 'geometry.chamfer_l2')} fscore={_metric(metrics, 'geometry.fscore_tau')} vol_iou={_metric(metrics, 'geometry.volumetric_iou')}",
        f"  true_fscore={_metric(metrics, 'geometry.true.fscore_tau')} recoverable_fscore={_metric(metrics, 'geometry.recoverable.fscore_tau')} ambiguity_gap={_metric(metrics, 'geometry.ambiguity_gap_chamfer_l2')}",
        f"  psnr={_metric(metrics, 'novel_view.psnr')} ssim={_metric(metrics, 'novel_view.ssim')} lpips={_metric(metrics, 'novel_view.lpips')}",
        f"  editable={_metric(metrics, 'editability.editable_reconstruction_index')}",
        f"  uv={_metric(metrics, 'appearance.uv_valid')} pbr={_metric(metrics, 'appearance.pbr_channel_coverage_ratio')} texture_only={_metric(metrics, 'appearance.attribution_texture_only_detail_score')}",
        f"  export={_metric(metrics, 'export.qa_score')} cost_ms={_metric(metrics, 'cost.total_wall_ms')}",
        f"  deps={_dependency_summary(bundle)}",
        f"  failures={len(bundle.failures)}",
    ]
    return "\n".join(parts)


def markdown_report(bundles: Iterable[EvaluationBundle], *, title: str = "Evaluation Report") -> str:
    bundle_tuple = tuple(bundles)
    rows = []
    for bundle in bundle_tuple:
        metrics = bundle.metric_index()
        rows.append(
            "| {candidate} | {mode} | {status} | {rank} | {selected} | {score} | {min_iou} | {avg_iou} | {boundary} | {sdf} | {true_fscore} | {recoverable_fscore} | {gap} | {chamfer} | {fscore} | {vol_iou} | {psnr} | {ssim} | {lpips} | {editable} | {uv} | {pbr} | {texture_only} | {export} | {cost} | {failures} |".format(
                candidate=bundle.candidate_id,
                mode=bundle.mode,
                status=bundle.status,
                rank=_selection(bundle, "rank"),
                selected=_selection(bundle, "selected"),
                score=_selection(bundle, "score_total"),
                min_iou=_metric(metrics, "silhouette.min_view_iou"),
                avg_iou=_metric(metrics, "silhouette.average_iou"),
                boundary=_metric(metrics, "silhouette.mean_boundary_iou"),
                sdf=_metric(metrics, "silhouette.mean_signed_distance_loss"),
                true_fscore=_metric(metrics, "geometry.true.fscore_tau"),
                recoverable_fscore=_metric(metrics, "geometry.recoverable.fscore_tau"),
                gap=_metric(metrics, "geometry.ambiguity_gap_chamfer_l2"),
                chamfer=_metric(metrics, "geometry.chamfer_l2"),
                fscore=_metric(metrics, "geometry.fscore_tau"),
                vol_iou=_metric(metrics, "geometry.volumetric_iou"),
                psnr=_metric(metrics, "novel_view.psnr"),
                ssim=_metric(metrics, "novel_view.ssim"),
                lpips=_metric(metrics, "novel_view.lpips"),
                editable=_metric(metrics, "editability.editable_reconstruction_index"),
                uv=_metric(metrics, "appearance.uv_valid"),
                pbr=_metric(metrics, "appearance.pbr_channel_coverage_ratio"),
                texture_only=_metric(metrics, "appearance.attribution_texture_only_detail_score"),
                export=_metric(metrics, "export.qa_score"),
                cost=_metric(metrics, "cost.total_wall_ms"),
                failures=len(bundle.failures),
            )
        )
    return "\n".join(
        [
            f"# {title}",
            "",
            "| Candidate | Mode | Status | Rank | Selected | Score | Min IoU | Avg IoU | Boundary | SDF Loss | True F-score | Recoverable F-score | Ambiguity Gap | Chamfer | F-score | Vol IoU | PSNR | SSIM | LPIPS | Editable | UV | PBR | Texture-only | Export | Cost ms | Failures |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            *_failure_lines(bundle_tuple),
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


def _selection(bundle: EvaluationBundle, key: str) -> str:
    value = bundle.selection.get(key)
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _dependency_summary(bundle: EvaluationBundle) -> str:
    if not bundle.dependency_state:
        return "n/a"
    parts = []
    for name, payload in sorted(bundle.dependency_state.items()):
        available = None
        if isinstance(payload, dict):
            available = payload.get("available")
        if available is None:
            parts.append(str(name))
        else:
            parts.append(f"{name}={'ok' if available else 'missing'}")
    return ", ".join(parts)


def _failure_lines(bundles: tuple[EvaluationBundle, ...]) -> list[str]:
    lines = ["## Failure Notes", ""]
    emitted = False
    for bundle in bundles:
        for failure in bundle.failures:
            emitted = True
            metrics = ", ".join(
                f"{key}={value}" for key, value in failure.evidence_metrics.items()
            )
            lines.append(
                f"- `{bundle.candidate_id}` `{failure.code}` ({failure.severity}, {failure.subsystem}) {metrics}".rstrip()
            )
    if not emitted:
        return []
    lines.append("")
    return lines
