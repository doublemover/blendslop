"""Static HTML report generation for refinement runs."""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .contracts import ExperimentResult, RefinementRunManifest
from .parameter_search import rank_results

try:
    from PIL import Image, ImageDraw

    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency branch
    PIL_AVAILABLE = False
    Image = None
    ImageDraw = None

try:
    from blender_blocking.integration.image_processing.image_loader import load_image
    from blender_blocking.validation.silhouette_iou import mask_from_image_array
except ImportError:  # pragma: no cover
    from integration.image_processing.image_loader import load_image
    from validation.silhouette_iou import mask_from_image_array


@dataclass(frozen=True)
class ReportOptions:
    top_k: int = 10
    include_failures: str = "top"
    write_overlays: bool = True


def generate_report(
    *,
    run_root: Path,
    results: Sequence[ExperimentResult],
    manifest: RefinementRunManifest | Mapping[str, object] | None = None,
    objective: str = "quality_win",
    options: ReportOptions = ReportOptions(),
) -> Path:
    run_root = Path(run_root)
    assets = run_root / "assets"
    overlays = assets / "overlays"
    diffs = assets / "diffs"
    thumbs = assets / "thumbnails"
    for directory in (assets, overlays, diffs, thumbs):
        directory.mkdir(parents=True, exist_ok=True)
    css_path = assets / "report.css"
    _write_css(css_path)
    scored = rank_results(results, objective=objective)
    rows = [_row(index + 1, result, score) for index, (result, score) in enumerate(scored)]
    visual_rows = _visual_rows(scored, options)
    if options.write_overlays and PIL_AVAILABLE:
        for result, _score in visual_rows:
            _write_visual_assets(result, run_root, overlays, diffs, thumbs)
    html_text = _render_html(
        run_root=run_root,
        rows=rows,
        visual_rows=visual_rows,
        manifest=manifest.to_dict() if hasattr(manifest, "to_dict") else manifest or {},
        objective=objective,
    )
    report_path = run_root / "report.html"
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def _render_html(
    *,
    run_root: Path,
    rows: Sequence[Mapping[str, object]],
    visual_rows: Sequence[tuple[ExperimentResult, Mapping[str, object]]],
    manifest: Mapping[str, object],
    objective: str,
) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    sections = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\">",
        "<title>Blendslop Refinement Report</title>",
        "<link rel=\"stylesheet\" href=\"assets/report.css\">",
        "</head><body>",
        "<h1>Blendslop Refinement Report</h1>",
        "<section><h2>Summary</h2>",
        _kv_table(
            {
                "run_id": manifest.get("run_id", ""),
                "suite": manifest.get("suite", ""),
                "track": manifest.get("track", ""),
                "search": manifest.get("search", ""),
                "objective": objective,
                "counts": json.dumps(counts, sort_keys=True),
                "git": json.dumps(manifest.get("git", {}), sort_keys=True),
            }
        ),
        "</section>",
        "<section><h2>Leaderboard</h2>",
        _leaderboard_table(rows),
        "</section>",
        "<section><h2>Visual Compare</h2>",
        _visual_grid(run_root, visual_rows),
        "</section>",
        "<section><h2>Autopsy</h2>",
        _autopsy_summary(rows),
        "</section>",
        "<section><h2>Reproduction</h2>",
        "<p>Every variant directory contains <code>command.txt</code>, <code>config.json</code>, and <code>result.json</code> when available.</p>",
        "</section>",
        "</body></html>",
    ]
    return "\n".join(sections)


def _leaderboard_table(rows: Sequence[Mapping[str, object]]) -> str:
    headers = [
        "rank",
        "case",
        "variant",
        "mode",
        "status",
        "score",
        "avg",
        "min",
        "front",
        "side",
        "top",
        "topology",
        "editability",
        "elapsed",
        "autopsy",
    ]
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{html.escape(header)}</th>" for header in headers)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        values = [
            row["rank"],
            row["case_id"],
            row["variant_id"],
            row["mode"],
            row["status"],
            f"{float(row['score']):.3f}",
            f"{float(row['average_iou']):.3f}",
            f"{float(row['min_iou']):.3f}",
            f"{float(row['front_iou']):.3f}",
            f"{float(row['side_iou']):.3f}",
            f"{float(row['top_iou']):.3f}",
            f"{float(row['topology_score']):.3f}",
            f"{float(row['editability_score']):.3f}",
            f"{float(row['elapsed_s']):.3f}",
            row.get("autopsy_category", ""),
        ]
        out.extend(f"<td>{html.escape(str(value))}</td>" for value in values)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def _visual_grid(
    run_root: Path,
    visual_rows: Sequence[tuple[ExperimentResult, Mapping[str, object]]],
) -> str:
    if not visual_rows:
        return "<p>No visual rows available.</p>"
    out = ["<div class=\"visual-grid\">"]
    for result, score in visual_rows:
        out.append("<article class=\"candidate-card\">")
        out.append(
            f"<h3>{html.escape(result.variant_id)} <small>{html.escape(result.mode)}</small></h3>"
        )
        out.append(
            f"<p>score={float(score.get('total', 0.0)):.3f} avg={result.avg_iou:.3f} min={result.min_iou:.3f}</p>"
        )
        for view in ("front", "side", "top"):
            out.append("<div class=\"view-row\">")
            for kind, mapping in (
                ("ref", result.reference_paths),
                ("render", result.render_paths),
                ("overlay", {"overlay": run_root / "assets" / "overlays" / f"{result.case_id}-{result.variant_id}-{view}.png"}),
                ("diff", {"diff": run_root / "assets" / "diffs" / f"{result.case_id}-{result.variant_id}-{view}.png"}),
            ):
                path = mapping.get(view) if kind in {"ref", "render"} else next(iter(mapping.values()))
                if path and Path(path).exists():
                    rel = _rel(Path(path), run_root)
                    out.append(f"<figure><img src=\"{html.escape(rel)}\"><figcaption>{kind} {view}</figcaption></figure>")
                else:
                    out.append(f"<figure class=\"missing\"><figcaption>{kind} {view} missing</figcaption></figure>")
            out.append("</div>")
        out.append("</article>")
    out.append("</div>")
    return "\n".join(out)


def _autopsy_summary(rows: Sequence[Mapping[str, object]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row.get("autopsy_category", "") or "none")
        counts[category] = counts.get(category, 0) + 1
    return _kv_table(counts)


def _visual_rows(
    scored: Sequence[tuple[ExperimentResult, Mapping[str, object]]],
    options: ReportOptions,
) -> list[tuple[ExperimentResult, Mapping[str, object]]]:
    rows = list(scored[: max(1, options.top_k)])
    if options.include_failures == "all":
        seen = {result.variant_id for result, _score in rows}
        rows.extend(
            (result, score)
            for result, score in scored
            if result.status != "pass" and result.variant_id not in seen
        )
    return rows


def _write_visual_assets(
    result: ExperimentResult,
    run_root: Path,
    overlays: Path,
    diffs: Path,
    thumbs: Path,
) -> None:
    for view in ("front", "side", "top"):
        ref = result.reference_paths.get(view)
        render = result.render_paths.get(view)
        if not ref or not render or not Path(ref).exists() or not Path(render).exists():
            continue
        try:
            ref_mask = mask_from_image_array(load_image(str(ref)))
            render_mask = mask_from_image_array(load_image(str(render)))
            if ref_mask.shape != render_mask.shape:
                render_mask = _resize_mask(render_mask, ref_mask.shape)
            overlay = _overlay(ref_mask, render_mask)
            diff = _diff(ref_mask, render_mask)
            overlay.save(overlays / f"{result.case_id}-{result.variant_id}-{view}.png")
            diff.save(diffs / f"{result.case_id}-{result.variant_id}-{view}.png")
            _thumbnail(Path(ref), thumbs / f"{result.case_id}-{result.variant_id}-{view}-ref.png")
            _thumbnail(Path(render), thumbs / f"{result.case_id}-{result.variant_id}-{view}-render.png")
        except Exception:
            continue


def _overlay(ref_mask: np.ndarray, render_mask: np.ndarray) -> "Image.Image":
    ref = np.asarray(ref_mask).astype(bool)
    render = np.asarray(render_mask).astype(bool)
    rgb = np.zeros(ref.shape + (3,), dtype=np.uint8)
    rgb[np.logical_and(ref, render)] = (255, 255, 255)
    rgb[np.logical_and(ref, ~render)] = (0, 220, 80)
    rgb[np.logical_and(~ref, render)] = (220, 0, 220)
    return Image.fromarray(rgb, mode="RGB")


def _diff(ref_mask: np.ndarray, render_mask: np.ndarray) -> "Image.Image":
    ref = np.asarray(ref_mask).astype(bool)
    render = np.asarray(render_mask).astype(bool)
    rgb = np.zeros(ref.shape + (3,), dtype=np.uint8)
    rgb[np.logical_and(ref, render)] = (255, 255, 255)
    rgb[np.logical_and(ref, ~render)] = (40, 120, 255)
    rgb[np.logical_and(~ref, render)] = (255, 60, 40)
    return Image.fromarray(rgb, mode="RGB")


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def _thumbnail(source: Path, target: Path) -> None:
    image = Image.open(source)
    image.thumbnail((240, 240))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)


def _row(rank: int, result: ExperimentResult, score: Mapping[str, object]) -> dict[str, object]:
    autopsy = result.autopsy if isinstance(result.autopsy, Mapping) else {}
    return {
        "rank": rank,
        "case_id": result.case_id,
        "variant_id": result.variant_id,
        "mode": result.mode,
        "status": result.status,
        "score": float(score.get("total", 0.0)),
        "average_iou": result.avg_iou,
        "min_iou": result.min_iou,
        "front_iou": result.view_iou("front") or 0.0,
        "side_iou": result.view_iou("side") or 0.0,
        "top_iou": result.view_iou("top") or 0.0,
        "topology_score": _metric(result, "topology_score"),
        "editability_score": _metric(result, "editability_score"),
        "elapsed_s": result.elapsed_s,
        "autopsy_category": autopsy.get("category", ""),
    }


def _metric(result: ExperimentResult, key: str) -> float:
    value = result.metrics.get(key)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _kv_table(items: Mapping[str, object]) -> str:
    out = ["<table class=\"kv\"><tbody>"]
    for key, value in items.items():
        out.append(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def _write_css(path: Path) -> None:
    path.write_text(
        """
body { font-family: system-ui, Segoe UI, sans-serif; margin: 24px; color: #171717; background: #f7f7f5; }
h1, h2, h3 { margin: 0.8rem 0 0.5rem; }
table { border-collapse: collapse; width: 100%; background: white; }
th, td { border: 1px solid #d8d8d2; padding: 4px 6px; font-size: 12px; vertical-align: top; }
th { background: #ecece6; text-align: left; }
code { background: #eee; padding: 1px 3px; border-radius: 3px; }
.visual-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
.candidate-card { border: 1px solid #d6d6cf; background: white; padding: 12px; border-radius: 6px; }
.view-row { display: grid; grid-template-columns: repeat(4, minmax(70px, 1fr)); gap: 6px; margin: 8px 0; }
figure { margin: 0; border: 1px solid #ddd; min-height: 72px; background: #fafafa; }
figcaption { font-size: 11px; padding: 2px 4px; color: #555; }
img { max-width: 100%; display: block; }
.missing { display: flex; align-items: center; justify-content: center; color: #777; }
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()
