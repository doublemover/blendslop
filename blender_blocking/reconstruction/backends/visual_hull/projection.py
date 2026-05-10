from __future__ import annotations

from typing import Any, Mapping


def collect_visual_hull_projection_metrics(
    *,
    target: Any,
    grid: Any,
    max_metric_voxels: int,
    boundary_refine: bool,
    boundary_dilate_px: int | None,
    metrics: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    try:
        from reconstruction.point_cloud import (
            visual_hull_projection_metrics_from_target,
            visual_hull_view_diagnostics_from_target,
        )

        per_view_metrics = visual_hull_projection_metrics_from_target(
            target,
            grid,
            max_metric_voxels=max_metric_voxels,
            boundary_refine=boundary_refine,
        )
        skipped_metric = per_view_metrics.pop("_skipped", None)
        if skipped_metric:
            metrics["projection_metrics_skipped"] = skipped_metric
            warnings.append(
                str(skipped_metric.get("reason", "projection metrics skipped"))
            )
            return per_view_metrics
        diagnostics = visual_hull_view_diagnostics_from_target(
            target,
            grid,
            per_view_metrics=per_view_metrics,
            boundary_refine=boundary_refine,
            boundary_dilate_px=boundary_dilate_px,
        )
        metrics["visual_hull_view_diagnostics"] = diagnostics
        if diagnostics.get("axis_or_transform_suspect"):
            warnings.append("visual hull view diagnostics flagged axis_or_transform_suspect")
        if diagnostics.get("catastrophic_view_failure"):
            warnings.append("visual hull view diagnostics flagged catastrophic_view_failure")
        return per_view_metrics
    except Exception as exc:
        warnings.append(f"projection metrics failed: {exc}")
        return {}
