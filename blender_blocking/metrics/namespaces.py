"""Helpers for canonical quality metric namespaces."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence


REQUIRED_RENDER_VIEWS = ("front", "side", "top")


def set_metric_path(
    metrics: MutableMapping[str, Any],
    path: str,
    value: Any,
    *,
    flat_alias: bool = True,
) -> None:
    """Write a metric as both a dotted key and nested JSON path."""
    if value is None:
        return
    if flat_alias:
        metrics[path] = value
    parts = [part for part in str(path).split(".") if part]
    if not parts:
        return
    current: MutableMapping[str, Any] = metrics
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def get_metric_path(metrics: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Read a metric from a dotted key or nested JSON path."""
    if path in metrics:
        return metrics[path]
    current: Any = metrics
    for part in str(path).split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return default
    return current


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validation_mode(metrics: Mapping[str, Any]) -> str:
    return str(get_metric_path(metrics, "validation_mode", "") or "")


def render_view_iou(metrics: Mapping[str, Any], view: str) -> float | None:
    value = get_metric_path(metrics, f"render.per_view.{view}.area_iou")
    if value is None:
        value = metrics.get(f"{view}_iou")
    if value is None:
        views = metrics.get("views", {})
        if isinstance(views, Mapping):
            view_data = views.get(view, {})
            if isinstance(view_data, Mapping):
                value = view_data.get("iou", view_data.get("area_iou"))
    return optional_float(value)


def render_required_view_values(
    metrics: Mapping[str, Any],
    *,
    required_views: Sequence[str] = REQUIRED_RENDER_VIEWS,
) -> tuple[float | None, ...]:
    return tuple(render_view_iou(metrics, view) for view in required_views)


def has_complete_required_render_views(
    metrics: Mapping[str, Any],
    *,
    required_views: Sequence[str] = REQUIRED_RENDER_VIEWS,
) -> bool:
    return all(value is not None for value in render_required_view_values(metrics, required_views=required_views))


def render_min_view_iou(
    metrics: Mapping[str, Any],
    *,
    required_views: Sequence[str] = REQUIRED_RENDER_VIEWS,
) -> float | None:
    values = [
        value
        for value in render_required_view_values(metrics, required_views=required_views)
        if value is not None
    ]
    if values:
        return min(values)
    return optional_float(get_metric_path(metrics, "render.min_view_iou"))


def render_average_iou(metrics: Mapping[str, Any]) -> float | None:
    value = get_metric_path(metrics, "render.average_iou")
    if value is None and validation_mode(metrics) == "render-iou":
        value = metrics.get("average_iou")
    return optional_float(value)


def set_render_aggregate_metrics(
    metrics: MutableMapping[str, Any],
    *,
    required_views: Sequence[str] = REQUIRED_RENDER_VIEWS,
) -> None:
    """Derive canonical render aggregates from required per-view metrics."""
    boundary_values = _required_view_metric_values(
        metrics,
        "boundary_iou",
        required_views=required_views,
    )
    if boundary_values:
        boundary_mean = sum(boundary_values) / len(boundary_values)
        set_metric_path(metrics, "render.boundary_iou_mean", boundary_mean)
        set_metric_path(metrics, "render.boundary_iou_min", min(boundary_values))
        metrics["boundary_iou_mean"] = boundary_mean
    sdf_values = _required_view_metric_values(
        metrics,
        "signed_distance_loss",
        required_views=required_views,
    )
    if sdf_values:
        sdf_mean = sum(sdf_values) / len(sdf_values)
        set_metric_path(metrics, "render.signed_distance_loss_mean", sdf_mean)
        set_metric_path(metrics, "render.signed_distance_loss_max", max(sdf_values))
        metrics["signed_distance_loss_mean"] = sdf_mean


def _required_view_metric_values(
    metrics: Mapping[str, Any],
    metric: str,
    *,
    required_views: Sequence[str],
) -> list[float]:
    values: list[float] = []
    for view in required_views:
        value = optional_float(get_metric_path(metrics, f"render.per_view.{view}.{metric}"))
        if value is not None:
            values.append(value)
    return values


def required_render_metrics_missing(
    metrics: Mapping[str, Any],
    *,
    required_views: Sequence[str] = REQUIRED_RENDER_VIEWS,
) -> bool:
    if validation_mode(metrics) == "render-iou":
        return not has_complete_required_render_views(metrics, required_views=required_views)
    if any(render_view_iou(metrics, view) is not None for view in required_views):
        return not has_complete_required_render_views(metrics, required_views=required_views)
    return True


def namespace_metric_key(name: str) -> str:
    """Translate legacy source metric names into canonical public namespaces."""
    aliases = {
        "area_iou_mean": "backend.area_iou_mean",
        "area_iou_min": "backend.area_iou_min",
        "boundary_iou_mean": "backend.boundary_iou_mean",
        "topology_score": "topology.score",
        "topology_score_mean": "topology.score",
        "editability_score": "editability.qa_score",
        "export_roundtrip_score": "editability.export_roundtrip_score",
        "elapsed_s": "cost.backend.elapsed_s",
    }
    return aliases.get(name, name)
