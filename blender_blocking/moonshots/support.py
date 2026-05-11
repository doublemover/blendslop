"""Shared helpers for deterministic moonshot sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
        return payload if isinstance(payload, Mapping) else {}
    return {}


def candidate_rows(candidate: Any) -> tuple[Mapping[str, Any], ...]:
    if candidate is None:
        return ()
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, Mapping)):
        rows: list[Mapping[str, Any]] = []
        for item in candidate:
            rows.extend(candidate_rows(item))
        return tuple(rows)
    payload = as_mapping(candidate)
    if not payload:
        return ()
    for key in ("rows", "results", "candidates"):
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            nested = tuple(as_mapping(item) for item in value)
            nested = tuple(item for item in nested if item)
            if nested:
                return nested
    return (payload,)


def selected_candidate_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    backend = row.get("backend_result")
    if isinstance(backend, Mapping):
        selected = backend.get("selected")
        if isinstance(selected, Mapping):
            return selected
        return backend
    selected = row.get("selected")
    if isinstance(selected, Mapping):
        return selected
    return row


def metric_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    source = selected_candidate_payload(row)
    metric = source.get("metric_result")
    if isinstance(metric, Mapping):
        return metric
    metric = row.get("metrics")
    return metric if isinstance(metric, Mapping) else {}


def metric_extras(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = metric_payload(row)
    extras = metrics.get("extras")
    return extras if isinstance(extras, Mapping) else {}


def nested_get(value: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return default
        current = current.get(part)
    return default if current is None else current


def float_value(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def row_metric(row: Mapping[str, Any], *paths: str, default: float = 0.0) -> float:
    metrics = metric_payload(row)
    for path in paths:
        value = nested_get(metrics, path)
        if value is not None:
            return float_value(value, default)
        value = metrics.get(path.replace(".", "_"))
        if value is not None:
            return float_value(value, default)
    source = selected_candidate_payload(row)
    for path in paths:
        value = nested_get(source, path)
        if value is not None:
            return float_value(value, default)
    return default


def per_view_area_iou(row: Mapping[str, Any]) -> dict[str, float]:
    output = per_view_metric(row, "area_iou", aliases=("iou",))
    metric_result = metric_payload({"backend_result": selected_candidate_payload(row)})
    per_view = metric_result.get("per_view") if isinstance(metric_result, Mapping) else None
    if isinstance(per_view, Mapping):
        for view, payload in per_view.items():
            if isinstance(payload, Mapping) and payload.get("area_iou") is not None:
                output.setdefault(str(view), float_value(payload.get("area_iou")))
    return output


def per_view_boundary_iou(row: Mapping[str, Any]) -> dict[str, float]:
    return per_view_metric(row, "boundary_iou")


def per_view_signed_distance_loss(row: Mapping[str, Any]) -> dict[str, float]:
    return per_view_metric(row, "signed_distance_loss")


def per_view_metric(
    row: Mapping[str, Any],
    metric_name: str,
    *,
    aliases: Sequence[str] = (),
) -> dict[str, float]:
    metrics = metric_payload(row)
    views = metrics.get("views")
    output: dict[str, float] = {}
    names = (metric_name,) + tuple(aliases)
    if isinstance(views, Mapping):
        for view, payload in views.items():
            if not isinstance(payload, Mapping):
                continue
            value = None
            for name in names:
                value = payload.get(name)
                if value is not None:
                    break
            if value is not None:
                output[str(view)] = float_value(value)
    for view in ("front", "side", "top"):
        value = None
        for name in names:
            value = nested_get(metrics, f"render.per_view.{view}.{name}")
            if value is not None:
                break
        if value is None and metric_name in {"area_iou", "iou"}:
            value = metrics.get(f"{view}_iou")
        if value is not None:
            output.setdefault(view, float_value(value))
    metric_result = metric_payload({"backend_result": selected_candidate_payload(row)})
    per_view = metric_result.get("per_view") if isinstance(metric_result, Mapping) else None
    if isinstance(per_view, Mapping):
        for view, payload in per_view.items():
            if not isinstance(payload, Mapping):
                continue
            for name in names:
                if payload.get(name) is not None:
                    output.setdefault(str(view), float_value(payload.get(name)))
                    break
    return output


def row_failures(row: Mapping[str, Any]) -> tuple[str, ...]:
    failures: list[str] = []
    for key in ("promotion_blockers", "parent_selection_blockers", "errors", "warnings"):
        value = row.get(key)
        if isinstance(value, str) and value:
            failures.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            failures.extend(str(item) for item in value if item)
    autopsy = row.get("autopsy")
    if isinstance(autopsy, Mapping):
        category = autopsy.get("category")
        if category:
            failures.append(str(category))
        findings = autopsy.get("findings")
        if isinstance(findings, Sequence) and not isinstance(findings, (str, bytes)):
            for finding in findings:
                if isinstance(finding, Mapping) and finding.get("category"):
                    failures.append(str(finding.get("category")))
    backend = selected_candidate_payload(row)
    for key in ("failures", "errors", "warnings"):
        value = backend.get(key)
        if isinstance(value, str) and value:
            failures.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            failures.extend(str(item) for item in value if item)
    return tuple(dict.fromkeys(failures))


def target_signals(request: Any, *, include_profile_rows: bool = False) -> Mapping[str, Mapping[str, Any]]:
    target = getattr(request, "target", None)
    if target is not None:
        try:
            from blender_blocking.reconstruction.target_signals import collect_target_signals
        except Exception:  # pragma: no cover - legacy script import path
            from reconstruction.target_signals import collect_target_signals  # type: ignore

        try:
            return collect_target_signals(
                target,
                include_profile_rows=include_profile_rows,
            )
        except Exception:
            pass
    config = getattr(request, "config", {}) or {}
    if isinstance(config, Mapping):
        signals = config.get("target_signals")
        if isinstance(signals, Mapping):
            return _normalize_signals(signals)
        target_payload = config.get("target")
        if isinstance(target_payload, Mapping):
            signals = target_payload.get("signals")
            if isinstance(signals, Mapping):
                return _normalize_signals(signals)
    for row in candidate_rows(getattr(request, "candidate", None)):
        extras = metric_extras(row)
        signals = extras.get("signal_summary")
        if isinstance(signals, Mapping):
            return _normalize_signals(signals)
        backend = selected_candidate_payload(row)
        signals = nested_get(backend, "target.signals")
        if isinstance(signals, Mapping):
            return _normalize_signals(signals)
    return _empty_signals()


def _normalize_signals(signals: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    empty = _empty_signals()
    return {
        key: dict(signals.get(key, empty[key])) if isinstance(signals.get(key, empty[key]), Mapping) else dict(empty[key])
        for key in empty
    }


def _empty_signals() -> Mapping[str, Mapping[str, Any]]:
    return {
        "surface": {"available": False, "surface_point_count": 0, "constraint_count": 0},
        "profile": {
            "available": False,
            "view_count": 0,
            "band_samples": 0,
            "interval_count": 0,
            "hole_count": 0,
            "mean_width": 0.0,
            "max_width": 0.0,
            "complexity": 0.0,
        },
        "constraints": {"available": False, "constraint_count": 0, "constraint_views": {}},
        "uncertainty": {
            "available": False,
            "overall_confidence_mean": 1.0,
            "overall_boundary_uncertainty_mean": 0.0,
            "consistency": 0.75,
            "view_details": {},
        },
        "topology": {"available": False, "score": 0.75, "complexity": 0.0},
    }


def topology_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = metric_payload(row)
    topology = metrics.get("topology")
    if isinstance(topology, Mapping):
        return topology
    extras = metric_extras(row)
    topology = extras.get("topology")
    if isinstance(topology, Mapping):
        return topology
    mesh_quality = metrics.get("mesh_quality")
    if isinstance(mesh_quality, Mapping):
        return mesh_quality
    return {}


def artifact_exists(row: Mapping[str, Any], key: str) -> bool:
    source = selected_candidate_payload(row)
    value = source.get(key)
    artifacts = source.get("artifacts")
    if not value and isinstance(artifacts, Mapping):
        value = artifacts.get(key)
    if not value:
        return False
    return Path(str(value)).exists()
