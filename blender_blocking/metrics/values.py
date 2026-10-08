"""Shared metric value coercion helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def float_or(value: Any, default: float = 0.0) -> float:
    parsed = optional_float(value)
    return float(default) if parsed is None else parsed


def finite_float_or_none(value: Any) -> float | None:
    parsed = optional_float(value)
    if parsed is None or not math.isfinite(parsed):
        return None
    return parsed


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
