"""Shared bounded config coercion helpers."""

from __future__ import annotations

import math
from typing import Any


_MISSING = object()


def _fallback(default: Any) -> Any:
    return None if default is _MISSING else default


def coerce_int(
    value: Any,
    name: str,
    errors: list[str],
    *,
    default: int | object = _MISSING,
    min_value: int | None = None,
    max_value: int | None = None,
    default_on_bounds: bool = False,
) -> int | None:
    if isinstance(value, bool):
        errors.append(f"{name} must be an integer, got {value!r}")
        return _fallback(default)
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        errors.append(f"{name} must be an integer, got {type(value)!r}")
        return _fallback(default)
    out_of_bounds = False
    if min_value is not None and parsed < min_value:
        errors.append(f"{name} must be >= {min_value}, got {parsed}")
        out_of_bounds = True
    if max_value is not None and parsed > max_value:
        errors.append(f"{name} must be <= {max_value}, got {parsed}")
        out_of_bounds = True
    if out_of_bounds and default_on_bounds:
        return _fallback(default)
    return parsed


def coerce_float(
    value: Any,
    name: str,
    errors: list[str],
    *,
    default: float | object = _MISSING,
    min_value: float | None = None,
    max_value: float | None = None,
    default_on_bounds: bool = False,
    real_number_message: bool = False,
) -> float | None:
    if isinstance(value, bool):
        noun = "a real number" if real_number_message else "a float"
        errors.append(f"{name} must be {noun}, got {value!r}")
        return _fallback(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        noun = "a real number" if real_number_message else "a float"
        detail = value if real_number_message else type(value)
        errors.append(f"{name} must be {noun}, got {detail!r}")
        return _fallback(default)
    if not math.isfinite(parsed):
        errors.append(f"{name} must be finite, got {parsed!r}")
        return _fallback(default)
    out_of_bounds = False
    if min_value is not None and parsed < min_value:
        errors.append(f"{name} must be >= {min_value}, got {parsed}")
        out_of_bounds = True
    if max_value is not None and parsed > max_value:
        errors.append(f"{name} must be <= {max_value}, got {parsed}")
        out_of_bounds = True
    if out_of_bounds and default_on_bounds:
        return _fallback(default)
    return parsed


def coerce_optional_int(
    value: Any,
    name: str,
    errors: list[str],
    *,
    default: int | None = None,
    min_value: int | None = None,
    max_value: int | None = None,
    default_on_bounds: bool = False,
) -> int | None:
    if value is None:
        return default
    return coerce_int(
        value,
        name,
        errors,
        default=default,
        min_value=min_value,
        max_value=max_value,
        default_on_bounds=default_on_bounds,
    )


def coerce_optional_float(
    value: Any,
    name: str,
    errors: list[str],
    *,
    default: float | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    default_on_bounds: bool = False,
    real_number_message: bool = False,
) -> float | None:
    if value is None:
        return default
    return coerce_float(
        value,
        name,
        errors,
        default=default,
        min_value=min_value,
        max_value=max_value,
        default_on_bounds=default_on_bounds,
        real_number_message=real_number_message,
    )
