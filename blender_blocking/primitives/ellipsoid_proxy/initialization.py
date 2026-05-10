"""Initialization and adaptive-count policies for proxy primitives."""

from __future__ import annotations

from .fit import _adaptive_point_count, _adaptive_primitive_count, _prepare_initialization_points

__all__ = [
    "_adaptive_point_count",
    "_adaptive_primitive_count",
    "_prepare_initialization_points",
]
