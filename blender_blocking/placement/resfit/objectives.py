"""Objective hook builders for residual primitive fitting."""

from __future__ import annotations

from .pipeline import (
    _build_constraint_penalty_hook,
    _build_profile_silhouette_hook,
    _build_topology_penalty_hook,
    _build_uncertainty_penalty_hook,
)

__all__ = [
    "_build_constraint_penalty_hook",
    "_build_profile_silhouette_hook",
    "_build_topology_penalty_hook",
    "_build_uncertainty_penalty_hook",
]
