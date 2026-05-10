"""Profile-band adapters for residual primitive initialization and losses."""

from __future__ import annotations

from .pipeline import (
    _build_per_view_profile_summary,
    _dominant_interval,
    _family_supports_profile_init,
    _profile_rows_to_slices,
    _profile_width_scale,
)

__all__ = [
    "_build_per_view_profile_summary",
    "_dominant_interval",
    "_family_supports_profile_init",
    "_profile_rows_to_slices",
    "_profile_width_scale",
]
