"""Slice sampling utilities for elliptical profiles."""

from __future__ import annotations

from typing import List

import numpy as np

from geometry.profile_models import EllipticalProfileU, EllipticalSlice


def sample_elliptical_slices(
    profile: EllipticalProfileU,
    num_slices: int,
    sampling: str = "endpoints",
) -> List[EllipticalSlice]:
    """Sample an EllipticalProfileU into per-slice radii."""
    if num_slices <= 0:
        raise ValueError("num_slices must be >= 1")

    if num_slices == 1:
        t_values = np.array([0.5], dtype=np.float32)
    elif sampling == "endpoints":
        t_values = np.linspace(0.0, 1.0, num_slices, dtype=np.float32)
    elif sampling == "cell_centers":
        t_values = (np.arange(num_slices, dtype=np.float32) + 0.5) / float(num_slices)
    else:
        raise ValueError(f"Unknown sampling policy: {sampling}")

    heights_t = np.asarray(profile.heights_t, dtype=np.float32)
    rx_values = np.asarray(profile.rx, dtype=np.float32)
    ry_values = np.asarray(profile.ry, dtype=np.float32)

    rx_interp = np.interp(t_values, heights_t, rx_values)
    ry_interp = np.interp(t_values, heights_t, ry_values)

    cx = np.interp(t_values, heights_t, profile.cx) if profile.cx is not None else np.zeros(len(t_values))
    cy = np.interp(t_values, heights_t, profile.cy) if profile.cy is not None else np.zeros(len(t_values))
    slices: List[EllipticalSlice] = []
    for t, rx, ry, x, y in zip(t_values, rx_interp, ry_interp, cx, cy):
        z = profile.z0 + float(t) * profile.world_height
        slices.append(EllipticalSlice(z=z, rx=float(rx), ry=float(ry), cx=float(x), cy=float(y)))

    return slices
