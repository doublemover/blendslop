"""Connected loft section plans with explicit artist-facing surface modes.

Smooth mode interpolates geometry, never just normals. Sharp mode preserves
the supplied section corners. Stepped mode inserts connected planar shoulders
instead of stacking overlapping, individually capped cylinders.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

from geometry.profile_models import EllipticalSlice

SURFACE_MODES = ("smooth", "stepped", "sharp")


def prepare_loft_surface(
    slices: Sequence[EllipticalSlice],
    mode: str = "smooth",
    subdivisions: int = 4,
) -> list[EllipticalSlice]:
    """Return deterministic sections without overshooting source radii.

Subdivisions are a bounded per-interval budget, not a fitting operation.
Step shoulders require nested, concentric ellipse boundaries. Crossing
boundaries are rejected rather than emitting a folded zero-height annulus.
"""
    if mode not in SURFACE_MODES:
        raise ValueError(f"surface_mode must be one of {SURFACE_MODES}")
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or not 1 <= subdivisions <= 16:
        raise ValueError("surface_subdivisions must be an integer in [1, 16]")
    if not slices:
        raise ValueError("slices must not be empty")
    columns = np.asarray([
        (s.z, s.rx, s.ry, s.cx or 0.0, s.cy or 0.0) for s in slices
    ], dtype=float)
    if not np.isfinite(columns).all() or (columns[:, 1:3] < 0).any():
        raise ValueError("loft sections require finite coordinates and nonnegative radii")
    if len(slices) > 1 and (np.diff(columns[:, 0]) <= 0).any():
        raise ValueError("source loft sections must have strictly increasing heights")
    if mode == "sharp" or len(slices) == 1:
        return list(slices)
    if mode == "stepped":
        output = [slices[0]]
        for a, b in zip(slices[:-1], slices[1:]):
            if (a.cx or 0.0, a.cy or 0.0) != (b.cx or 0.0, b.cy or 0.0):
                raise ValueError("stepped shoulders require concentric sections")
            if (a.rx, a.ry) != (b.rx, b.ry) and (b.rx - a.rx) * (b.ry - a.ry) <= 0:
                raise ValueError("stepped shoulders require nested ellipse boundaries")
            if (a.rx, a.ry) != (b.rx, b.ry):
                if min(a.rx, a.ry, b.rx, b.ry) <= 0:
                    raise ValueError("stepped shoulders require nonzero radii")
                midpoint = (float(a.z) + float(b.z)) / 2.0
                output.extend((replace(a, z=midpoint), replace(b, z=midpoint)))
            output.append(b)
        return output
    if subdivisions == 1:
        return list(slices)
    z = columns[:, 0]
    sample_z = np.concatenate([
        np.linspace(a, b, subdivisions, endpoint=False)
        for a, b in zip(z[:-1], z[1:])
    ] + [z[-1:]])
    interpolated = PchipInterpolator(z, columns[:, 1:], axis=0)(sample_z)
    return [EllipticalSlice(
        z=float(height), rx=float(values[0]), ry=float(values[1]),
        cx=float(values[2]), cy=float(values[3]),
    ) for height, values in zip(sample_z, interpolated)]
