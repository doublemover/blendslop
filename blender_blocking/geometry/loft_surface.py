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
    regularization_window: int = 0,
    regularization_max_deviation_u: float = 0.,
) -> list[EllipticalSlice]:
    """Return deterministic sections without overshooting source radii.

    Subdivisions are a bounded per-interval budget, not a fitting operation.
    Smooth regularization bounds displacement at each source section; the
    interpolation between sections remains a hypothesis. Step shoulders require
    nested, concentric ellipse boundaries. Crossing boundaries are rejected.
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
    if (isinstance(regularization_window, bool) or not isinstance(regularization_window, int) or
            regularization_window not in (0,) and (not 5 <= regularization_window <= 129 or regularization_window % 2 != 1) or
            not np.isfinite(regularization_max_deviation_u) or regularization_max_deviation_u < 0 or
            regularization_window > 0 and regularization_max_deviation_u <= 0):
        raise ValueError("regularization requires an odd 5..129 window and positive world deviation budget")
    if mode == "smooth" and regularization_window and len(slices) >= 5:
        from scipy.signal import savgol_filter
        window = min(regularization_window, len(slices) if len(slices) % 2 else len(slices) - 1)
        if window >= 5:
            uniform_z = np.linspace(columns[0, 0], columns[-1, 0], len(slices))
            for axis in (1, 2):
                samples = np.interp(uniform_z, columns[:, 0], columns[:, axis])
                filtered = savgol_filter(samples, window, 3, mode="interp")
                proposed = np.interp(columns[:, 0], uniform_z, filtered)
                proposed = np.clip(proposed, columns[:, axis] - regularization_max_deviation_u,
                                   columns[:, axis] + regularization_max_deviation_u)
                proposed[[0, -1]] = columns[[0, -1], axis]
                columns[:, axis] = np.maximum(proposed, 0.)
            slices = [replace(s, rx=float(row[1]), ry=float(row[2])) for s, row in zip(slices, columns)]
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
