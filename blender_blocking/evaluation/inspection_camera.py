"""Fresh display-only square orthographic framing; frozen gate cameras stay intact."""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np


def inspection_frame_bounds(record, vertices):
    """Observe projected extents and full clipping from an explicit camera record."""
    matrix = np.asarray(record.get("matrix_world"), float)
    points = np.asarray(vertices, float)
    size = record.get("resolution", [512, 512])
    if (record.get("projection", "ORTHO") != "ORTHO" or matrix.shape != (4, 4)
            or not np.isfinite(matrix).all() or not np.array_equal(matrix[3], [0., 0., 0., 1.])
            or points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= 1000000
            or not np.isfinite(points).all()):
        raise ValueError("inspection framing needs finite world points and an affine orthographic camera")
    if (len(size) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in size)
            or size[0] != size[1] or list(record.get("pixel_aspect", [1., 1.])) != [1., 1.]):
        raise ValueError("inspection framing currently supports square pixels and square frames only")
    scale = float(record["ortho_scale"])
    clips = [float(record[key]) for key in ("clip_start", "clip_end")]
    shifts = [float(record.get(key, 0.)) for key in ("shift_x", "shift_y")]
    if (not all(math.isfinite(x) for x in [scale, *clips, *shifts]) or scale <= 0
            or not 0 < clips[0] < clips[1]):
        raise ValueError("inspection framing requires finite scale/shifts and explicit ordered full clips")
    basis = matrix[:3, :3]
    if not np.allclose(basis.T @ basis, np.eye(3), rtol=0., atol=2e-6) or np.linalg.det(basis) <= 0:
        raise ValueError("inspection framing requires an ordinary proper native camera basis")
    local = (points - matrix[:3, 3]) @ np.linalg.inv(basis).T
    xy = local[:, :2] - np.asarray(shifts) * scale
    depth = -local[:, 2]
    return {"projected_min": xy.min(axis=0).tolist(), "projected_max": xy.max(axis=0).tolist(),
            "depth_min": float(depth.min()), "depth_max": float(depth.max()),
            "projection_enclosed": bool((np.abs(xy) <= scale / 2).all()),
            "clipping_enclosed": bool((depth >= clips[0]).all() and (depth <= clips[1]).all()),
            "scope": "projected/clipping observation of these points only; no silhouette or family acceptance"}


def unclipped_inspection_camera(record, vertices, *, padding_fraction=.08):
    """Center combined source/candidate projected bounds with explicit padding.

    This opt-in declaration is for new human inspection passes only. It keeps
    orientation/full clip planes, sets zero shifts and changes translation/scale.
    Inputs are copied; original masks, cameras and scored gates are never edited.
    Camera-depth failures refuse rather than manufacture an unclipped claim.
    """
    if (isinstance(padding_fraction, bool) or not isinstance(padding_fraction, (int, float))
            or not math.isfinite(padding_fraction) or not 0 < padding_fraction <= .5):
        raise ValueError("inspection padding must be finite and in (0, .5]")
    inspection_frame_bounds(record, vertices)  # Validate before deriving anything.
    matrix = np.asarray(record["matrix_world"], float).copy()
    points = np.asarray(vertices, float)
    local = (points - matrix[:3, 3]) @ np.linalg.inv(matrix[:3, :3]).T
    lower, upper = local[:, :2].min(axis=0), local[:, :2].max(axis=0)
    extent = float((upper - lower).max())
    if extent <= 0:
        raise ValueError("inspection points have no positive projected extent")
    center = (lower + upper) / 2
    matrix[:3, 3] += matrix[:3, :2] @ center
    result = {key: deepcopy(record[key]) for key in ("projection", "resolution", "pixel_aspect",
              "clip_start", "clip_end") if key in record}
    result.update(projection="ORTHO", matrix_world=matrix.tolist(), ortho_scale=extent * (1 + 2 * padding_fraction),
                  shift_x=0., shift_y=0.)
    bounds = inspection_frame_bounds(result, points)
    if not bounds["projection_enclosed"] or not bounds["clipping_enclosed"]:
        raise ValueError("display-only framing does not enclose the points inside the declared full clips")
    return result
