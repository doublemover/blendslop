"""Declared observed-evidence and compact-geometry prior, without benchmark truth."""
from __future__ import annotations


def prefer_candidate(candidate, incumbent, *, policy="silhouette_only", tolerance=.006):
    a, b = candidate.metric_result, incumbent.metric_result
    if policy == "silhouette_only":
        return a.area_iou_min >= b.area_iou_min-.002 and a.area_iou_mean > b.area_iou_mean+.002
    if policy != "structure_v1":
        raise ValueError("unsupported geometry selection policy")
    if not all(row.get("passed", False) for row in a.per_view.values() if row.get("required", True)):
        return False
    # Clear evidence improvement remains preferable without a simplicity tie.
    if a.area_iou_min >= b.area_iou_min-.002 and a.area_iou_mean > b.area_iou_mean+.002:
        return True
    if (a.area_iou_min < b.area_iou_min-tolerance or
        a.area_iou_mean < b.area_iou_mean-tolerance or
        a.boundary_iou_mean < b.boundary_iou_mean-tolerance):
        return False
    from .native_geometry import geometry_arrays
    if candidate.geometry is None or incumbent.geometry is None:
        return False
    # Compactness is a stated prior among data-consistent alternatives, never
    # additional observation or proof of hidden shape. Require a meaningful gap.
    return len(geometry_arrays(candidate.geometry).faces) < .5*len(geometry_arrays(incumbent.geometry).faces)


def missing_observed_points(target, prediction_masks, points):
    """Grow only where an actually observed foreground ray is unexplained."""
    import numpy as np
    from .projection_contract import project_vertices
    from .visibility import valid_evidence, point_support
    points = np.asarray(points, float)
    demand = np.zeros(len(points), bool)
    for c in target.constraints:
        xy = np.floor(project_vertices(target,c,points)+.5).astype(int)
        reference = np.asarray(getattr(c.mask, "mask", c.mask), bool)
        h,w = reference.shape
        inside = (xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        ids = np.flatnonzero(inside); x,y = xy[ids].T
        demand[ids] |= reference[y,x] & valid_evidence(c)[y,x] & ~prediction_masks[c.view][y,x]
    allowed, _ = point_support(target,points)
    return demand & allowed
