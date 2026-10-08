"""Camera-row occupied interval unions, preserving separated parts and gaps."""
from __future__ import annotations
import numpy as np


def union_intervals(intervals, tolerance=1e-10):
    result = []
    for a, b in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if result and a <= result[-1][1] + tolerance:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def intersect_intervals(left, right):
    return union_intervals((max(a,c), min(b,d)) for a,b in left for c,d in right if min(b,d) > max(a,c))


def interval_disagreement(target, predicted, scale):
    target, predicted = union_intervals(target), union_intervals(predicted)
    intersection = sum(b-a for a,b in intersect_intervals(target, predicted))
    union = sum(b-a for a,b in union_intervals((*target, *predicted)))
    coverage = 1 - intersection / union if union > 0 else 0.
    if target and predicted:
        a, b = np.asarray(target).ravel(), np.asarray(predicted).ravel()
        distance = np.abs(a[:,None]-b[None,:])
        endpoints = .5 * (distance.min(axis=0).mean()+distance.min(axis=1).mean()) / max(scale, 1e-8)
    else:
        endpoints = 0. if not target and not predicted else 1.
    return float(coverage + endpoints)


def projected_row_intervals(primitive, axes, vertical, mesh=None):
    center = np.asarray(getattr(primitive, "center", getattr(primitive, "position", (0,0,0))), float)
    if type(primitive).__name__ in {"EllipsoidPrimitive", "AnisotropicGaussianPrimitive"}:
        covariance = primitive.covariance() if callable(primitive.covariance) else primitive.covariance
        s = np.asarray(covariance)[np.ix_(axes, axes)]
        delta = vertical - center[axes[1]]
        fraction = 1 - delta * delta / max(s[1,1], 1e-12)
        if fraction <= 0:
            return []
        midpoint = center[axes[0]] + s[0,1] / max(s[1,1],1e-12) * delta
        radius = np.sqrt(max(0, (s[0,0] - s[0,1]**2 / max(s[1,1],1e-12)) * fraction))
        return [(midpoint-radius, midpoint+radius)]
    if mesh is None:
        from blender_blocking.reconstruction.mesh_io import mesh_arrays_from_object
        mesh = mesh_arrays_from_object(primitive.to_mesh_data(16))
    vertices, faces = mesh
    xy = np.asarray(vertices)[:, axes]
    intervals = []
    for face in faces:
        polygon = xy[np.asarray(face, int)]
        next_points = np.roll(polygon, -1, axis=0)
        denominator = next_points[:,1] - polygon[:,1]
        crosses = (np.minimum(polygon[:,1],next_points[:,1]) <= vertical) & (np.maximum(polygon[:,1],next_points[:,1]) >= vertical) & (np.abs(denominator)>1e-12)
        if np.count_nonzero(crosses) >= 2:
            points = polygon[crosses,0] + (vertical-polygon[crosses,1]) / denominator[crosses] * (next_points[crosses,0]-polygon[crosses,0])
            intervals.append((points.min(), points.max()))
    return union_intervals(intervals)
