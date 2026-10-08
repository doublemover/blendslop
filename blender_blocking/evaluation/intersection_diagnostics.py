"""Bounded contact diagnostics for pairs reported by a native triangle screen.

Classification never removes a native intersection or certifies a solid. The
relative tolerance and all ambiguous cases are reported for review.
"""
from __future__ import annotations
from collections import Counter
import numpy as np

INTERSECTION_DIAGNOSTIC_VERSION = "triangle_pair_contact_diagnostic_v1"


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _coplanar_polygon(left, right, normal, tolerance):
    axes = [i for i in range(3) if i != int(np.argmax(np.abs(normal)))]
    polygon = [point.copy() for point in left[:, axes]]
    clip = right[:, axes].copy()
    if _cross2(clip[1] - clip[0], clip[2] - clip[0]) < 0:
        clip = clip[::-1]
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        edge = b - a
        epsilon = tolerance * np.linalg.norm(edge)
        output = []
        if not polygon:
            break
        previous = polygon[-1]
        before = _cross2(edge, previous - a)
        for point in polygon:
            after = _cross2(edge, point - a)
            before_in, after_in = before >= -epsilon, after >= -epsilon
            if before_in != after_in:
                denominator = before - after
                if abs(denominator) > np.finfo(float).tiny:
                    output.append(previous + (point - previous) * before / denominator)
            if after_in:
                output.append(point.copy())
            previous, before = point, after
        polygon = output
    return np.array(polygon).reshape(-1, 2)


def _plane_cut(triangle, distances, tolerance):
    cut = []
    for index, point in enumerate(triangle):
        other = (index + 1) % 3
        if abs(distances[index]) <= tolerance:
            cut.append(point)
        if distances[index] * distances[other] < 0:
            weight = distances[index] / (distances[index] - distances[other])
            cut.append(point + weight * (triangle[other] - point))
    return np.array(cut).reshape(-1, 3)


def _strictly_inside(point, triangle, normal, tolerance):
    for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
        edge = b - a
        if float(np.dot(np.cross(edge, point - a), normal)) <= tolerance * np.linalg.norm(edge):
            return False
    return True


def classify_triangle_contact(left, right, *, relative_tolerance=1e-9):
    left, right = np.asarray(left, float), np.asarray(right, float)
    if (left.shape != (3, 3) or right.shape != (3, 3) or
            not np.isfinite(left).all() or not np.isfinite(right).all() or
            not np.isfinite(relative_tolerance) or relative_tolerance <= 0):
        raise ValueError("contact diagnostics require two finite triangles and positive tolerance")
    scale = max(np.linalg.norm(np.roll(t, -1, axis=0) - t, axis=1).max() for t in (left, right))
    tolerance = max(float(scale * relative_tolerance), np.finfo(float).eps * max(scale, 1.) * 16)
    normals = [np.cross(t[1] - t[0], t[2] - t[0]) for t in (left, right)]
    if any(np.linalg.norm(n) <= tolerance * max(scale, tolerance) for n in normals):
        return {"classification": "indeterminate_degenerate", "tolerance": tolerance}
    na, nb = [n / np.linalg.norm(n) for n in normals]
    da = (left - right[0]) @ nb
    db = (right - left[0]) @ na
    if (np.all(da > tolerance) or np.all(da < -tolerance) or
            np.all(db > tolerance) or np.all(db < -tolerance)):
        return {"classification": "no_contact_at_tolerance", "tolerance": tolerance}
    line = np.cross(na, nb)
    if np.linalg.norm(line) <= relative_tolerance:
        if max(np.abs(da).max(), np.abs(db).max()) > tolerance:
            return {"classification": "indeterminate_near_parallel", "tolerance": tolerance}
        polygon = _coplanar_polygon(left, right, na, tolerance)
        if not len(polygon):
            kind = "no_contact_at_tolerance"
            area = 0.
        else:
            area = abs(sum(_cross2(a, b) for a, b in zip(polygon, np.roll(polygon, -1, axis=0)))) / 2
            # Projection drops one axis; restore area in the triangle plane.
            area /= np.max(np.abs(na))
            diameter = np.linalg.norm(polygon[:, None] - polygon[None, :], axis=2).max()
            kind = ("coplanar_area_overlap" if area > tolerance * max(scale, tolerance) else
                    "boundary_segment_contact" if diameter > tolerance else "point_contact")
        return {"classification": kind, "tolerance": tolerance, "coplanar_overlap_area": float(area)}
    line /= np.linalg.norm(line)
    a, b = _plane_cut(left, da, tolerance), _plane_cut(right, db, tolerance)
    if not len(a) or not len(b):
        return {"classification": "indeterminate_empty_plane_cut", "tolerance": tolerance}
    ap, bp = a @ line, b @ line
    lower, upper = max(ap.min(), bp.min()), min(ap.max(), bp.max())
    if upper < lower - tolerance:
        kind = "no_contact_at_tolerance"
    elif upper - lower <= tolerance:
        kind = "point_contact"
    else:
        midpoint = a[0] + line * ((lower + upper) / 2 - ap[0])
        kind = ("proper_crossing" if _strictly_inside(midpoint, left, na, tolerance) and
                _strictly_inside(midpoint, right, nb, tolerance) else "boundary_segment_contact")
    return {"classification": kind, "tolerance": tolerance,
            "intersection_length": float(max(0., upper - lower))}


def intersection_pair_diagnostics(vertices, faces, pairs, *, max_pairs=1000, relative_tolerance=1e-9):
    vertices, faces, pairs = np.asarray(vertices, float), np.asarray(faces), np.asarray(pairs)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu" or max_pairs < 0:
        raise ValueError("native pairs must be integer Nx2 indices and max_pairs nonnegative")
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= len(faces)):
        raise ValueError("intersection pair is outside the triangle array")
    records = []
    for pair in pairs[:max_pairs]:
        item = classify_triangle_contact(vertices[faces[pair[0]]], vertices[faces[pair[1]]],
                                         relative_tolerance=relative_tolerance)
        records.append({"pair": pair.tolist(), **item})
    return {"version": INTERSECTION_DIAGNOSTIC_VERSION,
            "status": "complete" if len(records) == len(pairs) else "bounded_partial",
            "pair_count": len(pairs), "classified_count": len(records), "limit": max_pairs,
            "relative_tolerance": relative_tolerance,
            "counts": dict(Counter(row["classification"] for row in records)), "records": records,
            "qualification_policy": "diagnostic_only; all native pairs still prevent single-solid qualification"}
