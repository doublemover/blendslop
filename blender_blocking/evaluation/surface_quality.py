"""Shared-world-frame sample-to-triangle distances and oriented normal angles.

Unlike point-to-point Chamfer, nearest target triangles do not create a random
point-sampling distance floor. Sampling is deterministic and area weighted.
Silhouette, solid-boundary and artist-edit acceptance remain separate.
"""

from __future__ import annotations

import numpy as np

from blender_blocking.evaluation.comparable_geometry import sample_surface


def oriented_normal_angles(source, target):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("normal arrays must have matching (N, 3) shapes")
    lengths = np.linalg.norm(source, axis=1) * np.linalg.norm(target, axis=1)
    if not np.isfinite(np.r_[source, target]).all() or (lengths <= 0).any():
        raise ValueError("normal directions must be finite and nonzero")
    cos = np.einsum("ij,ij->i", source, target) / lengths
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def compare_surface_arrays(reference, candidate, *, count=4096, seed=61007,
                           mean_distance_max=.003, normal_p95_max=2.5):
    """Compare immutable geometry arrays using native Blender triangle BVHs.

No independently normalized mesh, nearest sampled point or flipped-normal
equivalence is admitted. Caps/seams are included; zero-area faces are excluded
from sample allocation only and remain a topology failure elsewhere.
"""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    if (isinstance(count, bool) or not isinstance(count, int) or not 256 <= count <= 16384
            or not np.isfinite([mean_distance_max, normal_p95_max]).all()
            or mean_distance_max <= 0 or not 0 < normal_p95_max <= 180):
        raise ValueError("invalid bounded surface metric budget")
    results = []
    for source, target in ((reference, candidate), (candidate, reference)):
        points, normals = sample_surface(source.vertices, source.faces, count=count, seed=seed)
        tree = BVHTree.FromPolygons(target.vertices.tolist(), target.faces.tolist(), all_triangles=True)
        nearest = [tree.find_nearest(Vector(point)) for point in points]
        if any(row[0] is None or row[1] is None or row[3] is None for row in nearest):
            raise ValueError("surface BVH returned unavailable nearest-triangle evidence")
        distances = np.asarray([row[3] for row in nearest], float)
        target_normals = np.asarray([tuple(row[1]) for row in nearest], float)
        angles = oriented_normal_angles(normals, target_normals)
        results.append((distances, angles))
    distances = np.concatenate([r[0] for r in results])
    angles = np.concatenate([r[1] for r in results])
    mean, p95 = float(distances.mean()), float(np.percentile(angles, 95))
    return {
        "protocol": "shared_world_area_sample_to_triangle_v1",
        "sample_count_per_direction": count, "seed": seed,
        "reference_geometry_hash": reference.content_hash,
        "candidate_geometry_hash": candidate.content_hash,
        "normal_orientation": "oriented; opposite normals are 180 degrees",
        "normal_interpretation": "geometric face normals, not shading normals",
        "units": "unchanged Blender world coordinates; no fitting or normalization",
        "symmetric_mean_distance_world": mean,
        "distance_p95_world": float(np.percentile(distances, 95)),
        "sampled_max_distance_world": float(distances.max()),
        "normal_angle_mean_degrees": float(angles.mean()),
        "normal_angle_p95_degrees": p95,
        "frozen_limits": {"symmetric_mean_distance_world_max": mean_distance_max,
                          "normal_angle_p95_degrees_max": normal_p95_max},
        "surface_passed": bool(mean <= mean_distance_max and p95 <= normal_p95_max),
        "silhouette_verdict": "independent",
        "topology_verdict": "independent; BVH proximity does not qualify a solid boundary",
        "editability_verdict": "independent",
    }
