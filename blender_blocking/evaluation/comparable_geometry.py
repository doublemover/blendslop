"""Deterministic mesh comparison with an explicit scale/translation contract."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from blender_blocking.evaluation.geometry import surface_distance_report


def export_evaluated_object(obj, path):
    """Export Blender world coordinates directly; OBJ importer axis defaults do not apply."""
    from blender_blocking.reconstruction.native_geometry import evaluated_arrays
    from blender_blocking.reconstruction.mesh_io import write_obj
    data = evaluated_arrays(obj)
    return write_obj(path, {"vertices": data.vertices, "faces": data.faces},
                     header=("Blendslop Z-up world coordinates",))


def read_obj(path):
    vertices, triangles = [], []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "v":
            vertices.append([float(x) for x in words[1:4]])
        elif words[0] == "f":
            face = [int(x.split("/")[0]) for x in words[1:]]
            face = [x - 1 if x > 0 else len(vertices) + x for x in face]
            triangles.extend((face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1))
    return np.asarray(vertices, float), np.asarray(triangles, int).reshape(-1, 3)


def normalize_mesh(vertices):
    """Uniform similarity only: preserve aspect ratio, orientation, and concavities."""
    vertices = np.asarray(vertices, float)
    if not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("mesh vertices must be nonempty and finite")
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    scale = float((hi - lo).max())
    if scale <= 1e-12:
        raise ValueError("mesh has zero extent")
    center = (lo + hi) / 2
    return (vertices - center) / scale, {"center": center.tolist(), "scale": scale}


def sample_surface(vertices, triangles, *, count=8192, seed=1234):
    """Area weighted triangle samples, rather than tessellation dependent vertices."""
    tri = vertices[triangles]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    good = double_area > 1e-14
    tri, cross, double_area = tri[good], cross[good], double_area[good]
    if not len(tri):
        raise ValueError("mesh has no nondegenerate triangles")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(tri), count, p=double_area / double_area.sum())
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, 0])
    weights = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    points = np.einsum("ni,nij->nj", weights, tri[chosen])
    return points, cross[chosen] / double_area[chosen, None]


def occupancy(vertices, triangles, *, resolution=24):
    """Closed-mesh parity on the same unit cube, using Blender's installed BVH."""
    from mathutils.bvhtree import BVHTree
    from mathutils import Vector
    tree = BVHTree.FromPolygons(vertices.tolist(), triangles.tolist(), all_triangles=True)
    axis = (np.arange(resolution) + .5) / resolution - .5
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    result = np.zeros(len(grid), bool)
    direction = Vector((1, .000013, .000017)).normalized()
    for index, point in enumerate(grid):
        origin = Vector(point)
        hits = 0
        for _ in range(256):
            hit, _, _, _ = tree.ray_cast(origin, direction, 2)
            if hit is None:
                break
            hits += 1
            origin = hit + direction * 1e-7
        else:
            raise ValueError("occupancy ray intersection limit exceeded")
        result[index] = hits % 2 == 1
    return result.reshape((resolution,) * 3)


def compare_meshes(reference_path, candidate_path, *, count=8192, seed=1234, tolerance=.02):
    rv, rt = read_obj(reference_path)
    cv, ct = read_obj(candidate_path)
    rv, ref_transform = normalize_mesh(rv)
    cv, cand_transform = normalize_mesh(cv)
    rp, rn = sample_surface(rv, rt, count=count, seed=seed)
    cp, cn = sample_surface(cv, ct, count=count, seed=seed)
    result = surface_distance_report(rp, cp, tolerance=tolerance,
                                     reference_normals=rn, candidate_normals=cn).to_dict()
    result["source"] = "bbox_uniform_normalized_mesh_surface"
    result["normalization"] = {"protocol": "bbox_center_longest_extent_v1",
                               "reference": ref_transform, "candidate": cand_transform,
                               "rotation_alignment": False, "anisotropic_alignment": False}
    result["sampling"] = {"protocol": "area_weighted_triangles", "seed": seed, "count": count}
    from blender_blocking.metrics.topology import mesh_topology_report
    topology = mesh_topology_report(cv, ct).to_dict()
    result["candidate_topology"] = topology
    reference_topology = mesh_topology_report(rv, rt).to_dict()
    result["reference_topology"] = reference_topology
    try:
        for label, report in (("reference", reference_topology), ("candidate", topology)):
            if not report.get("watertight") or report.get("connected_components") != 1:
                raise ValueError(f"{label} must be one watertight component; overlapping part parity is not solid-union volume")
        from blender_blocking.evaluation.geometry import volumetric_iou
        result["volumetric_iou"] = volumetric_iou(occupancy(rv, rt), occupancy(cv, ct))
        result["occupancy_sampling"] = {"resolution": 24, "bounds": [-.5, .5], "protocol": "cell_center_ray_parity"}
    except (ImportError, ValueError) as exc:
        result["volumetric_iou"] = None
        result["warnings"].append(str(exc))
    return result
