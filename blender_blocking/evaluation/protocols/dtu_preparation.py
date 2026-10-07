"""Bounded, seeded port of the pinned DTU Python sampling/masking protocol.

Reference: https://github.com/jzhangbs/DTUeval-python
Revision: 45b4fecbe02e8333f6653c6b655c3160eb8404a4 (MIT; see adjacent notice).
This is a reference port, not a claim of bitwise MATLAB evaluator reproduction.
Wrapper silhouette culling must supply a separately verified receipt.
"""
from __future__ import annotations

import hashlib
from itertools import product
from pathlib import Path
from typing import Mapping

import numpy as np

from .core import apply_transform, points

DTU_REFERENCE_REVISION = "45b4fecbe02e8333f6653c6b655c3160eb8404a4"
DTU_REFERENCE_SHA256 = "927f7dd95eb8a61db65f910c8a8f7d577e599ad92f35139fd1d4cd75186e59ba"
DTU_PREPARATION_VERSION = "blendslop_dtu_python_reference_port_v1"


class PreparationLimitExceeded(ValueError):
    """Preparation refuses to truncate or silently decimate an oversized input."""


def array_sha256(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _triangles(value, vertex_count):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "iu":
        raise ValueError("faces must be an integer Nx3 array")
    faces = raw.astype(np.int64, copy=False)
    if len(faces) and (faces.min() < 0 or faces.max() >= vertex_count):
        raise ValueError("face index outside prediction vertices")
    return faces


def sample_dtu_mesh(vertices, faces, *, spacing_mm=.2, max_points=2_000_000):
    """Triangle mid-cell lattice plus every original vertex, in native mm.

    The edge lengths and twice-area determine each triangle's lattice spacing.
    Degenerate triangles add no interior samples. No area-random substitute or
    automatic coarsening is used when the explicit allocation limit is reached.
    """
    vertices = points(vertices)
    faces = _triangles(faces, len(vertices))
    if not np.isfinite(spacing_mm) or spacing_mm <= 0 or max_points < 1:
        raise ValueError("spacing and allocation limit must be positive")
    if len(vertices) > max_points:
        raise PreparationLimitExceeded("prediction vertices exceed sampling limit")
    chunks = [vertices]
    count = len(vertices)
    for face in faces:
        origin, b, c = vertices[face]
        u, v = b - origin, c - origin
        area2 = np.linalg.norm(np.cross(u, v))
        if area2 <= 0:
            continue
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        threshold = spacing_mm * np.sqrt(lu * lv / area2)
        n1, n2 = int(np.floor(lu / threshold)), int(np.floor(lv / threshold))
        if n1 == 0 or n2 == 0:
            continue
        # Compute each row length before allocating; this also handles thin
        # triangles with a long edge without a giant rectangular temporary.
        for row in range(n1 + 1):
            alpha = (row + .5) / n1
            if alpha >= 1:
                break
            upper = min(n2 + 1, max(0, int(np.ceil((1 - alpha) * n2 - .5))))
            if upper == 0:
                continue
            if upper > max_points - count + 1:
                raise PreparationLimitExceeded("DTU triangle samples exceed sampling limit")
            beta = (np.arange(upper, dtype=np.float64) + .5) / n2
            beta = beta[alpha + beta < 1]
            count += len(beta)
            if count > max_points:
                raise PreparationLimitExceeded("DTU triangle samples exceed sampling limit")
            chunks.append(origin + alpha * u + beta[:, None] * v)
    return np.concatenate(chunks, axis=0)


def thin_dtu_points(value, *, spacing_mm=.2, seed=1234, max_points=2_000_000):
    """Seeded radius-greedy thinning; keep the upstream shuffled output order."""
    data = points(value).copy()
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("spacing must be positive and finite")
    if len(data) > max_points:
        raise PreparationLimitExceeded("point cloud exceeds thinning limit")
    np.random.default_rng(seed).shuffle(data, axis=0)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    keep = np.ones(len(data), dtype=bool)
    if cKDTree is not None:
        tree = cKDTree(data)
        for index in range(len(data)):
            if keep[index]:
                keep[tree.query_ball_point(data[index], spacing_mm)] = False
                keep[index] = True
        backend = "scipy_ckdtree_radius_greedy"
    else:
        cells = {}
        radius2 = spacing_mm ** 2
        for index, point in enumerate(data):
            cell = tuple(int(x) for x in np.floor(point / spacing_mm))
            neighbors = (cells.get(tuple(cell[k] + offset[k] for k in range(3)), ())
                         for offset in product((-1, 0, 1), repeat=3))
            if any(float(np.dot(point - data[j], point - data[j])) <= radius2
                   for group in neighbors for j in group):
                keep[index] = False
            else:
                cells.setdefault(cell, []).append(index)
        backend = "spatial_hash_radius_greedy"
    return data[keep], {"backend": backend, "seed": int(seed),
                        "input_count": len(data), "output_count": int(keep.sum())}


def dtu_directional_sets(native_prediction, assets: Mapping, *, patch_mm=60.):
    """Build the asymmetric sets, including BB padding, ObsMask and ground plane."""
    data = points(native_prediction)
    reference = points(assets["reference"])
    bb = np.asarray(assets["observation_bb"], dtype=np.float32)
    res = np.asarray(assets["observation_res"], dtype=np.float64).reshape(-1)
    obs = np.asarray(assets["obs_mask"])
    plane = np.asarray(assets["ground_plane"], dtype=np.float64).reshape(-1)
    if (bb.shape != (2, 3) or not np.isfinite(bb).all() or np.any(bb[1] <= bb[0]) or
            res.size not in (1, 3) or not np.isfinite(res).all() or np.any(res <= 0) or
            obs.ndim != 3 or not obs.size or obs.dtype.kind not in "biuf" or
            not np.isfinite(obs).all() or plane.size != 4 or not np.isfinite(plane).all() or
            not np.isfinite(patch_mm) or patch_mm < 0):
        raise ValueError("invalid native DTU observation assets")
    inbound = np.all((data >= bb[:1] - patch_mm) & (data < bb[1:] + patch_mm * 2), axis=1)
    data_in = data[inbound]
    grid = np.rint((data_in - bb[:1]) / res).astype(np.int64)
    grid_in = np.all((grid >= 0) & (grid < np.array(obs.shape)), axis=1)
    valid_grid = grid[grid_in]
    observed = obs[valid_grid[:, 0], valid_grid[:, 1], valid_grid[:, 2]].astype(bool)
    homogeneous = np.column_stack((reference, np.ones(len(reference))))
    above = (homogeneous @ plane) > 0
    return {"prediction_queries": data_in[grid_in][observed],
            "reference_targets": reference,
            "reference_queries": reference[above],
            "prediction_targets": data_in}


def silhouette_keep_mask(value, cameras, masks, *, require_all_views=True):
    """Calibrated nearest-pixel projection helper, with explicit view admission.

    Cameras are native-coordinate 3x4 projective matrices. Missing/behind/outside
    observations reject a point for the all-view rule. This helper does not by
    itself certify any external method's wrapper semantics.
    """
    data = points(value)
    if not cameras or len(cameras) != len(masks):
        raise ValueError("one mask per calibrated camera is required")
    per_view = []
    for camera, mask in zip(cameras, masks):
        matrix = np.asarray(camera, dtype=float)
        image = np.asarray(mask)
        if matrix.shape != (3, 4) or not np.isfinite(matrix).all() or image.ndim != 2 or not image.size:
            raise ValueError("invalid camera or silhouette mask")
        projected = data @ matrix[:, :3].T + matrix[:, 3]
        positive = projected[:, 2] > 0
        pixel = np.zeros((len(data), 2), dtype=np.int64)
        pixel[positive] = np.floor(projected[positive, :2] / projected[positive, 2:3] + .5).astype(np.int64)
        visible = positive & (pixel[:, 0] >= 0) & (pixel[:, 0] < image.shape[1]) & (pixel[:, 1] >= 0) & (pixel[:, 1] < image.shape[0])
        admitted = np.zeros(len(data), dtype=bool)
        admitted[visible] = image[pixel[visible, 1], pixel[visible, 0]].astype(bool)
        per_view.append(admitted)
    return np.all(per_view, axis=0) if require_all_views else np.any(per_view, axis=0)


def prepare_dtu_prediction(*, mode, prediction, assets, faces=None, seed=1234,
                           max_points=2_000_000, culling=None):
    """Prepare real native assets; wrapper culling and resource failures stay explicit."""
    from .dtu import DTU_MODES, DTU_CONTRACT
    if mode not in DTU_MODES:
        raise ValueError("unknown DTU wrapper mode")
    missing = [key for key in DTU_CONTRACT["required_assets"] if key not in assets]
    if missing:
        return {"status": "unavailable", "reason": "missing native assets", "missing": missing}
    original = points(prediction)
    native = apply_transform(original, assets["scale_mat"])
    triangles = None if faces is None else _triangles(faces, len(native))
    culling_hashes = None
    if DTU_MODES[mode]["silhouette_culling"]:
        receipt = (culling or {}).get("receipt", {})
        keep = np.asarray((culling or {}).get("keep", []))
        expected_hash = array_sha256(original if triangles is None else triangles)
        if (receipt.get("verified_wrapper") is not True or not receipt.get("source_sha256") or
                receipt.get("input_sha256") != expected_hash or
                receipt.get("prediction_sha256") != array_sha256(original) or
                receipt.get("native_prediction_sha256") != array_sha256(native) or
                not receipt.get("camera_sha256") or not receipt.get("mask_sha256") or keep.dtype.kind != "b" or
                keep.shape != (len(native) if triangles is None else len(triangles),) or
                receipt.get("keep_sha256") != array_sha256(keep) or
                receipt.get("selection") != ("points" if triangles is None else "triangles")):
            return {"status": "unavailable", "reason": "hash-matched verified wrapper culling required"}
        if triangles is None:
            native = native[keep]
        else:
            triangles = triangles[keep]
            # Keep only vertices referenced by the culled mesh, preserving their
            # original order and remapping triangle connectivity explicitly.
            used = np.unique(triangles)
            remap = np.full(len(native), -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            native, triangles = native[used], remap[triangles]
        culling_hashes = dict(receipt)
    if not len(native):
        return {"status": "unavailable", "reason": "empty prediction after wrapper culling"}
    try:
        sampled = native if triangles is None else sample_dtu_mesh(native, triangles, max_points=max_points)
        thinned, thinning = thin_dtu_points(sampled, seed=seed, max_points=max_points)
        sets = dtu_directional_sets(thinned, assets)
    except PreparationLimitExceeded as exc:
        return {"status": "unavailable", "reason": str(exc), "limit_exceeded": True}
    empty = [key for key, value in sets.items() if not len(value)]
    if empty:
        return {"status": "unavailable", "reason": "empty asymmetric direction", "empty": empty}
    receipt = {"preparation_version": DTU_PREPARATION_VERSION,
               "compatibility": "pinned_dtu_python_reference_port",
               "implementation_sha256": implementation_sha256(),
               "reference_revision": DTU_REFERENCE_REVISION, "source_sha256": DTU_REFERENCE_SHA256,
               "sampling": "point_cloud" if faces is None else "triangle_midcell_lattice_plus_vertices",
               "spacing_mm": .2, "patch_mm": 60., "thinning": thinning,
               "silhouette_culling": DTU_MODES[mode]["silhouette_culling"],
               "wrapper_culling": culling_hashes,
               "prediction_sha256": array_sha256(original),
               "native_prediction_sha256": array_sha256(apply_transform(original, assets["scale_mat"])),
               "faces_sha256": None if faces is None else array_sha256(_triangles(faces, len(original))),
               "asset_sha256": {key: array_sha256(assets[key]) for key in DTU_CONTRACT["required_assets"]},
               "directional_sha256": {key: array_sha256(value) for key, value in sets.items()}}
    return {"status": "available", "prediction": original, "directional_sets": sets,
            "sampling_receipt": receipt}
