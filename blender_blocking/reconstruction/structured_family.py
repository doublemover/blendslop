"""Observed nonconvex frozen-family proposals; reference geometry is excluded."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import time

import numpy as np

from .frozen_family import _camera, coverage_contour_points, validate_family_budget

STRUCTURED_FAMILIES = ("torus", "concave_arch")


def _torus_program(target, masks, cameras, coverage_masks, max_evaluations, max_elapsed_s):
    from scipy.ndimage import binary_fill_holes
    from scipy.optimize import least_squares
    from primitives.shape_program import ShapeNode, ShapeProgram
    coverage = None if coverage_masks is None else coverage_masks.get("top")
    if coverage is None:
        raise ValueError("torus ring fit requires explicit complete top coverage")
    top = np.asarray(coverage) >= .5
    holes = binary_fill_holes(top) & ~top
    if np.count_nonzero(holes) < 4:
        raise ValueError("torus ring fit requires an observed hole")
    if int(np.argmin(target.bounds.size)) != 2:
        raise ValueError("initial torus proposal requires a canonical Z-axis ring")
    pixels = coverage_contour_points(coverage)
    height, width = top.shape
    record = cameras["top"]
    matrix, scale = _camera(record)
    if (not np.allclose(matrix[:3, 0], [1., 0., 0.], atol=1e-5)
            or not np.allclose(matrix[:3, 1], [0., 1., 0.], atol=1e-5)):
        raise ValueError("torus fit requires the canonical top camera X/Y basis")
    points = np.column_stack(((pixels[:, 0]/width-.5)*scale,
                             (.5-pixels[:, 1]/height)*scale))+matrix[:2, 3]
    center = np.asarray(target.bounds.center, float)
    radii = np.linalg.norm(points-center[:2], axis=1)
    split = .5*(float(radii.min())+float(radii.max()))
    outer = radii > split
    if min(int(outer.sum()), int((~outer).sum())) < 8:
        raise ValueError("ring requires independent outer and inner contour support")
    outer_radius, inner_radius = float(np.median(radii[outer])), float(np.median(radii[~outer]))
    major_seed = .5*(outer_radius+inner_radius)
    minor_seed = .5*(outer_radius-inner_radius)
    if not 0 < minor_seed < .95*major_seed:
        raise ValueError("observed annular radii are incompatible with a torus")
    length = float(max(target.bounds.size))
    depth_radius = float(target.bounds.size[2]*.5)
    offset = np.log(.95*major_seed/minor_seed-1)
    sign = np.where(outer, 1., -1.)
    # Equal component weights prevent the longer outer ring from dominating.
    weights = np.where(outer, 1./np.sqrt(outer.sum()), 1./np.sqrt((~outer).sum()))
    calls, score, best, termination = 0, float("inf"), np.zeros(4), "solver_completed"
    start = time.perf_counter()

    def decode(x):
        major = major_seed*np.exp(x[2])
        minor = major*.95/(1+np.exp(-x[3]+offset))
        return center[:2]+x[:2]*length, float(major), float(minor)

    class AllowanceEnded(Exception):
        pass

    def residual(x):
        nonlocal calls, score, best, termination
        if calls >= max_evaluations or (calls and time.perf_counter()-start >= max_elapsed_s):
            termination = "ring_evaluation_allowance" if calls >= max_evaluations else "ring_elapsed_allowance"
            raise AllowanceEnded
        calls += 1
        xy, major, minor = decode(x)
        contour = (np.linalg.norm(points-xy, axis=1)-major-sign*minor)*weights/length
        result = np.r_[contour, (minor-depth_radius)/length]
        current = float(result @ result)
        if current < score:
            score, best = current, x.copy()
        return result

    try:
        least_squares(residual, best, bounds=(-np.ones(4)*2, np.ones(4)*2),
                      x_scale="jac", max_nfev=max_evaluations, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    except AllowanceEnded:
        pass
    xy, major, minor = decode(best)
    parameters = {"x": float(xy[0]), "y": float(xy[1]), "z": float(center[2]),
                  "major_radius": major, "minor_radius": minor, "rotation": np.eye(3).tolist()}
    metadata = {"proposal": "observed_annular_circle_fit", "support_evaluations": calls,
                "support_elapsed_s": time.perf_counter()-start, "support_squared_residual": score,
                "termination": termination, "observed_hole_pixels": int(holes.sum()),
                "inner_contour_samples": int((~outer).sum()), "outer_contour_samples": int(outer.sum()),
                "evidence_scope": "explicit top outer/inner crossings and observed front/side depth bounds",
                "coverage_views": ["front", "side", "top"], "input_bounds": target.bounds.to_dict(),
                "camera_basis_scope": "canonical top image right +X/up +Y; Z-axis ring only",
                "camera_sha256": hashlib.sha256(json.dumps(cameras, sort_keys=True).encode()).hexdigest(),
                "full_five_view_admission": "unrun; required after actual native geometry"}
    return ShapeProgram("1", "frozen-family-torus",
                        (ShapeNode("observed_torus", "add", "torus", parameters=parameters),), metadata=metadata)


def structured_family_program(family, target, masks, cameras, *, coverage_masks=None,
                             max_evaluations=256, max_elapsed_s=3.):
    """Fit an observed ring or reuse the existing measured planar extrusion."""
    validate_family_budget(max_evaluations, max_elapsed_s)
    if family == "torus":
        return _torus_program(target, masks, cameras, coverage_masks, max_evaluations, max_elapsed_s)
    if family == "concave_arch":
        from primitives.shape_program import ShapeProgram
        from .polygon_proposals import planar_extrusion_programs
        proposals = planar_extrusion_programs(target, ShapeProgram("1", "frozen-family-concave_arch", ()),
                                              maximum_thickness_ratio=.3)
        if len(proposals) != 1 or proposals[0].node_count() != 1:
            raise ValueError("arch proposal needs one complete measured planar component")
        return replace(proposals[0], metadata={**proposals[0].metadata,
            "support_evaluations": 0, "support_elapsed_s": 0., "support_squared_residual": None,
            "evidence_scope": "known front pixel-cell exterior outline and independently observed axis thickness",
            "coverage_views": sorted(coverage_masks or {}), "input_bounds": target.bounds.to_dict(),
            "planar_thickness_ratio_cap": .3,
            "camera_sha256": hashlib.sha256(json.dumps(cameras, sort_keys=True).encode()).hexdigest(),
            "full_five_view_admission": "unrun; required after actual native geometry"})
    raise ValueError("unsupported nonconvex frozen family")
