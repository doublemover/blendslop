"""Small whole-shape proposals fitted to observed orthographic support bounds.

Support fitting ignores concavity. Its output is a proposal requiring complete
mask, empty-region and actual mesh/render checks before admission.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np


@dataclass(frozen=True)
class SupportEvidence:
    directions: np.ndarray
    values: np.ndarray
    tolerances: np.ndarray
    censored: np.ndarray
    weights: np.ndarray
    length_scale: float


def primitive_support(family, directions, center, dimensions, rotation):
    """World directional support; dimensions are positive local half sizes.

    Cylinder dimensions are (radius, half-height), and frustum dimensions are
    (bottom radius, top radius, half-height). The local Z column is the axis.
    """
    directions = np.asarray(directions, float)
    local = directions @ np.asarray(rotation, float)
    dimensions = np.asarray(dimensions, float)
    if family == "box":
        radial = np.abs(local) @ dimensions
    elif family == "ellipsoid":
        radial = np.linalg.norm(local * dimensions, axis=1)
    elif family == "cylinder":
        radial = dimensions[1] * np.abs(local[:, 2]) + dimensions[0] * np.linalg.norm(local[:, :2], axis=1)
    elif family == "frustum":
        cross = np.linalg.norm(local[:, :2], axis=1)
        radial = np.maximum(-dimensions[2] * local[:, 2] + dimensions[0] * cross,
                             dimensions[2] * local[:, 2] + dimensions[1] * cross)
    else:
        raise ValueError(f"unsupported whole support family {family!r}")
    return directions @ np.asarray(center, float) + radial


def support_evidence(target, *, directions_per_view=24):
    from .projection_contract import pixel_cell_viewport
    from .visibility import valid_evidence
    directions, values, tolerances, censored = [], [], [], []
    for constraint in target.constraints:
        mask = np.asarray(getattr(constraint.mask, "mask", constraint.mask), bool)
        valid = valid_evidence(constraint)
        y, x = np.nonzero(mask & valid)
        if not len(x):
            continue
        h, w = mask.shape
        axes, (u0, u1, v0, v1) = pixel_cell_viewport(target, constraint)
        sx, sy = (u1-u0)/w, (v1-v0)/h
        points = np.column_stack((u0+(x+.5)*sx, v1-(y+.5)*sy))
        unknown_y, unknown_x = np.nonzero(~valid)
        unknown = np.column_stack((u0+(unknown_x+.5)*sx, v1-(unknown_y+.5)*sy))
        for angle in np.linspace(0., 2.*np.pi, max(8, int(directions_per_view)), endpoint=False):
            uv = np.array([np.cos(angle), np.sin(angle)])
            footprint = .5 * (abs(uv[0])*sx + abs(uv[1])*sy)
            observed = float(np.max(points @ uv)) + footprint
            # Unknown pixels farther along this direction censor the upper
            # support bound. Their stored foreground values are never read.
            is_censored = bool(len(unknown) and np.max(unknown @ uv)+footprint >= observed-footprint)
            world = np.zeros(3); world[list(axes)] = uv
            directions.append(world); values.append(observed)
            tolerances.append(footprint); censored.append(is_censored)
    if not directions:
        raise ValueError("whole-shape fitting has no known foreground support")
    lo, hi = np.asarray(target.bounds.to_min_max(), float)
    length = float(np.max(hi-lo))
    if not np.isfinite(length) or length <= 0.:
        raise ValueError("whole-shape support fitting needs positive metric bounds")
    return SupportEvidence(np.asarray(directions), np.asarray(values), np.asarray(tolerances),
                           np.asarray(censored), np.ones(len(values)), length)


def support_residual(evidence, prediction):
    delta = np.asarray(prediction, float) - evidence.values
    residual = np.sign(delta) * np.maximum(np.abs(delta) - evidence.tolerances, 0.)
    # Censored evidence provides a lower bound only, never a forced exact edge.
    residual = np.where(evidence.censored, np.minimum(delta + evidence.tolerances, 0.), residual)
    return residual / evidence.length_scale * np.sqrt(evidence.weights)


def fit_whole_support(family, evidence, *, center, dimensions, rotation,
                      max_evaluations=96, max_elapsed_s=1.):
    """Fit a coupled local pose/size model, retaining every best scored state.

    Counts include numeric-Jacobian calls; a deadline/count stop returns the best
    completed proposal instead of erasing useful fitting work.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    start = time.perf_counter()
    center, dimensions, rotation = np.asarray(center, float), np.asarray(dimensions, float), np.asarray(rotation, float)
    if np.any(dimensions <= 0.) or not np.isfinite(dimensions).all():
        raise ValueError("whole-shape dimensions must be finite and positive")
    if max_evaluations < 1 or (max_elapsed_s is not None and max_elapsed_s <= 0.):
        raise ValueError("whole-shape fit allowance must be positive")
    initial = np.zeros(6 + len(dimensions))
    calls, best, best_score = 0, initial.copy(), float("inf")
    stop = "solver_completed"

    def decode(x):
        return (center + x[:3]*evidence.length_scale,
                dimensions*np.exp(x[6:]), rotation @ Rotation.from_rotvec(x[3:6]).as_matrix())

    class AllowanceEnded(Exception):
        pass

    def residual(x):
        nonlocal calls, best, best_score, stop
        if calls >= max_evaluations:
            stop = "support_evaluation_allowance"
            raise AllowanceEnded
        if calls and max_elapsed_s is not None and time.perf_counter()-start >= max_elapsed_s:
            stop = "support_elapsed_allowance"
            raise AllowanceEnded
        calls += 1
        c, d, r = decode(x)
        value = support_residual(evidence, primitive_support(family, evidence.directions, c, d, r))
        score = float(value @ value)
        if score < best_score:
            best, best_score = x.copy(), score
        return value

    try:
        least_squares(residual, initial, bounds=(-np.ones_like(initial)*2., np.ones_like(initial)*2.),
                      x_scale="jac", max_nfev=max_evaluations, ftol=1e-8, xtol=1e-8, gtol=1e-8)
    except AllowanceEnded:
        pass
    c, d, r = decode(best)
    return {"family": family, "center": c, "dimensions": d, "rotation": r,
            "support_squared_residual": best_score, "support_evaluations": calls,
            "support_elapsed_s": time.perf_counter()-start, "termination": stop,
            "evidence_scope": "observed support bounds; full masks and retained geometry admission still required"}
