"""Bounded convex family proposals fitted only to observed masks and cameras.

The family label selects a hypothesis, never a dimension. Reference geometry and
fixture parameters are deliberately absent from this interface. Supports ignore
concavity, so every proposal still needs full rendered-mask admission.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import time

import numpy as np

from .oriented_support import SupportEvidence, fit_whole_support, support_residual
from .projection_contract import AXES, bounds_from_calibrated_masks, observed_holes
from .types import Bounds2D, OrthographicCameraSpec, ReconstructionTarget, ViewConstraint

CONVEX_FAMILIES = (
    "sphere", "anisotropic_ellipsoid", "cylinder", "tapered_frustum",
    "capsule", "rounded_box", "thin_plate",
)
SUPPORTED_FAMILIES = CONVEX_FAMILIES + ("torus", "concave_arch")
FAMILY_PROPOSAL_ROUTES = {name: "run_frozen_family_reconstruction.py" for name in SUPPORTED_FAMILIES}
FAMILY_PROPOSAL_ROUTES["asymmetric_multipart_solid"] = "run_frozen_multipart_reconstruction.py"
FROZEN_FAMILIES = (
    "sphere", "anisotropic_ellipsoid", "cylinder", "tapered_frustum",
    "smooth_vase", "torus", "capsule", "rounded_box", "thin_plate",
    "concave_arch", "asymmetric_multipart_solid", "rounded_triangle_dot",
)


def coverage_contour_points(coverage, *, allow_clipped=False):
    """Half-coverage crossings in pixel-cell coordinates, without smoothing."""
    image = np.asarray(coverage, float)
    if (image.ndim != 2 or min(image.shape) < 3 or not np.isfinite(image).all()
            or (image < 0).any() or (image > 1).any()):
        raise ValueError("coverage must be a finite bounded image")
    foreground = image >= .5
    if not allow_clipped and (foreground[0].any() or foreground[-1].any()
            or foreground[:, 0].any() or foreground[:, -1].any()):
        raise ValueError("clipped coverage has no complete convex support")
    points = []
    for axis in (0, 1):
        lower = image[:-1, :] if axis == 0 else image[:, :-1]
        upper = image[1:, :] if axis == 0 else image[:, 1:]
        y, x = np.nonzero((lower >= .5) != (upper >= .5))
        fraction = (.5 - lower[y, x]) / (upper[y, x] - lower[y, x])
        points.append(np.column_stack((x + .5 + (fraction if axis == 1 else 0),
                                       y + .5 + (fraction if axis == 0 else 0))))
    result = np.concatenate(points)
    if len(result) < 8:
        raise ValueError("observed coverage has insufficient complete contour")
    return result


def _camera(record):
    matrix = np.asarray(record.get("matrix_world"), float)
    scale = float(record.get("ortho_scale", 0))
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.isfinite(scale) or not scale > 0
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or np.linalg.det(matrix[:3, :3]) < .99999):
        raise ValueError("fitting requires a finite proper orthographic camera")
    return matrix, scale


def observed_target(masks, cameras, *, coverage_masks=None):
    """Build metric bounds exclusively from three calibrated input images."""
    if not set(AXES).issubset(masks) or not set(masks).issubset(cameras):
        raise ValueError("front, side, top masks and matching cameras are required")
    constraints, boxes, records, canonical_masks = [], {}, {}, {}
    for view, axes in AXES.items():
        mask = np.asarray(masks[view], bool)
        if mask.ndim != 2 or not mask.any():
            raise ValueError("canonical observed mask must be nonempty and 2D")
        matrix, scale = _camera(cameras[view])
        if (not np.allclose(matrix[:3, 0], np.eye(3)[axes[0]], atol=1e-5)
                or not np.allclose(matrix[:3, 1], np.eye(3)[axes[1]], atol=1e-5)):
            raise ValueError("canonical input camera is rotated or flipped")
        center = matrix[:3, 3]
        u, v = center[list(axes)]
        viewport = (u-scale/2, u+scale/2, v-scale/2, v+scale/2)
        y, x = np.nonzero(mask)
        boxes[view] = Bounds2D(float(x.min()), float(y.min()),
                              float(x.max()+1), float(y.max()+1))
        records[view] = {"world_bounds": viewport, "world_units": "metres"}
        camera = OrthographicCameraSpec(
            view, {"front": "y", "side": "x", "top": "z"}[view],
            resolution=(mask.shape[1], mask.shape[0]),
            bounds=Bounds2D(viewport[0], viewport[2], viewport[1], viewport[3]))
        coverage = None if coverage_masks is None else coverage_masks.get(view)
        constraints.append(ViewConstraint(view, mask.copy(), camera, bbox=boxes[view],
                                          coverage_mask=coverage))
        canonical_masks[view] = mask
    bounds = bounds_from_calibrated_masks(canonical_masks, boxes, records,
                                          coverage_masks=coverage_masks)
    return ReconstructionTarget(tuple(constraints), bounds=bounds,
                                extras={"view_calibration": records})


def camera_support_evidence(masks, cameras, bounds, *, coverage_masks=None,
                            directions_per_view=32):
    """Measured support in each declared camera plane, including oblique inputs.

    With explicit coverage, half-coverage crossings replace hard pixel edges.
    A tenth-pixel allowance is measurement precision, not a surface pass limit.
    """
    if not 8 <= directions_per_view <= 128:
        raise ValueError("support direction allowance must be between 8 and 128")
    directions, values, tolerances, censored = [], [], [], []
    for view in sorted(masks):
        mask = np.asarray(masks[view], bool)
        if mask.ndim != 2 or not mask.any():
            raise ValueError("support mask must be nonempty and 2D")
        matrix, scale = _camera(cameras[view])
        h, w = mask.shape
        sx, sy = scale/w, scale/h
        coverage = None if coverage_masks is None else coverage_masks.get(view)
        if coverage is not None:
            if np.asarray(coverage).shape != mask.shape:
                raise ValueError("coverage and observed mask shape differ")
            pixels = coverage_contour_points(coverage, allow_clipped=True)
        else:
            y, x = np.nonzero(mask)
            pixels = np.column_stack((x+.5, y+.5))
        visible = mask if coverage is None else np.asarray(coverage) >= .5
        edge = np.zeros(mask.shape, bool)
        edge[0] = edge[-1] = True
        edge[:, 0] = edge[:, -1] = True
        by, bx = np.nonzero(visible & edge)
        boundary_pixels = np.column_stack((bx+.5, by+.5))
        if len(boundary_pixels):
            pixels = np.concatenate((pixels, boundary_pixels))
        boundary_uv = np.column_stack(((boundary_pixels[:, 0]/w-.5)*scale,
                                       (.5-boundary_pixels[:, 1]/h)*scale))
        uv_points = np.column_stack(((pixels[:, 0]/w-.5)*scale,
                                     (.5-pixels[:, 1]/h)*scale))
        for angle in np.linspace(0, 2*np.pi, directions_per_view, endpoint=False):
            uv = np.array([np.cos(angle), np.sin(angle)])
            world = matrix[:3, :2] @ uv
            footprint = .5*(abs(uv[0])*sx + abs(uv[1])*sy)
            value = float(np.max(uv_points @ uv) + world @ matrix[:3, 3])
            directions.append(world)
            values.append(value if coverage is not None else value+footprint)
            tolerances.append(.1*max(sx, sy) if coverage is not None else footprint)
            censored.append(bool(len(boundary_uv) and np.max(boundary_uv @ uv)
                                 >= np.max(uv_points @ uv)-max(sx, sy)))
    length = float(max(bounds.size))
    return SupportEvidence(np.asarray(directions), np.asarray(values),
                           np.asarray(tolerances), np.asarray(censored),
                           np.ones(len(values)), length)


def rounded_box_support(directions, center, half_sizes, radius):
    """Axis-aligned box Minkowski-summed with a sphere of world radius."""
    half_sizes = np.asarray(half_sizes, float)
    if half_sizes.shape != (3,) or not 0 < radius < min(half_sizes):
        raise ValueError("rounded radius must fit inside every positive half size")
    directions = np.asarray(directions, float)
    return (directions @ np.asarray(center) + np.abs(directions) @ (half_sizes-radius)
            + radius*np.linalg.norm(directions, axis=1))


def _fit_rounded_box(evidence, center, half_sizes, max_evaluations, max_elapsed_s):
    from scipy.optimize import least_squares
    start = time.perf_counter()
    calls, best, score, termination = 0, np.zeros(7), float("inf"), "solver_completed"

    def decode(x):
        sizes = half_sizes*np.exp(x[3:6])
        fraction = .95/(1+np.exp(-x[6]+np.log(8.5)))
        return center+x[:3]*evidence.length_scale, sizes, float(min(sizes)*fraction)

    class AllowanceEnded(Exception):
        pass

    def residual(x):
        nonlocal calls, best, score, termination
        if calls >= max_evaluations or (calls and time.perf_counter()-start >= max_elapsed_s):
            termination = "support_evaluation_allowance" if calls >= max_evaluations else "support_elapsed_allowance"
            raise AllowanceEnded
        calls += 1
        c, sizes, radius = decode(x)
        result = support_residual(evidence, rounded_box_support(evidence.directions, c, sizes, radius))
        current = float(result @ result)
        if current < score:
            best, score = x.copy(), current
        return result

    try:
        least_squares(residual, best, bounds=(-np.ones(7)*3, np.ones(7)*3),
                      x_scale="jac", max_nfev=max_evaluations,
                      ftol=1e-8, xtol=1e-8, gtol=1e-8)
    except AllowanceEnded:
        pass
    c, sizes, radius = decode(best)
    return {"center": c, "dimensions": sizes, "radius": radius,
            "rotation": np.eye(3), "support_squared_residual": score,
            "support_evaluations": calls, "support_elapsed_s": time.perf_counter()-start,
            "termination": termination, "family": "rounded_box"}


def validate_family_budget(max_evaluations, max_elapsed_s):
    """Reject ambiguous or unbounded public fit allowances before proposal work."""
    if (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, numbers.Integral)
            or not 1 <= max_evaluations <= 512
            or isinstance(max_elapsed_s, bool) or not isinstance(max_elapsed_s, numbers.Real)
            or not math.isfinite(max_elapsed_s) or not 0 < max_elapsed_s <= 10):
        raise ValueError("family fit exceeds the bounded evaluation/time contract")


def fitted_family_program(family, target, masks, cameras, *, coverage_masks=None,
                          max_evaluations=256, max_elapsed_s=3.):
    """Return one bounded editable hypothesis; no reference data is accepted."""
    from primitives.shape_program import ShapeNode, ShapeProgram
    if family not in SUPPORTED_FAMILIES:
        raise ValueError("family requires a structured proposal outside this runner")
    validate_family_budget(max_evaluations, max_elapsed_s)
    if family not in CONVEX_FAMILIES:
        from .structured_family import structured_family_program
        return structured_family_program(family, target, masks, cameras, coverage_masks=coverage_masks,
                                         max_evaluations=max_evaluations, max_elapsed_s=max_elapsed_s)
    if any(count >= 4 for count in observed_holes(target).values()):
        raise ValueError("convex family cannot preserve an observed enclosed hole")
    evidence = camera_support_evidence(masks, cameras, target.bounds,
                                      coverage_masks=coverage_masks)
    center = np.asarray(target.bounds.center, float)
    half_sizes = np.asarray(target.bounds.size, float)*.5
    primitive = {"anisotropic_ellipsoid": "ellipsoid", "tapered_frustum": "frustum",
                 "thin_plate": "box", "sphere": "ellipsoid"}.get(family, family)
    rotation = np.eye(3)
    if primitive in {"cylinder", "frustum", "capsule"}:
        axis = int(np.argmax(half_sizes))
        rotation = (np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]) if axis == 0
                    else np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]) if axis == 1
                    else np.eye(3))
        local = np.abs(rotation).T @ half_sizes
        radius = float(np.sqrt(local[0]*local[1]))
        dimensions = ([radius, radius*.65, local[2]] if primitive == "frustum"
                      else [radius, max(local[2]-radius, .01*local[2])] if primitive == "capsule"
                      else [radius, local[2]])
    else:
        dimensions = half_sizes
    if family == "rounded_box":
        fit = _fit_rounded_box(evidence, center, half_sizes, max_evaluations, max_elapsed_s)
    else:
        fit = fit_whole_support(primitive, evidence, center=center, dimensions=dimensions,
                               rotation=rotation, max_evaluations=max_evaluations,
                               max_elapsed_s=max_elapsed_s)
    dimensions = fit["dimensions"]
    if primitive in {"box", "ellipsoid", "rounded_box"}:
        sizes = dimensions*2
    elif primitive == "capsule":
        sizes = [2*dimensions[0], 2*dimensions[0], 2*(dimensions[0]+dimensions[1])]
    else:
        radius = max(dimensions[:2]) if primitive == "frustum" else dimensions[0]
        sizes = [2*radius, 2*radius, 2*dimensions[-1]]
    parameters = {**dict(zip(("x", "y", "z"), fit["center"].tolist())),
                  **dict(zip(("width_world", "depth_world", "height_world"), np.asarray(sizes).tolist())),
                  "rotation": fit["rotation"].tolist()}
    if primitive in {"cylinder", "frustum"}:
        parameters.update(radius_bottom=float(dimensions[0]),
                          radius_top=float(dimensions[1] if primitive == "frustum" else dimensions[0]))
    if primitive == "capsule":
        parameters.update(radius_world=float(dimensions[0]), segment_height_world=float(2*dimensions[1]))
    if family == "rounded_box":
        parameters["corner_radius_world"] = fit["radius"]
        parameters["bevel_segments"] = 8
        parameters["weighted_normals"] = True
        parameters["weighted_normals_keep_sharp"] = False
    metadata = {key: value for key, value in fit.items()
                if key not in {"center", "dimensions", "rotation"}}
    metadata.update(evidence_scope="observed five-view mask supports; full silhouette gates still required",
                    coverage_views=sorted(coverage_masks or {}), input_bounds=target.bounds.to_dict(),
                    support_censored_count=int(evidence.censored.sum()),
                    support_censored_by_view={view: int(evidence.censored[i*32:(i+1)*32].sum())
                                              for i, view in enumerate(sorted(masks))},
                    camera_sha256=hashlib.sha256(json.dumps(cameras, sort_keys=True).encode()).hexdigest())
    return ShapeProgram("1", "frozen-family-"+family,
                        (ShapeNode("observed_"+family, "add", primitive, parameters=parameters),),
                        metadata=metadata)


def initial_family_rows(selected, reused=()):
    """Preserve a complete workload map with distinct unsupported and unrun rows."""
    unknown = (set(selected) | set(reused)) - set(FROZEN_FAMILIES)
    if unknown or set(selected) & set(reused):
        raise ValueError("unknown or overlapping selected/reused families")
    return {name: {"status": "reused" if name in reused else "pending" if name in selected
                   else "unrun" if name in FAMILY_PROPOSAL_ROUTES or name in {"smooth_vase", "rounded_triangle_dot"}
                   else "unsupported", "proposal_runner": FAMILY_PROPOSAL_ROUTES.get(name), "aggregate_accepted": False}
            for name in FROZEN_FAMILIES}


def retained_family_program(wire):
    """Restore the saved one-part recipe without fitting or coordinate changes."""
    from primitives.shape_program import ShapeNode, ShapeProgram
    if wire.get("constraints") or wire.get("residual_patches") or len(wire.get("root_nodes", ())) != 1:
        raise ValueError("retained family replay requires one unconstrained primitive")
    node = wire["root_nodes"][0]
    if node.get("children") or node.get("operation") != "add":
        raise ValueError("retained family replay requires one additive leaf")
    restored = ShapeNode(node["node_id"], node["operation"], node["primitive_type"],
                         parameters=dict(node["parameters"]), name=node.get("name", ""),
                         editable=node.get("editable", True))
    return ShapeProgram(wire["schema_version"], wire["program_id"], (restored,),
                        metadata=dict(wire.get("metadata", {})))


def retained_triangle_indices(source, replayed):
    """Restore ordering only after exact vertices and oriented surfaces agree."""
    from blender_blocking.evaluation.reference_noise import oriented_surface_identity
    if not np.array_equal(source.vertices, replayed.vertices):
        raise ValueError("replayed world vertices differ from frozen candidate")
    if oriented_surface_identity(source) != oriented_surface_identity(replayed):
        raise ValueError("replayed oriented surface differs from frozen candidate")
    return source.faces.copy()
