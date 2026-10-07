"""Canonical orthographic image/world mapping, including explicit viewports."""
from __future__ import annotations

import numpy as np

AXES = {"front": (0, 2), "side": (1, 2), "top": (0, 1)}


def validate_view_calibration(records):
    for view, record in records.items():
        if view not in AXES:
            raise ValueError(f"unsupported calibrated view {view!r}")
        if record.get("projection", "orthographic") != "orthographic":
            raise ValueError(f"{view}: perspective evidence requires a perspective solver; orthographic fitting refused")

        if record.get("world_units", "metres") not in {"metres", "meters", "m"}:
            raise ValueError(f"{view}: calibrated world_units must be metres; mixed or unconverted units refused")
        if "axes" in record and tuple(record["axes"]) != AXES[view]:
            raise ValueError(f"{view}: unsupported view axes; canonical orientation required")
        if record.get("orientation", "canonical_positive_axes") != "canonical_positive_axes":
            raise ValueError(f"{view}: rotated/flipped camera is unsupported; explicit canonical orientation required")
        values = np.asarray(record.get("world_bounds", ()), dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all() or values[1] <= values[0] or values[3] <= values[2]:
            raise ValueError(f"{view}: world_bounds must contain finite increasing viewport intervals")


def viewport_for_constraint(target, constraint):
    """Return image axes and complete viewport; object bounds never mean canvas bounds."""
    axes = AXES[constraint.view]
    explicit = getattr(getattr(constraint, "camera", None), "bounds", None)
    if explicit is not None:
        return axes, (explicit.x0, explicit.x1, explicit.y0, explicit.y1)
    from reconstruction.point_cloud import target_bounds
    bounds = target_bounds(target)
    lo, hi = np.asarray(bounds.to_min_max(), dtype=float)
    u0, u1, v0, v1 = lo[axes[0]], hi[axes[0]], lo[axes[1]], hi[axes[1]]
    mask = np.asarray(getattr(constraint.mask, "mask", constraint.mask))
    height, width = mask.shape
    bbox = getattr(constraint, "bbox", None)
    if bbox is None:
        return axes, (u0, u1, v0, v1)
    sx = (u1-u0) / max(1., bbox.width-1.)
    sy = (v1-v0) / max(1., bbox.height-1.)
    xmin = u0 - bbox.x0*sx
    ymax = v1 + bbox.y0*sy
    return axes, (xmin, xmin+(width-1)*sx, ymax-(height-1)*sy, ymax)


def project_vertices(target, constraint, vertices, *, output_shape=None):
    axes, (xmin, xmax, ymin, ymax) = viewport_for_constraint(target, constraint)
    height, width = np.asarray(getattr(constraint.mask, "mask", constraint.mask)).shape
    xy = np.asarray(vertices, dtype=float)[:, axes].copy()
    if getattr(getattr(constraint, "camera", None), "bounds", None) is not None:
        xy[:, 0] = (xy[:, 0]-xmin)/(xmax-xmin)*width - 0.5
        xy[:, 1] = (ymax-xy[:, 1])/(ymax-ymin)*height - 0.5
    else:
        xy[:, 0] = (xy[:, 0]-xmin)/(xmax-xmin)*(width-1)
        xy[:, 1] = (ymax-xy[:, 1])/(ymax-ymin)*(height-1)
    if output_shape is not None:
        out_height, out_width = output_shape
        if out_height <= 0 or out_width <= 0:
            raise ValueError("output_shape must contain positive height and width")
        # Resize in pixel-center coordinates, including the legacy viewport path.
        xy = (xy + 0.5) * np.array([out_width / width, out_height / height]) - 0.5
    return xy


def pixel_cell_viewport(target,constraint):
    """Complete cell-edge viewport, including legacy endpoint-center cameras."""
    axes,(u0,u1,v0,v1)=viewport_for_constraint(target,constraint)
    if getattr(getattr(constraint,'camera',None),'bounds',None) is None:
        height,width=np.asarray(getattr(constraint.mask,'mask',constraint.mask)).shape
        half_x=(u1-u0)/max(1,width-1)*.5
        half_y=(v1-v0)/max(1,height-1)*.5
        return axes,(u0-half_x,u1+half_x,v0-half_y,v1+half_y)
    return axes,(u0,u1,v0,v1)


def bounds_from_calibrated_masks(masks, bboxes, records):
    """Infer only observed axis extents; camera bounds contain no hidden mesh dimensions."""
    from reconstruction.types import Bounds3D
    validate_view_calibration(records)
    observed = {axis: [] for axis in range(3)}
    for view, mask in masks.items():
        if view not in records or view not in bboxes:
            raise ValueError(f"{view}: calibration and a nonempty mask are required for every supplied view")
        height, width = mask.shape
        xmin,xmax,ymin,ymax = records[view]["world_bounds"]
        box = bboxes[view]
        sx, sy = (xmax-xmin)/max(1,width), (ymax-ymin)/max(1,height)
        axes = AXES[view]
        observed[axes[0]].append((xmin+box.x0*sx, xmin+box.x1*sx, view, sx))
        observed[axes[1]].append((ymax-box.y1*sy, ymax-box.y0*sy, view, sy))
    lows, highs = [], []
    for axis, values in observed.items():
        if not values:
            raise ValueError(f"axis {axis}: unobserved metric extent; provide another calibrated view")
        low, high = max(v[0] for v in values), min(v[1] for v in values)
        spans = [v[1]-v[0] for v in values]
        tolerance = max(max(v[3] for v in values)*3, max(spans)*.05)
        if high <= low or max(v[0] for v in values)-min(v[0] for v in values) > tolerance or max(v[1] for v in values)-min(v[1] for v in values) > tolerance:
            raise ValueError(f"axis {axis}: inconsistent calibrated silhouette extents across {[v[2] for v in values]}")
        lows.append(low)
        highs.append(high)
    return Bounds3D.from_min_max(lows, highs)


def observed_holes(target):
    from scipy.ndimage import binary_fill_holes
    from scipy.ndimage import label
    from .visibility import valid_evidence
    result = {}
    for c in target.constraints:
        mask = np.asarray(getattr(c.mask, "mask", c.mask), bool)
        valid = valid_evidence(c)
        foreground = mask & valid
        holes = binary_fill_holes(foreground) & ~foreground
        labels, count = label(holes)
        # An enclosed region containing unknown pixels is not a known hole.
        result[c.view] = sum(int(np.count_nonzero(labels == i)) for i in range(1, count+1)
                             if valid[labels == i].all())
    return result


def permits_bounds_seed(target):
    holes = observed_holes(target)
    meaningful = {view: count for view, count in holes.items() if count >= 4}
    return not meaningful, {"observed_hole_pixels": meaningful,
                            "reason": "observed_negative_space" if meaningful else "no_observed_holes"}


def calibrated_profile(request):
    """Build a labeled profile hypothesis from known row edges and bounds."""
    from geometry.profile_models import EllipticalProfileU
    from .profile_evidence import observed_profile_rows, complete_profile_hypothesis
    bounds = request.target.bounds
    count = max(2, int(request.config.get("num_samples", 100)))
    t = np.linspace(0., 1., count)
    zs = bounds.min_z + t*(bounds.max_z-bounds.min_z)
    radii, centers, evidence = {}, {}, {}
    for c in request.target.constraints:
        if c.view not in ("front", "side"):
            continue
        rows = observed_profile_rows(request.target, c, zs)
        center = bounds.center[AXES[c.view][0]]
        radii[c.view], positions = complete_profile_hypothesis(rows, default_center=center)
        centers[c.view] = positions - center
        evidence[c.view] = rows
    if set(radii) != {"front", "side"}:
        raise ValueError("calibrated profile loft requires front and side; circular fallback would invent unobserved geometry")
    return EllipticalProfileU(heights_t=t, rx=radii["front"], ry=radii["side"],
        world_height=bounds.max_z-bounds.min_z, z0=bounds.min_z,
        cx=centers["front"], cy=centers["side"], meta={"source": "known_calibrated_mask_rows",
        "row_evidence": evidence, "censored_completion": "interpolated_exact_sections_with_known_foreground_lower_bounds"})


def visible_search_bounds(masks, bboxes, records, valid_masks):
    """Use full observations for tight extents; partial viewports bound search only."""
    from reconstruction.types import Bounds3D
    validate_view_calibration(records)
    observed = {axis: [] for axis in range(3)}
    domains = {axis: [] for axis in range(3)}
    for view, mask in masks.items():
        h, w = mask.shape
        u0, u1, v0, v1 = records[view]['world_bounds']
        a, b = AXES[view]
        domains[a].append((u0, u1)); domains[b].append((v0, v1))
        valid = np.asarray(valid_masks.get(view, np.ones(mask.shape, bool)), bool)
        if valid.shape != mask.shape: raise ValueError('visibility mask dimensions differ')
        if valid.all() and view in bboxes:
            box = bboxes[view]
            observed[a].append((u0+box.x0/w*(u1-u0), u0+box.x1/w*(u1-u0)))
            observed[b].append((v1-box.y1/h*(v1-v0), v1-box.y0/h*(v1-v0)))
    lows, highs = [], []
    for axis in range(3):
        values = observed[axis] or domains[axis]
        if not values: raise ValueError(f'axis {axis}: no calibrated search domain')
        low, high = max(p[0] for p in values), min(p[1] for p in values)
        if high <= low: raise ValueError('calibrated observed domains do not overlap')
        lows.append(low); highs.append(high)
    return Bounds3D.from_min_max(lows, highs)
