"""Foreground/background assertions restricted to observed camera pixels."""
from __future__ import annotations
import numpy as np


def valid_evidence(constraint):
    mask = np.asarray(getattr(constraint.mask, "mask", constraint.mask))
    supplied = getattr(constraint, "valid_mask", None)
    if supplied is None:
        supplied = getattr(constraint, "diagnostics", {}).get("valid_mask")
    if supplied is None:
        return np.ones(mask.shape, dtype=bool)
    valid = np.asarray(supplied, bool)
    if valid.shape != mask.shape:
        raise ValueError("observed-pixel mask must match its silhouette")
    return valid


def evaluate_visible_pair(reference, prediction, constraint, **kwargs):
    from blender_blocking.evaluation.silhouette_eval import evaluate_silhouette_pair
    reference, prediction = np.asarray(reference, bool), np.asarray(prediction, bool)
    valid = valid_evidence(constraint)
    if reference.shape != valid.shape or prediction.shape != valid.shape:
        raise ValueError("candidate/reference visibility shapes differ")
    kwargs.setdefault("view", constraint.view)
    result = evaluate_silhouette_pair(reference & valid, prediction & valid, **kwargs)
    result.update(valid_pixel_count=int(valid.sum()), evidence_coverage=float(valid.mean()))
    if not valid.any():
        result.update(passed=True, required=False, reason="no observed pixels; excluded from objective",
            area_iou=None, boundary_iou=None, signed_distance_loss=None, evidence_weight=0.)
        result["pass"] = True
    elif not valid.all():
        from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
        interior = binary_erosion(valid, border_value=1)
        rb = (reference ^ binary_erosion(reference)) & interior
        pb = (prediction ^ binary_erosion(prediction)) & interior
        rr, pr = binary_dilation(rb, iterations=2) & valid, binary_dilation(pb, iterations=2) & valid
        union = np.count_nonzero(rr | pr)
        result["boundary_iou"] = float(np.count_nonzero(rr & pr) / union) if union else 1.
        d = distance_transform_edt(~rb) / max(reference.shape) if rb.any() else np.ones(reference.shape)
        result["signed_distance_loss"] = float(np.mean(d[valid] * (reference[valid] != prediction[valid])))
        cfg = kwargs.get("config")
        area = getattr(cfg, "min_area_iou", 0)
        boundary = getattr(cfg, "min_boundary_iou", None)
        distance = getattr(cfg, "max_signed_distance_loss", None)
        passed = result["area_iou"] >= area and (boundary is None or result["boundary_iou"] >= boundary) and (distance is None or result["signed_distance_loss"] <= distance)
        result.update(passed=bool(passed), reason="" if passed else "observed-pixel geometric gate failed")
        result["pass"] = bool(passed)
    return result


def point_support(target, points):
    """Reject known background rays; outside/unknown pixels make no assertion."""
    from .projection_contract import project_vertices
    points = np.asarray(points, float)
    allowed = np.ones(len(points), bool)
    observed = np.zeros(len(points), np.int32)
    for constraint in target.constraints:
        mask = np.asarray(getattr(constraint.mask, "mask", constraint.mask), bool)
        xy = np.floor(project_vertices(target, constraint, points) + .5).astype(int)
        inside = (xy[:,0] >= 0) & (xy[:,0] < mask.shape[1]) & (xy[:,1] >= 0) & (xy[:,1] < mask.shape[0])
        indices = np.flatnonzero(inside)
        if not len(indices):
            continue
        u, v = xy[indices,0], xy[indices,1]
        known = valid_evidence(constraint)[v,u]
        observed[indices] += known
        allowed[indices] &= ~known | mask[v,u]
    return allowed, observed


def prepare_evidence_inputs(views, supplied, calibration, crops=None, files=None):
    """Apply the same pixel crop to images, validity and orthographic viewport."""
    import hashlib
    from PIL import Image
    from copy import deepcopy
    result, valid, records = {}, {}, {}
    calibration = deepcopy(calibration)
    supplied = dict(supplied or {})
    for view, path in (files or {}).items():
        if view in supplied:
            raise ValueError(f"{view}: supply one validity source")
        supplied[view] = np.asarray(Image.open(path).convert('L')) > 0
    if set(supplied) - set(views) or set(crops or {}) - set(views):
        raise ValueError("validity/crop supplied for an absent view")
    for view, image in views.items():
        height, width = image.shape[:2]
        mask = np.asarray(supplied.get(view, np.ones((height, width), bool)), bool)
        if mask.shape != (height, width):
            raise ValueError(f"{view}: validity must match original image dimensions; implicit resizing refused")
        crop = tuple((crops or {}).get(view, (0, 0, width, height)))
        if len(crop) != 4 or any(int(x) != x for x in crop):
            raise ValueError("view crop must contain four integer pixel edges")
        x0, y0, x1, y1 = map(int, crop)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("view crop is outside the original image")
        if crop != (0, 0, width, height):
            if view not in calibration:
                raise ValueError("cropped evidence requires its original camera calibration")
            u0, u1, v0, v1 = calibration[view]['world_bounds']
            calibration[view]['world_bounds'] = [u0+x0/width*(u1-u0), u0+x1/width*(u1-u0),
                v1-y1/height*(v1-v0), v1-y0/height*(v1-v0)]
        result[view] = image[y0:y1, x0:x1].copy()
        if view in supplied:
            valid[view] = mask[y0:y1, x0:x1].copy()
        records[view] = {'original_shape': [height, width], 'crop_pixel_edges': list(crop),
            'validity_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
            'camera': calibration.get(view), 'supplied_validity': view in supplied}
    return result, valid, calibration, records


def observed_area_score(rows):
    values = [float(r['area_iou']) for r in rows if r.get('area_iou') is not None]
    return min(values) if values else 0.
