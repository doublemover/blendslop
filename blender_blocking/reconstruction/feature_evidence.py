"""Required known-empty features, separate from whole-image silhouette scores."""
from __future__ import annotations
import numpy as np


def known_hole_regions(constraint, *, minimum_pixels=4):
    from scipy.ndimage import binary_fill_holes, label
    from .visibility import valid_evidence
    mask = np.asarray(getattr(constraint.mask, "mask", constraint.mask), bool)
    valid = valid_evidence(constraint)
    foreground = mask & valid
    labels, count = label(binary_fill_holes(foreground) & ~foreground)
    return [labels == i for i in range(1, count+1)
            if np.count_nonzero(labels == i) >= minimum_pixels and valid[labels == i].all()]


def known_empty_feature_guard(constraint, prediction, *, maximum_filled_fraction=.1):
    from scipy.ndimage import binary_erosion
    prediction = np.asarray(prediction, bool)
    rows = []
    for region in known_hole_regions(constraint):
        core = binary_erosion(region)
        # A one-pixel boundary allowance must not erase an entire thin hole.
        core = core if core.any() else region
        filled = float(np.count_nonzero(prediction & core) / np.count_nonzero(core))
        rows.append({"known_hole_pixels": int(region.sum()), "checked_core_pixels": int(core.sum()),
                     "filled_fraction": filled, "passed": filled <= maximum_filled_fraction})
    return {"passed": all(row["passed"] for row in rows), "regions": rows,
            "maximum_filled_fraction": maximum_filled_fraction,
            "scope": "known enclosed empty evidence only; unknown regions omitted"}


def mesh_empty_features(target, data):
    from .projected_metrics import projected_mesh_masks
    masks = projected_mesh_masks(target, data.vertices, data.faces)
    rows = {c.view: known_empty_feature_guard(c, masks[c.view]) for c in target.constraints}
    return {"passed": all(row["passed"] for row in rows.values()), "per_view": rows}


def hole_capable_request(request):
    """Eligibility follows effective representation options, not backend labels."""
    from .quality_config import quality_config
    config = quality_config(request.config)
    return request.backend_name != "profile_loft" or bool(config.get("contour_sections", False))
