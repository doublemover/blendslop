"""Profile observations with unknown edges explicitly represented as censored."""
from __future__ import annotations
import numpy as np


def measure_profile_row(mask, valid, coverage=None):
    mask, valid = np.asarray(mask, bool), np.asarray(valid, bool)
    if mask.ndim != 1 or mask.shape != valid.shape:
        raise ValueError("profile row and validity must be matching 1D arrays")
    if coverage is not None:
        from .coverage_evidence import coverage_interval
        left_edge, right_edge = coverage_interval(coverage, valid)
        record = measure_profile_row(np.asarray(coverage) >= .5, valid)
        record.update(left_edge_px=left_edge, right_edge_px=right_edge,
                      exact_width_px=(0. if not record["foreground"] and valid.all()
                                      else None if left_edge is None or right_edge is None
                                      else right_edge - left_edge),
                      center_px=None if left_edge is None or right_edge is None else .5 * (left_edge + right_edge),
                      width_lower_px=float(np.count_nonzero(
                          (np.asarray(coverage) >= 1 - 1e-6) & valid)),
                      edge_model="declared_linear_half_coverage_crossing")
        return record
    foreground = np.flatnonzero(mask & valid)
    if not len(foreground):
        return {"observed": bool(valid.any()), "foreground": False,
                "width_lower_px": 0., "left_edge_px": None, "right_edge_px": None,
                "exact_width_px": 0. if valid.all() else None, "center_px": None}
    left, right = int(foreground[0]), int(foreground[-1]) + 1
    left_exact = left > 0 and valid[left - 1] and not mask[left - 1]
    right_exact = right < len(mask) and valid[right] and not mask[right]
    return {"observed": True, "foreground": True,
            "width_lower_px": float(right - left),
            "left_edge_px": float(left) if left_exact else None,
            "right_edge_px": float(right) if right_exact else None,
            "exact_width_px": float(right-left) if left_exact and right_exact else None,
            "center_px": .5 * (left + right)}


def observed_profile_rows(target, constraint, world_heights):
    from .projection_contract import pixel_cell_viewport
    from .visibility import valid_evidence
    mask = np.asarray(getattr(constraint.mask, "mask", constraint.mask), bool)
    valid = valid_evidence(constraint)
    from .coverage_evidence import constraint_coverage
    coverage = constraint_coverage(constraint, valid)
    h, w = mask.shape
    _, (xmin, xmax, ymin, ymax) = pixel_cell_viewport(target, constraint)
    sx = (xmax - xmin) / w
    records = []
    foreground_rows = np.flatnonzero(np.any((mask if coverage is None else coverage >= .5) & valid, axis=1))
    for z in world_heights:
        pixel = (ymax - z) / (ymax - ymin) * h - .5
        row = int(np.floor(pixel + .5))
        # A mask bbox denotes cell edges. At its outer vertical edge, sampling
        # the adjacent empty cell invents a zero-radius tip. Use the nearest
        # observed foreground row only within that outer half-cell; interior
        # empty rows and unknown pixels remain evidence, including cavities.
        if len(foreground_rows):
            first, last = int(foreground_rows[0]), int(foreground_rows[-1])
            epsilon = 8 * np.finfo(float).eps * max(1., abs(pixel))
            if first - .5 - epsilon <= pixel < first:
                row = first
            elif last < pixel <= last + .5 + epsilon:
                row = last
        if not 0 <= row < h:
            record = measure_profile_row(np.zeros(w, bool), np.zeros(w, bool))
        else:
            record = measure_profile_row(mask[row], valid[row], None if coverage is None else coverage[row])
        # The outer partially covered row mixes vertical cap coverage into
        # its horizontal edge signal. Its half-crossing remains recorded, but
        # is not an exact section. The flat-cap loft uses the nearest reliable
        # interior section as a labeled proposal prior. Unknown rows are never
        # normalized or promoted to exact observations.
        if (coverage is not None and len(foreground_rows)
                and row in (foreground_rows[0], foreground_rows[-1])
                and 0 <= row < h and valid[row].all()
                and np.max(coverage[row]) < 1 - 1 / 255
                and record["foreground"]):
            record["filtered_width_px"] = record["exact_width_px"]
            record["exact_width_px"] = None
            record["completion_reason"] = "partial_terminal_coverage_flat_cap_prior"
        record.update(sample_row=row, z_world=float(z), radius_lower_world=record["width_lower_px"] * sx * .5,
                      within_viewport=bool(0<=row<h),
                      exact_radius_world=None if record["exact_width_px"] is None else record["exact_width_px"] * sx * .5,
                      center_world=None if record["center_px"] is None else xmin + record["center_px"] * sx)
        records.append(record)
    return records


def complete_profile_hypothesis(records, *, default_center):
    """Interpolate from exact rows; retain lower bounds and label completion.

    This is a proposal prior, never a conversion of unknown pixels into observed
    dimensions. No reliable foreground cross-section means no supported seed.
    """
    heights = np.asarray([r["z_world"] for r in records])
    measured = [r for r in records if r["foreground"] and r["exact_radius_world"] is not None]
    if not measured:
        raise ValueError("profile has no fully measured foreground section; censored-only loft seed unavailable")
    source_z = np.asarray([r["z_world"] for r in measured])
    radii = np.interp(heights, source_z, [r["exact_radius_world"] for r in measured])
    centers = np.interp(heights, source_z, [r["center_world"] for r in measured])
    for i, row in enumerate(records):
        if row["exact_radius_world"] is not None:
            radii[i] = row["exact_radius_world"]
            if row["center_world"] is not None:
                centers[i] = row["center_world"]
            elif radii[i] == 0.:
                centers[i] = default_center
        else:
            radii[i] = max(radii[i], row["radius_lower_world"])
    return radii, centers
