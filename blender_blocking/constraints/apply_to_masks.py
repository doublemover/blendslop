"""Apply human correction constraints to silhouette masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .model import (
    ConstraintSatisfaction,
    ConstraintSet,
    ScribbleConstraint,
    satisfaction_for_constraint,
    summarize_satisfaction,
)


@dataclass(frozen=True)
class MaskConstraintResult:
    mask: np.ndarray
    confidence: np.ndarray
    satisfaction: Tuple[ConstraintSatisfaction, ...]
    report: Dict[str, Any]


def scribble_pixel_mask(shape: Tuple[int, int], scribble: ScribbleConstraint) -> np.ndarray:
    """Rasterize a scribble brush into a boolean HxW mask."""
    if len(shape) != 2:
        raise ValueError("shape must be (height, width)")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("shape values must be positive")

    result = np.zeros((height, width), dtype=bool)
    radius = float(scribble.brush_radius_px)
    radius_sq = radius * radius
    for x_float, y_float in scribble.points_px:
        x_center = float(x_float)
        y_center = float(y_float)
        x0 = max(0, int(np.floor(x_center - radius)))
        x1 = min(width - 1, int(np.ceil(x_center + radius)))
        y0 = max(0, int(np.floor(y_center - radius)))
        y1 = min(height - 1, int(np.ceil(y_center + radius)))
        if x1 < x0 or y1 < y0:
            continue
        yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        if radius <= 0:
            disk = (int(round(x_center)) == xx) & (int(round(y_center)) == yy)
        else:
            disk = (xx - x_center) ** 2 + (yy - y_center) ** 2 <= radius_sq
        result[y0 : y1 + 1, x0 : x1 + 1] |= disk
    return result


def apply_constraints_to_mask(
    mask: np.ndarray,
    constraints: ConstraintSet,
    view: str,
    *,
    confidence: Optional[np.ndarray] = None,
    copy: bool = True,
    unknown_confidence: float = 0.25,
) -> MaskConstraintResult:
    """Apply scribble constraints to a single mask.

    Foreground and background scribbles force mask pixels. Unknown scribbles
    lower confidence without changing the hard foreground/background decision.
    """
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError("mask must be a 2D array")
    result_mask = mask_array.astype(bool, copy=copy)

    if confidence is None:
        result_confidence = np.ones(result_mask.shape, dtype=np.float32)
    else:
        result_confidence = np.asarray(confidence, dtype=np.float32).copy()
        if result_confidence.shape != result_mask.shape:
            raise ValueError("confidence shape must match mask shape")

    unknown_confidence = float(np.clip(unknown_confidence, 0.0, 1.0))

    scribbles = constraints.scribbles_for_view(view)
    forced_value = np.full(result_mask.shape, -1, dtype=np.int8)
    forced_confidence = np.full(result_mask.shape, -np.inf, dtype=np.float32)
    forced_conflict = {id(scribble): False for scribble in scribbles}

    for scribble in scribbles:
        pixels = scribble_pixel_mask(result_mask.shape, scribble)
        if not pixels.any():
            continue
        if scribble.kind == "unknown":
            result_confidence[pixels] = np.minimum(
                result_confidence[pixels], np.float32(unknown_confidence)
            )
            continue

        desired = 1 if scribble.kind == "foreground" else 0
        conflict = pixels & (forced_value >= 0) & (forced_value != desired)
        stronger = pixels & (scribble.confidence >= forced_confidence)
        weaker_conflict = conflict & ~stronger
        tied_conflict = conflict & (scribble.confidence == forced_confidence)
        if weaker_conflict.any() or tied_conflict.any():
            forced_conflict[id(scribble)] = True

        update = pixels & stronger
        forced_value[update] = desired
        forced_confidence[update] = np.float32(scribble.confidence)
        result_confidence[update] = 1.0

    result_mask[forced_value == 1] = True
    result_mask[forced_value == 0] = False

    satisfaction = []
    for scribble in scribbles:
        pixels = scribble_pixel_mask(result_mask.shape, scribble)
        painted = int(pixels.sum())
        if scribble.kind == "unknown":
            changed = painted > 0 and float(result_confidence[pixels].max()) <= unknown_confidence
            satisfaction.append(
                satisfaction_for_constraint(
                    scribble,
                    satisfied=changed,
                    score=1.0 if changed else 0.0,
                    message="unknown scribble lowered mask confidence",
                    details={"view": view, "pixels": painted},
                )
            )
            continue

        desired_bool = scribble.kind == "foreground"
        if painted == 0:
            ok = False
            score = 0.0
        else:
            ok_pixels = result_mask[pixels] == desired_bool
            score = float(np.count_nonzero(ok_pixels)) / float(painted)
            ok = score == 1.0 and not forced_conflict[id(scribble)]
        satisfaction.append(
            satisfaction_for_constraint(
                scribble,
                satisfied=ok,
                score=score,
                message=f"{scribble.kind} scribble applied to mask",
                details={"view": view, "pixels": painted},
            )
        )

    report = summarize_satisfaction(satisfaction)
    report["view"] = view
    report["forced_foreground_pixels"] = int(np.count_nonzero(forced_value == 1))
    report["forced_background_pixels"] = int(np.count_nonzero(forced_value == 0))
    return MaskConstraintResult(
        mask=result_mask,
        confidence=result_confidence,
        satisfaction=tuple(satisfaction),
        report=report,
    )


def apply_constraints_to_masks(
    masks: Mapping[str, np.ndarray],
    constraints: ConstraintSet,
    *,
    confidences: Optional[Mapping[str, np.ndarray]] = None,
    unknown_confidence: float = 0.25,
) -> Dict[str, MaskConstraintResult]:
    """Apply view-scoped mask constraints to a mapping of view name to mask."""
    results: Dict[str, MaskConstraintResult] = {}
    for view, mask in masks.items():
        confidence = None if confidences is None else confidences.get(view)
        results[view] = apply_constraints_to_mask(
            mask,
            constraints,
            view,
            confidence=confidence,
            unknown_confidence=unknown_confidence,
        )
    return results


def mask_extraction_hints(constraints: ConstraintSet, view: str) -> Dict[str, Any]:
    """Return extraction options that can be merged into a silhouette extractor config."""
    hints: Dict[str, Any] = {}
    polarity = [item for item in constraints.polarities if item.view == view]
    if polarity:
        selected = max(polarity, key=lambda item: item.confidence)
        hints["polarity"] = selected.polarity
        if selected.polarity == "foreground_dark":
            hints["polarity"] = "dark_foreground"
            hints["invert_policy"] = "invert"
        elif selected.polarity == "foreground_light":
            hints["polarity"] = "light_foreground"
            hints["invert_policy"] = "no_invert"
    preserve_holes = any(
        item.kind == "preserve_holes" and (item.view is None or item.view == view)
        for item in constraints.components
    )
    preserve_components = any(
        item.kind == "preserve_components" and (item.view is None or item.view == view)
        for item in constraints.components
    )
    if preserve_holes:
        hints["fill_holes"] = False
    if preserve_components:
        hints["largest_component_only"] = False
    bbox = constraints.bbox_for_view(view)
    if bbox is not None:
        hints["bbox_px"] = bbox.bbox_px
        hints["bbox_tolerance_px"] = bbox.tolerance_px
    return hints
