"""Silhouette comparison metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class SilhouetteMetricResult:
    """Per-view silhouette metric result."""

    view: str
    area_iou: float
    boundary_iou: Optional[float]
    soft_iou: Optional[float]
    signed_distance_loss: Optional[float]
    intersection: int
    union: int
    ref_area: int
    render_area: int
    pass_required: bool
    warnings: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "view": self.view,
            "area_iou": self.area_iou,
            "boundary_iou": self.boundary_iou,
            "soft_iou": self.soft_iou,
            "signed_distance_loss": self.signed_distance_loss,
            "intersection": self.intersection,
            "union": self.union,
            "ref_area": self.ref_area,
            "render_area": self.render_area,
            "pass_required": self.pass_required,
            "warnings": list(self.warnings),
        }


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool, copy=False)


def _boundary_band(mask: np.ndarray, dilation_radius: int) -> np.ndarray:
    mask_bool = _as_bool(mask)
    if not mask_bool.any():
        return np.zeros(mask_bool.shape, dtype=bool)
    radius = max(1, int(dilation_radius))
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask_u8 = mask_bool.astype(np.uint8)
    eroded = cv2.erode(mask_u8, kernel)
    boundary = np.logical_xor(mask_bool, eroded.astype(bool))
    dilated = cv2.dilate(boundary.astype(np.uint8), kernel)
    return dilated.astype(bool)


def boundary_iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    dilation_radius: int = 2,
    confidence_a: Optional[np.ndarray] = None,
    confidence_b: Optional[np.ndarray] = None,
) -> Tuple[float, Tuple[str, ...]]:
    """Compute Boundary IoU on dilated mask boundary bands."""
    a = _as_bool(mask_a)
    b = _as_bool(mask_b)
    if a.shape != b.shape:
        raise ValueError("Masks must have matching shapes for Boundary IoU")

    warnings = []
    if not a.any() and not b.any():
        warnings.append("both_boundaries_empty")
        return 0.0, tuple(warnings)
    if not a.any() or not b.any():
        warnings.append("one_boundary_empty")

    band_a = _boundary_band(a, dilation_radius)
    band_b = _boundary_band(b, dilation_radius)
    if confidence_a is not None or confidence_b is not None:
        conf_a = (
            np.asarray(confidence_a, dtype=np.float32)
            if confidence_a is not None
            else np.ones(a.shape, dtype=np.float32)
        )
        conf_b = (
            np.asarray(confidence_b, dtype=np.float32)
            if confidence_b is not None
            else np.ones(b.shape, dtype=np.float32)
        )
        if conf_a.shape != a.shape or conf_b.shape != b.shape:
            raise ValueError("Confidence maps must match mask shape")
        weights = np.clip((conf_a + conf_b) * 0.5, 0.0, 1.0)
        intersection = float((np.logical_and(band_a, band_b) * weights).sum())
        union = float((np.logical_or(band_a, band_b) * weights).sum())
    else:
        intersection = float(np.logical_and(band_a, band_b).sum())
        union = float(np.logical_or(band_a, band_b).sum())

    if union <= 0.0:
        warnings.append("empty_boundary_union")
        return 0.0, tuple(warnings)
    return float(intersection / union), tuple(warnings)


def soft_iou(
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Expected IoU for probability masks in [0, 1]."""
    a = np.clip(np.asarray(prob_a, dtype=np.float32), 0.0, 1.0)
    b = np.clip(np.asarray(prob_b, dtype=np.float32), 0.0, 1.0)
    if a.shape != b.shape:
        raise ValueError("Probability masks must have matching shapes")
    intersection = np.minimum(a, b).sum(dtype=np.float64)
    union = np.maximum(a, b).sum(dtype=np.float64)
    if union <= eps:
        return 0.0
    return float(intersection / union)


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    mask_bool = _as_bool(mask)
    inside = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((~mask_bool).astype(np.uint8), cv2.DIST_L2, 3)
    return inside.astype(np.float32) - outside.astype(np.float32)


def signed_distance_silhouette_loss(
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    normalize: bool = True,
) -> float:
    """Mean absolute signed-distance disagreement between silhouettes."""
    ref = _as_bool(reference_mask)
    cand = _as_bool(candidate_mask)
    if ref.shape != cand.shape:
        raise ValueError("Masks must have matching shapes for signed-distance loss")
    ref_sdf = _signed_distance(ref)
    cand_sdf = _signed_distance(cand)
    loss = float(np.mean(np.abs(ref_sdf - cand_sdf)))
    if normalize:
        denom = float(max(ref.shape)) if ref.shape else 1.0
        loss /= max(denom, 1.0)
    return loss
