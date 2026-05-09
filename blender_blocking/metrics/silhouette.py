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
    boundary_iou: Optional[float] = None
    soft_iou: Optional[float] = None
    signed_distance_loss: Optional[float] = None
    intersection: int = 0
    union: int = 0
    ref_area: int = 0
    render_area: int = 0
    required: bool = True
    passed: bool = True
    reason: str = ""
    warnings: Tuple[str, ...] = ()

    @property
    def pass_required(self) -> bool:
        return self.required

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
            "required": self.required,
            "passed": self.passed,
            "pass": self.passed,
            "reason": self.reason,
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


def area_iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> Tuple[float, int, int, int, int]:
    """Compute hard-mask area IoU plus raw area counts."""
    a = _as_bool(mask_a)
    b = _as_bool(mask_b)
    if a.shape != b.shape:
        raise ValueError("Masks must have matching shapes for area IoU")
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    area_a = int(a.sum())
    area_b = int(b.sum())
    value = float(intersection / union) if union else 0.0
    return value, intersection, union, area_a, area_b


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


def silhouette_metric_result(
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    view: str = "",
    required: bool = True,
    min_area_iou: float = 0.0,
    min_boundary_iou: Optional[float] = None,
    max_signed_distance_loss: Optional[float] = None,
    reference_probability: Optional[np.ndarray] = None,
    candidate_probability: Optional[np.ndarray] = None,
    reference_confidence: Optional[np.ndarray] = None,
    candidate_confidence: Optional[np.ndarray] = None,
    boundary_dilation_radius: int = 2,
) -> SilhouetteMetricResult:
    """Build the standard per-view silhouette metric contract."""
    area, intersection, union, ref_area, render_area = area_iou(
        reference_mask, candidate_mask
    )
    b_iou, warnings = boundary_iou(
        reference_mask,
        candidate_mask,
        dilation_radius=boundary_dilation_radius,
        confidence_a=reference_confidence,
        confidence_b=candidate_confidence,
    )
    s_iou = None
    if reference_probability is not None or candidate_probability is not None:
        ref_prob = (
            reference_probability
            if reference_probability is not None
            else _as_bool(reference_mask).astype(np.float32)
        )
        cand_prob = (
            candidate_probability
            if candidate_probability is not None
            else _as_bool(candidate_mask).astype(np.float32)
        )
        s_iou = soft_iou(ref_prob, cand_prob)

    sdf_loss = signed_distance_silhouette_loss(reference_mask, candidate_mask)

    reasons = []
    if area < float(min_area_iou):
        reasons.append(f"area_iou {area:.4f} below {float(min_area_iou):.4f}")
    if min_boundary_iou is not None and b_iou < float(min_boundary_iou):
        reasons.append(
            f"boundary_iou {b_iou:.4f} below {float(min_boundary_iou):.4f}"
        )
    if (
        max_signed_distance_loss is not None
        and sdf_loss > float(max_signed_distance_loss)
    ):
        reasons.append(
            "signed_distance_loss "
            f"{sdf_loss:.4f} above {float(max_signed_distance_loss):.4f}"
        )

    return SilhouetteMetricResult(
        view=view,
        area_iou=area,
        boundary_iou=b_iou,
        soft_iou=s_iou,
        signed_distance_loss=sdf_loss,
        intersection=intersection,
        union=union,
        ref_area=ref_area,
        render_area=render_area,
        required=bool(required),
        passed=not reasons,
        reason="; ".join(reasons),
        warnings=warnings,
    )
