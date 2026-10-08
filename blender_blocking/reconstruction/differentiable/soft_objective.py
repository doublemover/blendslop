"""Smooth silhouette proposal objective; hard metrics remain acceptance scores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

try:
    from primitives.soft_silhouette import _signed_distance
except ImportError:  # pragma: no cover - package import path
    from ...primitives.soft_silhouette import _signed_distance


@dataclass(frozen=True)
class SoftMaskTarget:
    mask: np.ndarray
    binary: np.ndarray
    distance: np.ndarray
    valid: np.ndarray
    pixel_weights: np.ndarray | None = None

    @classmethod
    def from_mask(cls, mask: np.ndarray, valid=None,pixel_weights=None) -> 'SoftMaskTarget':
        values = np.asarray(mask, dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all() or values.size == 0:
            raise ValueError('target silhouette must be a nonempty finite 2D array')
        values = np.clip(values, 0.0, 1.0).copy()
        binary = (values >= 0.5).astype(np.float64)
        distance = np.abs(_signed_distance(binary.astype(bool))) / max(values.shape)
        valid = np.ones(values.shape,bool) if valid is None else np.asarray(valid,bool)
        if valid.shape != values.shape:
            raise ValueError("soft target visibility shape differs")
        if not valid.all():
            from scipy.ndimage import binary_erosion, distance_transform_edt
            boundary = (binary.astype(bool) ^ binary_erosion(binary.astype(bool))) & binary_erosion(valid,border_value=1)
            distance = distance_transform_edt(~boundary) / max(values.shape) if boundary.any() else np.zeros(values.shape)
        if pixel_weights is not None:
            pixel_weights=np.asarray(pixel_weights,float)
            if pixel_weights.shape!=values.shape or not np.isfinite(pixel_weights[valid]).all() or np.any(pixel_weights[valid]<0.):
                raise ValueError('soft target pixel reliability must match observed pixels and be finite nonnegative')
            pixel_weights=np.where(valid,pixel_weights,0.)
        return cls(values, binary, distance, valid.copy(),pixel_weights)


def soft_mask_loss_and_gradient(
    prediction: np.ndarray,
    target: SoftMaskTarget,
    *,
    l2_weight: float = 1.0,
    iou_weight: float = 1.0,
    distance_weight: float = 0.25,
    epsilon: float = 1e-8,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Mask L2, product soft IoU, and fixed target-distance disagreement.

    The distance target is fixed for the whole fit. No gradient is invented for
    adaptive hard thresholds, boundary IoU, or candidate signed-distance metrics.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    if pred.shape != target.mask.shape or not np.isfinite(pred).all():
        raise ValueError('prediction must be finite and match the target')
    weights = np.asarray((l2_weight, iou_weight, distance_weight), dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError('soft objective weights must be finite and nonnegative')
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError('soft IoU epsilon must be finite and positive')
    valid = target.valid if target.pixel_weights is None else target.pixel_weights*target.valid
    pixels = float(np.sum(valid))
    if pixels == 0:
        return 0., np.zeros_like(pred), {"mask_l2":0.,"product_soft_iou_loss":0.,"target_distance_disagreement":0.}
    difference = pred - target.mask
    l2 = float(np.sum(valid * difference * difference) / pixels)
    intersection = float(np.sum(valid * pred * target.mask))
    union = float(np.sum(valid * pred) + np.sum(valid * target.mask) - intersection + epsilon)
    iou = 1.0 - intersection / union
    disagreement = pred * (1.0 - target.binary) + (1.0 - pred) * target.binary
    distance = float(np.sum(valid * target.distance * disagreement) / pixels)
    gradient = (l2_weight * 2.0 * difference / pixels
        - iou_weight * (target.mask * union - intersection * (1.0 - target.mask)) / union**2
        + distance_weight * target.distance * (1.0 - 2.0 * target.binary) / pixels)
    gradient *= valid
    terms = {'mask_l2': l2, 'product_soft_iou_loss': iou,
             'target_distance_disagreement': distance}
    return float(l2_weight * l2 + iou_weight * iou + distance_weight * distance), gradient, terms


class SoftSilhouetteObjective:
    """Fixed targets and normalized configured view weights for gradient proposals."""

    def __init__(self, masks: Mapping[str, np.ndarray], weights: object,
                 view_weights: Mapping[str, float] | None = None, valid_masks=None,pixel_weights=None) -> None:
        self.targets = {name: SoftMaskTarget.from_mask(mask, (valid_masks or {}).get(name),(pixel_weights or {}).get(name)) for name, mask in masks.items()
                        if np.asarray((valid_masks or {}).get(name, np.ones(np.shape(mask),bool))).any()}
        configured = view_weights or {}
        self.view_weights = {}
        for name in self.targets:
            try:
                value = float(configured.get(name, 1.0))
            except (TypeError, ValueError):
                value = 1.0
            # Matches actual loss behavior for invalid weights.
            self.view_weights[name] = float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 1.0
        total = sum(self.view_weights.values())
        if total <= 0.0 and self.view_weights:
            self.view_weights = {name: 1.0 / len(self.view_weights) for name in self.targets}
        elif total > 0.0:
            self.view_weights = {name: value / total for name, value in self.view_weights.items()}
        self.weights = weights

    def evaluate(self, masks: Mapping[str, np.ndarray]) -> tuple[float, dict[str, np.ndarray], dict[str, float]]:
        total = 0.0
        gradients: dict[str, np.ndarray] = {}
        terms: dict[str, float] = {}
        for name, target in self.targets.items():
            if name not in masks:
                raise ValueError(f'missing rendered silhouette for {name}')
            value, gradient, view_terms = soft_mask_loss_and_gradient(masks[name], target,
                l2_weight=float(self.weights.silhouette_l2),
                iou_weight=float(self.weights.soft_iou),
                distance_weight=float(self.weights.signed_distance))
            weight = self.view_weights[name]
            total += weight * value
            gradients[name] = weight * gradient
            for term, term_value in view_terms.items():
                terms[term] = terms.get(term, 0.0) + weight * term_value
        return float(total), gradients, terms
