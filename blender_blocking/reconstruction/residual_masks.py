"""Residual silhouette classification helpers for editable shape programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ResidualMaskReport:
    view: str
    missing_area: int
    extra_area: int
    residual_area: int
    category: str
    confidence: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "view": self.view,
            "missing_area": self.missing_area,
            "extra_area": self.extra_area,
            "residual_area": self.residual_area,
            "category": self.category,
            "confidence": self.confidence,
            "notes": list(self.notes),
        }


def classify_residual_mask(
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    view: str = "",
    metadata: Mapping[str, object] | None = None,
) -> ResidualMaskReport:
    ref = np.asarray(reference_mask).astype(bool, copy=False)
    cand = np.asarray(candidate_mask).astype(bool, copy=False)
    if ref.shape != cand.shape:
        raise ValueError("reference and candidate masks must have matching shapes")
    missing = np.logical_and(ref, np.logical_not(cand))
    extra = np.logical_and(np.logical_not(ref), cand)
    missing_area = int(missing.sum())
    extra_area = int(extra.sum())
    residual_area = missing_area + extra_area
    ref_area = max(1, int(ref.sum()))
    ratio = residual_area / float(ref_area)
    category = _classify(missing, extra, ratio, metadata or {})
    confidence = float(min(1.0, ratio * 4.0))
    return ResidualMaskReport(
        view=view,
        missing_area=missing_area,
        extra_area=extra_area,
        residual_area=residual_area,
        category=category,
        confidence=confidence,
    )


def _classify(
    missing: np.ndarray,
    extra: np.ndarray,
    residual_ratio: float,
    metadata: Mapping[str, object],
) -> str:
    if residual_ratio <= 0.005:
        return "none"
    if bool(metadata.get("hidden_concavity_ambiguous")):
        return "ambiguous_hidden_geometry"
    missing_components = _component_count(missing)
    extra_components = _component_count(extra)
    if missing.sum() > extra.sum() * 2:
        if missing_components > 4:
            return "missing_repeated_detail"
        if _thin_structure_hint(missing):
            return "missing_thin_support"
        return "missing_protrusion"
    if extra.sum() > missing.sum() * 2:
        if extra_components > 4:
            return "extra_fragments"
        return "extra_bulk"
    if missing_components > 1 or extra_components > 1:
        return "symmetry_or_component_mismatch"
    return "boundary_mismatch"


def _component_count(mask: np.ndarray) -> int:
    try:
        import cv2

        num_labels, _labels = cv2.connectedComponents(mask.astype(np.uint8), 8)
        return max(0, int(num_labels) - 1)
    except Exception:
        return 1 if bool(np.asarray(mask).any()) else 0


def _thin_structure_hint(mask: np.ndarray) -> bool:
    ys, xs = np.where(np.asarray(mask).astype(bool, copy=False))
    if xs.size == 0 or ys.size == 0:
        return False
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    long_side = max(width, height)
    short_side = max(1, min(width, height))
    return long_side / float(short_side) >= 4.0
