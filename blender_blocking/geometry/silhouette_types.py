"""Typed data contracts for silhouette extraction and canonicalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class MaskBBoxPx:
    """Pixel-space half-open bounding box for a mask."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)

    def to_dict(self) -> Dict[str, int]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> Optional["MaskBBoxPx"]:
        mask_bool = np.asarray(mask).astype(bool, copy=False)
        ys, xs = np.where(mask_bool)
        if xs.size == 0 or ys.size == 0:
            return None
        return cls(
            x0=int(xs.min()),
            y0=int(ys.min()),
            x1=int(xs.max()) + 1,
            y1=int(ys.max()) + 1,
        )


@dataclass(frozen=True)
class SilhouetteCandidate:
    """Intermediate extraction candidate before final selection."""

    mask: np.ndarray
    source: str
    polarity: str
    threshold: Optional[float]
    score: float
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class SilhouetteMask:
    """Selected hard silhouette mask with extraction metadata."""

    mask: np.ndarray
    bbox: Optional[MaskBBoxPx]
    source: str
    polarity: str
    threshold: Optional[float]
    score: float
    diagnostics: Mapping[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "bbox": None if self.bbox is None else self.bbox.to_dict(),
            "source": self.source,
            "polarity": self.polarity,
            "threshold": self.threshold,
            "score": self.score,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class UncertainMask:
    """Silhouette probabilities and confidence maps paired with a hard mask."""

    foreground_prob: np.ndarray
    hard_mask: np.ndarray
    confidence: np.ndarray
    boundary_uncertainty: np.ndarray
    source: str
    threshold: Optional[float]
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class CanonicalSilhouette:
    """Canonical fixed-size silhouette and its source transform."""

    mask: np.ndarray
    source_bbox: Optional[MaskBBoxPx]
    output_size: int
    padding_frac: float
    anchor: str
    transform: Mapping[str, float]
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_bbox": None
            if self.source_bbox is None
            else self.source_bbox.to_dict(),
            "output_size": self.output_size,
            "padding_frac": self.padding_frac,
            "anchor": self.anchor,
            "transform": dict(self.transform),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class MaskIdentity:
    """Optional canonicalization cache identity supplied by callers."""

    shape: Tuple[int, ...]
    dtype: str
    strides: Tuple[int, ...]
    digest: Optional[str] = None
    version: Optional[str] = None
    extraction_config_hash: Optional[str] = None

    @classmethod
    def from_array(
        cls,
        mask: np.ndarray,
        *,
        digest: Optional[str] = None,
        version: Optional[str] = None,
        extraction_config_hash: Optional[str] = None,
    ) -> "MaskIdentity":
        arr = np.asarray(mask)
        return cls(
            shape=tuple(int(v) for v in arr.shape),
            dtype=str(arr.dtype),
            strides=tuple(int(v) for v in arr.strides),
            digest=digest,
            version=version,
            extraction_config_hash=extraction_config_hash,
        )
