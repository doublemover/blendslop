"""Canonical silhouette IoU utilities."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import CanonicalizeConfig, SilhouetteExtractConfig
from geometry.silhouette_pipeline import (
    build_uncertain_mask,
    canonicalize_silhouette,
    extract_silhouette_mask,
)
from geometry.silhouette_types import CanonicalSilhouette, MaskIdentity, UncertainMask
from metrics.silhouette import (
    SilhouetteMetricResult,
    boundary_iou,
    signed_distance_silhouette_loss,
    soft_iou,
)

_CANONICALIZE_CACHE_MAX = 128
_CANONICALIZE_CACHE: "OrderedDict[Tuple[object, ...], np.ndarray]" = OrderedDict()
_CANONICALIZE_CACHE_HITS = 0
_CANONICALIZE_CACHE_MISSES = 0


@dataclass(frozen=True)
class IoUResult:
    """Result of an IoU comparison."""

    iou: float
    intersection: int
    union: int
    warnings: Tuple[str, ...]
    ref_area: int = 0
    candidate_area: int = 0
    empty_classification: str = "non_empty"

    def to_dict(self) -> Dict[str, object]:
        """Return a serializable dict of result details."""
        return {
            "iou": self.iou,
            "intersection": self.intersection,
            "union": self.union,
            "ref_area": self.ref_area,
            "candidate_area": self.candidate_area,
            "empty_classification": self.empty_classification,
            "warnings": list(self.warnings),
        }


def mask_from_image_array(
    image: np.ndarray, *, extract_config: Optional[SilhouetteExtractConfig] = None
) -> np.ndarray:
    """Convert an image array into a boolean silhouette mask.

    Defaults now match the geometry extraction policy instead of using a raw,
    no-cleanup validation-only policy. Callers that need raw masks can pass an
    explicit SilhouetteExtractConfig with cleanup disabled.
    """
    if extract_config is None:
        extract_config = SilhouetteExtractConfig()
    return extract_silhouette_mask(image, extract_config).mask


def uncertain_mask_from_image_array(
    image: np.ndarray, *, extract_config: Optional[SilhouetteExtractConfig] = None
) -> UncertainMask:
    """Convert an image array into a hard/probability/confidence silhouette."""
    if extract_config is None:
        extract_config = SilhouetteExtractConfig()
    return build_uncertain_mask(image, extract_config)


def canonicalize_mask_with_metadata(
    mask: np.ndarray,
    *,
    canonicalize_config: Optional[CanonicalizeConfig] = None,
    output_size: Optional[int] = None,
    padding_frac: Optional[float] = None,
    anchor: Optional[str] = None,
    morph_close_px: int = 0,
) -> CanonicalSilhouette:
    """Canonicalize a boolean mask and return transform metadata."""
    overrides: Dict[str, object] = {"morph_close_px": morph_close_px}
    if output_size is not None:
        overrides["output_size"] = output_size
    if padding_frac is not None:
        overrides["padding_frac"] = padding_frac
    if anchor is not None:
        overrides["anchor"] = anchor
    return canonicalize_silhouette(mask, canonicalize_config, **overrides)


def canonicalize_mask(
    mask: np.ndarray,
    *,
    output_size: int = 256,
    padding_frac: float = 0.1,
    anchor: str = "bottom_center",
    morph_close_px: int = 0,
    canonicalize_config: Optional[CanonicalizeConfig] = None,
) -> np.ndarray:
    """Canonicalize a boolean mask into a fixed-size canvas."""
    if canonicalize_config is not None:
        return canonicalize_mask_with_metadata(
            mask,
            canonicalize_config=canonicalize_config,
            morph_close_px=morph_close_px,
        ).mask
    return canonicalize_mask_with_metadata(
        mask,
        canonicalize_config=canonicalize_config,
        output_size=output_size,
        padding_frac=padding_frac,
        anchor=anchor,
        morph_close_px=morph_close_px,
    ).mask


def _hash_mask(mask: np.ndarray) -> str:
    """Return a stable hash for a boolean mask."""
    arr = np.asarray(mask).astype(bool, copy=False)
    if arr.flags.c_contiguous:
        mask_bool = arr
    else:
        mask_bool = np.ascontiguousarray(arr)
    return hashlib.sha1(mask_bool.tobytes()).hexdigest()


def _make_cache_key(
    mask: np.ndarray,
    *,
    output_size: int,
    padding_frac: float,
    anchor: str,
    morph_close_px: int,
    mask_identity: Optional[MaskIdentity] = None,
    digest: Optional[str] = None,
) -> Tuple[object, ...]:
    """Build a cache key from mask identity/content and canonicalization params."""
    mask_arr = np.asarray(mask)
    identity_digest = digest or (mask_identity.digest if mask_identity else None)
    if identity_digest is None:
        identity_digest = _hash_mask(mask_arr)
    identity = mask_identity or MaskIdentity.from_array(mask_arr, digest=identity_digest)
    return (
        identity_digest,
        identity.shape,
        identity.dtype,
        identity.strides,
        identity.version,
        identity.extraction_config_hash,
        int(output_size),
        float(padding_frac),
        str(anchor),
        int(morph_close_px),
    )


def canonicalize_mask_cached(
    mask: np.ndarray,
    *,
    output_size: int = 256,
    padding_frac: float = 0.1,
    anchor: str = "bottom_center",
    morph_close_px: int = 0,
    canonicalize_config: Optional[CanonicalizeConfig] = None,
    mask_identity: Optional[MaskIdentity] = None,
    digest: Optional[str] = None,
) -> np.ndarray:
    """Canonicalize a mask using a small LRU cache."""
    global _CANONICALIZE_CACHE_HITS
    global _CANONICALIZE_CACHE_MISSES

    if canonicalize_config is not None:
        output_size = canonicalize_config.output_size
        padding_frac = canonicalize_config.padding_frac
        anchor = canonicalize_config.anchor
        if not canonicalize_config.use_cache:
            return canonicalize_mask(
                mask,
                output_size=output_size,
                padding_frac=padding_frac,
                anchor=anchor,
                morph_close_px=morph_close_px,
                canonicalize_config=canonicalize_config,
            )

    key = _make_cache_key(
        mask,
        output_size=output_size,
        padding_frac=padding_frac,
        anchor=anchor,
        morph_close_px=morph_close_px,
        mask_identity=mask_identity,
        digest=digest,
    )

    cached = _CANONICALIZE_CACHE.get(key)
    if cached is not None:
        _CANONICALIZE_CACHE_HITS += 1
        _CANONICALIZE_CACHE.move_to_end(key)
        return cached

    _CANONICALIZE_CACHE_MISSES += 1
    result = canonicalize_mask(
        mask,
        output_size=output_size,
        padding_frac=padding_frac,
        anchor=anchor,
        morph_close_px=morph_close_px,
        canonicalize_config=canonicalize_config,
    )
    _CANONICALIZE_CACHE[key] = result
    if len(_CANONICALIZE_CACHE) > _CANONICALIZE_CACHE_MAX:
        _CANONICALIZE_CACHE.popitem(last=False)
    return result


def reset_canonicalize_cache() -> None:
    """Clear the canonicalization cache and counters."""
    global _CANONICALIZE_CACHE_HITS
    global _CANONICALIZE_CACHE_MISSES

    _CANONICALIZE_CACHE.clear()
    _CANONICALIZE_CACHE_HITS = 0
    _CANONICALIZE_CACHE_MISSES = 0


def get_canonicalize_cache_stats() -> Dict[str, int]:
    """Return cache hit/miss counters and current size."""
    return {
        "hits": _CANONICALIZE_CACHE_HITS,
        "misses": _CANONICALIZE_CACHE_MISSES,
        "size": len(_CANONICALIZE_CACHE),
        "capacity": _CANONICALIZE_CACHE_MAX,
    }


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> IoUResult:
    """Compute intersection-over-union between two boolean masks."""
    a = np.asarray(mask_a).astype(bool, copy=False)
    b = np.asarray(mask_b).astype(bool, copy=False)
    if a.shape != b.shape:
        raise ValueError("Masks must have matching shapes for IoU")

    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    ref_area = int(a.sum())
    candidate_area = int(b.sum())

    warnings: List[str] = []
    empty_classification = "non_empty"
    if union == 0:
        empty_classification = "both_empty"
        warnings.append("Both masks are empty")
        return IoUResult(
            iou=0.0,
            intersection=0,
            union=0,
            warnings=tuple(warnings),
            ref_area=ref_area,
            candidate_area=candidate_area,
            empty_classification=empty_classification,
        )

    if ref_area == 0:
        empty_classification = "reference_empty"
        warnings.append("Reference mask is empty")
    elif candidate_area == 0:
        empty_classification = "candidate_empty"
        warnings.append("Candidate mask is empty")

    iou = float(intersection) / float(union)
    return IoUResult(
        iou=iou,
        intersection=intersection,
        union=union,
        warnings=tuple(warnings),
        ref_area=ref_area,
        candidate_area=candidate_area,
        empty_classification=empty_classification,
    )


def compute_silhouette_metrics(
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    view: str = "unknown",
    pass_required: bool = True,
    boundary_radius: int = 2,
    reference_prob: Optional[np.ndarray] = None,
    candidate_prob: Optional[np.ndarray] = None,
    reference_confidence: Optional[np.ndarray] = None,
    candidate_confidence: Optional[np.ndarray] = None,
) -> SilhouetteMetricResult:
    """Compute hard and soft silhouette metrics for one view."""
    area = compute_mask_iou(reference_mask, candidate_mask)
    b_iou, b_warnings = boundary_iou(
        reference_mask,
        candidate_mask,
        dilation_radius=boundary_radius,
        confidence_a=reference_confidence,
        confidence_b=candidate_confidence,
    )
    soft = None
    if reference_prob is not None and candidate_prob is not None:
        soft = soft_iou(reference_prob, candidate_prob)
    sd_loss = signed_distance_silhouette_loss(reference_mask, candidate_mask)
    warnings = tuple(area.warnings) + tuple(b_warnings)
    return SilhouetteMetricResult(
        view=view,
        area_iou=area.iou,
        boundary_iou=b_iou,
        soft_iou=soft,
        signed_distance_loss=sd_loss,
        intersection=area.intersection,
        union=area.union,
        ref_area=area.ref_area,
        render_area=area.candidate_area,
        pass_required=pass_required,
        warnings=warnings,
    )
