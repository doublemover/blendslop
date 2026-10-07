"""Candidate-scored silhouette extraction and canonicalization pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from geometry.silhouette_types import (
    CanonicalSilhouette,
    MaskBBoxPx,
    SilhouetteCandidate,
    SilhouetteMask,
    UncertainMask,
)


@dataclass(frozen=True)
class ExtractionPolicy:
    """Normalized extraction settings accepted by the pipeline."""

    prefer_alpha: bool = True
    alpha_threshold: int = 127
    gray_threshold: Optional[int] = None
    invert_policy: str = "auto"
    polarity: str = "auto"
    alpha_min_coverage: float = 0.001
    min_area_frac: float = 0.0005
    max_area_frac: float = 0.98
    max_border_contact_frac: float = 0.85
    morph_close_px: int = 0
    morph_open_px: int = 0
    fill_holes: bool = False
    largest_component_only: bool = False
    min_component_area_px: int = 0
    min_component_area_frac: float = 0.0
    adaptive_threshold: bool = True


@dataclass(frozen=True)
class CanonicalizationPolicy:
    """Normalized canonicalization settings."""

    output_size: int = 256
    padding_frac: float = 0.1
    anchor: str = "bottom_center"
    interp: str = "nearest"
    morph_close_px: int = 0


def _as_policy(config: Any = None, **overrides: Any) -> ExtractionPolicy:
    values = {
        "prefer_alpha": True,
        "alpha_threshold": 127,
        "gray_threshold": None,
        "invert_policy": "auto",
        "polarity": "auto",
        "alpha_min_coverage": 0.001,
        "min_area_frac": 0.0005,
        "max_area_frac": 0.98,
        "max_border_contact_frac": 0.85,
        "morph_close_px": 0,
        "morph_open_px": 0,
        "fill_holes": False,
        "largest_component_only": False,
        "min_component_area_px": 0,
        "min_component_area_frac": 0.0,
        "adaptive_threshold": True,
    }
    if config is not None:
        if isinstance(config, Mapping):
            source = config
        else:
            source = config.to_dict() if hasattr(config, "to_dict") else vars(config)
        for key in values:
            if key in source:
                values[key] = source[key]
            elif hasattr(config, key):
                values[key] = getattr(config, key)
    values.update({key: value for key, value in overrides.items() if key in values})
    return ExtractionPolicy(**values)


def _as_canonical_policy(config: Any = None, **overrides: Any) -> CanonicalizationPolicy:
    values = {
        "output_size": 256,
        "padding_frac": 0.1,
        "anchor": "bottom_center",
        "interp": "nearest",
        "morph_close_px": 0,
    }
    if config is not None:
        if isinstance(config, Mapping):
            source = config
        else:
            source = config.to_dict() if hasattr(config, "to_dict") else vars(config)
        for key in values:
            if key in source:
                values[key] = source[key]
            elif hasattr(config, key):
                values[key] = getattr(config, key)
    values.update({key: value for key, value in overrides.items() if key in values})
    return CanonicalizationPolicy(**values)


def normalize_image_array(image: np.ndarray) -> np.ndarray:
    """Return an ndarray and validate the supported image shape."""
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    if arr.ndim == 3 and arr.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported channel count: {arr.shape[2]}")
    return arr


def _ensure_uint8(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == np.uint8:
        return arr
    max_val = float(np.max(arr)) if arr.size else 0.0
    if max_val <= 1.0:
        return (arr.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def luma_from_image(image: np.ndarray) -> np.ndarray:
    """Extract uint8 luma from a grayscale, RGB, or RGBA image."""
    arr = normalize_image_array(image)
    if arr.ndim == 2:
        return _ensure_uint8(arr)
    rgb = _ensure_uint8(arr[:, :, :3])
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def alpha_from_image(image: np.ndarray) -> Optional[np.ndarray]:
    arr = normalize_image_array(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return _ensure_uint8(arr[:, :, 3])
    return None


def otsu_threshold(gray: np.ndarray) -> float:
    gray_u8 = _ensure_uint8(gray)
    threshold, _ = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(threshold)


def _kernel_for(size: int) -> Optional[np.ndarray]:
    if size <= 0:
        return None
    k = max(3, int(size))
    if k % 2 == 0:
        k += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _component_stats(mask_uint8: np.ndarray) -> Dict[str, object]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_uint8 > 0).astype(np.uint8), 8
    )
    areas = [
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, num_labels)
        if int(stats[label, cv2.CC_STAT_AREA]) > 0
    ]
    total = int(sum(areas))
    largest = max(areas) if areas else 0
    return {
        "labels": labels,
        "areas": areas,
        "component_count": len(areas),
        "largest_area": largest,
        "largest_component_frac": float(largest / total) if total else 0.0,
    }


def _remove_small_components(mask_uint8: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask_uint8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_uint8 > 0).astype(np.uint8), 8
    )
    kept = np.zeros_like(mask_uint8)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            kept[labels == label] = 255
    return kept


def _largest_component(mask_uint8: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_uint8 > 0).astype(np.uint8), 8
    )
    if num_labels <= 1:
        return np.zeros_like(mask_uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    return np.where(labels == label, 255, 0).astype(np.uint8)


def _fill_holes(mask_uint8: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask_uint8)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, -1)
    return filled


def cleanup_mask(mask: np.ndarray, policy: ExtractionPolicy) -> np.ndarray:
    """Apply deterministic morphology and component cleanup."""
    mask_uint8 = (np.asarray(mask).astype(bool) * 255).astype(np.uint8)
    close_kernel = _kernel_for(policy.morph_close_px)
    if close_kernel is not None:
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, close_kernel)
    open_kernel = _kernel_for(policy.morph_open_px)
    if open_kernel is not None:
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, open_kernel)
    min_area = max(
        int(policy.min_component_area_px),
        int(round(mask_uint8.size * float(policy.min_component_area_frac))),
    )
    mask_uint8 = _remove_small_components(mask_uint8, min_area)
    if policy.largest_component_only:
        mask_uint8 = _largest_component(mask_uint8)
    if policy.fill_holes:
        mask_uint8 = _fill_holes(mask_uint8)
    return mask_uint8 > 0


def _border_contact_frac(mask: np.ndarray) -> float:
    mask_bool = np.asarray(mask).astype(bool, copy=False)
    if mask_bool.size == 0:
        return 0.0
    border = np.zeros(mask_bool.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_pixels = int(np.logical_and(mask_bool, border).sum())
    perimeter_pixels = int(border.sum())
    return float(border_pixels / perimeter_pixels) if perimeter_pixels else 0.0


def _contour_and_hole_counts(mask: np.ndarray) -> Tuple[int, int]:
    mask_uint8 = (np.asarray(mask).astype(bool) * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0, 0
    contour_count = len(contours)
    holes = sum(1 for item in hierarchy[0] if int(item[3]) >= 0)
    return contour_count, holes


def _candidate_diagnostics(mask: np.ndarray) -> Dict[str, object]:
    mask_bool = np.asarray(mask).astype(bool, copy=False)
    area = int(mask_bool.sum())
    area_frac = float(area / mask_bool.size) if mask_bool.size else 0.0
    bbox = MaskBBoxPx.from_mask(mask_bool)
    bbox_area_frac = float(bbox.area / mask_bool.size) if bbox and mask_bool.size else 0.0
    compactness = float(area / bbox.area) if bbox and bbox.area else 0.0
    contour_count, hole_count = _contour_and_hole_counts(mask_bool)
    stats = _component_stats((mask_bool * 255).astype(np.uint8))
    return {
        "area": area,
        "area_frac": area_frac,
        "bbox": None if bbox is None else bbox.to_dict(),
        "bbox_area_frac": bbox_area_frac,
        "bbox_compactness": compactness,
        "border_contact_frac": _border_contact_frac(mask_bool),
        "contour_count": contour_count,
        "hole_count": hole_count,
        "component_count": stats["component_count"],
        "largest_component_frac": stats["largest_component_frac"],
    }


def _score_candidate(
    mask: np.ndarray,
    *,
    policy: ExtractionPolicy,
    source: str,
    polarity: str,
) -> Tuple[float, Dict[str, object]]:
    diagnostics = _candidate_diagnostics(mask)
    area_frac = float(diagnostics["area_frac"])
    border_frac = float(diagnostics["border_contact_frac"])
    compactness = float(diagnostics["bbox_compactness"])
    largest_frac = float(diagnostics["largest_component_frac"])
    contour_count = int(diagnostics["contour_count"])
    hole_count = int(diagnostics["hole_count"])

    hard_rejects = []
    if area_frac < policy.min_area_frac:
        hard_rejects.append("area_below_min")
    if area_frac > policy.max_area_frac:
        hard_rejects.append("area_above_max")

    score = 1.0
    score -= min(0.85, abs(area_frac - 0.30) * 0.9)
    score += 0.35 * largest_frac
    score += 0.20 * min(1.0, compactness)
    score -= 0.40 * min(1.0, border_frac / max(policy.max_border_contact_frac, 1e-6))
    score -= 0.08 * max(0, contour_count - 1)
    score -= 0.03 * hole_count
    if source == "alpha" and policy.prefer_alpha:
        score += 0.20
    if policy.polarity != "auto" and polarity == policy.polarity:
        score += 0.25
    elif policy.polarity not in ("auto", polarity):
        score -= 0.35
    if border_frac > policy.max_border_contact_frac:
        score -= 0.35
        diagnostics["border_policy_warning"] = "border_contact_above_limit"
    if hard_rejects:
        score -= 10.0
        diagnostics["reject_reasons"] = tuple(hard_rejects)
    diagnostics["score_terms"] = {
        "area_frac": area_frac,
        "largest_component_frac": largest_frac,
        "bbox_compactness": compactness,
        "border_contact_frac": border_frac,
        "contour_count": contour_count,
        "hole_count": hole_count,
    }
    return float(score), diagnostics


def _build_candidate(
    mask: np.ndarray,
    *,
    source: str,
    polarity: str,
    threshold: Optional[float],
    policy: ExtractionPolicy,
) -> SilhouetteCandidate:
    cleaned = cleanup_mask(mask, policy)
    score, diagnostics = _score_candidate(
        cleaned, policy=policy, source=source, polarity=polarity
    )
    diagnostics = {
        **diagnostics,
        "source": source,
        "polarity": polarity,
        "threshold": threshold,
    }
    return SilhouetteCandidate(
        mask=cleaned,
        source=source,
        polarity=polarity,
        threshold=threshold,
        score=score,
        diagnostics=diagnostics,
    )


def _alpha_candidates(image: np.ndarray, policy: ExtractionPolicy) -> Iterable[SilhouetteCandidate]:
    alpha = alpha_from_image(image)
    if alpha is None or not policy.prefer_alpha:
        return ()
    raw = alpha > int(policy.alpha_threshold)
    coverage = float(raw.sum() / raw.size) if raw.size else 0.0
    if coverage < policy.alpha_min_coverage or coverage > policy.max_area_frac:
        return ()
    if np.ptp(alpha) == 0:
        return ()
    return (
        _build_candidate(
            raw,
            source="alpha",
            polarity="alpha_foreground",
            threshold=float(policy.alpha_threshold),
            policy=policy,
        ),
    )


def _luma_candidates(image: np.ndarray, policy: ExtractionPolicy) -> Sequence[SilhouetteCandidate]:
    gray = luma_from_image(image)
    threshold = float(policy.gray_threshold) if policy.gray_threshold is not None else otsu_threshold(gray)
    candidates = []
    allowed = {policy.polarity} if policy.polarity != "auto" else {
        "dark_foreground",
        "light_foreground",
    }
    if policy.invert_policy == "invert":
        allowed = {"dark_foreground"}
    elif policy.invert_policy == "no_invert":
        allowed = {"light_foreground"}

    if "dark_foreground" in allowed:
        candidates.append(
            _build_candidate(
                gray <= threshold,
                source="luma",
                polarity="dark_foreground",
                threshold=threshold,
                policy=policy,
            )
        )
    if "light_foreground" in allowed:
        candidates.append(
            _build_candidate(
                gray > threshold,
                source="luma",
                polarity="light_foreground",
                threshold=threshold,
                policy=policy,
            )
        )
    if policy.adaptive_threshold and policy.gray_threshold is None and min(gray.shape) >= 9:
        block = max(9, (min(gray.shape) // 8) | 1)
        adaptive_dark = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 2
        )
        adaptive_light = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 2
        )
        if policy.polarity in ("auto", "dark_foreground"):
            candidates.append(
                _build_candidate(
                    adaptive_dark > 0,
                    source="adaptive_luma",
                    polarity="dark_foreground",
                    threshold=None,
                    policy=policy,
                )
            )
        if policy.polarity in ("auto", "light_foreground"):
            candidates.append(
                _build_candidate(
                    adaptive_light > 0,
                    source="adaptive_luma",
                    polarity="light_foreground",
                    threshold=None,
                    policy=policy,
                )
            )
    return candidates


def extract_silhouette_candidates(
    image: np.ndarray,
    config: Any = None,
    **overrides: Any,
) -> Tuple[SilhouetteCandidate, ...]:
    """Generate cleaned and scored silhouette candidates."""
    arr = normalize_image_array(image)
    policy = _as_policy(config, **overrides)
    candidates = list(_alpha_candidates(arr, policy))
    candidates.extend(_luma_candidates(arr, policy))
    return tuple(sorted(candidates, key=lambda item: item.score, reverse=True))


def extract_silhouette_mask(
    image: np.ndarray,
    config: Any = None,
    **overrides: Any,
) -> SilhouetteMask:
    """Extract the best-scored hard silhouette with diagnostics."""
    candidates = extract_silhouette_candidates(image, config, **overrides)
    if not candidates:
        arr = normalize_image_array(image)
        empty = np.zeros(arr.shape[:2], dtype=bool)
        return SilhouetteMask(
            mask=empty,
            bbox=None,
            source="none",
            polarity="none",
            threshold=None,
            score=float("-inf"),
            diagnostics={"error": "no_silhouette_candidates"},
        )
    best = candidates[0]
    margin = (
        float(best.score - candidates[1].score) if len(candidates) > 1 else float("inf")
    )
    diagnostics = {
        **best.diagnostics,
        "candidate_count": len(candidates),
        "score_margin": margin,
        "candidates": [
            {
                "source": item.source,
                "polarity": item.polarity,
                "threshold": item.threshold,
                "score": item.score,
                "diagnostics": dict(item.diagnostics),
            }
            for item in candidates
        ],
    }
    return SilhouetteMask(
        mask=best.mask.astype(bool, copy=False),
        bbox=MaskBBoxPx.from_mask(best.mask),
        source=best.source,
        polarity=best.polarity,
        threshold=best.threshold,
        score=best.score,
        diagnostics=diagnostics,
    )


def _distance_confidence(hard_mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(hard_mask).astype(np.uint8)
    inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 3)
    dist = inside + outside
    if dist.size == 0:
        return dist.astype(np.float32)
    return np.clip(dist / max(1.0, float(np.percentile(dist, 95))), 0.0, 1.0).astype(
        np.float32
    )


def build_uncertain_mask(
    image: np.ndarray,
    config: Any = None,
    **overrides: Any,
) -> UncertainMask:
    """Extract hard mask plus foreground probability and confidence maps."""
    selected = extract_silhouette_mask(image, config, **overrides)
    arr = normalize_image_array(image)
    hard = selected.mask.astype(bool, copy=False)

    if selected.source == "alpha":
        alpha = alpha_from_image(arr)
        prob = (
            alpha.astype(np.float32) / 255.0
            if alpha is not None
            else hard.astype(np.float32)
        )
        confidence = np.clip(np.abs(prob - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
    else:
        gray = luma_from_image(arr).astype(np.float32)
        threshold = selected.threshold if selected.threshold is not None else otsu_threshold(gray)
        spread = max(8.0, float(np.std(gray)))
        signed = (float(threshold) - gray) / spread
        if selected.polarity == "light_foreground":
            signed = -signed
        prob = (1.0 / (1.0 + np.exp(-signed))).astype(np.float32)
        confidence = np.clip(np.abs(gray - float(threshold)) / (2.0 * spread), 0.0, 1.0)
        confidence = confidence.astype(np.float32)

    boundary_confidence = _distance_confidence(hard)
    boundary_uncertainty = (1.0 - boundary_confidence).astype(np.float32)
    confidence = np.minimum(confidence, np.maximum(boundary_confidence, 0.25)).astype(
        np.float32
    )

    diagnostics = {
        **dict(selected.diagnostics),
        "uncertainty": {
            "foreground_prob_min": float(np.min(prob)) if prob.size else 0.0,
            "foreground_prob_max": float(np.max(prob)) if prob.size else 0.0,
            "confidence_mean": float(np.mean(confidence)) if confidence.size else 0.0,
            "boundary_uncertainty_mean": float(np.mean(boundary_uncertainty))
            if boundary_uncertainty.size
            else 0.0,
        },
    }
    return UncertainMask(
        foreground_prob=prob,
        hard_mask=hard,
        confidence=confidence,
        boundary_uncertainty=boundary_uncertainty,
        source=selected.source,
        threshold=selected.threshold,
        diagnostics=diagnostics,
    )


def canonicalize_silhouette(
    mask: np.ndarray,
    config: Any = None,
    **overrides: Any,
) -> CanonicalSilhouette:
    """Canonicalize a mask and retain transform metadata."""
    policy = _as_canonical_policy(config, **overrides)
    output_size = int(policy.output_size)
    if output_size < 1:
        raise ValueError("output_size must be >= 1")
    if policy.anchor not in ("bottom_center", "center"):
        raise ValueError(f"Unknown anchor: {policy.anchor}")
    if policy.interp != "nearest":
        raise ValueError(f"Unsupported interpolation: {policy.interp}")

    mask_bool = np.asarray(mask).astype(bool, copy=False)
    source_bbox = MaskBBoxPx.from_mask(mask_bool)
    warnings = []
    if source_bbox is None:
        warnings.append("empty_mask")
        return CanonicalSilhouette(
            mask=np.zeros((output_size, output_size), dtype=bool),
            source_bbox=None,
            output_size=output_size,
            padding_frac=float(policy.padding_frac),
            anchor=policy.anchor,
            transform={
                "scale": 0.0,
                "x_start": 0.0,
                "y_start": 0.0,
                "pad_px": 0.0,
                "crop_width": 0.0,
                "crop_height": 0.0,
            },
            warnings=tuple(warnings),
        )

    cropped = mask_bool[source_bbox.y0 : source_bbox.y1, source_bbox.x0 : source_bbox.x1]
    crop_h, crop_w = cropped.shape
    pad = int(round(max(crop_h, crop_w) * float(policy.padding_frac)))
    padded = np.pad(
        cropped, ((pad, pad), (pad, pad)), mode="constant", constant_values=False
    )
    padded_h, padded_w = padded.shape
    scale = output_size / float(max(padded_h, padded_w))
    new_w = max(1, int(round(padded_w * scale)))
    new_h = max(1, int(round(padded_h * scale)))
    resized = cv2.resize(
        padded.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    canvas = np.zeros((output_size, output_size), dtype=bool)
    if policy.anchor == "bottom_center":
        x_start = (output_size - new_w) // 2
        y_start = output_size - new_h
    else:
        x_start = (output_size - new_w) // 2
        y_start = (output_size - new_h) // 2
    x_start = max(0, min(output_size - new_w, x_start))
    y_start = max(0, min(output_size - new_h, y_start))
    canvas[y_start : y_start + new_h, x_start : x_start + new_w] = resized

    if policy.morph_close_px > 0:
        kernel = _kernel_for(policy.morph_close_px)
        if kernel is not None:
            canvas = cv2.morphologyEx(
                canvas.astype(np.uint8), cv2.MORPH_CLOSE, kernel
            ).astype(bool)

    return CanonicalSilhouette(
        mask=canvas,
        source_bbox=source_bbox,
        output_size=output_size,
        padding_frac=float(policy.padding_frac),
        anchor=policy.anchor,
        transform={
            "scale": float(scale),
            "x_start": float(x_start),
            "y_start": float(y_start),
            "pad_px": float(pad),
            "crop_x0": float(source_bbox.x0),
            "crop_y0": float(source_bbox.y0),
            "crop_width": float(crop_w),
            "crop_height": float(crop_h),
            "resized_width": float(new_w),
            "resized_height": float(new_h),
        },
        warnings=tuple(warnings),
    )


def canonicalize_mask_array(mask: np.ndarray, config: Any = None, **overrides: Any) -> np.ndarray:
    """Compatibility helper returning only the canonical hard mask."""
    return canonicalize_silhouette(mask, config, **overrides).mask
