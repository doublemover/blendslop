"""Distributional silhouette profile-band extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .types import ProfileIntervalPx, UncertainProfileBand


@dataclass(frozen=True)
class ProfileBandDistributionConfig:
    thresholds: tuple[float, ...] = (0.35, 0.45, 0.5, 0.55, 0.65)
    include_morphology: bool = True
    morphology_radius: int = 1
    min_threshold: float = 0.01
    max_threshold: float = 0.99


def distributional_profile_bands(
    mask: Any,
    *,
    sample_count: int,
    view: str = "",
    probability: Any = None,
    confidence: Any = None,
    config: ProfileBandDistributionConfig = ProfileBandDistributionConfig(),
) -> tuple[UncertainProfileBand, ...]:
    """Extract profile bands from a small distribution of plausible masks."""

    hard = np.asarray(mask).astype(bool, copy=False)
    if hard.ndim != 2:
        raise ValueError("mask must be 2D")
    probability_map = _optional_map(probability, hard.shape, "probability")
    confidence_map = _optional_map(confidence, hard.shape, "confidence")
    variants = _profile_variants(
        hard,
        probability=probability_map,
        config=config,
    )
    if not variants:
        variants = (hard,)
    height = hard.shape[0]
    if height == 0:
        return ()
    rows = _sample_rows(height, sample_count)
    denom = max(1, height - 1)
    bands = []
    for row in rows:
        row_index = int(row)
        confidence_row = (
            None if confidence_map is None else confidence_map[row_index, :]
        )
        probability_row = (
            None if probability_map is None else probability_map[row_index, :]
        )
        row_variants = tuple(variant[row_index, :] for variant in variants)
        base_row = hard[row_index, :]
        intervals = _row_intervals(base_row, confidence=confidence_row)
        holes = _holes_between_intervals(intervals, confidence=confidence_row)
        stats = _row_distribution_stats(
            row_variants,
            probability=probability_row,
            confidence=confidence_row,
            thresholds=config.thresholds,
        )
        bands.append(
            UncertainProfileBand(
                t=float(1.0 - float(row_index) / denom),
                intervals=intervals,
                center_x=stats["center_median"],
                width_px=float(stats["width_median"] or 0.0),
                holes=holes,
                moments=stats["moments"],
                confidence=float(stats["confidence"]),
                source_view=view,
                center_std=float(stats["center_std"] or 0.0),
                width_std=float(stats["width_std"] or 0.0),
            )
        )
    return tuple(bands)


def profile_distribution_summary(
    bands: Sequence[UncertainProfileBand],
) -> dict[str, float]:
    if not bands:
        return {
            "band_count": 0.0,
            "width_std_mean": 0.0,
            "center_std_mean": 0.0,
            "variant_count": 0.0,
        }
    width_std = [float(getattr(band, "width_std", 0.0) or 0.0) for band in bands]
    center_std = [float(getattr(band, "center_std", 0.0) or 0.0) for band in bands]
    variant_counts = [
        float(getattr(band, "moments", {}).get("variant_count", 0.0) or 0.0)
        for band in bands
    ]
    return {
        "band_count": float(len(bands)),
        "width_std_mean": float(np.mean(width_std)),
        "width_std_max": float(np.max(width_std)),
        "center_std_mean": float(np.mean(center_std)),
        "center_std_max": float(np.max(center_std)),
        "variant_count": float(max(variant_counts) if variant_counts else 0.0),
    }


def _profile_variants(
    hard: np.ndarray,
    *,
    probability: np.ndarray | None,
    config: ProfileBandDistributionConfig,
) -> tuple[np.ndarray, ...]:
    variants: list[np.ndarray] = [np.asarray(hard, dtype=bool)]
    if probability is not None:
        for threshold in config.thresholds:
            threshold_value = float(np.clip(threshold, config.min_threshold, config.max_threshold))
            variants.append(probability >= threshold_value)
    if config.include_morphology:
        radius = max(1, int(config.morphology_radius))
        seeds = tuple(variants)
        for variant in seeds:
            variants.append(_binary_erosion(variant, radius=radius))
            variants.append(_binary_dilation(variant, radius=radius))
    return _unique_masks(variants)


def _row_distribution_stats(
    rows: Sequence[np.ndarray],
    *,
    probability: np.ndarray | None,
    confidence: np.ndarray | None,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    widths = []
    centers = []
    component_counts = []
    for row in rows:
        intervals = _row_intervals(row)
        width = float(sum(interval.width for interval in intervals))
        widths.append(width)
        component_counts.append(float(len(intervals)))
        centers.append(_weighted_center(intervals))
    widths_array = np.asarray(widths, dtype=np.float64)
    centers_array = np.asarray(
        [center for center in centers if center is not None],
        dtype=np.float64,
    )
    components_array = np.asarray(component_counts, dtype=np.float64)
    center_median = (
        float(np.median(centers_array)) if centers_array.size else None
    )
    width_median = float(np.median(widths_array)) if widths_array.size else 0.0
    confidence_value = _row_confidence(rows, confidence)
    moments: dict[str, float] = {
        "variant_count": float(len(rows)),
        "threshold_min": float(min(thresholds)) if thresholds else 0.0,
        "threshold_max": float(max(thresholds)) if thresholds else 0.0,
        "width_min": float(np.min(widths_array)) if widths_array.size else 0.0,
        "width_p05": _percentile(widths_array, 0.05),
        "width_p50": width_median,
        "width_p95": _percentile(widths_array, 0.95),
        "width_max": float(np.max(widths_array)) if widths_array.size else 0.0,
        "width_variance": float(np.var(widths_array)) if widths_array.size else 0.0,
        "component_count_mean": float(np.mean(components_array)) if components_array.size else 0.0,
        "component_count_std": float(np.std(components_array)) if components_array.size else 0.0,
    }
    if centers_array.size:
        moments.update(
            {
                "center_p05": _percentile(centers_array, 0.05),
                "center_p50": float(np.median(centers_array)),
                "center_p95": _percentile(centers_array, 0.95),
                "center_variance": float(np.var(centers_array)),
            }
        )
    if probability is not None:
        prob = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
        moments["probability_width_px"] = float(np.sum(prob))
        moments["probability_entropy"] = _probability_entropy(prob)
    if confidence is not None:
        conf = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
        moments["row_confidence_mean"] = float(np.mean(conf)) if conf.size else 0.0
    return {
        "center_median": center_median,
        "width_median": width_median,
        "center_std": float(np.std(centers_array)) if centers_array.size else 0.0,
        "width_std": float(np.std(widths_array)) if widths_array.size else 0.0,
        "confidence": confidence_value,
        "moments": moments,
    }


def _row_intervals(
    row: Any,
    *,
    confidence: np.ndarray | None = None,
) -> tuple[ProfileIntervalPx, ...]:
    values = np.asarray(row).astype(bool, copy=False)
    if values.size == 0 or not values.any():
        return ()
    confidence_values = None if confidence is None else np.asarray(confidence, dtype=np.float32)
    padded = np.pad(values.astype(np.int8), (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0]
    intervals = []
    for start, stop in zip(starts, stops):
        interval_confidence = 1.0
        if confidence_values is not None and stop > start:
            interval_confidence = float(np.mean(confidence_values[start:stop]))
        intervals.append(
            ProfileIntervalPx(
                float(start),
                float(stop),
                confidence=interval_confidence,
                source="distributional_mask",
            )
        )
    return tuple(intervals)


def _holes_between_intervals(
    intervals: Sequence[ProfileIntervalPx],
    *,
    confidence: np.ndarray | None = None,
) -> tuple[ProfileIntervalPx, ...]:
    holes = []
    confidence_values = None if confidence is None else np.asarray(confidence, dtype=np.float32)
    for left, right in zip(intervals, intervals[1:]):
        if right.x0 <= left.x1:
            continue
        start = int(round(left.x1))
        stop = int(round(right.x0))
        hole_confidence = 1.0
        if confidence_values is not None and stop > start:
            hole_confidence = float(np.mean(confidence_values[start:stop]))
        holes.append(
            ProfileIntervalPx(
                float(left.x1),
                float(right.x0),
                confidence=hole_confidence,
                source="distributional_hole",
            )
        )
    return tuple(holes)


def _weighted_center(intervals: Sequence[ProfileIntervalPx]) -> float | None:
    if not intervals:
        return None
    widths = np.asarray([interval.width for interval in intervals], dtype=np.float64)
    centers = np.asarray([interval.center for interval in intervals], dtype=np.float64)
    if float(widths.sum()) <= 0.0:
        return None
    return float(np.average(centers, weights=widths))


def _row_confidence(rows: Sequence[np.ndarray], confidence: np.ndarray | None) -> float:
    if confidence is None:
        return 1.0
    conf = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
    occupied = np.logical_or.reduce([np.asarray(row, dtype=bool) for row in rows])
    if occupied.shape == conf.shape and occupied.any():
        return float(np.mean(conf[occupied]))
    return float(np.mean(conf)) if conf.size else 0.0


def _sample_rows(height: int, sample_count: int) -> np.ndarray:
    count = max(1, int(sample_count))
    if count == 1:
        return np.array([height // 2], dtype=np.int64)
    return np.linspace(0, height - 1, count).round().astype(np.int64)


def _optional_map(values: Any, shape: tuple[int, int], name: str) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} shape must match mask shape")
    return np.clip(array, 0.0, 1.0)


def _binary_erosion(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(1, int(radius))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_and.reduce(_neighbors(padded, result.shape))
    return result


def _binary_dilation(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(1, int(radius))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_or.reduce(_neighbors(padded, result.shape))
    return result


def _neighbors(padded: np.ndarray, shape: Sequence[int]) -> list[np.ndarray]:
    height, width = int(shape[0]), int(shape[1])
    return [
        padded[dy : dy + height, dx : dx + width]
        for dy in range(3)
        for dx in range(3)
    ]


def _unique_masks(masks: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    seen: set[bytes] = set()
    unique = []
    for mask in masks:
        array = np.asarray(mask, dtype=bool)
        key = array.tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append(array)
    return tuple(unique)


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, float(np.clip(q, 0.0, 1.0)) * 100.0))


def _probability_entropy(values: np.ndarray) -> float:
    clipped = np.clip(values.astype(np.float64, copy=False), 1.0e-9, 1.0 - 1.0e-9)
    entropy = -clipped * np.log2(clipped) - (1.0 - clipped) * np.log2(1.0 - clipped)
    return float(np.mean(entropy)) if entropy.size else 0.0
