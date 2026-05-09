"""Content-adaptive patch selection and fusion for reconstruction refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PatchBox:
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

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) * 0.5, (self.y0 + self.y1) * 0.5)

    def expand(self, margin: int, *, width: int, height: int) -> "PatchBox":
        return PatchBox(
            max(0, self.x0 - margin),
            max(0, self.y0 - margin),
            min(width, self.x1 + margin),
            min(height, self.y1 + margin),
        )

    def intersection_area(self, other: "PatchBox") -> int:
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        return max(0, x1 - x0) * max(0, y1 - y0)

    def iou(self, other: "PatchBox") -> float:
        intersection = self.intersection_area(other)
        union = self.area + other.area - intersection
        return float(intersection / union) if union > 0 else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "center": list(self.center),
        }


@dataclass(frozen=True)
class AdaptivePatch:
    patch_id: str
    box: PatchBox
    crop_box: PatchBox
    score: float
    scale: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "patch_id": self.patch_id,
            "box": self.box.to_dict(),
            "crop_box": self.crop_box.to_dict(),
            "score": self.score,
            "scale": self.scale,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PatchFusionResult:
    prediction: np.ndarray
    contribution_weight: np.ndarray
    uncertainty: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "prediction_shape": list(self.prediction.shape),
            "contribution_weight_shape": list(self.contribution_weight.shape),
            "uncertainty_shape": list(self.uncertainty.shape),
            "metadata": dict(self.metadata),
        }


def score_map_from_signals(
    *,
    boundary: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
    residual: np.ndarray | None = None,
    image: np.ndarray | None = None,
    boundary_weight: float = 1.0,
    uncertainty_weight: float = 0.8,
    residual_weight: float = 1.2,
    edge_weight: float = 0.5,
) -> np.ndarray:
    """Combine boundary, uncertainty, residual, and image-edge signals."""
    shape = _first_shape(boundary, uncertainty, residual, image)
    if shape is None:
        raise ValueError("at least one signal array is required")
    score = np.zeros(shape, dtype=np.float64)
    if boundary is not None:
        score += float(boundary_weight) * _normalize(_as_2d(boundary, shape))
    if uncertainty is not None:
        score += float(uncertainty_weight) * _normalize(_as_2d(uncertainty, shape))
    if residual is not None:
        score += float(residual_weight) * _normalize(np.abs(_as_2d(residual, shape)))
    if image is not None:
        score += float(edge_weight) * _normalize(_edge_magnitude(image, shape))
    return _normalize(score)


def select_adaptive_patches(
    score_map: np.ndarray,
    *,
    patch_sizes: Sequence[int] = (96, 160, 256),
    max_patches: int = 12,
    margin_fraction: float = 0.15,
    min_center_distance_fraction: float = 0.35,
    max_iou: float = 0.35,
    min_score: float = 0.05,
) -> tuple[AdaptivePatch, ...]:
    """Select high-value patches with deterministic NMS over the score map."""
    score = np.asarray(score_map, dtype=np.float64)
    if score.ndim != 2:
        raise ValueError("score_map must be 2D")
    height, width = score.shape
    sizes = tuple(max(1, int(size)) for size in patch_sizes)
    candidates = []
    for size in sizes:
        window = min(size, width, height)
        radius = max(1, int(round(window * min_center_distance_fraction)))
        peaks = _top_peaks(score, max_patches=max_patches * 4, radius=radius, min_score=min_score)
        for y, x, value in peaks:
            half = window // 2
            box = PatchBox(
                max(0, x - half),
                max(0, y - half),
                min(width, x - half + window),
                min(height, y - half + window),
            )
            box = _shift_box_inside(box, width=width, height=height)
            margin = int(round(window * margin_fraction))
            crop_box = box.expand(margin, width=width, height=height)
            candidates.append((float(value), window / float(min(width, height)), box, crop_box))
    candidates.sort(key=lambda item: (-item[0], item[2].y0, item[2].x0, item[2].area))
    selected: list[AdaptivePatch] = []
    for value, scale, box, crop_box in candidates:
        if len(selected) >= max_patches:
            break
        if any(box.iou(existing.box) > max_iou for existing in selected):
            continue
        selected.append(
            AdaptivePatch(
                patch_id=f"patch_{len(selected):03d}",
                box=box,
                crop_box=crop_box,
                score=value,
                scale=scale,
                metadata={
                    "margin_px": max(0, crop_box.width - box.width) // 2,
                    "score_source": "content_adaptive_nms",
                },
            )
        )
    return tuple(selected)


def fuse_patch_predictions(
    global_prediction: np.ndarray,
    patches: Sequence[AdaptivePatch],
    patch_predictions: Mapping[str, np.ndarray],
    *,
    align_mean: bool = True,
    feather_fraction: float = 0.12,
) -> PatchFusionResult:
    """Fuse patch predictions into a global prediction with feathered weights."""
    global_array = np.asarray(global_prediction, dtype=np.float64)
    if global_array.ndim != 2:
        raise ValueError("global_prediction must be 2D")
    accum = np.zeros_like(global_array, dtype=np.float64)
    weight_sum = np.zeros_like(global_array, dtype=np.float64)
    stack_values: list[np.ndarray] = [global_array]
    used = []
    for patch in patches:
        raw = patch_predictions.get(patch.patch_id)
        if raw is None:
            continue
        patch_array = _resize_nearest(np.asarray(raw, dtype=np.float64), patch.crop_box.height, patch.crop_box.width)
        y0, y1 = patch.crop_box.y0, patch.crop_box.y1
        x0, x1 = patch.crop_box.x0, patch.crop_box.x1
        global_crop = global_array[y0:y1, x0:x1]
        if patch_array.shape != global_crop.shape:
            patch_array = _resize_nearest(patch_array, global_crop.shape[0], global_crop.shape[1])
        if align_mean and patch_array.size and global_crop.size:
            patch_array = patch_array + (float(np.mean(global_crop)) - float(np.mean(patch_array)))
        weights = _feather_weights(patch_array.shape, feather_fraction=feather_fraction)
        accum[y0:y1, x0:x1] += patch_array * weights
        weight_sum[y0:y1, x0:x1] += weights
        full = np.full_like(global_array, np.nan, dtype=np.float64)
        full[y0:y1, x0:x1] = patch_array
        stack_values.append(full)
        used.append(patch.patch_id)
    fused = np.where(weight_sum > 0.0, accum / np.maximum(weight_sum, 1e-12), global_array)
    uncertainty = _patch_disagreement(stack_values, fallback=np.zeros_like(global_array))
    return PatchFusionResult(
        prediction=fused,
        contribution_weight=weight_sum,
        uncertainty=uncertainty,
        metadata={
            "patch_count": len(patches),
            "used_patch_count": len(used),
            "used_patch_ids": used,
            "align_mean": align_mean,
            "feather_fraction": feather_fraction,
        },
    )


def _first_shape(*arrays: np.ndarray | None) -> tuple[int, int] | None:
    for array in arrays:
        if array is None:
            continue
        values = np.asarray(array)
        if values.ndim >= 2:
            return int(values.shape[0]), int(values.shape[1])
    return None


def _as_2d(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 3:
        values = np.mean(values[..., : min(values.shape[2], 3)], axis=2)
    if values.ndim != 2:
        raise ValueError("signal arrays must be 2D or image-like 3D")
    if values.shape != shape:
        values = _resize_nearest(values, shape[0], shape[1])
    return values


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float64)
    min_value = float(np.min(array[finite]))
    max_value = float(np.max(array[finite]))
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.float64)
    out = (array - min_value) / (max_value - min_value)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0)


def _edge_magnitude(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    gray = _as_2d(image, shape)
    dy = np.zeros_like(gray)
    dx = np.zeros_like(gray)
    dy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    dx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    return np.sqrt(dx * dx + dy * dy)


def _top_peaks(
    score: np.ndarray,
    *,
    max_patches: int,
    radius: int,
    min_score: float,
) -> list[tuple[int, int, float]]:
    work = np.array(score, dtype=np.float64, copy=True)
    peaks: list[tuple[int, int, float]] = []
    for _ in range(max(1, max_patches)):
        index = int(np.argmax(work))
        value = float(work.flat[index])
        if value < min_score:
            break
        y, x = np.unravel_index(index, work.shape)
        peaks.append((int(y), int(x), value))
        y0 = max(0, int(y) - radius)
        y1 = min(work.shape[0], int(y) + radius + 1)
        x0 = max(0, int(x) - radius)
        x1 = min(work.shape[1], int(x) + radius + 1)
        work[y0:y1, x0:x1] = -np.inf
    return peaks


def _shift_box_inside(box: PatchBox, *, width: int, height: int) -> PatchBox:
    x0, y0, x1, y1 = box.x0, box.y0, box.x1, box.y1
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return PatchBox(max(0, x0), max(0, y0), min(width, x1), min(height, y1))


def _resize_nearest(values: np.ndarray, height: int, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (height, width):
        return array
    if height <= 0 or width <= 0:
        return np.empty((max(0, height), max(0, width)), dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        return np.zeros((height, width), dtype=np.float64)
    y_idx = np.linspace(0, max(0, array.shape[0] - 1), height).round().astype(int)
    x_idx = np.linspace(0, max(0, array.shape[1] - 1), width).round().astype(int)
    return array[np.ix_(y_idx, x_idx)]


def _feather_weights(
    shape: tuple[int, int],
    *,
    feather_fraction: float,
) -> np.ndarray:
    height, width = shape
    if height <= 0 or width <= 0:
        return np.empty(shape, dtype=np.float64)
    yy = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float64)
    xx = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float64)
    y_width = max(1.0, height * max(0.0, feather_fraction))
    x_width = max(1.0, width * max(0.0, feather_fraction))
    wy = np.clip((yy + 1.0) / y_width, 0.0, 1.0)
    wx = np.clip((xx + 1.0) / x_width, 0.0, 1.0)
    return np.outer(wy, wx)


def _patch_disagreement(
    layers: Sequence[np.ndarray],
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    stack = np.stack(layers, axis=0)
    valid = np.isfinite(stack)
    counts = np.sum(valid, axis=0)
    filled = np.where(valid, stack, 0.0)
    mean = np.sum(filled, axis=0) / np.maximum(counts, 1)
    variance = np.sum(np.where(valid, (stack - mean) ** 2, 0.0), axis=0) / np.maximum(counts, 1)
    return np.where(counts > 1, np.sqrt(variance), fallback)
