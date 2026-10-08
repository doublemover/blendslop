"""Content-adaptive patch selection and fusion for reconstruction refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from metrics.silhouette import silhouette_metric_result
except ModuleNotFoundError:  # pragma: no cover - package entrypoint import path
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from metrics.silhouette import silhouette_metric_result


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


@dataclass(frozen=True)
class ContentAdaptiveRefinementResult:
    prediction: np.ndarray
    refined_mask: np.ndarray
    patches: tuple[AdaptivePatch, ...]
    fusion: PatchFusionResult
    metric_before: Mapping[str, Any]
    metric_after: Mapping[str, Any]
    accepted: bool
    improvement: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "prediction_shape": list(self.prediction.shape),
            "refined_mask_shape": list(self.refined_mask.shape),
            "patches": [patch.to_dict() for patch in self.patches],
            "fusion": self.fusion.to_dict(),
            "metric_before": dict(self.metric_before),
            "metric_after": dict(self.metric_after),
            "accepted": self.accepted,
            "improvement": dict(self.improvement),
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


def run_content_adaptive_patch_refinement(
    reference_mask: np.ndarray,
    *,
    candidate_prediction: np.ndarray | None = None,
    candidate_mask: np.ndarray | None = None,
    image: np.ndarray | None = None,
    patch_sizes: Sequence[int] = (96, 160, 256),
    max_patches: int = 12,
    threshold: float = 0.5,
    correction_strength: float = 1.0,
    min_area_iou_delta: float = 0.01,
    min_boundary_iou_delta: float = 0.0,
    alignment_mode: str | None = "none",
    feather_fraction: float = 0.12,
) -> ContentAdaptiveRefinementResult:
    """Run a deterministic residual-patch correction pass over a silhouette field.

    The pass is deliberately pure Python and image-space.  Backends can use it
    as a measured refinement probe before spending Blender/optimization time:
    select high-residual local crops, apply local residual corrections, fuse
    them with feathered weights, and report whether the refined silhouette
    actually improved.
    """
    reference = _as_probability(reference_mask)
    if candidate_prediction is None:
        if candidate_mask is None:
            raise ValueError("candidate_prediction or candidate_mask is required")
        candidate = _as_probability(candidate_mask)
    else:
        candidate = _as_probability(candidate_prediction)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate predictions must have matching shapes")

    candidate_bool = candidate >= float(threshold)
    reference_bool = reference >= float(threshold)
    residual = reference - candidate
    boundary = np.logical_xor(reference_bool, candidate_bool).astype(float)
    score = score_map_from_signals(
        boundary=boundary,
        residual=residual,
        image=image,
        boundary_weight=1.0,
        uncertainty_weight=0.0,
        residual_weight=1.4,
        edge_weight=0.35,
    )
    patches = select_adaptive_patches(
        score,
        patch_sizes=patch_sizes,
        max_patches=max_patches,
        min_score=0.01,
    )
    patch_predictions: dict[str, np.ndarray] = {}
    for patch in patches:
        y0, y1 = patch.crop_box.y0, patch.crop_box.y1
        x0, x1 = patch.crop_box.x0, patch.crop_box.x1
        crop = candidate[y0:y1, x0:x1]
        correction = residual[y0:y1, x0:x1] * float(correction_strength)
        patch_predictions[patch.patch_id] = np.clip(crop + correction, 0.0, 1.0)
    fusion = fuse_patch_predictions(
        candidate,
        patches,
        patch_predictions,
        align_mean=False,
        alignment_mode=alignment_mode,
        feather_fraction=feather_fraction,
        edge_weight_map=score,
        edge_weight_strength=0.5,
    )
    refined_prediction = np.clip(fusion.prediction, 0.0, 1.0)
    refined_mask = refined_prediction >= float(threshold)
    before = silhouette_metric_result(
        reference_bool,
        candidate_bool,
        view="content_adaptive",
    ).to_dict()
    after = silhouette_metric_result(
        reference_bool,
        refined_mask,
        view="content_adaptive",
    ).to_dict()
    improvement = {
        "area_iou_delta": _metric_delta(after, before, "area_iou"),
        "boundary_iou_delta": _metric_delta(after, before, "boundary_iou"),
        "signed_distance_loss_delta": _metric_delta(
            after,
            before,
            "signed_distance_loss",
        ),
    }
    accepted = bool(
        improvement["area_iou_delta"] >= float(min_area_iou_delta)
        and improvement["boundary_iou_delta"] >= float(min_boundary_iou_delta)
        and improvement["signed_distance_loss_delta"] <= 0.0
    )
    return ContentAdaptiveRefinementResult(
        prediction=refined_prediction,
        refined_mask=refined_mask,
        patches=patches,
        fusion=fusion,
        metric_before=before,
        metric_after=after,
        accepted=accepted,
        improvement=improvement,
        metadata={
            "threshold": float(threshold),
            "correction_strength": float(correction_strength),
            "patch_prediction_count": len(patch_predictions),
            "score_map_max": float(np.max(score)) if score.size else 0.0,
        },
    )


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
        peaks = _top_peaks(
            score, max_patches=max_patches * 4, radius=radius, min_score=min_score
        )
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
            candidates.append(
                (float(value), window / float(min(width, height)), box, crop_box)
            )
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
    alignment_mode: str | None = None,
    feather_fraction: float = 0.12,
    edge_weight_map: np.ndarray | None = None,
    edge_weight_strength: float = 0.0,
    min_alignment_pixels: int = 8,
) -> PatchFusionResult:
    """Fuse patch predictions into a global prediction with feathered weights."""
    global_array = np.asarray(global_prediction, dtype=np.float64)
    if global_array.ndim != 2:
        raise ValueError("global_prediction must be 2D")
    resolved_alignment = _resolve_alignment_mode(
        alignment_mode,
        align_mean=align_mean,
    )
    edge_weights = None
    if edge_weight_map is not None:
        edge_weights = _normalize(_as_2d(edge_weight_map, global_array.shape))
    accum = np.zeros_like(global_array, dtype=np.float64)
    weight_sum = np.zeros_like(global_array, dtype=np.float64)
    stack_values: list[np.ndarray] = [global_array]
    used = []
    alignments = []
    for patch in patches:
        raw = patch_predictions.get(patch.patch_id)
        if raw is None:
            continue
        patch_array = _resize_nearest(
            np.asarray(raw, dtype=np.float64),
            patch.crop_box.height,
            patch.crop_box.width,
        )
        y0, y1 = patch.crop_box.y0, patch.crop_box.y1
        x0, x1 = patch.crop_box.x0, patch.crop_box.x1
        global_crop = global_array[y0:y1, x0:x1]
        if patch_array.shape != global_crop.shape:
            patch_array = _resize_nearest(
                patch_array, global_crop.shape[0], global_crop.shape[1]
            )
        patch_array, alignment = _align_patch_prediction(
            patch_array,
            global_crop,
            mode=resolved_alignment,
            min_pixels=min_alignment_pixels,
        )
        weights = _feather_weights(patch_array.shape, feather_fraction=feather_fraction)
        if edge_weights is not None and edge_weight_strength > 0.0:
            edge_crop = edge_weights[y0:y1, x0:x1]
            if edge_crop.shape != weights.shape:
                edge_crop = _resize_nearest(
                    edge_crop, weights.shape[0], weights.shape[1]
                )
            weights = weights * (1.0 + float(edge_weight_strength) * edge_crop)
        accum[y0:y1, x0:x1] += patch_array * weights
        weight_sum[y0:y1, x0:x1] += weights
        full = np.full_like(global_array, np.nan, dtype=np.float64)
        full[y0:y1, x0:x1] = patch_array
        stack_values.append(full)
        used.append(patch.patch_id)
        alignments.append({"patch_id": patch.patch_id, **alignment})
    fused = np.where(
        weight_sum > 0.0, accum / np.maximum(weight_sum, 1e-12), global_array
    )
    uncertainty = _patch_disagreement(
        stack_values, fallback=np.zeros_like(global_array)
    )
    return PatchFusionResult(
        prediction=fused,
        contribution_weight=weight_sum,
        uncertainty=uncertainty,
        metadata={
            "patch_count": len(patches),
            "used_patch_count": len(used),
            "used_patch_ids": used,
            "alignment_mode": resolved_alignment,
            "alignments": alignments,
            "align_mean": align_mean,
            "feather_fraction": feather_fraction,
            "edge_weight_strength": float(edge_weight_strength),
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


def _as_probability(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask, dtype=np.float64)
    if array.ndim == 3:
        array = np.mean(array[..., : min(array.shape[2], 3)], axis=2)
    if array.ndim != 2:
        raise ValueError("mask/prediction arrays must be 2D or image-like 3D")
    max_value = float(np.max(array)) if array.size else 0.0
    if max_value > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _metric_delta(
    after: Mapping[str, Any],
    before: Mapping[str, Any],
    key: str,
) -> float:
    return _float(after.get(key)) - _float(before.get(key))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _resolve_alignment_mode(
    alignment_mode: str | None,
    *,
    align_mean: bool,
) -> str:
    if alignment_mode is None:
        return "mean" if align_mean else "none"
    mode = str(alignment_mode).strip().lower()
    if mode == "shift":
        mode = "mean"
    if mode not in {"none", "mean", "affine"}:
        raise ValueError("alignment_mode must be none, mean, or affine")
    return mode


def _align_patch_prediction(
    patch: np.ndarray,
    reference: np.ndarray,
    *,
    mode: str,
    min_pixels: int,
) -> tuple[np.ndarray, dict[str, object]]:
    patch_array = np.asarray(patch, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if mode == "none" or patch_array.size == 0 or reference_array.size == 0:
        return patch_array, {"mode": mode, "scale": 1.0, "shift": 0.0, "pixels": 0}
    valid = np.isfinite(patch_array) & np.isfinite(reference_array)
    pixels = int(np.sum(valid))
    if pixels < max(1, int(min_pixels)):
        return patch_array, {
            "mode": mode,
            "scale": 1.0,
            "shift": 0.0,
            "pixels": pixels,
            "status": "insufficient_overlap",
        }
    x = patch_array[valid].reshape(-1)
    y = reference_array[valid].reshape(-1)
    if mode == "mean":
        shift = float(np.mean(y) - np.mean(x))
        return patch_array + shift, {
            "mode": mode,
            "scale": 1.0,
            "shift": shift,
            "pixels": pixels,
            "status": "aligned",
        }
    variance = float(np.var(x))
    if variance <= 1e-12:
        shift = float(np.mean(y) - np.mean(x))
        return patch_array + shift, {
            "mode": "mean",
            "requested_mode": mode,
            "scale": 1.0,
            "shift": shift,
            "pixels": pixels,
            "status": "affine_degenerate_mean_fallback",
        }
    design = np.stack([x, np.ones_like(x)], axis=1)
    try:
        scale, shift = np.linalg.lstsq(design, y, rcond=None)[0]
    except Exception:
        shift = float(np.mean(y) - np.mean(x))
        return patch_array + shift, {
            "mode": "mean",
            "requested_mode": mode,
            "scale": 1.0,
            "shift": shift,
            "pixels": pixels,
            "status": "affine_solve_failed_mean_fallback",
        }
    scale = float(scale)
    shift = float(shift)
    if not np.isfinite(scale) or not np.isfinite(shift) or abs(scale) > 10.0:
        shift = float(np.mean(y) - np.mean(x))
        return patch_array + shift, {
            "mode": "mean",
            "requested_mode": mode,
            "scale": 1.0,
            "shift": shift,
            "pixels": pixels,
            "status": "affine_unstable_mean_fallback",
        }
    return patch_array * scale + shift, {
        "mode": mode,
        "scale": scale,
        "shift": shift,
        "pixels": pixels,
        "status": "aligned",
    }


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
    variance = np.sum(np.where(valid, (stack - mean) ** 2, 0.0), axis=0) / np.maximum(
        counts, 1
    )
    return np.where(counts > 1, np.sqrt(variance), fallback)
