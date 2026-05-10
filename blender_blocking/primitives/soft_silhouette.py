"""
CPU projected ellipse and Gaussian soft silhouette renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

try:
    from .analytic_primitives import AnisotropicGaussianPrimitive, EllipsoidPrimitive
except ImportError:  # pragma: no cover - supports direct script execution.
    from analytic_primitives import AnisotropicGaussianPrimitive, EllipsoidPrimitive


@dataclass(frozen=True)
class OrthographicCamera:
    """Minimal orthographic camera for CPU silhouette previews."""

    name: str = "front"
    axes: Tuple[int, int] = (0, 2)
    image_size: Tuple[int, int] = (128, 128)
    world_bounds: Tuple[float, float, float, float] = (-1.5, 1.5, -1.5, 1.5)

    def __post_init__(self) -> None:
        width, height = self.image_size
        if not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer)):
            raise TypeError("image_size values must be integers")
        if width <= 0 or height <= 0:
            raise ValueError("image_size values must be positive")
        bounds = np.asarray(self.world_bounds, dtype=np.float64)
        if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
            raise ValueError("world_bounds must be four finite values")
        if bounds[0] == bounds[1] or bounds[2] == bounds[3]:
            raise ValueError("world_bounds extents must be non-zero")
        if bounds[0] > bounds[1] or bounds[2] > bounds[3]:
            raise ValueError("world_bounds must be (min_x, max_x, min_y, max_y)")
        axes = self.axes
        if len(axes) != 2 or len(set(axes)) != 2:
            raise ValueError("axes must contain two distinct indices")
        if any(axis not in (0, 1, 2) for axis in axes):
            raise ValueError("axes must be a 2-tuple of values 0, 1, or 2")

    @classmethod
    def from_view(
        cls,
        view: str,
        image_size: Tuple[int, int] = (128, 128),
        world_bounds: Tuple[float, float, float, float] = (-1.5, 1.5, -1.5, 1.5),
    ) -> "OrthographicCamera":
        view_map = {
            "front": (0, 2),
            "side": (1, 2),
            "top": (0, 1),
        }
        if view not in view_map:
            raise ValueError(f"unknown orthographic view: {view}")
        return cls(name=view, axes=view_map[view], image_size=image_size, world_bounds=world_bounds)


def camera_grid(camera: OrthographicCamera) -> Tuple[np.ndarray, np.ndarray]:
    width, height = camera.image_size
    xmin, xmax, ymin, ymax = camera.world_bounds
    xs = np.linspace(xmin, xmax, width, dtype=np.float64)
    ys = np.linspace(ymax, ymin, height, dtype=np.float64)
    return np.meshgrid(xs, ys)


def primitive_projected_mean_cov(
    primitive: object,
    camera: OrthographicCamera,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Project an ellipsoid or Gaussian primitive into camera image axes."""
    def _to_float_matrix2(values: object) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape != (2, 2) and matrix.shape != (3, 3):
            raise ValueError(f"primitive covariance must be 2x2 or 3x3, got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("primitive covariance contains non-finite values")
        return matrix

    axes = list(camera.axes)
    if isinstance(primitive, EllipsoidPrimitive):
        center = primitive.center[axes]
        cov = _to_float_matrix2(primitive.covariance())[np.ix_(axes, axes)]
        opacity = float(np.clip(primitive.density * primitive.confidence, 0.0, 1.0))
        if not np.all(np.isfinite(center)):
            raise ValueError("primitive center contains non-finite values")
        return center, cov, opacity
    if isinstance(primitive, AnisotropicGaussianPrimitive):
        center = primitive.center[axes]
        cov = _to_float_matrix2(primitive.covariance)[np.ix_(axes, axes)]
        opacity = float(np.clip(primitive.opacity * primitive.confidence, 0.0, 1.0))
        if not np.all(np.isfinite(center)):
            raise ValueError("primitive center contains non-finite values")
        return center, cov, opacity

    if hasattr(primitive, "center") and hasattr(primitive, "radii"):
        center3 = np.asarray(getattr(primitive, "center"), dtype=np.float64)
        radii = np.asarray(getattr(primitive, "radii"), dtype=np.float64)
        rotation = np.asarray(getattr(primitive, "rotation", np.eye(3)), dtype=np.float64)
        if center3.shape != (3,):
            raise ValueError(
                f"unsupported center dimension for projected renderer: {center3.shape}"
            )
        if radii.shape != (3,):
            raise ValueError(f"unsupported radii dimension for projected renderer: {radii.shape}")
        cov3 = rotation @ np.diag(radii * radii) @ rotation.T
        cov3 = _to_float_matrix2(cov3)
        density = float(getattr(primitive, "density", 1.0))
        confidence = float(getattr(primitive, "confidence", 1.0))
        if not np.all(np.isfinite(density)) or not np.all(np.isfinite(confidence)):
            raise ValueError("primitive density/confidence contains non-finite values")
        return center3[axes], cov3[np.ix_(axes, axes)], np.clip(density * confidence, 0.0, 1.0)

    raise TypeError(f"unsupported primitive for projected renderer: {type(primitive)!r}")


def render_projected_soft_silhouette(
    primitives: Sequence[object],
    camera: OrthographicCamera,
    softness: float = 24.0,
    min_variance: float = 1e-6,
) -> np.ndarray:
    """
    Render a soft occupancy silhouette by alpha-compositing projected ellipses.

    The implicit ellipse footprint is q <= 1 where q is projected Mahalanobis
    distance. Values are composited as occupancy probabilities.
    """
    softness = float(softness)
    min_variance = float(min_variance)
    if not np.isfinite(softness) or softness <= 0.0:
        raise ValueError("softness must be a finite positive number")
    if not np.isfinite(min_variance) or min_variance <= 0.0:
        raise ValueError("min_variance must be a finite positive number")

    xx, yy = camera_grid(camera)
    pixels = np.stack([xx, yy], axis=2)
    occupancy = np.zeros(camera.image_size[::-1], dtype=np.float64)

    for primitive in primitives:
        mean, cov, opacity = primitive_projected_mean_cov(primitive, camera)
        if opacity <= 0.0:
            continue
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, min_variance)
        inv_cov = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        delta = pixels - mean[None, None, :]
        q = np.einsum("hwi,ij,hwj->hw", delta, inv_cov, delta)
        alpha = opacity / (1.0 + np.exp(np.clip(softness * (q - 1.0), -60.0, 60.0)))
        occupancy = 1.0 - (1.0 - occupancy) * (1.0 - alpha)

    return np.clip(occupancy, 0.0, 1.0)


def render_multi_view_soft_silhouettes(
    primitives: Sequence[object],
    cameras: Sequence[OrthographicCamera] | None = None,
    image_size: Tuple[int, int] = (128, 128),
    world_bounds: Tuple[float, float, float, float] = (-1.5, 1.5, -1.5, 1.5),
    softness: float = 24.0,
) -> Dict[str, np.ndarray]:
    """Render front/side/top masks unless cameras are supplied."""
    if cameras is None:
        cameras = (
            OrthographicCamera.from_view("front", image_size, world_bounds),
            OrthographicCamera.from_view("side", image_size, world_bounds),
            OrthographicCamera.from_view("top", image_size, world_bounds),
        )
    return {
        camera.name: render_projected_soft_silhouette(primitives, camera, softness)
        for camera in cameras
    }


def soft_mask_metrics(predicted: np.ndarray, target: np.ndarray) -> Mapping[str, float]:
    """Return soft silhouette losses plus boundary/SDF diagnostics."""
    pred = np.asarray(predicted, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if pred.shape != tgt.shape:
        raise ValueError("predicted and target masks must have the same shape")
    pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    tgt = np.nan_to_num(tgt, nan=0.0, posinf=1.0, neginf=0.0)
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(tgt)):
        raise ValueError("predicted and target masks must be finite")
    pred = np.clip(pred, 0.0, 1.0)
    tgt = np.clip(tgt, 0.0, 1.0)
    intersection = float(np.minimum(pred, tgt).sum())
    union = float(np.maximum(pred, tgt).sum())
    pixel_count = float(pred.size)
    if pixel_count == 0.0:
        return {
            "soft_l1": 0.0,
            "soft_l2": 0.0,
            "soft_iou_loss": 0.0,
            "area_iou_loss": 0.0,
            "pred_area_ratio": 0.0,
            "target_area_ratio": 0.0,
            "area_abs_diff_ratio": 0.0,
        }
    hard_tgt = tgt >= 0.5
    hard_pred, hard_threshold = _adaptive_hard_prediction(pred, hard_tgt)
    hard_union = float(np.logical_or(hard_pred, hard_tgt).sum())
    hard_iou = (
        float(np.logical_and(hard_pred, hard_tgt).sum()) / hard_union
        if hard_union > 0.0
        else 1.0
    )
    soft_iou = intersection / union if union > 0.0 else 1.0
    boundary_score = _boundary_iou(hard_pred, hard_tgt)
    sdf_loss = _signed_distance_loss(hard_tgt, hard_pred)
    return {
        "soft_l1": float(np.mean(np.abs(pred - tgt))),
        "soft_l2": float(np.mean((pred - tgt) ** 2)),
        "soft_iou_loss": 1.0 - soft_iou,
        "area_iou_loss": 1.0 - hard_iou,
        "boundary_iou": boundary_score,
        "boundary_iou_loss": 1.0 - boundary_score,
        "signed_distance_loss": sdf_loss,
        "pred_area_ratio": float(pred.mean()),
        "target_area_ratio": float(tgt.mean()),
        "area_abs_diff_ratio": abs(float(pred.sum() - tgt.sum())) / pixel_count,
        "pred_boundary_area_ratio": float(_boundary_band(hard_pred).mean()),
        "target_boundary_area_ratio": float(_boundary_band(hard_tgt).mean()),
        "pred_hard_threshold": float(hard_threshold),
        "pred_hard_area_ratio": float(hard_pred.mean()),
        "target_hard_area_ratio": float(hard_tgt.mean()),
        "hard_intersection_ratio": float(np.logical_and(hard_pred, hard_tgt).sum())
        / pixel_count,
        "hard_union_ratio": hard_union / pixel_count,
        **_mask_geometry_metrics("pred", hard_pred),
        **_mask_geometry_metrics("target", hard_tgt),
    }


def _adaptive_hard_prediction(
    predicted: np.ndarray,
    target_hard: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Harden a soft mask for diagnostics without erasing low-opacity overlap."""
    pred = np.asarray(predicted, dtype=np.float64)
    default_threshold = 0.5
    hard = pred >= default_threshold
    if hard.any() or not np.asarray(target_hard, dtype=bool).any():
        return hard, default_threshold
    positive = pred[pred > 0.0]
    if positive.size == 0:
        return hard, default_threshold
    target_count = int(np.count_nonzero(target_hard))
    if target_count <= 0:
        return hard, default_threshold
    flat = pred.ravel()
    count = max(1, min(int(target_count), flat.size))
    kth = flat.size - count
    threshold = float(np.partition(flat, kth)[kth])
    if threshold <= 0.0 or not np.isfinite(threshold):
        threshold = float(np.min(positive))
    threshold = min(default_threshold, max(threshold, float(np.min(positive))))
    adaptive = pred >= threshold
    if adaptive.any():
        return adaptive, threshold
    return hard, default_threshold


def _mask_geometry_metrics(prefix: str, mask: np.ndarray) -> Mapping[str, float]:
    hard = np.asarray(mask, dtype=bool)
    if hard.size == 0 or not hard.any():
        return {
            f"{prefix}_bbox_x0": -1.0,
            f"{prefix}_bbox_y0": -1.0,
            f"{prefix}_bbox_x1": -1.0,
            f"{prefix}_bbox_y1": -1.0,
            f"{prefix}_centroid_x": -1.0,
            f"{prefix}_centroid_y": -1.0,
        }
    ys, xs = np.nonzero(hard)
    return {
        f"{prefix}_bbox_x0": float(np.min(xs)),
        f"{prefix}_bbox_y0": float(np.min(ys)),
        f"{prefix}_bbox_x1": float(np.max(xs)),
        f"{prefix}_bbox_y1": float(np.max(ys)),
        f"{prefix}_centroid_x": float(np.mean(xs)),
        f"{prefix}_centroid_y": float(np.mean(ys)),
    }


def _boundary_iou(mask_a: np.ndarray, mask_b: np.ndarray, radius: int = 2) -> float:
    """Boundary IoU for hard masks without requiring OpenCV at render time."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("boundary masks must have matching shapes")
    if not a.any() and not b.any():
        return 1.0
    band_a = _boundary_band(a, radius=radius)
    band_b = _boundary_band(b, radius=radius)
    union = float(np.logical_or(band_a, band_b).sum())
    if union <= 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.logical_and(band_a, band_b).sum() / union)


def _boundary_band(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    hard = np.asarray(mask, dtype=bool)
    if not hard.any():
        return np.zeros(hard.shape, dtype=bool)
    radius = max(1, int(radius))
    eroded = _binary_erosion(hard, radius=1)
    boundary = np.logical_xor(hard, eroded)
    return _binary_dilation(boundary, radius=radius)


def _binary_erosion(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(1, int(radius))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighbors = [
            padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ]
        result = np.logical_and.reduce(neighbors)
    return result


def _binary_dilation(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(1, int(radius))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighbors = [
            padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ]
        result = np.logical_or.reduce(neighbors)
    return result


def _signed_distance_loss(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=bool)
    cand = np.asarray(candidate, dtype=bool)
    if ref.shape != cand.shape:
        raise ValueError("signed-distance masks must have matching shapes")
    if ref.size == 0:
        return 0.0
    ref_sdf = _signed_distance(ref)
    cand_sdf = _signed_distance(cand)
    denom = float(max(ref.shape)) if ref.shape else 1.0
    return float(np.mean(np.abs(ref_sdf - cand_sdf)) / max(denom, 1.0))


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    hard = np.asarray(mask, dtype=bool)
    try:
        from scipy import ndimage
    except Exception:
        inside = _nearest_distance(~hard)
        outside = _nearest_distance(hard)
    else:
        inside = ndimage.distance_transform_edt(hard)
        outside = ndimage.distance_transform_edt(~hard)
    return inside.astype(np.float64, copy=False) - outside.astype(
        np.float64,
        copy=False,
    )


def _nearest_distance(target: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
    target_bool = np.asarray(target, dtype=bool)
    coords = np.argwhere(target_bool)
    output = np.zeros(target_bool.shape, dtype=np.float64)
    if not len(coords):
        fill = float(max(target_bool.shape) if target_bool.shape else 0.0)
        output.fill(fill)
        return output
    points = np.argwhere(np.ones(target_bool.shape, dtype=bool))
    distances = np.empty((len(points),), dtype=np.float64)
    coords_f = coords.astype(np.float64, copy=False)
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        block = points[start:end].astype(np.float64, copy=False)
        delta = block[:, None, :] - coords_f[None, :, :]
        distances[start:end] = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
    output[tuple(points.T)] = distances
    return output
