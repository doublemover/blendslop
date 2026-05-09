"""Surface and volume comparison metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class SurfaceDistanceReport:
    """Nearest-neighbor surface distance summary."""

    chamfer_l1: float
    chamfer_l2: float
    hausdorff: float
    forward_mean: float
    backward_mean: float
    sample_count_a: int
    sample_count_b: int

    def to_dict(self) -> dict[str, object]:
        return {
            "chamfer_l1": self.chamfer_l1,
            "chamfer_l2": self.chamfer_l2,
            "hausdorff": self.hausdorff,
            "forward_mean": self.forward_mean,
            "backward_mean": self.backward_mean,
            "sample_count_a": self.sample_count_a,
            "sample_count_b": self.sample_count_b,
        }


@dataclass(frozen=True)
class VolumeOverlapReport:
    """Boolean volume overlap metrics."""

    intersection: int
    union: int
    volume_a: int
    volume_b: int
    iou: float
    dice: float
    false_negative_rate: float
    false_positive_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "intersection": self.intersection,
            "union": self.union,
            "volume_a": self.volume_a,
            "volume_b": self.volume_b,
            "iou": self.iou,
            "dice": self.dice,
            "false_negative_rate": self.false_negative_rate,
            "false_positive_rate": self.false_positive_rate,
        }


def bounded_sample_points(points: np.ndarray, limit: Optional[int]) -> np.ndarray:
    """Deterministically subsample point rows without random state."""
    arr = _points(points)
    if limit is None or limit <= 0 or len(arr) <= limit:
        return arr.copy()
    indices = np.linspace(0, len(arr) - 1, int(limit)).round().astype(np.int64)
    return arr[indices]


def chamfer_distance(
    points_a: np.ndarray | Sequence[Sequence[float]],
    points_b: np.ndarray | Sequence[Sequence[float]],
    *,
    max_points: Optional[int] = 4096,
) -> SurfaceDistanceReport:
    """Compute symmetric nearest-neighbor distances for two point clouds."""
    a = bounded_sample_points(np.asarray(points_a, dtype=float), max_points)
    b = bounded_sample_points(np.asarray(points_b, dtype=float), max_points)
    if len(a) == 0 or len(b) == 0:
        return SurfaceDistanceReport(
            chamfer_l1=float("inf"),
            chamfer_l2=float("inf"),
            hausdorff=float("inf"),
            forward_mean=float("inf"),
            backward_mean=float("inf"),
            sample_count_a=int(len(a)),
            sample_count_b=int(len(b)),
        )

    forward = _nearest_distances(a, b)
    backward = _nearest_distances(b, a)
    forward_mean = float(np.mean(forward))
    backward_mean = float(np.mean(backward))
    chamfer_l1 = forward_mean + backward_mean
    chamfer_l2 = float(np.mean(forward * forward) + np.mean(backward * backward))
    hausdorff = float(max(np.max(forward), np.max(backward)))
    return SurfaceDistanceReport(
        chamfer_l1=chamfer_l1,
        chamfer_l2=chamfer_l2,
        hausdorff=hausdorff,
        forward_mean=forward_mean,
        backward_mean=backward_mean,
        sample_count_a=int(len(a)),
        sample_count_b=int(len(b)),
    )


def volume_overlap(
    volume_a: np.ndarray,
    volume_b: np.ndarray,
    *,
    threshold_a: float = 0.5,
    threshold_b: float = 0.5,
) -> VolumeOverlapReport:
    """Compute IoU/Dice and asymmetric error rates for aligned volumes."""
    a = np.asarray(volume_a)
    b = np.asarray(volume_b)
    if a.shape != b.shape:
        raise ValueError("volumes must have matching shapes")
    a_bool = a > threshold_a if a.dtype != np.dtype(bool) else a.astype(bool)
    b_bool = b > threshold_b if b.dtype != np.dtype(bool) else b.astype(bool)
    intersection = int(np.logical_and(a_bool, b_bool).sum())
    union = int(np.logical_or(a_bool, b_bool).sum())
    volume_a_count = int(a_bool.sum())
    volume_b_count = int(b_bool.sum())
    iou = float(intersection / union) if union else 0.0
    denom = volume_a_count + volume_b_count
    dice = float(2 * intersection / denom) if denom else 0.0
    fn = int(np.logical_and(a_bool, np.logical_not(b_bool)).sum())
    fp = int(np.logical_and(np.logical_not(a_bool), b_bool).sum())
    false_negative_rate = float(fn / volume_a_count) if volume_a_count else 0.0
    false_positive_rate = float(fp / max(1, volume_b_count))
    return VolumeOverlapReport(
        intersection=intersection,
        union=union,
        volume_a=volume_a_count,
        volume_b=volume_b_count,
        iou=iou,
        dice=dice,
        false_negative_rate=false_negative_rate,
        false_positive_rate=false_positive_rate,
    )


def surface_score(report: SurfaceDistanceReport, budgets: Mapping[str, float]) -> float:
    """Convert a surface report into a pass/fail-like score in [0, 1]."""
    chamfer_budget = float(budgets.get("chamfer_l1_max", 1.0))
    hausdorff_budget = float(budgets.get("hausdorff_max", chamfer_budget * 4.0))
    chamfer_term = 1.0 - min(1.0, report.chamfer_l1 / max(chamfer_budget, 1e-12))
    hausdorff_term = 1.0 - min(1.0, report.hausdorff / max(hausdorff_budget, 1e-12))
    return float(max(0.0, 0.7 * chamfer_term + 0.3 * hausdorff_term))


def _points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return arr


def _nearest_distances(source: np.ndarray, target: np.ndarray, block_size: int = 1024) -> np.ndarray:
    distances = np.empty((len(source),), dtype=np.float64)
    for start in range(0, len(source), block_size):
        end = min(len(source), start + block_size)
        delta = source[start:end, None, :] - target[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", delta, delta)
        distances[start:end] = np.sqrt(np.min(dist2, axis=1))
    return distances
