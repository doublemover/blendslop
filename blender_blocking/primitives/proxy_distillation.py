"""Distill Gaussian/ellipsoid proxies into comparable SDF and occupancy fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ProxyFieldDistillation:
    resolution: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    occupancy_ratio: float
    surface_voxel_count: int
    target_surface_sdf_mean: float
    target_surface_band_ratio: float
    arbitration_score: float
    sdf_statistics: Mapping[str, float] = field(default_factory=dict)
    primitive_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "resolution": self.resolution,
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
            "occupancy_ratio": self.occupancy_ratio,
            "surface_voxel_count": self.surface_voxel_count,
            "target_surface_sdf_mean": self.target_surface_sdf_mean,
            "target_surface_band_ratio": self.target_surface_band_ratio,
            "arbitration_score": self.arbitration_score,
            "sdf_statistics": dict(self.sdf_statistics),
            "primitive_count": self.primitive_count,
        }


def distill_proxy_field(
    primitives: Sequence[object],
    *,
    target_points: Any | None = None,
    resolution: int = 24,
    padding: float = 0.08,
    surface_band: float | None = None,
) -> tuple[ProxyFieldDistillation, np.ndarray, np.ndarray, np.ndarray]:
    """Sample primitive proxies into an SDF field and occupancy grid."""

    resolution = max(4, int(resolution))
    points = _coerce_points(target_points)
    bounds_min, bounds_max = _distillation_bounds(primitives, points, padding=padding)
    grid = _grid_points(resolution, bounds_min, bounds_max)
    flat = grid.reshape(-1, 3)
    sdf = _sdf_for_primitives(primitives, flat).reshape((resolution, resolution, resolution))
    occupancy = sdf <= 0.0
    voxel_size = np.linalg.norm((bounds_max - bounds_min) / max(1, resolution - 1))
    band = float(surface_band if surface_band is not None else voxel_size * 1.5)
    surface_voxel_count = int(np.count_nonzero(np.abs(sdf) <= band))
    target_sdf = _sdf_for_primitives(primitives, points) if len(points) else np.asarray(())
    if len(target_sdf):
        target_abs = np.abs(target_sdf)
        target_mean = float(np.mean(target_abs))
        target_band_ratio = float(np.count_nonzero(target_abs <= band) / len(target_abs))
        scale = max(1e-6, float(np.percentile(np.linalg.norm(points, axis=1), 95)))
        coverage = float(1.0 / (1.0 + target_mean / scale))
    else:
        target_mean = 0.0
        target_band_ratio = 0.0
        coverage = 0.0
    arbitration = float(np.clip(coverage * 0.7 + target_band_ratio * 0.3, 0.0, 1.0))
    report = ProxyFieldDistillation(
        resolution=resolution,
        bounds_min=tuple(float(v) for v in bounds_min),
        bounds_max=tuple(float(v) for v in bounds_max),
        occupancy_ratio=float(np.count_nonzero(occupancy) / max(1, occupancy.size)),
        surface_voxel_count=surface_voxel_count,
        target_surface_sdf_mean=target_mean,
        target_surface_band_ratio=target_band_ratio,
        arbitration_score=arbitration,
        sdf_statistics=_sdf_statistics(sdf),
        primitive_count=len(tuple(primitives)),
    )
    return report, grid, sdf, occupancy


def write_proxy_field_npz(
    path: Path,
    *,
    report: ProxyFieldDistillation,
    points: np.ndarray,
    sdf: np.ndarray,
    occupancy: np.ndarray,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=points.astype(np.float32, copy=False),
        sdf=sdf.astype(np.float32, copy=False),
        occupancy=occupancy.astype(np.uint8, copy=False),
        bounds_min=np.asarray(report.bounds_min, dtype=np.float32),
        bounds_max=np.asarray(report.bounds_max, dtype=np.float32),
    )
    return path


def _sdf_for_primitives(primitives: Sequence[object], points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.asarray((), dtype=np.float32)
    rows = []
    for primitive in primitives:
        if hasattr(primitive, "sdf_batch"):
            rows.append(np.asarray(primitive.sdf_batch(points), dtype=float))
    if not rows:
        return np.full((len(points),), np.inf, dtype=np.float32)
    return np.min(np.vstack(rows), axis=0).astype(np.float32, copy=False)


def _distillation_bounds(
    primitives: Sequence[object],
    points: np.ndarray,
    *,
    padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    if len(points):
        mins.append(np.min(points, axis=0))
        maxs.append(np.max(points, axis=0))
    for primitive in primitives:
        center = np.asarray(getattr(primitive, "center", (0.0, 0.0, 0.0)), dtype=float)
        radii = np.asarray(getattr(primitive, "radii", (0.5, 0.5, 0.5)), dtype=float)
        if center.shape == (3,) and radii.shape == (3,):
            radii = np.maximum(np.abs(radii), 1e-6)
            mins.append(center - radii)
            maxs.append(center + radii)
    if not mins:
        mins.append(np.asarray((-0.5, -0.5, -0.5), dtype=float))
        maxs.append(np.asarray((0.5, 0.5, 0.5), dtype=float))
    bounds_min = np.min(np.vstack(mins), axis=0)
    bounds_max = np.max(np.vstack(maxs), axis=0)
    extent = np.maximum(bounds_max - bounds_min, 1e-6)
    pad = extent * max(0.0, float(padding))
    return bounds_min - pad, bounds_max + pad


def _grid_points(
    resolution: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> np.ndarray:
    axes = [
        np.linspace(bounds_min[index], bounds_max[index], resolution, dtype=np.float32)
        for index in range(3)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=-1)


def _coerce_points(points: Any | None) -> np.ndarray:
    if points is None:
        return np.zeros((0, 3), dtype=np.float32)
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float32)
    return array


def _sdf_statistics(sdf: np.ndarray) -> dict[str, float]:
    finite = np.asarray(sdf[np.isfinite(sdf)], dtype=float)
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }
