"""
Initialization helpers for modular residual primitive fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    from primitives.analytic_primitives import (
        AnisotropicGaussianPrimitive,
        EllipsoidPrimitive,
    )
    from primitives.superfrustum import SuperFrustum
except ImportError:  # pragma: no cover - package import path.
    from ..primitives.analytic_primitives import (
        AnisotropicGaussianPrimitive,
        EllipsoidPrimitive,
    )
    from ..primitives.superfrustum import SuperFrustum


def _validate_finite_float(value: object, name: str, errors: list[str]) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a real number, got {value!r}")
        return None
    if not np.isfinite(parsed):
        errors.append(f"{name} must be finite, got {value!r}")
        return None
    return parsed


def _validate_positive_int(value: object, name: str, errors: list[str]) -> int | None:
    if isinstance(value, bool):
        errors.append(f"{name} must be an integer, got {value!r}")
        return None
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        errors.append(f"{name} must be an integer, got {value!r}")
        return None
    if parsed <= 0:
        errors.append(f"{name} must be > 0, got {value!r}")
    return parsed


@dataclass(frozen=True)
class PrimitiveInitializationConfig:
    """Configuration for deterministic primitive seeding."""

    primitive_count: int = 5
    target_point_count: int = 4096
    min_radius: float = 0.05
    covariance_floor: float = 1e-4
    kmeans_iterations: int = 8

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        _validate_positive_int(self.primitive_count, "primitive_count", errors)
        _validate_positive_int(self.target_point_count, "target_point_count", errors)
        _validate_positive_int(self.kmeans_iterations, "kmeans_iterations", errors)
        min_radius = _validate_finite_float(self.min_radius, "min_radius", errors)
        covariance_floor = _validate_finite_float(
            self.covariance_floor, "covariance_floor", errors
        )
        if min_radius is not None and min_radius <= 0.0:
            errors.append(f"min_radius must be > 0.0, got {self.min_radius!r}")
        if covariance_floor is not None and covariance_floor < 0.0:
            errors.append(
                f"covariance_floor must be >= 0.0, got {self.covariance_floor!r}"
            )
        return tuple(errors)


def bounded_point_sample(points: np.ndarray, limit: int) -> np.ndarray:
    """Return a deterministic bounded sample without random state."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if limit <= 0 or len(points) <= limit:
        return points.copy()
    indices = np.linspace(0, len(points) - 1, limit).round().astype(int)
    return points[indices]


def deterministic_kmeans(
    points: np.ndarray,
    cluster_count: int,
    iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic k-means used for primitive centers."""
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    cluster_count = max(1, min(int(cluster_count), len(points)))
    order = np.argsort(points[:, 2], kind="mergesort")
    seed_indices = np.linspace(0, len(points) - 1, cluster_count).round().astype(int)
    centers = points[order[seed_indices]].copy()

    labels = np.zeros((len(points),), dtype=np.int64)
    for _ in range(max(1, iterations)):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        for idx in range(cluster_count):
            cluster_points = points[labels == idx]
            if len(cluster_points) > 0:
                centers[idx] = cluster_points.mean(axis=0)

    return centers, labels


def _cluster_covariance(
    points: np.ndarray,
    center: np.ndarray,
    floor: float,
) -> np.ndarray:
    if len(points) <= 1:
        return np.eye(3, dtype=np.float64) * max(floor, 1e-6)
    centered = points - center[None, :]
    cov = centered.T @ centered / max(1, len(points) - 1)
    cov += np.eye(3, dtype=np.float64) * floor
    return cov


def initialize_superfrusta_from_points(
    points: np.ndarray,
    config: PrimitiveInitializationConfig = PrimitiveInitializationConfig(),
) -> List[SuperFrustum]:
    """Seed SuperFrusta by vertical point bands."""
    points = bounded_point_sample(points, config.target_point_count)
    if len(points) == 0:
        return []
    count = max(1, config.primitive_count)
    z_coords = points[:, 2]
    z_min, z_max = float(z_coords.min()), float(z_coords.max())
    if abs(z_max - z_min) < 1e-9:
        z_max = z_min + 1.0

    primitives: List[SuperFrustum] = []
    for idx in range(count):
        z0 = z_min + idx * (z_max - z_min) / count
        z1 = z_min + (idx + 1) * (z_max - z_min) / count
        if idx == count - 1:
            mask = (z_coords >= z0) & (z_coords <= z1)
        else:
            mask = (z_coords >= z0) & (z_coords < z1)
        band = points[mask]
        if len(band) == 0:
            continue
        center = band.mean(axis=0)
        xy_radius = np.linalg.norm(band[:, :2] - center[None, :2], axis=1)
        radius = max(float(np.percentile(xy_radius, 75)), config.min_radius)
        height = max(float(z1 - z0), config.min_radius * 2.0)
        primitives.append(
            SuperFrustum(
                position=tuple(center),
                orientation=(0.0, 0.0),
                radius_bottom=radius,
                radius_top=max(radius * 0.9, config.min_radius),
                height=height,
            )
        )
    return primitives


def initialize_ellipsoids_from_points(
    points: np.ndarray,
    config: PrimitiveInitializationConfig = PrimitiveInitializationConfig(),
) -> List[EllipsoidPrimitive]:
    """Seed ellipsoids from deterministic clusters and local PCA."""
    points = bounded_point_sample(points, config.target_point_count)
    centers, labels = deterministic_kmeans(
        points, config.primitive_count, config.kmeans_iterations
    )
    ellipsoids: List[EllipsoidPrimitive] = []
    for idx, center in enumerate(centers):
        cluster = points[labels == idx]
        if len(cluster) == 0:
            continue
        cov = _cluster_covariance(cluster, center, config.covariance_floor)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        radii = np.maximum(2.0 * np.sqrt(np.maximum(eigvals, 0.0)), config.min_radius)
        ellipsoids.append(
            EllipsoidPrimitive(
                center=center,
                radii=radii,
                rotation=eigvecs,
                density=min(1.0, len(cluster) / max(1, len(points))),
                confidence=1.0,
            )
        )
    return ellipsoids


def initialize_gaussians_from_points(
    points: np.ndarray,
    config: PrimitiveInitializationConfig = PrimitiveInitializationConfig(),
) -> List[AnisotropicGaussianPrimitive]:
    """Seed anisotropic Gaussians from deterministic clusters and local PCA."""
    points = bounded_point_sample(points, config.target_point_count)
    centers, labels = deterministic_kmeans(
        points, config.primitive_count, config.kmeans_iterations
    )
    gaussians: List[AnisotropicGaussianPrimitive] = []
    for idx, center in enumerate(centers):
        cluster = points[labels == idx]
        if len(cluster) == 0:
            continue
        cov = _cluster_covariance(cluster, center, config.covariance_floor)
        gaussians.append(
            AnisotropicGaussianPrimitive(
                center=center,
                covariance=cov,
                opacity=min(1.0, max(0.1, len(cluster) / max(1, len(points)))),
                confidence=1.0,
            )
        )
    return gaussians


def initialize_from_profile_bands(
    slice_data: Sequence[Dict[str, Any]],
    config: PrimitiveInitializationConfig = PrimitiveInitializationConfig(),
) -> List[SuperFrustum]:
    """Seed SuperFrusta from existing profile/slice analysis dictionaries."""
    if config.primitive_count <= 0:
        raise ValueError("primitive_count must be >= 1")
    if not slice_data:
        return []

    per_primitive = max(1, len(slice_data) // config.primitive_count)
    primitives: List[SuperFrustum] = []
    for idx in range(config.primitive_count):
        start = idx * per_primitive
        end = min((idx + 1) * per_primitive, len(slice_data))
        segment = slice_data[start:end]
        if not segment:
            continue
        centers = np.asarray([entry["center"] for entry in segment], dtype=np.float64)
        radii = np.asarray([entry["radius"] for entry in segment], dtype=np.float64)
        center = centers.mean(axis=0)
        height = (
            float(abs(centers[-1, 2] - centers[0, 2]))
            if len(centers) > 1
            else max(config.min_radius * 2.0, float(radii[0]) * 2.0)
        )
        primitives.append(
            SuperFrustum(
                position=tuple(center),
                orientation=(0.0, 0.0),
                radius_bottom=max(float(radii[0]), config.min_radius),
                radius_top=max(float(radii[-1]), config.min_radius),
                height=max(height, config.min_radius * 2.0),
            )
        )
    return primitives
