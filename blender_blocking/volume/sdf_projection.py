"""Signed-distance projection helpers for volume reconstruction outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .chunks import ChunkedVolumeGrid
from .contracts import VolumeGrid
from .dense import DenseVolumeGrid
from .meshing import extract_surface_voxels
from .sparse_hash import SparseHashVolumeGrid


SIGN_CONVENTION = "negative_inside_positive_outside"


@dataclass(frozen=True)
class SDFProjectionReport:
    """Metadata for an occupancy-to-SDF projection."""

    status: str
    method: str
    sign_convention: str
    source_backend: str
    source_value_type: str
    output_backend: str
    shape: Tuple[int, int, int]
    voxel_size: Tuple[float, float, float]
    occupancy_threshold: float
    occupied_voxels: int
    total_voxels: int
    occupied_ratio: float
    surface_voxels: int
    positive_voxels: int
    negative_voxels: int
    zero_voxels: int
    field_min: float
    field_max: float
    field_mean: float
    narrow_band_voxels: Optional[int] = None
    narrow_band_world: Optional[float] = None
    warnings: Tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "method": self.method,
            "sign_convention": self.sign_convention,
            "source_backend": self.source_backend,
            "source_value_type": self.source_value_type,
            "output_backend": self.output_backend,
            "shape": [int(v) for v in self.shape],
            "voxel_size": [float(v) for v in self.voxel_size],
            "occupancy_threshold": float(self.occupancy_threshold),
            "occupied_voxels": int(self.occupied_voxels),
            "total_voxels": int(self.total_voxels),
            "occupied_ratio": float(self.occupied_ratio),
            "surface_voxels": int(self.surface_voxels),
            "positive_voxels": int(self.positive_voxels),
            "negative_voxels": int(self.negative_voxels),
            "zero_voxels": int(self.zero_voxels),
            "field_min": float(self.field_min),
            "field_max": float(self.field_max),
            "field_mean": float(self.field_mean),
            "narrow_band_voxels": (
                None
                if self.narrow_band_voxels is None
                else int(self.narrow_band_voxels)
            ),
            "narrow_band_world": (
                None
                if self.narrow_band_world is None
                else float(self.narrow_band_world)
            ),
            "warnings": list(self.warnings),
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class SDFProjectionResult:
    """Projected signed-distance grid plus structured diagnostics."""

    grid: VolumeGrid
    report: SDFProjectionReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_backend": getattr(self.grid, "backend", self.grid.__class__.__name__),
            "grid_value_type": self.grid.value_type,
            "report": self.report.to_dict(),
        }


def signed_distance_grid_from_volume(
    grid: VolumeGrid,
    *,
    occupancy_threshold: float = 0.5,
    prefer_scipy: bool = True,
    narrow_band_voxels: Optional[int] = None,
    output_backend: str = "dense",
    numpy_max_voxels: int = 250_000,
) -> SDFProjectionResult:
    """Project an occupancy-like volume into a signed-distance scalar field.

    The resulting field uses the common meshing convention of negative values
    inside the occupied volume and positive values outside it.  Distances are
    expressed in world units using the source grid's voxel spacing.
    """
    dense = grid.to_dense()
    occupied = occupancy_mask_from_volume(
        grid,
        occupancy_threshold=occupancy_threshold,
        dense=dense,
    )
    field, method, warnings = signed_distance_field_from_occupancy(
        occupied,
        voxel_size=tuple(float(v) for v in grid.transform.voxel_size),
        prefer_scipy=prefer_scipy,
        narrow_band_voxels=narrow_band_voxels,
        numpy_max_voxels=numpy_max_voxels,
    )
    output_grid = _grid_from_field(
        field,
        grid,
        output_backend=output_backend,
        value_type="signed_distance",
        default_value=0.0,
    )
    report = _projection_report(
        source_grid=grid,
        output_grid=output_grid,
        occupied=occupied,
        field=field,
        method=method,
        occupancy_threshold=occupancy_threshold,
        narrow_band_voxels=narrow_band_voxels,
        warnings=warnings,
    )
    return SDFProjectionResult(grid=output_grid, report=report)


def signed_distance_field_from_occupancy(
    occupied: np.ndarray,
    *,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    prefer_scipy: bool = True,
    narrow_band_voxels: Optional[int] = None,
    numpy_max_voxels: int = 250_000,
) -> tuple[np.ndarray, str, tuple[str, ...]]:
    """Return a signed-distance field for a boolean occupancy array."""
    occupied = np.asarray(occupied, dtype=bool)
    if occupied.ndim != 3:
        raise ValueError("occupied must be a 3D array")
    voxel_size = _validate_voxel_size(voxel_size)
    warnings: list[str] = []
    method = "scipy_edt"

    total_voxels = int(occupied.size)
    domain_fill = _domain_fill_distance(occupied.shape, voxel_size)
    if total_voxels == 0:
        return np.empty(occupied.shape, dtype=np.float32), "empty", ()
    if not occupied.any():
        field = np.full(occupied.shape, domain_fill, dtype=np.float32)
        warnings.append("source occupancy is empty; SDF is positive everywhere")
        return _apply_narrow_band(
            field,
            voxel_size=voxel_size,
            narrow_band_voxels=narrow_band_voxels,
        ), "empty_positive", tuple(warnings)
    if occupied.all():
        field = np.full(occupied.shape, -domain_fill, dtype=np.float32)
        warnings.append("source occupancy fills the domain; SDF is negative everywhere")
        return _apply_narrow_band(
            field,
            voxel_size=voxel_size,
            narrow_band_voxels=narrow_band_voxels,
        ), "full_negative", tuple(warnings)

    if prefer_scipy:
        try:
            from scipy import ndimage

            outside = ndimage.distance_transform_edt(
                np.logical_not(occupied),
                sampling=voxel_size,
            )
            inside = ndimage.distance_transform_edt(
                occupied,
                sampling=voxel_size,
            )
        except Exception as exc:
            warnings.append(
                "scipy distance_transform_edt unavailable; using bounded NumPy "
                f"fallback: {exc}"
            )
            method = "numpy_bruteforce"
            outside = _numpy_distance_transform(
                np.logical_not(occupied),
                voxel_size=voxel_size,
                max_voxels=numpy_max_voxels,
            )
            inside = _numpy_distance_transform(
                occupied,
                voxel_size=voxel_size,
                max_voxels=numpy_max_voxels,
            )
    else:
        method = "numpy_bruteforce"
        outside = _numpy_distance_transform(
            np.logical_not(occupied),
            voxel_size=voxel_size,
            max_voxels=numpy_max_voxels,
        )
        inside = _numpy_distance_transform(
            occupied,
            voxel_size=voxel_size,
            max_voxels=numpy_max_voxels,
        )

    field = np.asarray(outside, dtype=np.float32) - np.asarray(inside, dtype=np.float32)
    field = _apply_narrow_band(
        field,
        voxel_size=voxel_size,
        narrow_band_voxels=narrow_band_voxels,
    )
    return field.astype(np.float32, copy=False), method, tuple(warnings)


def occupancy_mask_from_volume(
    grid: VolumeGrid,
    *,
    occupancy_threshold: float = 0.5,
    dense: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalize a volume grid to a boolean occupancy mask."""
    data = np.asarray(grid.to_dense() if dense is None else dense)
    if data.ndim != 3:
        raise ValueError("volume data must be a 3D array")
    if grid.value_type == "signed_distance":
        return data <= 0.0
    if np.issubdtype(data.dtype, np.bool_):
        return data.astype(bool, copy=False)
    if grid.value_type in {"occupancy_prob", "confidence", "view_agreement"}:
        return np.asarray(data, dtype=float) >= float(occupancy_threshold)
    return data != grid.default_value


def occupancy_grid_from_signed_distance(
    grid: VolumeGrid,
    *,
    threshold: float = 0.0,
    output_backend: str = "dense",
) -> VolumeGrid:
    """Threshold a signed-distance grid back to boolean occupancy."""
    if grid.value_type != "signed_distance":
        raise ValueError("grid.value_type must be 'signed_distance'")
    dense = np.asarray(grid.to_dense(), dtype=float) <= float(threshold)
    return _grid_from_field(
        dense,
        grid,
        output_backend=output_backend,
        value_type="occupancy_bool",
        default_value=False,
    )


def _grid_from_field(
    field: np.ndarray,
    source_grid: VolumeGrid,
    *,
    output_backend: str,
    value_type: str,
    default_value: Any,
) -> VolumeGrid:
    backend = str(output_backend).strip().lower().replace("-", "_")
    field = np.asarray(field)
    if backend in {"", "dense"}:
        return DenseVolumeGrid(
            field,
            source_grid.bounds,
            transform=source_grid.transform,
            value_type=value_type,
            default_value=default_value,
            chunk_size=source_grid.chunk_size,
        )
    if backend == "chunked":
        return ChunkedVolumeGrid.from_dense(
            field,
            source_grid.bounds,
            transform=source_grid.transform,
            value_type=value_type,
            default_value=default_value,
            chunk_size=source_grid.chunk_size,
        )
    if backend in {"sparse", "sparse_hash"}:
        return SparseHashVolumeGrid.from_dense(
            field,
            source_grid.bounds,
            transform=source_grid.transform,
            value_type=value_type,
            default_value=default_value,
            chunk_size=source_grid.chunk_size,
        )
    raise ValueError("output_backend must be dense, chunked, or sparse_hash")


def _projection_report(
    *,
    source_grid: VolumeGrid,
    output_grid: VolumeGrid,
    occupied: np.ndarray,
    field: np.ndarray,
    method: str,
    occupancy_threshold: float,
    narrow_band_voxels: Optional[int],
    warnings: tuple[str, ...],
) -> SDFProjectionReport:
    total_voxels = int(occupied.size)
    occupied_voxels = int(np.count_nonzero(occupied))
    surface = extract_surface_voxels(occupied, prefer_scipy=True)
    narrow_band_world = None
    if narrow_band_voxels is not None:
        narrow_band_world = _narrow_band_world(
            voxel_size=tuple(float(v) for v in source_grid.transform.voxel_size),
            narrow_band_voxels=narrow_band_voxels,
        )
    finite = np.asarray(field, dtype=float)
    return SDFProjectionReport(
        status="ok" if not warnings else "degraded",
        method=method,
        sign_convention=SIGN_CONVENTION,
        source_backend=str(getattr(source_grid, "backend", source_grid.__class__.__name__)),
        source_value_type=str(source_grid.value_type),
        output_backend=str(getattr(output_grid, "backend", output_grid.__class__.__name__)),
        shape=tuple(int(v) for v in field.shape),
        voxel_size=tuple(float(v) for v in source_grid.transform.voxel_size),
        occupancy_threshold=float(occupancy_threshold),
        occupied_voxels=occupied_voxels,
        total_voxels=total_voxels,
        occupied_ratio=(
            float(occupied_voxels) / float(total_voxels) if total_voxels else 0.0
        ),
        surface_voxels=int(np.count_nonzero(surface)),
        positive_voxels=int(np.count_nonzero(finite > 0.0)),
        negative_voxels=int(np.count_nonzero(finite < 0.0)),
        zero_voxels=int(np.count_nonzero(finite == 0.0)),
        field_min=float(np.min(finite)) if finite.size else 0.0,
        field_max=float(np.max(finite)) if finite.size else 0.0,
        field_mean=float(np.mean(finite)) if finite.size else 0.0,
        narrow_band_voxels=(
            None if narrow_band_voxels is None else int(narrow_band_voxels)
        ),
        narrow_band_world=narrow_band_world,
        warnings=warnings,
        extra={
            "source_active_voxels": int(source_grid.active_voxel_count()),
            "output_active_voxels": int(output_grid.active_voxel_count()),
        },
    )


def _numpy_distance_transform(
    mask: np.ndarray,
    *,
    voxel_size: Tuple[float, float, float],
    max_voxels: int,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D array")
    if int(mask.size) > int(max_voxels):
        raise RuntimeError(
            "NumPy signed-distance fallback is limited to "
            f"{int(max_voxels)} voxels; install scipy for larger volumes"
        )
    output = np.zeros(mask.shape, dtype=np.float32)
    query = np.argwhere(mask)
    if len(query) == 0:
        return output
    zeros = np.argwhere(np.logical_not(mask))
    if len(zeros) == 0:
        output[mask] = _domain_fill_distance(mask.shape, voxel_size)
        return output

    sampling = np.asarray(voxel_size, dtype=np.float64)
    batch_size = max(1, int(1_000_000 // max(1, len(zeros))))
    for start in range(0, len(query), batch_size):
        batch = query[start : start + batch_size]
        diff = (batch[:, None, :] - zeros[None, :, :]) * sampling[None, None, :]
        distances = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
        output[batch[:, 0], batch[:, 1], batch[:, 2]] = distances.astype(np.float32)
    return output


def _validate_voxel_size(voxel_size: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if len(voxel_size) != 3:
        raise ValueError("voxel_size must have three components")
    values = tuple(float(v) for v in voxel_size)
    if any(v <= 0.0 for v in values):
        raise ValueError("voxel_size components must be positive")
    return values


def _domain_fill_distance(
    shape: tuple[int, ...],
    voxel_size: Tuple[float, float, float],
) -> float:
    shape_array = np.asarray(shape, dtype=float)
    sampling = np.asarray(voxel_size, dtype=float)
    return float(np.linalg.norm(np.maximum(shape_array - 1.0, 1.0) * sampling))


def _apply_narrow_band(
    field: np.ndarray,
    *,
    voxel_size: Tuple[float, float, float],
    narrow_band_voxels: Optional[int],
) -> np.ndarray:
    if narrow_band_voxels is None:
        return np.asarray(field, dtype=np.float32)
    band = _narrow_band_world(
        voxel_size=voxel_size,
        narrow_band_voxels=narrow_band_voxels,
    )
    if band <= 0.0:
        raise ValueError("narrow_band_voxels must be positive when provided")
    return np.clip(np.asarray(field, dtype=np.float32), -band, band)


def _narrow_band_world(
    *,
    voxel_size: Tuple[float, float, float],
    narrow_band_voxels: int,
) -> float:
    return float(max(voxel_size) * int(narrow_band_voxels))
