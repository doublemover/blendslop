"""Shared contracts for volume grids and volume-derived meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Mapping, Optional, Protocol, Tuple

import numpy as np


SUPPORTED_VALUE_TYPES = {
    "occupancy_bool",
    "occupancy_prob",
    "signed_distance",
    "confidence",
    "view_agreement",
    "surface_flag",
}


@dataclass(frozen=True)
class Bounds3D:
    """Axis-aligned world-space bounds."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    def __post_init__(self) -> None:
        if self.max_x <= self.min_x:
            raise ValueError("max_x must be greater than min_x")
        if self.max_y <= self.min_y:
            raise ValueError("max_y must be greater than min_y")
        if self.max_z <= self.min_z:
            raise ValueError("max_z must be greater than min_z")

    @classmethod
    def from_min_max(cls, minimum: np.ndarray, maximum: np.ndarray) -> "Bounds3D":
        minimum = np.asarray(minimum, dtype=float)
        maximum = np.asarray(maximum, dtype=float)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("minimum and maximum must have shape (3,)")
        return cls(
            float(minimum[0]),
            float(maximum[0]),
            float(minimum[1]),
            float(maximum[1]),
            float(minimum[2]),
            float(maximum[2]),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Bounds3D":
        return cls(
            float(data["min_x"]),
            float(data["max_x"]),
            float(data["min_y"]),
            float(data["max_y"]),
            float(data["min_z"]),
            float(data["max_z"]),
        )

    @property
    def minimum(self) -> np.ndarray:
        return np.array([self.min_x, self.min_y, self.min_z], dtype=float)

    @property
    def maximum(self) -> np.ndarray:
        return np.array([self.max_x, self.max_y, self.max_z], dtype=float)

    @property
    def size(self) -> np.ndarray:
        return self.maximum - self.minimum

    @property
    def center(self) -> np.ndarray:
        return (self.minimum + self.maximum) * 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "min_z": self.min_z,
            "max_z": self.max_z,
        }


@dataclass(frozen=True)
class VoxelTransform:
    """Affine transform for axis-aligned voxel centers."""

    origin: Tuple[float, float, float]
    voxel_size: Tuple[float, float, float]
    shape: Tuple[int, int, int]

    def __post_init__(self) -> None:
        if len(self.origin) != 3:
            raise ValueError("origin must have three components")
        if len(self.voxel_size) != 3:
            raise ValueError("voxel_size must have three components")
        if len(self.shape) != 3:
            raise ValueError("shape must have three components")
        if any(size <= 0 for size in self.voxel_size):
            raise ValueError("voxel_size components must be positive")
        if any(dim <= 0 for dim in self.shape):
            raise ValueError("shape components must be positive")

    @classmethod
    def from_bounds_shape(
        cls, bounds: Bounds3D, shape: Tuple[int, int, int]
    ) -> "VoxelTransform":
        if any(dim <= 0 for dim in shape):
            raise ValueError("shape components must be positive")
        extent = bounds.size
        voxel_size = extent / np.asarray(shape, dtype=float)
        origin = bounds.minimum + voxel_size * 0.5
        return cls(
            tuple(float(v) for v in origin),
            tuple(float(v) for v in voxel_size),
            tuple(int(v) for v in shape),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VoxelTransform":
        return cls(
            tuple(float(v) for v in data["origin"]),
            tuple(float(v) for v in data["voxel_size"]),
            tuple(int(v) for v in data["shape"]),
        )

    @property
    def origin_array(self) -> np.ndarray:
        return np.asarray(self.origin, dtype=float)

    @property
    def voxel_size_array(self) -> np.ndarray:
        return np.asarray(self.voxel_size, dtype=float)

    def index_to_world(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=float)
        return self.origin_array + indices * self.voxel_size_array

    def world_to_index(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return (points - self.origin_array) / self.voxel_size_array

    def world_to_nearest_index(self, points: np.ndarray) -> np.ndarray:
        return np.rint(self.world_to_index(points)).astype(np.int64)

    def contains_indices(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices)
        shape = np.asarray(self.shape, dtype=np.int64)
        return np.all((indices >= 0) & (indices < shape), axis=-1)

    def to_dict(self) -> Dict[str, object]:
        return {
            "origin": list(self.origin),
            "voxel_size": list(self.voxel_size),
            "shape": list(self.shape),
        }


@dataclass(frozen=True, order=True)
class ChunkKey:
    """Integer key for a cubic chunk in chunk index space."""

    ix: int
    iy: int
    iz: int

    @classmethod
    def from_iterable(cls, values: Any) -> "ChunkKey":
        ix, iy, iz = values
        return cls(int(ix), int(iy), int(iz))

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.ix, self.iy, self.iz)


@dataclass(frozen=True)
class Chunk:
    """Chunk array plus placement metadata."""

    key: ChunkKey
    data: np.ndarray
    origin_index: Tuple[int, int, int]
    valid_shape: Tuple[int, int, int]


@dataclass(frozen=True)
class VolumeStats:
    total_voxels: int
    active_voxels: int
    active_chunks: int
    stored_chunks: int
    dense_shape: Tuple[int, int, int]
    chunk_size: int
    dtype: str
    value_type: str
    default_value: Any

    @property
    def occupancy_fraction(self) -> float:
        if self.total_voxels == 0:
            return 0.0
        return float(self.active_voxels) / float(self.total_voxels)

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_voxels": self.total_voxels,
            "active_voxels": self.active_voxels,
            "active_chunks": self.active_chunks,
            "stored_chunks": self.stored_chunks,
            "dense_shape": list(self.dense_shape),
            "chunk_size": self.chunk_size,
            "dtype": self.dtype,
            "value_type": self.value_type,
            "default_value": self.default_value,
            "occupancy_fraction": self.occupancy_fraction,
        }


@dataclass(frozen=True)
class VolumeMetadata:
    format_version: int
    backend: str
    value_type: str
    dtype: str
    bounds: Bounds3D
    transform: VoxelTransform
    shape: Tuple[int, int, int]
    chunk_size: int
    default_value: Any
    active_voxels: int
    source_candidate_id: Optional[str] = None
    source_masks: Tuple[str, ...] = ()
    source_views: Tuple[str, ...] = ()
    generation_seed: Optional[int] = None
    hashes: Mapping[str, str] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VolumeMetadata":
        return cls(
            format_version=int(data["format_version"]),
            backend=str(data["backend"]),
            value_type=str(data["value_type"]),
            dtype=str(data["dtype"]),
            bounds=Bounds3D.from_dict(data["bounds"]),
            transform=VoxelTransform.from_dict(data["transform"]),
            shape=tuple(int(v) for v in data["shape"]),
            chunk_size=int(data["chunk_size"]),
            default_value=data["default_value"],
            active_voxels=int(data["active_voxels"]),
            source_candidate_id=data.get("source_candidate_id"),
            source_masks=tuple(str(v) for v in data.get("source_masks", ())),
            source_views=tuple(str(v) for v in data.get("source_views", ())),
            generation_seed=data.get("generation_seed"),
            hashes=dict(data.get("hashes", {})),
            extra=dict(data.get("extra", {})),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "format_version": self.format_version,
            "backend": self.backend,
            "value_type": self.value_type,
            "dtype": self.dtype,
            "bounds": self.bounds.to_dict(),
            "transform": self.transform.to_dict(),
            "shape": list(self.shape),
            "chunk_size": self.chunk_size,
            "default_value": self.default_value,
            "active_voxels": self.active_voxels,
            "source_candidate_id": self.source_candidate_id,
            "source_masks": list(self.source_masks),
            "source_views": list(self.source_views),
            "generation_seed": self.generation_seed,
            "hashes": dict(self.hashes),
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class MeshExtractionResult:
    status: str
    method: str
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None
    message: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status == "ok"

    @classmethod
    def unavailable(cls, method: str, message: str) -> "MeshExtractionResult":
        return cls(
            status="unavailable",
            method=method,
            vertices=np.empty((0, 3), dtype=float),
            faces=np.empty((0, 3), dtype=np.int64),
            message=message,
        )

    @classmethod
    def skipped(cls, method: str, message: str) -> "MeshExtractionResult":
        return cls(
            status="skipped",
            method=method,
            vertices=np.empty((0, 3), dtype=float),
            faces=np.empty((0, 3), dtype=np.int64),
            message=message,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "method": self.method,
            "vertices": int(len(self.vertices)),
            "faces": int(len(self.faces)),
            "has_normals": self.normals is not None,
            "message": self.message,
            "metrics": dict(self.metrics),
        }


class VolumeGrid(Protocol):
    bounds: Bounds3D
    transform: VoxelTransform
    value_type: str
    default_value: Any
    chunk_size: int

    def get_chunk(self, key: ChunkKey) -> np.ndarray:
        ...

    def iter_active_chunks(self) -> Iterator[Chunk]:
        ...

    def sample_world(self, points: np.ndarray) -> np.ndarray:
        ...

    def active_voxel_count(self) -> int:
        ...

    def to_dense(self, max_voxels: Optional[int] = None) -> np.ndarray:
        ...
