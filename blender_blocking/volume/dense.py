"""Dense volume grid backend."""

from __future__ import annotations

from typing import Any, Iterator, Optional, Tuple

import numpy as np

from .contracts import (
    Bounds3D,
    Chunk,
    ChunkKey,
    SUPPORTED_VALUE_TYPES,
    VolumeStats,
    VoxelTransform,
)


def _normalize_default(value: Any, dtype: np.dtype) -> Any:
    if np.issubdtype(dtype, np.bool_):
        return bool(value)
    if np.issubdtype(dtype, np.integer):
        return int(value)
    if np.issubdtype(dtype, np.floating):
        return float(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _active_mask(array: np.ndarray, default_value: Any) -> np.ndarray:
    if np.issubdtype(array.dtype, np.floating) and isinstance(default_value, float):
        return ~np.isclose(array, default_value)
    return array != default_value


class DenseVolumeGrid:
    """In-memory dense volume grid."""

    backend = "dense"

    def __init__(
        self,
        data: np.ndarray,
        bounds: Bounds3D,
        *,
        transform: Optional[VoxelTransform] = None,
        value_type: str = "occupancy_bool",
        default_value: Any = False,
        chunk_size: int = 32,
    ) -> None:
        data = np.asarray(data)
        if data.ndim != 3:
            raise ValueError("data must be a 3D array")
        if value_type not in SUPPORTED_VALUE_TYPES:
            raise ValueError(f"unsupported value_type: {value_type}")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        self.data = data
        self.bounds = bounds
        self.transform = transform or VoxelTransform.from_bounds_shape(
            bounds, tuple(int(v) for v in data.shape)
        )
        if tuple(self.transform.shape) != tuple(data.shape):
            raise ValueError("transform shape must match data shape")

        self.value_type = value_type
        self.default_value = _normalize_default(default_value, data.dtype)
        self.chunk_size = int(chunk_size)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.data.shape)

    def get_chunk(self, key: ChunkKey) -> np.ndarray:
        slices, _origin, valid_shape = self._chunk_slices(key)
        if any(size <= 0 for size in valid_shape):
            return np.full(
                (self.chunk_size, self.chunk_size, self.chunk_size),
                self.default_value,
                dtype=self.data.dtype,
            )

        chunk = np.full(
            (self.chunk_size, self.chunk_size, self.chunk_size),
            self.default_value,
            dtype=self.data.dtype,
        )
        valid_slices = tuple(slice(0, size) for size in valid_shape)
        chunk[valid_slices] = self.data[slices]
        return chunk

    def iter_active_chunks(self) -> Iterator[Chunk]:
        for key in self._all_chunk_keys():
            slices, origin, valid_shape = self._chunk_slices(key)
            data = self.data[slices]
            if data.size and _active_mask(data, self.default_value).any():
                full_chunk = np.full(
                    (self.chunk_size, self.chunk_size, self.chunk_size),
                    self.default_value,
                    dtype=self.data.dtype,
                )
                valid_slices = tuple(slice(0, size) for size in valid_shape)
                full_chunk[valid_slices] = data
                yield Chunk(key, full_chunk, origin, valid_shape)

    def sample_world(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")

        indices = self.transform.world_to_nearest_index(points)
        valid = self.transform.contains_indices(indices)
        result = np.full((len(points),), self.default_value, dtype=self.data.dtype)
        if valid.any():
            valid_indices = indices[valid]
            result[valid] = self.data[
                valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
            ]
        return result

    def active_voxel_count(self) -> int:
        return int(_active_mask(self.data, self.default_value).sum())

    def to_dense(self, max_voxels: Optional[int] = None) -> np.ndarray:
        if max_voxels is not None and self.data.size > max_voxels:
            raise ValueError(
                f"dense volume has {self.data.size} voxels, exceeding max_voxels"
            )
        return self.data.copy()

    def stats(self) -> VolumeStats:
        active_chunks = sum(1 for _chunk in self.iter_active_chunks())
        return VolumeStats(
            total_voxels=int(self.data.size),
            active_voxels=self.active_voxel_count(),
            active_chunks=active_chunks,
            stored_chunks=active_chunks,
            dense_shape=self.shape,
            chunk_size=self.chunk_size,
            dtype=str(self.data.dtype),
            value_type=self.value_type,
            default_value=self.default_value,
        )

    def _all_chunk_keys(self) -> Iterator[ChunkKey]:
        sx, sy, sz = self.shape
        for ix in range((sx + self.chunk_size - 1) // self.chunk_size):
            for iy in range((sy + self.chunk_size - 1) // self.chunk_size):
                for iz in range((sz + self.chunk_size - 1) // self.chunk_size):
                    yield ChunkKey(ix, iy, iz)

    def _chunk_slices(
        self, key: ChunkKey
    ) -> Tuple[Tuple[slice, slice, slice], Tuple[int, int, int], Tuple[int, int, int]]:
        origin = (
            key.ix * self.chunk_size,
            key.iy * self.chunk_size,
            key.iz * self.chunk_size,
        )
        end = tuple(origin[axis] + self.chunk_size for axis in range(3))
        valid_end = tuple(min(end[axis], self.shape[axis]) for axis in range(3))
        valid_shape = tuple(max(0, valid_end[axis] - origin[axis]) for axis in range(3))
        slices = tuple(slice(origin[axis], valid_end[axis]) for axis in range(3))
        return slices, origin, valid_shape
