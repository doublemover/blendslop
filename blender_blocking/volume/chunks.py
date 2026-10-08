"""Chunked dense volume backend."""

from __future__ import annotations

from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import numpy as np

from .contracts import (
    Bounds3D,
    Chunk,
    ChunkKey,
    SUPPORTED_VALUE_TYPES,
    VolumeStats,
    VoxelTransform,
)
from .dense import _active_mask, _normalize_default
from .grid import all_chunk_keys, chunk_slices


class ChunkedVolumeGrid:
    """Grid backed by fixed-size chunks keyed by chunk coordinates."""

    backend = "chunked"

    def __init__(
        self,
        *,
        chunks: Optional[Mapping[ChunkKey, np.ndarray]] = None,
        shape: Tuple[int, int, int],
        bounds: Bounds3D,
        transform: Optional[VoxelTransform] = None,
        value_type: str = "occupancy_bool",
        dtype: Any = bool,
        default_value: Any = False,
        chunk_size: int = 32,
    ) -> None:
        if value_type not in SUPPORTED_VALUE_TYPES:
            raise ValueError(f"unsupported value_type: {value_type}")
        if any(dim <= 0 for dim in shape):
            raise ValueError("shape components must be positive")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        self.shape = tuple(int(v) for v in shape)
        self.bounds = bounds
        self.transform = transform or VoxelTransform.from_bounds_shape(bounds, self.shape)
        if tuple(self.transform.shape) != self.shape:
            raise ValueError("transform shape must match grid shape")

        self.value_type = value_type
        self.dtype = np.dtype(dtype)
        self.default_value = _normalize_default(default_value, self.dtype)
        self.chunk_size = int(chunk_size)
        self._chunks: Dict[ChunkKey, np.ndarray] = {}
        for key, chunk in (chunks or {}).items():
            self.set_chunk(key, chunk)

    @classmethod
    def from_dense(
        cls,
        data: np.ndarray,
        bounds: Bounds3D,
        *,
        transform: Optional[VoxelTransform] = None,
        value_type: str = "occupancy_bool",
        default_value: Any = False,
        chunk_size: int = 32,
    ) -> "ChunkedVolumeGrid":
        data = np.asarray(data)
        grid = cls(
            chunks={},
            shape=tuple(int(v) for v in data.shape),
            bounds=bounds,
            transform=transform,
            value_type=value_type,
            dtype=data.dtype,
            default_value=default_value,
            chunk_size=chunk_size,
        )
        for key in grid._all_chunk_keys():
            slices, _origin, valid_shape = grid._chunk_slices(key)
            chunk = np.full(
                (chunk_size, chunk_size, chunk_size),
                grid.default_value,
                dtype=grid.dtype,
            )
            valid_slices = tuple(slice(0, size) for size in valid_shape)
            chunk[valid_slices] = data[slices]
            grid.set_chunk(key, chunk)
        return grid

    def set_chunk(self, key: ChunkKey, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=self.dtype)
        expected = (self.chunk_size, self.chunk_size, self.chunk_size)
        if data.shape != expected:
            raise ValueError(f"chunk shape must be {expected}")
        self._chunks[ChunkKey.from_iterable(key.to_tuple())] = data.copy()

    def get_chunk(self, key: ChunkKey) -> np.ndarray:
        chunk = self._chunks.get(key)
        if chunk is None:
            return np.full(
                (self.chunk_size, self.chunk_size, self.chunk_size),
                self.default_value,
                dtype=self.dtype,
            )
        return chunk.copy()

    def iter_active_chunks(self) -> Iterator[Chunk]:
        for key in sorted(self._chunks):
            slices, origin, valid_shape = self._chunk_slices(key)
            chunk = self._chunks[key]
            valid_slices = tuple(slice(0, size) for size in valid_shape)
            valid_data = chunk[valid_slices]
            if valid_data.size and _active_mask(valid_data, self.default_value).any():
                yield Chunk(key, chunk.copy(), origin, valid_shape)

    def sample_world(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")

        indices = self.transform.world_to_nearest_index(points)
        valid = self.transform.contains_indices(indices)
        result = np.full((len(points),), self.default_value, dtype=self.dtype)
        for out_index in np.flatnonzero(valid):
            ix, iy, iz = (int(v) for v in indices[out_index])
            key = ChunkKey(ix // self.chunk_size, iy // self.chunk_size, iz // self.chunk_size)
            chunk = self._chunks.get(key)
            if chunk is None:
                continue
            result[out_index] = chunk[
                ix % self.chunk_size, iy % self.chunk_size, iz % self.chunk_size
            ]
        return result

    def active_voxel_count(self) -> int:
        count = 0
        for chunk in self.iter_active_chunks():
            valid_slices = tuple(slice(0, size) for size in chunk.valid_shape)
            count += int(_active_mask(chunk.data[valid_slices], self.default_value).sum())
        return count

    def to_dense(self, max_voxels: Optional[int] = None) -> np.ndarray:
        total = int(np.prod(self.shape))
        if max_voxels is not None and total > max_voxels:
            raise ValueError(f"dense volume has {total} voxels, exceeding max_voxels")
        dense = np.full(self.shape, self.default_value, dtype=self.dtype)
        for key, chunk in self._chunks.items():
            slices, _origin, valid_shape = self._chunk_slices(key)
            valid_slices = tuple(slice(0, size) for size in valid_shape)
            dense[slices] = chunk[valid_slices]
        return dense

    def stats(self) -> VolumeStats:
        active_chunks = sum(1 for _chunk in self.iter_active_chunks())
        return VolumeStats(
            total_voxels=int(np.prod(self.shape)),
            active_voxels=self.active_voxel_count(),
            active_chunks=active_chunks,
            stored_chunks=len(self._chunks),
            dense_shape=self.shape,
            chunk_size=self.chunk_size,
            dtype=str(self.dtype),
            value_type=self.value_type,
            default_value=self.default_value,
        )

    @property
    def chunks(self) -> Mapping[ChunkKey, np.ndarray]:
        return self._chunks

    def _all_chunk_keys(self) -> Iterator[ChunkKey]:
        yield from all_chunk_keys(self.shape, self.chunk_size)

    def _chunk_slices(
        self, key: ChunkKey
    ) -> Tuple[Tuple[slice, slice, slice], Tuple[int, int, int], Tuple[int, int, int]]:
        return chunk_slices(key, self.shape, self.chunk_size)
