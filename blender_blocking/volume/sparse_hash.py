"""Sparse hash volume backend with optional bool chunk packing support."""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Optional, Tuple

import numpy as np

from .chunks import ChunkedVolumeGrid
from .contracts import Bounds3D, Chunk, ChunkKey, VoxelTransform
from .dense import _active_mask


class SparseHashVolumeGrid(ChunkedVolumeGrid):
    """Hash-map sparse grid that stores only non-default chunks."""

    backend = "sparse_hash"

    def set_chunk(self, key: ChunkKey, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=self.dtype)
        expected = (self.chunk_size, self.chunk_size, self.chunk_size)
        if data.shape != expected:
            raise ValueError(f"chunk shape must be {expected}")
        if _active_mask(data, self.default_value).any():
            self._chunks[ChunkKey.from_iterable(key.to_tuple())] = data.copy()
        else:
            self._chunks.pop(key, None)

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
    ) -> "SparseHashVolumeGrid":
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

    @classmethod
    def from_chunks(
        cls,
        chunks: Mapping[ChunkKey, np.ndarray],
        *,
        shape: Tuple[int, int, int],
        bounds: Bounds3D,
        transform: Optional[VoxelTransform] = None,
        value_type: str = "occupancy_bool",
        dtype: Any = bool,
        default_value: Any = False,
        chunk_size: int = 32,
    ) -> "SparseHashVolumeGrid":
        return cls(
            chunks=chunks,
            shape=shape,
            bounds=bounds,
            transform=transform,
            value_type=value_type,
            dtype=dtype,
            default_value=default_value,
            chunk_size=chunk_size,
        )

    def iter_active_chunks(self) -> Iterator[Chunk]:
        yield from super().iter_active_chunks()

    def packed_bool_chunks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return sorted keys and np.packbits-compressed bool chunks."""
        if self.dtype != np.dtype(bool):
            raise TypeError("packed_bool_chunks is only valid for bool grids")
        keys = []
        packed = []
        for key in sorted(self._chunks):
            keys.append(key.to_tuple())
            packed.append(np.packbits(self._chunks[key].reshape(-1)))
        if not keys:
            packed_size = (self.chunk_size**3 + 7) // 8
            return (
                np.empty((0, 3), dtype=np.int64),
                np.empty((0, packed_size), dtype=np.uint8),
            )
        return np.asarray(keys, dtype=np.int64), np.asarray(packed, dtype=np.uint8)

    @staticmethod
    def unpack_bool_chunk(packed: np.ndarray, chunk_size: int) -> np.ndarray:
        unpacked = np.unpackbits(np.asarray(packed, dtype=np.uint8))
        return unpacked[: chunk_size**3].reshape(
            (chunk_size, chunk_size, chunk_size)
        ).astype(bool)
