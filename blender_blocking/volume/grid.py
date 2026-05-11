"""Shared chunk-grid math for volume backends."""

from __future__ import annotations

from typing import Iterator, Tuple

from .contracts import ChunkKey


def all_chunk_keys(
    shape: Tuple[int, int, int],
    chunk_size: int,
) -> Iterator[ChunkKey]:
    sx, sy, sz = (int(value) for value in shape)
    size = int(chunk_size)
    for ix in range((sx + size - 1) // size):
        for iy in range((sy + size - 1) // size):
            for iz in range((sz + size - 1) // size):
                yield ChunkKey(ix, iy, iz)


def chunk_slices(
    key: ChunkKey,
    shape: Tuple[int, int, int],
    chunk_size: int,
) -> Tuple[Tuple[slice, slice, slice], Tuple[int, int, int], Tuple[int, int, int]]:
    size = int(chunk_size)
    grid_shape = tuple(int(value) for value in shape)
    origin = (
        key.ix * size,
        key.iy * size,
        key.iz * size,
    )
    end = tuple(origin[axis] + size for axis in range(3))
    valid_end = tuple(min(end[axis], grid_shape[axis]) for axis in range(3))
    valid_shape = tuple(max(0, valid_end[axis] - origin[axis]) for axis in range(3))
    slices = tuple(slice(origin[axis], valid_end[axis]) for axis in range(3))
    return slices, origin, valid_shape
