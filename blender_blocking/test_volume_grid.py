"""Tests for shared volume chunk grid helpers."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from volume.chunks import ChunkedVolumeGrid
    from volume.contracts import Bounds3D, ChunkKey
    from volume.dense import DenseVolumeGrid
    from volume.grid import all_chunk_keys, chunk_slices
except ModuleNotFoundError:  # pragma: no cover - package unittest path
    from blender_blocking.volume.chunks import ChunkedVolumeGrid
    from blender_blocking.volume.contracts import Bounds3D, ChunkKey
    from blender_blocking.volume.dense import DenseVolumeGrid
    from blender_blocking.volume.grid import all_chunk_keys, chunk_slices


class VolumeGridTests(unittest.TestCase):
    def test_all_chunk_keys_matches_dense_and_chunked_backends(self) -> None:
        bounds = Bounds3D(0.0, 5.0, 0.0, 4.0, 0.0, 3.0)
        data = np.zeros((5, 4, 3), dtype=bool)
        dense = DenseVolumeGrid(data, bounds, chunk_size=2)
        chunked = ChunkedVolumeGrid(shape=data.shape, bounds=bounds, chunk_size=2)

        expected = tuple(all_chunk_keys(data.shape, 2))

        self.assertEqual(expected, tuple(dense._all_chunk_keys()))
        self.assertEqual(expected, tuple(chunked._all_chunk_keys()))
        self.assertEqual(len(expected), 12)

    def test_chunk_slices_matches_backend_edge_chunk_shape(self) -> None:
        bounds = Bounds3D(0.0, 5.0, 0.0, 4.0, 0.0, 3.0)
        data = np.zeros((5, 4, 3), dtype=bool)
        dense = DenseVolumeGrid(data, bounds, chunk_size=2)
        key = ChunkKey(2, 1, 1)

        shared = chunk_slices(key, data.shape, 2)

        self.assertEqual(shared, dense._chunk_slices(key))
        slices, origin, valid_shape = shared
        self.assertEqual(origin, (4, 2, 2))
        self.assertEqual(valid_shape, (1, 2, 1))
        self.assertEqual(
            [(item.start, item.stop) for item in slices],
            [(4, 5), (2, 4), (2, 3)],
        )


if __name__ == "__main__":
    unittest.main()
