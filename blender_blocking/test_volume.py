#!/usr/bin/env python3
"""Pure tests for volume grid contracts and interchange."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np

from volume import (
    Bounds3D,
    ChunkKey,
    ChunkedVolumeGrid,
    DenseVolumeGrid,
    SparseHashVolumeGrid,
    extract_mesh,
    extract_surface_voxels,
    load_volume,
    save_volume,
)


class VolumeGridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = Bounds3D(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

    def test_dense_sampling_and_chunks(self) -> None:
        data = np.zeros((4, 4, 4), dtype=bool)
        data[1, 1, 1] = True
        grid = DenseVolumeGrid(data, self.bounds, chunk_size=2)

        world_point = grid.transform.index_to_world(np.array([[1, 1, 1]]))
        self.assertTrue(bool(grid.sample_world(world_point)[0]))
        self.assertEqual(grid.active_voxel_count(), 1)
        self.assertEqual(len(list(grid.iter_active_chunks())), 1)

    def test_chunked_dense_conversion(self) -> None:
        data = np.zeros((5, 4, 3), dtype=np.float32)
        data[4, 3, 2] = 0.75
        grid = ChunkedVolumeGrid.from_dense(
            data,
            self.bounds,
            value_type="occupancy_prob",
            default_value=0.0,
            chunk_size=2,
        )

        np.testing.assert_array_equal(grid.to_dense(), data)
        self.assertEqual(grid.active_voxel_count(), 1)
        self.assertEqual(grid.get_chunk(ChunkKey(2, 1, 1)).shape, (2, 2, 2))

    def test_sparse_hash_round_trip(self) -> None:
        data = np.zeros((6, 6, 6), dtype=bool)
        data[1:3, 1:3, 1:3] = True
        grid = SparseHashVolumeGrid.from_dense(data, self.bounds, chunk_size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = save_volume(grid, tmpdir, generation_seed=123)
            loaded = load_volume(tmpdir)

        self.assertEqual(metadata.backend, "sparse_hash")
        np.testing.assert_array_equal(loaded.to_dense(), data)

    def test_vectorized_surface_extraction(self) -> None:
        cube = np.ones((3, 3, 3), dtype=bool)
        surface = extract_surface_voxels(cube, prefer_scipy=False)
        self.assertEqual(int(surface.sum()), 26)
        self.assertFalse(bool(surface[1, 1, 1]))

    def test_meshing_unavailable_or_structured_result(self) -> None:
        data = np.zeros((3, 3, 3), dtype=bool)
        data[1, 1, 1] = True
        grid = DenseVolumeGrid(data, self.bounds)
        result = extract_mesh(grid, method="point_cloud_only")

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.method, "point_cloud_only")
        self.assertEqual(result.faces.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
