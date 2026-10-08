#!/usr/bin/env python3
"""Pure tests for deterministic volume chunk caching."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from reconstruction.point_cloud import visual_hull_grid_from_target
from reconstruction.types import (
    Bounds3D as ReconstructionBounds3D,
    OrthographicCameraSpec,
    ReconstructionTarget,
    ViewConstraint,
)
from volume import ChunkCacheKey, VolumeChunkCache, chunk_cache_key


class VolumeChunkCacheTests(unittest.TestCase):
    def test_cache_key_is_stable_for_json_equivalent_payloads(self) -> None:
        first = chunk_cache_key("visual hull", {"b": 2, "a": [1, 2]})
        second = chunk_cache_key("visual hull", {"a": [1, 2], "b": 2})

        self.assertIsInstance(first, ChunkCacheKey)
        self.assertEqual(first.namespace, "visual_hull")
        self.assertEqual(first.digest, second.digest)

    def test_cache_round_trips_bool_chunk_and_reports_stats(self) -> None:
        chunk = np.zeros((2, 2, 2), dtype=bool)
        chunk[0, 0, 0] = True
        key = chunk_cache_key("unit", {"chunk": [0, 0, 0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = VolumeChunkCache(tmpdir, namespace="unit")
            self.assertIsNone(cache.load(key, expected_shape=(2, 2, 2)))
            self.assertTrue(cache.store(key, chunk, metadata={"case": "roundtrip"}))
            loaded = cache.load(key, expected_shape=(2, 2, 2))
            stats = cache.stats()

        np.testing.assert_array_equal(loaded, chunk)
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.writes, 1)
        self.assertGreater(stats.bytes_written, 0)

    def test_visual_hull_chunked_builder_reuses_cached_chunks(self) -> None:
        target = _small_target()
        with tempfile.TemporaryDirectory() as tmpdir:
            first = visual_hull_grid_from_target(
                target,
                resolution=6,
                chunk_size=2,
                backend="chunked",
                cache_directory=tmpdir,
                cache_namespace="vh-test",
            )
            second = visual_hull_grid_from_target(
                target,
                resolution=6,
                chunk_size=2,
                backend="chunked",
                cache_directory=tmpdir,
                cache_namespace="vh-test",
            )

        first_cache = first.chunk_cache_status.to_dict()
        second_cache = second.chunk_cache_status.to_dict()
        np.testing.assert_array_equal(first.to_dense(), second.to_dense())
        self.assertGreater(first_cache["writes"], 0)
        self.assertGreater(second_cache["hits"], 0)
        self.assertEqual(second_cache["write_errors"], 0)


def _small_target() -> ReconstructionTarget:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    return ReconstructionTarget(
        constraints=(
            ViewConstraint(
                view="front",
                mask=mask,
                camera=OrthographicCameraSpec(
                    view_name="front",
                    axis="z",
                    azimuth_deg=0.0,
                ),
            ),
        ),
        bounds=ReconstructionBounds3D(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
    )


if __name__ == "__main__":
    unittest.main()
