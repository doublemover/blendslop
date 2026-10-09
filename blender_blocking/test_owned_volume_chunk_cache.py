"""Real bounded NPZ cache publication, collision and retained failure contracts."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from utils.run_ownership import plan_run_reclamation
from volume.chunk_cache import VolumeChunkCache, chunk_cache_key
import volume.owned_chunk_cache as owned


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def owner_state(receipt, state):
    root = Path(receipt).parent
    manifest = read(root / "run-ownership.json")
    assert manifest["state"] == state, manifest
    assert read(root / "run-lease.json")["status"] == "released"
    assert plan_run_reclamation(root)["status"] == "dry_run_ready"
    return root, manifest


class OwnedVolumeChunkCacheTests(unittest.TestCase):
    def test_default_overwrite_and_stats_contract_stays_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = VolumeChunkCache(folder, namespace="unit")
            key = chunk_cache_key("unit", {"key": 1})
            self.assertTrue(cache.store(key, np.zeros((2, 2, 2), dtype=bool)))
            self.assertTrue(cache.store(key, np.ones((2, 2, 2), dtype=bool)))
            np.testing.assert_array_equal(cache.load(key, expected_shape=(2, 2, 2)), True)
            self.assertEqual(cache.stats().writes, 2)
            self.assertIsNone(cache.last_ownership_receipt)
            self.assertFalse((Path(folder) / "unit/.cache-runs").exists())

    def test_owned_store_loads_through_existing_reader_and_keeps_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = VolumeChunkCache(folder, namespace="unit", owned_writes=True)
            key = chunk_cache_key("unit", {"chunk": [1, 2, 3]})
            data = np.zeros((3, 2, 4), dtype=bool);data[1, 1, 1] = True
            before = data.copy()
            metadata = {"external": "input", "nested": {"value": 2}}
            self.assertTrue(cache.store(key, data, metadata=metadata))
            root, manifest = owner_state(cache.last_ownership_receipt, "succeeded")
            result = cache.load(key, expected_shape=data.shape)
            np.testing.assert_array_equal(result, before)
            np.testing.assert_array_equal(data, before)
            self.assertEqual(metadata, {"external": "input", "nested": {"value": 2}})
            self.assertEqual(manifest["published_output"]["path"], str(cache.path_for_key(key)))
            self.assertEqual(manifest["published_output"]["sha256"], hashlib.sha256(cache.path_for_key(key).read_bytes()).hexdigest())
            receipt = read(cache.last_ownership_receipt)
            self.assertEqual(receipt["subprocesses_started"], 0)
            self.assertEqual(receipt["array_uncompressed_bytes"], data.nbytes)
            self.assertLessEqual((root / "chunk.npz").stat().st_size, receipt["stage_byte_bound"])
            self.assertEqual(cache.stats().writes, 1)
            self.assertGreater(cache.stats().bytes_written, 0)

    def test_existing_key_collision_preserves_bytes_and_retains_failed_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            key = chunk_cache_key("unit", {"key": "existing"})
            old = VolumeChunkCache(folder, namespace="unit")
            self.assertTrue(old.store(key, np.zeros((2, 2, 2), dtype=bool)))
            before = old.path_for_key(key).read_bytes()
            cache = VolumeChunkCache(folder, namespace="unit", owned_writes=True)
            self.assertFalse(cache.store(key, np.ones((2, 2, 2), dtype=bool)))
            root, manifest = owner_state(cache.last_ownership_receipt, "failed")
            self.assertEqual(old.path_for_key(key).read_bytes(), before)
            self.assertTrue(read(cache.last_ownership_receipt)["publication_collision"])
            self.assertNotIn("published_output", manifest)
            self.assertEqual((cache.stats().writes, cache.stats().write_errors), (0, 1))
            self.assertTrue((root / "chunk.npz").is_file())
            self.assertEqual(next(row for row in manifest["artifacts"] if row["path"] == "chunk.npz")["category"], "diagnostic")

    def test_concurrent_same_key_has_one_winner_and_keeps_deterministic_path(self):
        with tempfile.TemporaryDirectory() as folder:
            key = chunk_cache_key("unit", {"key": "concurrent"})
            caches = [VolumeChunkCache(folder, namespace="unit", owned_writes=True) for _ in range(2)]
            data = np.arange(24, dtype=np.float32).reshape((3, 2, 4))
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(cache.store, key, data) for cache in caches]
                outcomes = [future.result(timeout=10.) for future in futures]
            self.assertEqual(sorted(outcomes), [False, True])
            self.assertNotEqual(caches[0].last_ownership_receipt, caches[1].last_ownership_receipt)
            for cache, succeeded in zip(caches, outcomes):
                owner_state(cache.last_ownership_receipt, "succeeded" if succeeded else "failed")
                self.assertEqual(cache.stats().writes, int(succeeded))
                self.assertEqual(cache.stats().write_errors, int(not succeeded))
            np.testing.assert_array_equal(caches[0].load(key, expected_shape=data.shape, expected_dtype=data.dtype), data)
            self.assertEqual([path.name for path in (Path(folder) / "unit").glob("*.npz")], [key.filename])

    def test_failed_serialization_keeps_old_entry_input_and_partial_history(self):
        with tempfile.TemporaryDirectory() as folder:
            key = chunk_cache_key("unit", {"key": "serialize-failure"})
            old = VolumeChunkCache(folder, namespace="unit")
            self.assertTrue(old.store(key, np.zeros((2, 2, 2), dtype=bool)))
            prior = old.path_for_key(key).read_bytes()
            data = np.ones((2, 2, 2), dtype=bool);snapshot = data.copy()
            metadata = {"input": "unchanged"}
            cache = VolumeChunkCache(folder, namespace="unit", owned_writes=True)
            def partial(stream, **kwargs):
                stream.write(b"partial generated NPZ")
                raise RuntimeError("fixture serialization failure")
            with patch.object(owned.np, "savez_compressed", side_effect=partial):
                self.assertFalse(cache.store(key, data, metadata=metadata))
            root, _ = owner_state(cache.last_ownership_receipt, "failed")
            self.assertEqual((root / "chunk.npz").read_bytes(), b"partial generated NPZ")
            self.assertIn("fixture serialization failure", read(cache.last_ownership_receipt)["error"])
            self.assertEqual(old.path_for_key(key).read_bytes(), prior)
            np.testing.assert_array_equal(data, snapshot)
            self.assertEqual(metadata, {"input": "unchanged"})
            self.assertEqual(cache.stats().write_errors, 1)

    def test_owned_bool_and_uncompressed_input_bounds_refuse_without_copy_or_serialization(self):
        with tempfile.TemporaryDirectory() as folder:
            for flag in (1, "true", None):
                with self.subTest(flag=flag), self.assertRaises(ValueError):
                    VolumeChunkCache(folder, owned_writes=flag)
            cache = VolumeChunkCache(folder, namespace="unit", owned_writes=True)
            key = chunk_cache_key("unit", {"key": "too-big"})
            huge = np.broadcast_to(np.array(True), (1, 1, owned.MAX_ARRAY_BYTES + 1))
            with patch.object(owned.np, "savez_compressed") as serializer:
                self.assertFalse(cache.store(key, huge))
                serializer.assert_not_called()
            self.assertIsNone(cache.last_ownership_receipt)
            self.assertFalse((Path(folder) / "unit/.cache-runs").exists())

    def test_stage_stream_stops_oversized_write_and_read(self):
        stream = io.BytesIO(b"initial")
        writer = owned._BoundedStream(stream, 8)
        writer.seek(0)
        with self.assertRaises(ValueError):
            writer.write(b"x" * 9)
        self.assertEqual(stream.getvalue(), b"initial")
        source = io.BytesIO(b"x" * 30)
        with self.assertRaises(ValueError):
            owned._BoundedStream(source, 8).read()
        self.assertEqual(source.tell(), 9)

    def test_tiny_compressed_stage_cannot_hide_huge_npy_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            stage = Path(folder) / "bogus.npz"
            array = np.zeros((1, 1, 1), dtype=bool);metadata = np.array("{}")
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(header,
                {"descr": np.dtype(bool).str, "fortran_order": False,
                 "shape": (1, 1, owned.MAX_ARRAY_BYTES + 1)})
            meta = io.BytesIO();np.save(meta, metadata)
            with zipfile.ZipFile(stage, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("data.npy", header.getvalue())
                archive.writestr("metadata.npy", meta.getvalue())
            with patch.object(owned.np, "load") as loader, self.assertRaises(ValueError):
                owned._verify_stage(stage, array, metadata, 65536)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
