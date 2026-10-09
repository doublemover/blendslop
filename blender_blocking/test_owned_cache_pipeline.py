"""Focused pure checks for opt-in owned cache writes through visual-hull config."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from config import BlockingConfig
from reconstruction.backends.visual_hull import VisualHullBackend
from reconstruction.backends.visual_hull.config import visual_hull_run_config
from reconstruction.point_cloud import visual_hull_grid_from_target
from reconstruction.types import CandidateRequest
from test_volume_chunk_cache import _small_target


class OwnedCachePipelineTests(unittest.TestCase):
    def test_option_is_strict_boolean_and_defaults_remain_legacy(self):
        config = BlockingConfig()
        config.validate()
        self.assertFalse(config.visual_hull.cache_owned_writes)
        self.assertFalse(config.to_dict()["visual_hull"]["cache_owned_writes"])
        self.assertFalse(visual_hull_run_config({}).cache_owned_writes)
        self.assertIsNone(visual_hull_run_config({"cache_owned_writes": True}).cache_directory)
        for invalid in (1, "true", None):
            with self.subTest(invalid=invalid):
                config.visual_hull.cache_owned_writes = invalid
                with self.assertRaisesRegex(ValueError, "cache_owned_writes must be a boolean"):
                    config.validate()
                with self.assertRaisesRegex(ValueError, "cache_owned_writes must be a boolean"):
                    visual_hull_run_config({"cache_owned_writes": invalid})
                with self.assertRaisesRegex(ValueError, "cache_owned_writes must be a boolean"):
                    visual_hull_grid_from_target(_small_target(), cache_owned_writes=invalid)

    def test_backend_retains_current_store_receipt_without_adopting_cache_hits(self):
        target = _small_target()
        mask_before = target.constraints[0].mask.copy()
        backend = VisualHullBackend()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BlockingConfig()
            hull = config.visual_hull
            hull.backend = "chunked"
            hull.resolution = 4
            hull.chunk_size = 4
            hull.boundary_refine = False
            hull.mesh_method = "points"
            hull.cache_directory = str(root / "legacy")
            hull.cache_namespace = "pipeline-test"
            config.validate()

            def reconstruct(candidate_id, overrides=None):
                payload = config.to_dict()["visual_hull"]
                payload["emit_editable_proxy"] = False
                payload.update(overrides or {})
                result = backend.reconstruct(CandidateRequest(
                    candidate_id=candidate_id, backend_name=backend.name,
                    target=target, config=payload,
                ))
                self.assertFalse(result.errors, result.errors)
                return result, result.metric_result.extras["chunk_cache"]

            legacy, legacy_metrics = reconstruct("legacy-default")
            self.assertEqual(legacy_metrics["writes"], 1)
            self.assertNotIn("ownership_status", legacy_metrics)
            self.assertFalse(list(root.rglob(".cache-runs")))

            hull.cache_directory = str(root / "owned")
            hull.cache_owned_writes = True
            first, first_metrics = reconstruct("owned-fresh")
            self.assertEqual(first_metrics["writes"], 1)
            self.assertTrue(first_metrics["owned_writes"])
            self.assertEqual(first_metrics["ownership_status"], "retained")
            first_receipt = Path(first_metrics["last_ownership_receipt"])
            self.assertEqual(json.loads(first_receipt.read_text())["status"], "succeeded")
            self.assertEqual(first.payload.chunk_cache_last_ownership_receipt, first_receipt)
            archives = list((root / "owned" / "pipeline-test").glob("*.npz"))
            self.assertEqual(len(archives), 1)
            archive_path = archives[0]
            archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            with np.load(archive_path, allow_pickle=False) as archive:
                self.assertEqual(archive["data"].shape, (4, 4, 4))
                self.assertEqual(archive["data"].dtype, np.dtype(bool))
                self.assertEqual(json.loads(str(archive["metadata"].item()))["resolution"], 4)

            second, second_metrics = reconstruct("owned-cache-hit")
            self.assertEqual(second_metrics["hits"], 1)
            self.assertEqual(second_metrics["writes"], 0)
            self.assertEqual(second_metrics["ownership_status"], "unrun")
            self.assertIsNone(second_metrics["last_ownership_receipt"])
            self.assertIsNone(second.payload.chunk_cache_last_ownership_receipt)
            self.assertEqual(list(root.rglob("cache-store.json")), [first_receipt])
            np.testing.assert_array_equal(first.payload.to_dense(), second.payload.to_dense())
            np.testing.assert_array_equal(legacy.payload.to_dense(), first.payload.to_dense())

            collision, collision_metrics = reconstruct("owned-collision", {"cache_read": False})
            self.assertEqual(collision_metrics["writes"], 0)
            self.assertEqual(collision_metrics["write_errors"], 1)
            self.assertEqual(collision_metrics["ownership_status"], "retained")
            failed_receipt = Path(collision_metrics["last_ownership_receipt"])
            self.assertNotEqual(failed_receipt, first_receipt)
            failed = json.loads(failed_receipt.read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertTrue(failed["publication_collision"])
            self.assertEqual(hashlib.sha256(archive_path.read_bytes()).hexdigest(), archive_hash)
            self.assertEqual(json.loads(first_receipt.read_text())["status"], "succeeded")
            np.testing.assert_array_equal(collision.payload.to_dense(), first.payload.to_dense())
        np.testing.assert_array_equal(target.constraints[0].mask, mask_before)


if __name__ == "__main__":
    unittest.main()
