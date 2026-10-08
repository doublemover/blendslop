"""Pure tests for the typed silhouette pipeline."""

from __future__ import annotations

import unittest

import numpy as np

from geometry.silhouette_pipeline import (
    build_uncertain_mask,
    canonicalize_silhouette,
    extract_silhouette_candidates,
    extract_silhouette_mask,
)
from metrics.silhouette import boundary_iou, signed_distance_silhouette_loss, soft_iou


class TestSilhouettePipeline(unittest.TestCase):
    def test_all_opaque_alpha_does_not_force_full_canvas(self) -> None:
        image = np.full((48, 48, 4), 255, dtype=np.uint8)
        image[:, :, :3] = 255
        image[12:36, 18:30, :3] = 0

        result = extract_silhouette_mask(image)

        self.assertEqual(result.source, "luma")
        self.assertLess(result.mask.mean(), 0.5)
        self.assertIsNotNone(result.bbox)

    def test_candidate_diagnostics_and_uncertainty_maps(self) -> None:
        image = np.full((32, 32), 255, dtype=np.uint8)
        image[8:24, 10:22] = 0

        candidates = extract_silhouette_candidates(image)
        uncertain = build_uncertain_mask(image)

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(uncertain.hard_mask.shape, image.shape)
        self.assertEqual(uncertain.foreground_prob.shape, image.shape)
        self.assertEqual(uncertain.confidence.shape, image.shape)
        self.assertEqual(uncertain.boundary_uncertainty.shape, image.shape)

    def test_canonicalization_returns_transform_metadata(self) -> None:
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:15, 10:20] = True

        canonical = canonicalize_silhouette(mask, output_size=40, padding_frac=0.0)

        self.assertEqual(canonical.mask.shape, (40, 40))
        self.assertEqual(canonical.source_bbox.to_xyxy(), (10, 5, 20, 15))
        self.assertGreater(canonical.transform["scale"], 0.0)

    def test_metric_helpers(self) -> None:
        ref = np.zeros((32, 32), dtype=bool)
        cand = np.zeros((32, 32), dtype=bool)
        ref[8:24, 8:24] = True
        cand[9:25, 8:24] = True

        b_iou, warnings = boundary_iou(ref, cand)
        self.assertGreater(b_iou, 0.0)
        self.assertFalse(warnings)
        self.assertGreater(soft_iou(ref.astype(float), cand.astype(float)), 0.0)
        self.assertGreater(signed_distance_silhouette_loss(ref, cand), 0.0)


if __name__ == "__main__":
    unittest.main()
