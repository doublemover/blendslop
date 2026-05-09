"""Pure tests for distributional profile-band extraction."""

from __future__ import annotations

import unittest

import numpy as np

from reconstruction.profile_bands import (
    ProfileBandDistributionConfig,
    distributional_profile_bands,
    profile_distribution_summary,
)
from reconstruction.target_builder import mask_to_profile_bands


class ProfileBandDistributionTests(unittest.TestCase):
    def test_distributional_bands_capture_uncertain_edge_width_variance(self) -> None:
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:15, 9:21] = True
        probability = np.zeros(mask.shape, dtype=np.float32)
        probability[:, :] = mask.astype(np.float32)
        probability[5:15, 7:9] = 0.4
        probability[5:15, 21:23] = 0.4
        confidence = np.ones(mask.shape, dtype=np.float32)
        confidence[5:15, 7:23] = 0.75

        bands = distributional_profile_bands(
            mask,
            sample_count=5,
            view="front",
            probability=probability,
            confidence=confidence,
            config=ProfileBandDistributionConfig(
                thresholds=(0.35, 0.5, 0.65),
                include_morphology=False,
            ),
        )
        occupied = [band for band in bands if band.width_px > 0.0]

        self.assertEqual(len(bands), 5)
        self.assertTrue(all(band.source_view == "front" for band in bands))
        self.assertTrue(any(band.width_std > 0.0 for band in occupied))
        self.assertTrue(any(band.moments["width_p95"] > band.moments["width_p05"] for band in occupied))
        self.assertTrue(all("probability_entropy" in band.moments for band in bands))
        summary = profile_distribution_summary(bands)
        self.assertEqual(summary["band_count"], 5.0)
        self.assertGreater(summary["variant_count"], 1.0)
        self.assertGreater(summary["width_std_max"], 0.0)

    def test_distributional_bands_preserve_multi_interval_rows_and_holes(self) -> None:
        mask = np.zeros((12, 24), dtype=bool)
        mask[4:8, 3:8] = True
        mask[4:8, 15:21] = True

        bands = distributional_profile_bands(
            mask,
            sample_count=3,
            view="front",
            config=ProfileBandDistributionConfig(include_morphology=False),
        )
        multi = [band for band in bands if len(band.intervals) > 1]

        self.assertTrue(multi)
        self.assertGreater(multi[0].moments["component_count_mean"], 1.0)
        self.assertEqual(len(multi[0].holes), 1)
        self.assertGreater(multi[0].holes[0].width, 0.0)

    def test_legacy_mask_to_profile_bands_remains_available(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 2:8] = True

        legacy = mask_to_profile_bands(mask, sample_count=3, view="front")
        distributional = distributional_profile_bands(mask, sample_count=3, view="front")

        self.assertEqual(len(legacy), 3)
        self.assertEqual(len(distributional), 3)
        self.assertTrue(all(hasattr(band, "width_std") for band in distributional))

    def test_shape_validation_rejects_mismatched_probability(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        with self.assertRaises(ValueError):
            distributional_profile_bands(
                mask,
                sample_count=2,
                probability=np.zeros((4, 4), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
