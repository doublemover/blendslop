"""Pure tests for reconstruction backend contracts and target building."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import BlockingConfig
from reconstruction.ensemble import CandidateConfig, EnsembleRunner
from reconstruction.registry import get_backend, list_backends, register_builtin_backends
from reconstruction.target_builder import build_target_from_images, mask_to_profile_bands
from reconstruction.types import CandidateBudget, UncertainProfileBand


def _rgb_rect() -> np.ndarray:
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[8:26, 10:23, :] = 0
    return image


class ReconstructionContractsTests(unittest.TestCase):
    def test_target_builder_preserves_intervals_and_artifacts(self) -> None:
        cfg = BlockingConfig()
        with tempfile.TemporaryDirectory() as tmp:
            result = build_target_from_images(
                {"front": _rgb_rect(), "side": _rgb_rect()},
                config=cfg,
                artifact_root=Path(tmp),
                bounds_minmax=((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
                profile_samples=8,
            )

        self.assertEqual(result.target.views(), ("front", "side"))
        self.assertIn("front", result.target.profile_bands)
        self.assertGreater(len(result.target.profile_bands["front"]), 0)
        self.assertTrue(any(path.name.endswith(".npy") for path in result.artifact_paths.values()))
        self.assertEqual(result.target.bounds.size, (2.0, 2.0, 2.0))
        self.assertIn("front", result.probabilities)
        self.assertEqual(result.probabilities["front"].shape, result.masks["front"].shape)
        self.assertIsNotNone(result.target.constraints[0].uncertainty)
        self.assertTrue(
            any(
                path.name.endswith("-probability.npy")
                for path in result.artifact_paths.values()
            )
        )
        self.assertIsInstance(result.target.profile_bands["front"][0], UncertainProfileBand)

    def test_profile_bands_capture_multiple_intervals(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[:, 2:5] = True
        mask[:, 12:16] = True
        bands = mask_to_profile_bands(mask, sample_count=3, view="front")

        self.assertEqual(len(bands), 3)
        self.assertEqual(len(bands[1].intervals), 2)
        self.assertEqual(len(bands[1].holes), 1)

    def test_profile_bands_preserve_uncertainty_moments(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[:, 4:14] = True
        probability = mask.astype(np.float32) * 0.75
        confidence = np.full(mask.shape, 0.5, dtype=np.float32)

        bands = mask_to_profile_bands(
            mask,
            sample_count=3,
            view="front",
            probability=probability,
            confidence=confidence,
        )

        self.assertIsInstance(bands[1], UncertainProfileBand)
        self.assertIn("probability_width_px", bands[1].moments)
        self.assertGreater(bands[1].width_std, 0.0)
        self.assertAlmostEqual(bands[1].confidence, 0.5)

    def test_builtin_registry_and_visual_hull_candidate(self) -> None:
        register_builtin_backends()
        names = {info.name for info in list_backends()}
        self.assertIn("visual_hull_voxel", names)

        cfg = BlockingConfig()
        target = build_target_from_images(
            {"front": _rgb_rect(), "side": _rgb_rect()},
            config=cfg,
            bounds_minmax=((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
            profile_samples=4,
        ).target
        backend = get_backend("visual_hull_voxel")
        with tempfile.TemporaryDirectory() as tmp:
            request = EnsembleRunner().build_requests(
                target=target,
                candidates=(
                    CandidateConfig(
                        backend_name="visual_hull_voxel",
                        config={"resolution": 8, "mesh_method": "points"},
                    ),
                ),
                artifact_root=Path(tmp),
                budget=CandidateBudget(memory_budget_mb=64),
            )[0]
            result = backend.reconstruct(request)

        self.assertEqual(result.status, "success")
        self.assertGreater(result.metric_result.extras["occupied_voxels"], 0)


if __name__ == "__main__":
    unittest.main()
