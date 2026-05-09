"""Tests for configuration validation (pure Python)."""

from __future__ import annotations

import unittest

from config import (
    CanonicalizeConfig,
    LoftMeshOptions,
    MeshJoinConfig,
    GaussianEllipsoidConfig,
    ProfileSamplingConfig,
    ReconstructionConfig,
    RenderConfig,
    SilhouetteExtractConfig,
    DifferentiableRenderConfig,
)


class TestConfigValidation(unittest.TestCase):
    def test_invalid_reconstruction_mode(self) -> None:
        cfg = ReconstructionConfig(reconstruction_mode="invalid")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_num_slices(self) -> None:
        cfg = ReconstructionConfig(num_slices=0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_mesh_join_mode(self) -> None:
        cfg = MeshJoinConfig(mode="noop")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_sampling_policy(self) -> None:
        cfg = ProfileSamplingConfig(sample_policy="bad_policy")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_cap_mode(self) -> None:
        cfg = LoftMeshOptions(cap_mode="bad_cap")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_render_resolution(self) -> None:
        cfg = RenderConfig(resolution=(16, 16))
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_canonicalize_output(self) -> None:
        cfg = CanonicalizeConfig(output_size=16)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_silhouette_threshold(self) -> None:
        cfg = SilhouetteExtractConfig(alpha_threshold=300)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_gaussian_renderer(self) -> None:
        cfg = GaussianEllipsoidConfig(renderer="bad_renderer")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_differentiable_backend(self) -> None:
        cfg = DifferentiableRenderConfig(backend="unsupported")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_optional_dependency_policy(self) -> None:
        cfg = DifferentiableRenderConfig(optional_dependency_policy="maybe")
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
