"""Tests for configuration validation (pure Python)."""

from __future__ import annotations

import unittest

from config import (
    CanonicalizeConfig,
    LoftMeshOptions,
    MeshJoinConfig,
    GaussianEllipsoidConfig,
    ProfileSamplingConfig,
    RefinementLabConfig,
    ReconstructionConfig,
    RenderConfig,
    ShapeProgramConfig,
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

    def test_invalid_shape_program_strategy(self) -> None:
        cfg = ShapeProgramConfig(root_strategy="raw_mesh_blob")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_shape_program_segments(self) -> None:
        cfg = ShapeProgramConfig(lathe_segments=4)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_shape_program_export_qa_target(self) -> None:
        cfg = ShapeProgramConfig(export_qa_targets=("obj", "fbx"))
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_refinement_search(self) -> None:
        cfg = RefinementLabConfig(default_search="bad_search")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_refinement_objective(self) -> None:
        cfg = RefinementLabConfig(default_objective="bad_objective")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_refinement_max_runs(self) -> None:
        cfg = RefinementLabConfig(max_runs=0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_refinement_top_k(self) -> None:
        cfg = RefinementLabConfig(top_k=0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_refinement_report_failures(self) -> None:
        cfg = RefinementLabConfig(report_failures="everything")
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
