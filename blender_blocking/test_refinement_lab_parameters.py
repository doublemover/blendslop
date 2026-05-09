"""Tests for refinement parameter catalog ownership."""

from __future__ import annotations

import unittest

from config import BlockingConfig
from refinement_lab.contracts import ExperimentVariant, ParameterSpec
from refinement_lab.parameters import (
    apply_variant_parameter_to_config,
    missing_catalog_entries,
    parameter_path,
    validate_parameter_catalog,
)
from refinement_lab.presets import get_track_preset, list_tracks
from refinement_lab.runner import _apply_variant_to_config


class RefinementParameterCatalogTests(unittest.TestCase):
    def test_all_track_parameters_are_cataloged_or_lab_only(self) -> None:
        tracks = [get_track_preset(name) for name in list_tracks()]

        issues = validate_parameter_catalog(tracks=tracks, config=BlockingConfig())

        self.assertEqual([issue.to_dict() for issue in issues], [])

    def test_catalog_reports_missing_non_lab_parameter(self) -> None:
        issues = missing_catalog_entries(
            (
                ParameterSpec(name="unknown_real_param", cli_arg="--unknown"),
                ParameterSpec(name="scratch_note", lab_only=True),
            ),
            track="test-track",
        )

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].parameter, "unknown_real_param")
        self.assertEqual(issues[0].track, "test-track")

    def test_apply_parameter_updates_config_with_coercions(self) -> None:
        cfg = BlockingConfig()

        self.assertTrue(apply_variant_parameter_to_config(cfg, "vh_resolution", 72))
        self.assertTrue(
            apply_variant_parameter_to_config(
                cfg,
                "primitive_loss_weights_json",
                '{"silhouette": 2.0}',
            )
        )
        self.assertTrue(
            apply_variant_parameter_to_config(
                cfg,
                "primitive_families",
                "ellipsoid,superquadric",
            )
        )
        self.assertTrue(
            apply_variant_parameter_to_config(
                cfg,
                "shape_evaluate_texture_materials",
                True,
            )
        )
        self.assertTrue(
            apply_variant_parameter_to_config(cfg, "shape_uv_strict", True)
        )
        self.assertTrue(
            apply_variant_parameter_to_config(cfg, "shape_material_target", "simple")
        )
        self.assertTrue(
            apply_variant_parameter_to_config(
                cfg,
                "shape_max_texture_memory_mb",
                96,
            )
        )
        self.assertFalse(
            apply_variant_parameter_to_config(cfg, "unknown_lab_hint", True)
        )

        self.assertEqual(cfg.visual_hull.resolution, 72)
        self.assertEqual(cfg.primitive_fit.loss_weights, {"silhouette": 2.0})
        self.assertEqual(
            cfg.primitive_fit.primitive_families,
            ("ellipsoid", "superquadric"),
        )
        self.assertTrue(cfg.shape_program.evaluate_texture_materials)
        self.assertTrue(cfg.shape_program.uv_strict)
        self.assertEqual(cfg.shape_program.material_target, "simple")
        self.assertEqual(cfg.shape_program.max_texture_memory_mb, 96)

    def test_runner_uses_shared_catalog_for_variants(self) -> None:
        cfg = BlockingConfig()
        variant = ExperimentVariant(
            variant_id="v",
            label="Variant",
            mode="visual_hull_voxel",
            parameters={
                "ensemble_candidates": ["visual_hull_voxel", "shape_program"],
                "vh_backend": "chunked",
                "vh_chunk_size": 16,
            },
        )

        _apply_variant_to_config(cfg, variant)

        self.assertEqual(cfg.reconstruction.reconstruction_mode, "visual_hull_voxel")
        self.assertEqual(cfg.visual_hull.backend, "chunked")
        self.assertEqual(cfg.visual_hull.chunk_size, 16)
        self.assertEqual(len(cfg.ensemble.candidates), 2)
        self.assertEqual(
            cfg.ensemble.candidates[1].backend_name,
            "shape_program",
        )
        self.assertEqual(parameter_path("vh_backend"), ("visual_hull", "backend"))


if __name__ == "__main__":
    unittest.main()
