"""Tests for config defaults and validation (pure Python)."""

from __future__ import annotations

import unittest

from config import BlockingConfig


class TestConfigDefaults(unittest.TestCase):
    def test_defaults_match_legacy(self) -> None:
        cfg = BlockingConfig()
        self.assertEqual(cfg.reconstruction.unit_scale, 0.01)
        self.assertEqual(cfg.reconstruction.num_slices, 10)
        self.assertEqual(cfg.mesh_join.mode, "boolean")
        self.assertEqual(cfg.render_silhouette.resolution, (512, 512))
        self.assertEqual(cfg.canonicalize.output_size, 256)

    def test_validate(self) -> None:
        cfg = BlockingConfig()
        cfg.validate()

    def test_refinement_lab_defaults_are_temp_scoped(self) -> None:
        cfg = BlockingConfig()
        self.assertEqual(cfg.refinement_lab.default_output_root, "temp/refinement-runs")
        self.assertEqual(cfg.refinement_lab.default_suite, "default-vase")
        self.assertEqual(
            cfg.refinement_lab.default_track, "profile-loft-refinement"
        )
        self.assertEqual(cfg.refinement_lab.default_search, "grid")
        self.assertEqual(cfg.refinement_lab.default_objective, "quality_win")
        self.assertTrue(cfg.refinement_lab.html_report)
        self.assertTrue(cfg.refinement_lab.write_overlays)
        self.assertTrue(cfg.refinement_lab.write_bounds_debug)
        self.assertTrue(cfg.refinement_lab.write_autopsy)
        self.assertTrue(cfg.refinement_lab.append_leaderboard)
        self.assertFalse(cfg.refinement_lab.allow_subprocess_blender)
        self.assertIn("refinement_lab", cfg.to_dict())
        self.assertEqual(cfg.shape_program.root_strategy, "hybrid_profile_bounds")
        self.assertTrue(cfg.shape_program.compile_blender)
        self.assertFalse(cfg.shape_program.run_export_qa)
        self.assertEqual(cfg.shape_program.export_qa_targets, ("obj", "glb"))
        self.assertFalse(cfg.shape_program.evaluate_texture_materials)
        self.assertFalse(cfg.shape_program.uv_strict)
        self.assertEqual(cfg.shape_program.material_target, "pbr")
        self.assertIsNone(cfg.shape_program.texture_reference_dir)
        self.assertIsNone(cfg.shape_program.max_texture_memory_mb)
        self.assertIn("shape_program", cfg.to_dict())


if __name__ == "__main__":
    unittest.main()
