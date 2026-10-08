"""Tests for refinement lab suite and track presets."""

from __future__ import annotations

import unittest

from refinement_lab.preset_catalog import (
    get_suite_preset,
    get_track_preset,
    list_suites,
    list_tracks,
)


class RefinementLabPresetTests(unittest.TestCase):
    def test_required_suites_exist(self) -> None:
        suites = set(list_suites())
        for name in (
            "default-vase",
            "synthetic-blender-smoke",
            "synthetic-smoke",
            "synthetic-visual-hull",
            "synthetic-profile-band",
            "synthetic-primitive-fit",
            "synthetic-adversarial",
            "synthetic-adversarial-curriculum",
            "synthetic-material-appearance",
            "synthetic-nightly",
        ):
            self.assertIn(name, suites)

    def test_required_tracks_exist_and_validate_parameters(self) -> None:
        tracks = set(list_tracks())
        for name in (
            "mask-refinement",
            "profile-loft-refinement",
            "visual-hull-transform",
            "visual-hull-quality",
            "primitive-fit",
            "gaussian-proxy",
            "differentiable-refine",
            "ensemble-selection",
            "shape-program-editability",
            "content-adaptive-patches",
            "adversarial-curriculum-hardening",
            "sota-metric-bundle",
        ):
            preset = get_track_preset(name)
            self.assertIn(name, tracks)
            self.assertTrue(preset.modes)
            for parameter in preset.parameters:
                parameter.validate()

    def test_adversarial_curriculum_preset_targets_progressive_suites(self) -> None:
        suite = get_suite_preset("synthetic-adversarial-curriculum")
        track = get_track_preset("adversarial-curriculum-hardening")

        self.assertEqual(
            suite.synthetic_suites,
            ("adversarial-level-1", "adversarial-level-2", "adversarial-level-3"),
        )
        self.assertEqual(track.default_search, "successive_halving")
        self.assertIn("ensemble", track.modes)
        self.assertIn("moonshot", track.tags)

    def test_unknown_suite_fails(self) -> None:
        with self.assertRaises(KeyError):
            get_suite_preset("missing-suite")


if __name__ == "__main__":
    unittest.main()
