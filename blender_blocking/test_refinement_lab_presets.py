"""Tests for refinement lab suite and track presets."""

from __future__ import annotations

import unittest

from refinement_lab.presets import (
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
            "synthetic-smoke",
            "synthetic-visual-hull",
            "synthetic-profile-band",
            "synthetic-primitive-fit",
            "synthetic-adversarial",
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
            "sota-metric-bundle",
        ):
            preset = get_track_preset(name)
            self.assertIn(name, tracks)
            self.assertTrue(preset.modes)
            for parameter in preset.parameters:
                parameter.validate()

    def test_unknown_suite_fails(self) -> None:
        with self.assertRaises(KeyError):
            get_suite_preset("missing-suite")


if __name__ == "__main__":
    unittest.main()
