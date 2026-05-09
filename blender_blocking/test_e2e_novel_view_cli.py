#!/usr/bin/env python3
"""Pure tests for E2E novel-view CLI plumbing."""

from __future__ import annotations

import argparse
from pathlib import Path
import unittest

from integration.blender_ops.render_utils import parse_orbit_view_degrees
from test_e2e_validation import (
    _aggregate_novel_reports,
    _novel_threshold,
    _novel_view_gate,
    _novel_view_names_from_args,
    _parse_view_reference_entries,
)


class E2ENovelViewCliTests(unittest.TestCase):
    def test_orbit_view_parser_accepts_common_names(self) -> None:
        self.assertEqual(parse_orbit_view_degrees("orbit_045"), 45.0)
        self.assertEqual(parse_orbit_view_degrees("azimuth-135"), 135.0)
        self.assertEqual(parse_orbit_view_degrees("view_405"), 45.0)
        self.assertIsNone(parse_orbit_view_degrees("front"))

    def test_view_reference_entries_are_validated_and_normalized(self) -> None:
        refs = _parse_view_reference_entries(
            ("orbit_045=temp/ref.png", "front=C:/tmp/front.png")
        )

        self.assertEqual(refs["orbit_045"], Path("temp/ref.png"))
        self.assertEqual(refs["front"], Path("C:/tmp/front.png"))
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_view_reference_entries(("diagonal=temp/ref.png",))

    def test_novel_view_names_include_angles_references_and_manual_views(self) -> None:
        args = argparse.Namespace(
            novel_view=("orbit_090",),
            novel_view_angles=("45", "135"),
            novel_view_reference=("orbit_045=temp/ref.png",),
        )

        names = _novel_view_names_from_args(args)

        self.assertEqual(names, ("orbit_090", "orbit_045", "orbit_135"))

    def test_threshold_zero_disables_metric_gate(self) -> None:
        self.assertIsNone(_novel_threshold(0.0))
        self.assertIsNone(_novel_threshold(None))
        self.assertEqual(_novel_threshold(20.0), 20.0)

    def test_novel_view_gate_tracks_per_metric_failures(self) -> None:
        gate = _novel_view_gate(
            {"psnr": 18.0, "ssim": 0.7, "lpips": 0.4},
            psnr_threshold=20.0,
            ssim_threshold=0.65,
            lpips_threshold=0.35,
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(len(gate["failures"]), 2)
        self.assertIn("PSNR", gate["failures"][0])
        self.assertIn("LPIPS", gate["failures"][1])

    def test_aggregate_novel_reports_requires_all_views_and_summary_gate(self) -> None:
        summary = _aggregate_novel_reports(
            {
                "front": {
                    "psnr": 32.0,
                    "ssim": 0.91,
                    "lpips": 0.12,
                    "mse": 0.01,
                    "gate": {"passed": True},
                },
                "orbit_045": {
                    "psnr": 26.0,
                    "ssim": 0.82,
                    "lpips": 0.18,
                    "mse": 0.02,
                    "gate": {"passed": True},
                },
            },
            missing_views=(),
            psnr_threshold=20.0,
            ssim_threshold=0.65,
            lpips_threshold=0.35,
        )

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["image_count"], 2)
        self.assertAlmostEqual(summary["psnr"], 29.0)
        self.assertAlmostEqual(summary["ssim"], 0.865)


if __name__ == "__main__":
    unittest.main()
