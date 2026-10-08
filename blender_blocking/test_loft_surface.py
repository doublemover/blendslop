"""Pure geometric contracts for smooth, stepped and sharp loft plans."""

import unittest
import numpy as np

from config import LoftMeshOptions
from geometry.loft_surface import prepare_loft_surface
from geometry.profile_models import EllipticalSlice


class LoftSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.sections = [EllipticalSlice(z=z, rx=r, ry=r / 2)
                         for z, r in ((0, 1), (1, .5), (2, 1))]

    def test_smooth_preserves_landmarks_and_never_overshoots(self):
        actual = prepare_loft_surface(self.sections, "smooth", 4)
        self.assertEqual(len(actual), 9)
        for index, source in enumerate(self.sections):
            self.assertEqual(actual[index * 4].rx, source.rx)
            self.assertEqual(actual[index * 4].ry, source.ry)
            self.assertEqual(actual[index * 4].z, source.z)
        self.assertTrue(all(.5 <= row.rx <= 1 for row in actual))
        self.assertEqual(actual, prepare_loft_surface(self.sections, "smooth", 4))

    def test_smooth_curve_has_no_slope_jump_at_neck(self):
        actual = prepare_loft_surface(self.sections, "smooth", 16)
        left = (actual[16].rx - actual[15].rx) / (actual[16].z - actual[15].z)
        right = (actual[17].rx - actual[16].rx) / (actual[17].z - actual[16].z)
        self.assertAlmostEqual(left, -right)
        self.assertLess(abs(left), .04)

    def test_sharp_preserves_authored_corners_exactly(self):
        self.assertEqual(prepare_loft_surface(self.sections, "sharp"), self.sections)

    def test_steps_make_zero_height_nested_shoulders(self):
        actual = prepare_loft_surface(self.sections, "stepped")
        self.assertEqual([s.z for s in actual], [0, .5, .5, 1, 1.5, 1.5, 2])
        self.assertEqual([s.rx for s in actual], [1, 1, .5, .5, .5, 1, 1])

    def test_constant_steps_do_not_emit_duplicate_rings(self):
        sections = [EllipticalSlice(z=z, rx=1, ry=1) for z in (0, 1, 2)]
        self.assertEqual(prepare_loft_surface(sections, "stepped"), sections)

    def test_steps_reject_crossing_or_shifted_shoulders(self):
        for last in (EllipticalSlice(z=1, rx=2, ry=.5),
                     EllipticalSlice(z=1, rx=2, ry=1),
                     EllipticalSlice(z=1, rx=2, ry=2, cx=.1),
                     EllipticalSlice(z=1, rx=0, ry=0)):
            with self.subTest(last=last), self.assertRaises(ValueError):
                prepare_loft_surface([EllipticalSlice(z=0, rx=1, ry=1), last], "stepped")

    def test_invalid_source_and_unbounded_budget_fail(self):
        for sections in ([], [EllipticalSlice(z=0, rx=-1, ry=1)],
                         [EllipticalSlice(z=float("nan"), rx=1, ry=1)],
                         [self.sections[1], self.sections[0]]):
            with self.subTest(sections=sections), self.assertRaises(ValueError):
                prepare_loft_surface(sections)
        for value in (True, 0, 17, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare_loft_surface(self.sections, subdivisions=value)

    def test_config_roundtrip_and_invalid_mode(self):
        cfg = LoftMeshOptions(surface_mode="stepped", surface_subdivisions=8)
        cfg.validate()
        self.assertEqual(cfg.to_dict()["surface_mode"], "stepped")
        self.assertEqual(cfg.to_dict()["surface_subdivisions"], 8)
        with self.assertRaises(ValueError):
            LoftMeshOptions(surface_mode="melt_everything").validate()


if __name__ == "__main__":
    unittest.main()
