"""Pure tests for executable calibration diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.calibration import (
    detect_view_role_permutation,
    executable_calibration_sweep,
    mask_alignment_sweep,
    shift_mask,
)


def _square(
    *,
    shape: tuple[int, int] = (48, 48),
    y0: int = 12,
    x0: int = 12,
    size: int = 14,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[y0 : y0 + size, x0 : x0 + size] = True
    return mask


class CalibrationDiagnosticsTests(unittest.TestCase):
    def test_shift_mask_zero_fills_integer_translation(self) -> None:
        mask = _square(shape=(12, 12), y0=2, x0=3, size=3)

        shifted = shift_mask(mask, dx=2, dy=-1)

        self.assertEqual(int(shifted.sum()), int(mask.sum()))
        self.assertTrue(shifted[1:4, 5:8].all())
        self.assertFalse(shifted[2:5, 3:6].all())

    def test_mask_alignment_sweep_recovers_pixel_offset(self) -> None:
        reference = _square(y0=14, x0=10, size=16)
        candidate = shift_mask(reference, dx=2, dy=1)

        report = mask_alignment_sweep(
            {"front": reference},
            {"front": candidate},
            max_offset_px=3,
            step_px=1,
            min_area_iou_delta=0.01,
            min_boundary_iou_delta=0.01,
        )
        best = report.best_by_view["front"]

        self.assertEqual(report.status, "improved")
        self.assertEqual(best.offset_px, (-2, -1))
        self.assertTrue(best.accepted)
        self.assertGreater(
            best.metric_after["area_iou"],
            best.metric_before["area_iou"],
        )
        self.assertEqual(report.aggregate_after["area_iou"], 1.0)

    def test_view_role_permutation_detects_swapped_masks(self) -> None:
        front = _square(x0=6, y0=14, size=12)
        side = _square(x0=28, y0=14, size=12)

        report = detect_view_role_permutation(
            {"front": front, "side": side},
            {"front": side, "side": front},
        )

        self.assertTrue(report.swapped)
        self.assertEqual(report.assignment["front"], "side")
        self.assertEqual(report.assignment["side"], "front")
        self.assertEqual(report.identity_score, 0.0)
        self.assertEqual(report.best_score, 1.0)

    def test_executable_calibration_sweep_combines_alignment_and_roles(self) -> None:
        front = _square(x0=6, y0=14, size=12)
        side = _square(x0=28, y0=14, size=12)
        shifted_front = shift_mask(side, dx=2, dy=0)
        shifted_side = shift_mask(front, dx=-2, dy=0)

        report = executable_calibration_sweep(
            {"front": front, "side": side},
            {"front": shifted_front, "side": shifted_side},
            max_offset_px=2,
            step_px=1,
        )
        payload = report.to_dict()

        self.assertEqual(report.status, "view_roles_suspect")
        self.assertTrue(payload["view_role_permutation"]["swapped"])
        self.assertIn("mask_alignment", payload)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mask_alignment_sweep(
                {"front": np.zeros((8, 8), dtype=bool)},
                {"front": np.zeros((9, 8), dtype=bool)},
            )


if __name__ == "__main__":
    unittest.main()
