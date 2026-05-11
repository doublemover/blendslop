"""Tests for shared binary morphology helpers."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from geometry.morphology import (
        as_bool_mask,
        binary_dilation,
        binary_erosion,
        boundary_band,
    )
except ModuleNotFoundError:  # pragma: no cover - package unittest path
    from blender_blocking.geometry.morphology import (
        as_bool_mask,
        binary_dilation,
        binary_erosion,
        boundary_band,
    )


class GeometryMorphologyTests(unittest.TestCase):
    def test_as_bool_mask_validates_2d_and_coerces_values(self) -> None:
        mask = as_bool_mask([[0, 2], [3, 0]])

        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.tolist(), [[False, True], [True, False]])
        with self.assertRaises(ValueError):
            as_bool_mask(np.zeros((1, 1, 1), dtype=bool))

    def test_erosion_and_dilation_handle_center_and_border_pixels(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True

        self.assertFalse(binary_erosion(mask, radius=1).any())
        self.assertEqual(int(binary_dilation(mask, radius=1).sum()), 9)
        self.assertEqual(int(binary_dilation(mask, radius=2).sum()), 25)

        border = np.zeros((5, 5), dtype=bool)
        border[0, 0] = True
        self.assertEqual(int(binary_dilation(border, radius=1).sum()), 4)

    def test_boundary_band_preserves_empty_shape_and_expands_edges(self) -> None:
        empty = np.zeros((4, 4), dtype=bool)
        self.assertEqual(boundary_band(empty).shape, (4, 4))
        self.assertFalse(boundary_band(empty).any())

        square = np.zeros((5, 5), dtype=bool)
        square[1:4, 1:4] = True
        band = boundary_band(square, radius=1)
        self.assertGreater(int(band.sum()), int(square.sum()))
        self.assertTrue(band[0, 0])


if __name__ == "__main__":
    unittest.main()
