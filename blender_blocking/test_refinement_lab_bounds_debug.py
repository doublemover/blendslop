"""Tests for bounds and transform diagnostics."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from refinement_lab.bounds_debug import obj_bounds, project_world_points_to_view


class RefinementLabBoundsDebugTests(unittest.TestCase):
    def test_obj_bounds_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mesh.obj"
            path.write_text("v 0 0 0\nv 2 4 6\nf 1 2 2\n", encoding="utf-8")
            summary = obj_bounds(path)
        self.assertEqual(summary["vertex_count"], 2)
        self.assertEqual(summary["size"], [2.0, 4.0, 6.0])

    def test_projection_produces_mask(self) -> None:
        points = np.array([[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5]], dtype=float)
        bounds = {"min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1, "min_z": 0, "max_z": 1}
        mask = project_world_points_to_view(points, view="front", bounds=bounds, output_size=(32, 32))
        self.assertEqual(mask.shape, (32, 32))
        self.assertTrue(mask.any())


if __name__ == "__main__":
    unittest.main()
