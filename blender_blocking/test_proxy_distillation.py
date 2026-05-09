"""Tests for proxy SDF/occupancy distillation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from primitives.proxy_distillation import distill_proxy_field, write_proxy_field_npz


class _SphereProxy:
    center = np.asarray((0.0, 0.0, 0.0), dtype=float)
    radii = np.asarray((1.0, 1.0, 1.0), dtype=float)

    def sdf_batch(self, points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points - self.center[None, :], axis=1) - 1.0


class ProxyDistillationTests(unittest.TestCase):
    def test_distillation_reports_occupancy_and_surface_fit(self) -> None:
        theta = np.linspace(0.0, np.pi * 2.0, 16, endpoint=False)
        target_points = np.stack(
            (
                np.cos(theta),
                np.sin(theta),
                np.zeros_like(theta),
            ),
            axis=1,
        )

        report, grid, sdf, occupancy = distill_proxy_field(
            (_SphereProxy(),),
            target_points=target_points,
            resolution=16,
        )

        self.assertEqual(report.resolution, 16)
        self.assertEqual(grid.shape, (16, 16, 16, 3))
        self.assertEqual(sdf.shape, (16, 16, 16))
        self.assertEqual(occupancy.shape, (16, 16, 16))
        self.assertGreater(report.occupancy_ratio, 0.0)
        self.assertGreater(report.target_surface_band_ratio, 0.5)
        self.assertGreater(report.arbitration_score, 0.7)

    def test_distillation_npz_is_portable(self) -> None:
        report, grid, sdf, occupancy = distill_proxy_field(
            (_SphereProxy(),),
            resolution=8,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_proxy_field_npz(
                Path(tmp) / "proxy-field.npz",
                report=report,
                points=grid,
                sdf=sdf,
                occupancy=occupancy,
            )
            with np.load(path) as loaded:
                self.assertIn("sdf", loaded.files)
                self.assertEqual(tuple(loaded["occupancy"].shape), (8, 8, 8))


if __name__ == "__main__":
    unittest.main()
