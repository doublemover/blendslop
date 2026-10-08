"""Pure tests for deterministic CPU soft-silhouette rendering."""

from __future__ import annotations

import unittest

import numpy as np

from primitives.analytic_primitives import AnisotropicGaussianPrimitive, EllipsoidPrimitive
from primitives.soft_silhouette import (
    OrthographicCamera,
    render_multi_view_soft_silhouettes,
    render_projected_soft_silhouette,
    soft_mask_metrics,
)


class TestSoftSilhouetteDeterminism(unittest.TestCase):
    def test_rendered_silhouette_is_deterministic(self) -> None:
        camera = OrthographicCamera.from_view(
            "front", image_size=(64, 32), world_bounds=(-2.5, 2.5, -1.5, 1.5)
        )
        primitives = [
            EllipsoidPrimitive(
                center=(0.2, -0.1, 0.0),
                radii=(1.1, 0.7, 0.5),
                density=0.9,
                confidence=0.8,
            ),
            AnisotropicGaussianPrimitive(
                center=(-0.8, 0.4, 0.2),
                covariance=np.diag([0.15, 0.04, 0.06]),
                opacity=0.9,
                confidence=0.75,
            ),
        ]

        first = render_projected_soft_silhouette(primitives, camera, softness=18.0)
        second = render_projected_soft_silhouette(primitives, camera, softness=18.0)

        self.assertEqual(first.shape, (32, 64))
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(float(first.mean()), 0.01)

    def test_multi_view_render_defaults(self) -> None:
        primitives = [
            EllipsoidPrimitive(
                center=(0.0, 0.0, 0.0),
                radii=(0.6, 0.6, 0.6),
            )
        ]
        rendered = render_multi_view_soft_silhouettes(
            primitives,
            cameras=None,
            image_size=(40, 32),
        )

        self.assertEqual(set(rendered.keys()), {"front", "side", "top"})
        self.assertTrue(all(mask.shape == (32, 40) for mask in rendered.values()))
        rerender = render_multi_view_soft_silhouettes(
            primitives,
            image_size=(40, 32),
        )
        for view in rendered:
            np.testing.assert_array_equal(rendered[view], rerender[view])

    def test_soft_metrics_are_deterministic_for_identical_inputs(self) -> None:
        target = np.zeros((24, 24), dtype=np.float64)
        target[7:17, 6:18] = 1.0

        first = soft_mask_metrics(target, target)
        second = soft_mask_metrics(target.copy(), target.copy())

        self.assertAlmostEqual(first["soft_l1"], 0.0, places=12)
        self.assertAlmostEqual(first["soft_l2"], 0.0, places=12)
        self.assertAlmostEqual(first["soft_iou_loss"], 0.0, places=12)
        self.assertAlmostEqual(first["area_iou_loss"], 0.0, places=12)
        self.assertAlmostEqual(first["boundary_iou"], 1.0, places=12)
        self.assertAlmostEqual(first["boundary_iou_loss"], 0.0, places=12)
        self.assertAlmostEqual(first["signed_distance_loss"], 0.0, places=12)
        self.assertEqual(first, second)

    def test_soft_metrics_include_boundary_and_signed_distance_terms(self) -> None:
        target = np.zeros((32, 32), dtype=np.float64)
        target[9:23, 8:22] = 1.0
        shifted = np.zeros_like(target)
        shifted[9:23, 10:24] = 1.0

        metrics = soft_mask_metrics(shifted, target)

        self.assertIn("boundary_iou", metrics)
        self.assertIn("boundary_iou_loss", metrics)
        self.assertIn("signed_distance_loss", metrics)
        self.assertGreater(metrics["area_iou_loss"], 0.0)
        self.assertGreater(metrics["boundary_iou_loss"], 0.0)
        self.assertGreater(metrics["signed_distance_loss"], 0.0)
        self.assertLess(metrics["boundary_iou"], 1.0)
        self.assertAlmostEqual(
            metrics["boundary_iou_loss"],
            1.0 - metrics["boundary_iou"],
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
