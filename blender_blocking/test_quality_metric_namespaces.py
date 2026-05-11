"""Tests for quality metric namespace separation."""

from __future__ import annotations

import unittest

from e2e.matrix import _matrix_metrics
from refinement_lab.runner import _metrics_from_payload


class QualityMetricNamespaceTests(unittest.TestCase):
    def test_backend_status_metrics_do_not_populate_render_aliases(self) -> None:
        payload = {
            "validation_mode": "backend-status",
            "backend_result": {
                "status": "success",
                "metric_result": {
                    "area_iou_mean": 0.98,
                    "area_iou_min": 0.96,
                    "boundary_iou_mean": 0.91,
                },
            },
        }

        metrics = _matrix_metrics(payload, passed=True)

        self.assertNotIn("average_iou", metrics)
        self.assertNotIn("min_view_iou", metrics)
        self.assertEqual(metrics["backend"]["area_iou_mean"], 0.98)
        self.assertEqual(metrics["backend"]["area_iou_min"], 0.96)
        self.assertNotIn("render", metrics)

    def test_render_iou_min_view_is_required_view_minimum(self) -> None:
        payload = {
            "validation_mode": "render-iou",
            "average_iou": 0.7,
            "min_view_iou": 0.7,
            "views": {
                "front": {"iou": 0.8},
                "side": {"iou": 0.4},
                "top": {"iou": 0.9},
            },
        }

        metrics = _matrix_metrics(payload, passed=False)

        self.assertEqual(metrics["render"]["min_view_iou"], 0.4)
        self.assertEqual(metrics["min_view_iou"], 0.4)
        self.assertEqual(metrics["render"]["per_view"]["side"]["area_iou"], 0.4)

    def test_refinement_flattening_keeps_backend_and_render_separate(self) -> None:
        payload = {
            "validation_mode": "render-iou",
            "average_iou": 0.6,
            "views": {
                "front": {"iou": 0.6, "boundary_iou": 0.4},
                "side": {"iou": 0.5, "boundary_iou": 0.3},
                "top": {"iou": 0.7, "boundary_iou": 0.5},
            },
            "backend_result": {
                "status": "success",
                "metric_result": {
                    "area_iou_mean": 0.99,
                    "area_iou_min": 0.98,
                },
            },
        }

        metrics = _metrics_from_payload(payload)

        self.assertEqual(metrics["render"]["average_iou"], 0.6)
        self.assertEqual(metrics["render"]["min_view_iou"], 0.5)
        self.assertEqual(metrics["backend"]["area_iou_mean"], 0.99)
        self.assertNotEqual(
            metrics["backend"]["area_iou_min"],
            metrics["render"]["min_view_iou"],
        )


if __name__ == "__main__":
    unittest.main()
