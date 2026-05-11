"""Tests for quality metric namespace separation."""

from __future__ import annotations

import unittest

from blender_blocking.e2e.matrix import _matrix_metrics
from blender_blocking.refinement_lab.runner import _metrics_from_payload


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

    def test_backend_evaluation_bundle_silhouette_stays_out_of_render_namespace(self) -> None:
        payload = {
            "validation_mode": "backend-status",
            "backend_result": {
                "status": "success",
                "evaluation_bundle": {
                    "metric_groups": [
                        {
                            "metrics": [
                                {"name": "silhouette.average_iou", "value": 0.99},
                                {"name": "silhouette.min_view_iou", "value": 0.98},
                            ]
                        }
                    ]
                },
            },
        }

        metrics = _matrix_metrics(payload, passed=True)

        self.assertEqual(metrics["silhouette"]["average_iou"], 0.99)
        self.assertNotIn("average_iou", metrics)
        self.assertNotIn("min_view_iou", metrics)
        self.assertNotIn("area_iou_mean", metrics)
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
        self.assertAlmostEqual(metrics["render"]["boundary_iou_mean"], 0.4)
        self.assertEqual(metrics["backend"]["area_iou_mean"], 0.99)
        self.assertNotEqual(
            metrics["backend"]["area_iou_min"],
            metrics["render"]["min_view_iou"],
        )

    def test_render_per_view_boundary_and_sdf_emit_aggregates(self) -> None:
        payload = {
            "validation_mode": "render-iou",
            "views": {
                "front": {
                    "iou": 0.9,
                    "boundary_iou": 0.8,
                    "signed_distance_loss": 0.1,
                },
                "side": {
                    "iou": 0.7,
                    "boundary_iou": 0.4,
                    "signed_distance_loss": 0.3,
                },
                "top": {
                    "iou": 0.95,
                    "boundary_iou": 0.6,
                    "signed_distance_loss": 0.2,
                },
            },
        }

        matrix_metrics = _matrix_metrics(payload, passed=True)
        refinement_metrics = _metrics_from_payload(payload)

        for metrics in (matrix_metrics, refinement_metrics):
            self.assertAlmostEqual(metrics["render"]["boundary_iou_mean"], 0.6)
            self.assertAlmostEqual(metrics["render"]["boundary_iou_min"], 0.4)
            self.assertAlmostEqual(metrics["render"]["signed_distance_loss_mean"], 0.2)
            self.assertAlmostEqual(metrics["render"]["signed_distance_loss_max"], 0.3)

    def test_refinement_missing_renderable_mesh_does_not_use_backend_bundle_as_render_iou(self) -> None:
        payload = {
            "validation_mode": "render-iou",
            "status": "failed",
            "failure_code": "missing_renderable_mesh",
            "backend_result": {
                "status": "success",
                "metric_result": {
                    "area_iou_mean": 0.99,
                    "area_iou_min": 0.98,
                },
                "evaluation_bundle": {
                    "metric_groups": [
                        {
                            "metrics": [
                                {"name": "silhouette.average_iou", "value": 0.97},
                                {"name": "silhouette.min_view_iou", "value": 0.96},
                            ]
                        }
                    ]
                },
            },
        }

        metrics = _metrics_from_payload(payload)

        self.assertEqual(metrics["backend"]["area_iou_mean"], 0.99)
        self.assertEqual(metrics["silhouette"]["average_iou"], 0.97)
        self.assertNotIn("average_iou", metrics)
        self.assertNotIn("min_view_iou", metrics)
        self.assertNotIn("render", metrics)


if __name__ == "__main__":
    unittest.main()
