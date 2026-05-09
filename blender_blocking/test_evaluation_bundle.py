"""Pure tests for SOTA-style evaluation bundle metric groups."""

from __future__ import annotations

import unittest

from evaluation import bundle_from_candidate
from evaluation.export_qa import ExportQAReport, reports_from_payload
from evaluation.reporting import compact_console_summary, markdown_report
from reconstruction.types import CandidateMetrics, CandidateResult


class EvaluationBundleTests(unittest.TestCase):
    def test_export_qa_payload_becomes_first_class_metric_group(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-a",
            backend_name="shape_program",
            status="success",
            metric_result=CandidateMetrics(
                area_iou_min=0.9,
                area_iou_mean=0.92,
                boundary_iou_mean=0.8,
                editability_score=0.85,
                extras={
                    "export_qa": {
                        "target": "blend",
                        "status": "pass",
                        "reimport_status": "pass",
                        "object_count": 3,
                        "vertex_count": 120,
                        "face_count": 64,
                        "material_count": 2,
                    }
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        metrics = bundle.metric_index()

        self.assertEqual(bundle.status, "pass")
        self.assertIn("export.qa_score", metrics)
        self.assertTrue(metrics["export.status_ok"].value)
        self.assertTrue(metrics["export.reimport_ok"].value)
        self.assertIn("export.object_count", metrics)
        self.assertEqual(metrics["export.target_count"].value, 1)
        self.assertGreater(float(metrics["export.qa_score"].value), 0.9)
        self.assertIn("export=", compact_console_summary(bundle))
        self.assertIn("| Export |", markdown_report([bundle]))

    def test_export_qa_low_score_classifies_delivery_failure(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-b",
            backend_name="visual_hull_voxel",
            status="success",
            metric_result=CandidateMetrics(
                extras={
                    "asset_export": {
                        "target": "obj",
                        "status": "fail",
                        "reimport_status": "fail",
                        "errors": ["roundtrip_missing_mesh"],
                    }
                }
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        codes = {failure.code for failure in bundle.failures}

        self.assertEqual(bundle.status, "fail")
        self.assertIn("export_qa_low", codes)

    def test_export_qa_payload_accepts_multiple_target_shapes(self) -> None:
        reports = reports_from_payload(
            {
                "targets": {
                    "blend": {"status": "pass", "reimport_status": "pass"},
                    "gltf": {"status": "pass", "reimport_status": "warn"},
                }
            }
        )

        self.assertEqual(len(reports), 2)
        self.assertIsInstance(reports[0], ExportQAReport)
        self.assertEqual({report.target for report in reports}, {"blend", "gltf"})

    def test_success_without_required_silhouette_metrics_fails_bundle(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-c",
            backend_name="gaussian_ellipsoid_proxy",
            status="success",
            metric_result=CandidateMetrics(),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        codes = {failure.code for failure in bundle.failures}
        metrics = bundle.metric_index()

        self.assertEqual(bundle.status, "fail")
        self.assertEqual(metrics["silhouette.min_view_iou"].status, "fail")
        self.assertIn("silhouette_required_metrics_missing", codes)

    def test_required_per_view_boundary_and_sdf_are_first_class(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-d",
            backend_name="visual_hull_voxel",
            status="success",
            metric_result=CandidateMetrics(
                topology_score=0.9,
                editability_score=0.8,
                per_view={
                    "front": {
                        "area_iou": 0.82,
                        "boundary_iou": 0.62,
                        "signed_distance_loss": 0.12,
                        "required": True,
                        "passed": True,
                    },
                    "side": {
                        "area_iou": 0.78,
                        "boundary_iou": 0.55,
                        "signed_distance_loss": 0.16,
                        "required": True,
                        "passed": True,
                    },
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        metrics = bundle.metric_index()

        self.assertEqual(bundle.status, "pass")
        self.assertEqual(metrics["silhouette.required_view_count"].value, 2)
        self.assertEqual(metrics["silhouette.missing_required_metric_count"].value, 0)
        self.assertAlmostEqual(metrics["silhouette.min_boundary_iou"].value, 0.55)
        self.assertAlmostEqual(
            metrics["silhouette.mean_signed_distance_loss"].value,
            0.14,
        )

    def test_zero_iou_required_views_are_not_reported_as_missing_metrics(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-zero-view",
            backend_name="differentiable_refine",
            status="degraded",
            degraded=True,
            metric_result=CandidateMetrics(
                per_view={
                    "front": {
                        "area_iou": 0.0,
                        "boundary_iou": 0.05,
                        "signed_distance_loss": 0.9,
                        "required": True,
                        "passed": False,
                        "reason": "soft silhouette did not satisfy required view gate",
                    }
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        codes = {failure.code for failure in bundle.failures}

        self.assertIn("silhouette_required_views_failed", codes)
        self.assertNotIn("silhouette_required_metrics_missing", codes)
        self.assertEqual(bundle.status, "fail")

    def test_classified_failures_influence_bundle_status(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-boundary",
            backend_name="gaussian_ellipsoid_proxy",
            status="success",
            metric_result=CandidateMetrics(
                per_view={
                    "front": {
                        "area_iou": 0.92,
                        "boundary_iou": 0.30,
                        "signed_distance_loss": 0.05,
                        "required": True,
                        "passed": True,
                    },
                    "side": {
                        "area_iou": 0.91,
                        "boundary_iou": 0.32,
                        "signed_distance_loss": 0.05,
                        "required": True,
                        "passed": True,
                    },
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        codes = {failure.code for failure in bundle.failures}

        self.assertIn("silhouette_boundary_blobby", codes)
        self.assertEqual(bundle.status, "fail")

    def test_recoverability_payload_splits_true_and_envelope_geometry(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-e",
            backend_name="visual_hull_voxel",
            status="success",
            metric_result=CandidateMetrics(
                area_iou_min=0.8,
                area_iou_mean=0.84,
                boundary_iou_mean=0.7,
                extras={
                    "recoverability": {
                        "source": "synthetic-test",
                        "true_geometry": {
                            "chamfer_l2": 0.08,
                            "fscore_tau": 0.42,
                            "volumetric_iou": 0.5,
                        },
                        "recoverable_geometry": {
                            "chamfer_l2": 0.02,
                            "fscore_tau": 0.82,
                            "volumetric_iou": 0.76,
                        },
                    }
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        metrics = bundle.metric_index()
        codes = {failure.code for failure in bundle.failures}

        self.assertEqual(metrics["geometry.true.fscore_tau"].value, 0.42)
        self.assertEqual(metrics["geometry.recoverable.fscore_tau"].value, 0.82)
        self.assertAlmostEqual(metrics["geometry.ambiguity_gap_chamfer_l2"].value, 0.06)
        self.assertIn("geometry_true_recoverable_gap_large", codes)

    def test_visual_hull_diagnostics_are_first_class_failures(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-vh-diag",
            backend_name="visual_hull_voxel",
            status="success",
            metric_result=CandidateMetrics(
                per_view={
                    "front": {
                        "area_iou": 0.91,
                        "boundary_iou": 0.7,
                        "signed_distance_loss": 0.05,
                        "required": True,
                        "passed": True,
                    },
                    "top": {
                        "area_iou": 0.12,
                        "boundary_iou": 0.04,
                        "signed_distance_loss": 0.8,
                        "required": True,
                        "passed": False,
                    },
                },
                extras={
                    "visual_hull_view_diagnostics": {
                        "axis_or_transform_suspect": True,
                        "catastrophic_view_failure": True,
                        "failed_views": ["top"],
                        "top_like_failures": ["top"],
                    }
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        metrics = bundle.metric_index()
        codes = {failure.code for failure in bundle.failures}

        self.assertEqual(
            metrics["diagnostics.visual_hull.catastrophic_view_failure"].value,
            True,
        )
        self.assertIn("visual_hull_axis_or_transform_suspect", codes)
        self.assertIn("visual_hull_catastrophic_view_failure", codes)
        self.assertEqual(bundle.status, "fail")


if __name__ == "__main__":
    unittest.main()
