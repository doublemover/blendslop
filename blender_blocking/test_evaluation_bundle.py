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


if __name__ == "__main__":
    unittest.main()
