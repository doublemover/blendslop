"""Pure tests for active cost recording and bundle metrics."""

from __future__ import annotations

import time
import unittest

from evaluation import bundle_from_candidate
from evaluation.cost_model import (
    CostRecorder,
    attach_cost_report_to_candidate,
    cost_report_from_candidate,
    cost_report_from_mapping,
)
from reconstruction.types import CandidateMetrics, CandidateResult


class CostModelTests(unittest.TestCase):
    def test_recorder_emits_stage_cache_and_throughput_report(self) -> None:
        recorder = CostRecorder(track_memory=False)
        with recorder.stage(
            "backend_reconstruct",
            work_units={"voxels": 1024},
            artifact_bytes=128,
        ):
            time.sleep(0.001)
        recorder.cache_hit("canonical_masks", entries=2, bytes_stored=256)
        recorder.cache_miss("canonical_masks", entries=3, bytes_stored=512)

        report = recorder.report(
            throughput={"voxels_per_ms": 10.0},
            include_unaccounted_time=False,
        )

        self.assertGreater(report.total_wall_ms, 0.0)
        self.assertEqual(len(report.stages), 1)
        self.assertEqual(report.stages[0].stage, "backend_reconstruct")
        self.assertEqual(report.stages[0].artifact_bytes, 128)
        self.assertEqual(report.stages[0].work_units["voxels"], 1024)
        self.assertEqual(report.cache["canonical_masks.hits"], 1.0)
        self.assertEqual(report.cache["canonical_masks.misses"], 1.0)
        self.assertEqual(report.cache["canonical_masks.hit_rate"], 0.5)
        self.assertEqual(report.cache["hit_rate"], 0.5)
        self.assertEqual(report.throughput["voxels_per_ms"], 10.0)

    def test_nested_spans_are_exclusive_and_missing_time_is_unavailable(self):
        from unittest.mock import patch
        with patch("evaluation.cost_model.perf_counter", side_effect=[0,1,2,5,8,9]):
            recorder = CostRecorder()
            with recorder.stage("outer"):
                with recorder.stage("inner"):
                    pass
            report = recorder.report()
        stages = {s.stage:s.wall_ms for s in report.stages}
        self.assertEqual(stages["inner"], 3000)
        self.assertEqual(stages["outer"], 4000)
        self.assertEqual(report.total_wall_ms, 9000)
        self.assertEqual(sum(stages.values()), report.total_wall_ms)
        self.assertFalse(report.diagnostic_allocations)
        self.assertIsNone(cost_report_from_candidate(CandidateResult("missing","none","skipped")).total_wall_ms)
        self.assertIsNone(cost_report_from_mapping({}).total_wall_ms)


    def test_fallback_does_not_fabricate_or_double_count_backend_time(self):
        from evaluation.cost_model import CostReport
        missing = CandidateResult("missing", "none", "skipped", metric_result=CandidateMetrics(
            extras={"optimization":{"objective_evaluations":3}, "objective":{"improved":True}}))
        self.assertIsNone(cost_report_from_candidate(missing).total_wall_ms)
        attached = attach_cost_report_to_candidate(missing, CostReport(total_wall_ms=None))
        self.assertIsNone(attached.metric_result.elapsed_s)
        measured = CandidateResult("measured", "none", "success", metric_result=CandidateMetrics(
            elapsed_s=2, extras={"optimization":{"elapsed_s":1}, "mesh_extraction":{"elapsed_s":.5}}))
        report = cost_report_from_candidate(measured)
        self.assertEqual(report.total_wall_ms, 2000)
        self.assertEqual(sum(stage.wall_ms for stage in report.stages), 2000)
        partial = CandidateResult("partial", "none", "success", metric_result=CandidateMetrics(
            extras={"optimization":{"elapsed_s":1}}))
        self.assertIsNone(cost_report_from_candidate(partial).total_wall_ms)
        self.assertIsNone(cost_report_from_mapping({"total_wall_ms":None,
            "stages":[{"stage":"partial", "wall_ms":10}]}).total_wall_ms)

    def test_mapping_round_trip_uses_explicit_total_or_stage_sum(self) -> None:
        payload = {
            "stages": [
                {
                    "stage": "load_inputs",
                    "status": "pass",
                    "wall_ms": 3.5,
                    "work_units": {"images": 3},
                },
                {
                    "stage": "render_validation",
                    "status": "warn",
                    "wall_ms": 6.5,
                    "notes": ["slow render"],
                },
            ],
            "cache": {"hits": 2, "misses": 1, "hit_rate": 2 / 3},
        }

        report = cost_report_from_mapping(payload)

        self.assertEqual(report.total_wall_ms, 10.0)
        self.assertEqual(len(report.stages), 2)
        self.assertEqual(report.stages[1].status, "warn")
        self.assertEqual(report.cache["hits"], 2)

    def test_candidate_explicit_cost_payload_becomes_bundle_metrics(self) -> None:
        result = CandidateResult(
            candidate_id="costed",
            backend_name="visual_hull_voxel",
            status="success",
            metric_result=CandidateMetrics(
                extras={
                    "cost": {
                        "total_wall_ms": 42.0,
                        "peak_memory_mb": 12.0,
                        "stages": [
                            {
                                "stage": "mesh_extraction",
                                "status": "pass",
                                "wall_ms": 7.0,
                                "artifact_bytes": 2048,
                                "work_units": {"faces": 64},
                            }
                        ],
                        "cache": {"hits": 3, "misses": 1, "hit_rate": 0.75},
                        "throughput": {"voxels_per_ms": 20.0},
                    }
                }
            ),
        )

        report = cost_report_from_candidate(result)
        bundle = bundle_from_candidate(result=result, repo="test")
        metrics = bundle.metric_index()

        self.assertEqual(report.total_wall_ms, 42.0)
        self.assertEqual(metrics["cost.total_wall_ms"].value, 42.0)
        self.assertEqual(metrics["cost.peak_memory_mb"].value, 12.0)
        self.assertEqual(metrics["cost.cache.hit_rate"].value, 0.75)
        self.assertEqual(metrics["cost.throughput.voxels_per_ms"].value, 20.0)
        self.assertEqual(metrics["cost.stage.mesh_extraction.wall_ms"].value, 7.0)
        self.assertEqual(
            metrics["cost.stage.mesh_extraction.artifact_bytes"].value,
            2048.0,
        )
        self.assertEqual(
            metrics["cost.stage.mesh_extraction.work_units.faces"].value,
            64.0,
        )

    def test_invalid_throughput_reason_does_not_become_numeric_metric(self) -> None:
        result = CandidateResult(
            candidate_id="invalid-throughput",
            backend_name="visual_hull_voxel",
            status="success",
            metric_result=CandidateMetrics(
                extras={
                    "cost": {
                        "total_wall_ms": 0.0,
                        "throughput": {
                            "invalid_reason": "missing_or_zero_total_wall_ms",
                        },
                    }
                }
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        metrics = bundle.metric_index()

        self.assertNotIn("cost.throughput.invalid_reason", metrics)
        self.assertFalse(metrics["cost.throughput.valid"].value)
        self.assertEqual(metrics["cost.throughput.valid"].status, "fail")

    def test_cost_report_attachment_preserves_candidate_metrics(self) -> None:
        result = CandidateResult(
            candidate_id="cost-attach",
            backend_name="profile_loft",
            status="success",
            metric_result=CandidateMetrics(
                area_iou_min=0.8,
                editability_score=0.7,
                extras={"existing": {"ok": True}},
            ),
        )
        recorder = CostRecorder(track_memory=False)
        with recorder.stage("build_target", work_units={"views": 2}):
            pass
        with recorder.stage("backend_reconstruct", work_units={"constraints": 2}):
            pass

        attached = attach_cost_report_to_candidate(result, recorder.report())
        report = cost_report_from_candidate(attached)

        self.assertIsNot(attached, result)
        self.assertEqual(attached.metric_result.area_iou_min, 0.8)
        self.assertEqual(attached.metric_result.extras["existing"], {"ok": True})
        self.assertIn("cost_report", attached.metric_result.extras)
        self.assertGreater(attached.metric_result.elapsed_s, 0.0)
        self.assertEqual([stage.stage for stage in report.stages[:2]], [
            "build_target",
            "backend_reconstruct",
        ])


if __name__ == "__main__":
    unittest.main()
