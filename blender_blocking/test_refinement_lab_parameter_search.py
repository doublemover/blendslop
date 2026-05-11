"""Tests for refinement objective scoring."""

from __future__ import annotations

from pathlib import Path
import unittest

from blender_blocking.refinement_lab.contracts import ExperimentResult
from blender_blocking.refinement_lab.parameter_search import (
    OBJECTIVES,
    promotion_decision,
    rank_results,
    score_result,
)


def _result(
    name: str,
    avg: float,
    front: float,
    side: float,
    top: float,
    *,
    backend_result: dict[str, object] | None = None,
    metrics: dict[str, object] | None = None,
) -> ExperimentResult:
    metric_payload = {
        "average_iou": avg,
        "front_iou": front,
        "side_iou": side,
        "top_iou": top,
        "topology_score": 1.0,
        "editability_score": 0.5,
    }
    if metrics:
        metric_payload.update(metrics)
    return ExperimentResult(
        run_id="r",
        case_id="c",
        variant_id=name,
        mode="profile_loft",
        status="pass",
        exit_code=0,
        started_utc="s",
        finished_utc="f",
        elapsed_s=1.0,
        metrics=metric_payload,
        backend_result=backend_result or {"status": "success"},
    )


class RefinementLabScoringTests(unittest.TestCase):
    def test_catastrophic_view_loses_quality_rank(self) -> None:
        good = _result("good", 0.8, 0.8, 0.8, 0.8)
        bad = _result("bad", 0.9, 0.99, 0.99, 0.02)
        ranked = rank_results([bad, good], objective="quality_win")
        self.assertEqual(ranked[0][0].variant_id, "good")

    def test_score_contains_terms(self) -> None:
        score = score_result(_result("r", 0.9, 0.9, 0.9, 0.9))
        self.assertIn("terms", score)
        self.assertGreater(score["total"], 0)

    def test_reliability_first_hard_zeroes_blocked_candidate(self) -> None:
        blocked = ExperimentResult(
            run_id="r",
            case_id="c",
            variant_id="blocked",
            mode="gaussian_ellipsoid_proxy",
            status="pass",
            exit_code=0,
            started_utc="s",
            finished_utc="f",
            elapsed_s=1.0,
            backend_result={"status": "success"},
            metrics={
                "validation_mode": "backend-status",
                "backend": {"area_iou_mean": 0.99, "area_iou_min": 0.99},
            },
        )

        score = score_result(blocked, objective="reliability_first")

        self.assertEqual(score["total"], 0.0)
        self.assertEqual(score["promotion"]["state"], "metric_only_candidate")

    def test_degraded_backend_loses_to_full_success(self) -> None:
        full = _result(
            "full",
            0.84,
            0.84,
            0.84,
            0.84,
            backend_result={"status": "success", "degraded": False},
        )
        degraded = _result(
            "degraded",
            0.92,
            0.92,
            0.92,
            0.92,
            backend_result={"status": "degraded", "degraded": True},
        )
        ranked = rank_results([degraded, full], objective="quality_win")
        self.assertEqual(ranked[0][0].variant_id, "full")
        self.assertEqual(ranked[1][1]["promotion"]["tier"], "degraded")

    def test_high_render_quality_degraded_primitive_fit_can_seed_exploration(self) -> None:
        result = ExperimentResult(
            run_id="r",
            case_id="c",
            variant_id="primitive-degraded",
            mode="primitive_fit_refine",
            status="pass",
            exit_code=0,
            started_utc="s",
            finished_utc="f",
            elapsed_s=1.0,
            backend_result={
                "status": "degraded",
                "degraded": True,
                "warnings": (
                    "primitive backend emitted renderable artifacts, but internal/proxy min IoU 0.310 is below 0.350",
                ),
            },
            metrics={
                "average_iou": 0.91,
                "front_iou": 0.90,
                "side_iou": 0.92,
                "top_iou": 0.91,
                "topology_score": 0.9,
                "editability_score": 0.7,
                "backend": {"area_iou_min": 0.31, "area_iou_mean": 0.36},
                "render": {
                    "min_view_iou": 0.90,
                    "per_view": {
                        "front": {"area_iou": 0.90},
                        "side": {"area_iou": 0.92},
                        "top": {"area_iou": 0.91},
                    },
                },
            },
        )

        decision = promotion_decision(result)

        self.assertFalse(decision.promotable)
        self.assertTrue(decision.parent_selectable)
        self.assertEqual(decision.tier, "degraded")
        self.assertIn("degraded_backend", decision.blockers)
        self.assertEqual(decision.blocking_for_parent_selection, ())

    def test_metric_only_candidate_cannot_hide_behind_area_iou(self) -> None:
        full = _result("full", 0.76, 0.76, 0.76, 0.76)
        metric_only = ExperimentResult(
            run_id="r",
            case_id="c",
            variant_id="metric-only",
            mode="gaussian_ellipsoid_proxy",
            status="pass",
            exit_code=0,
            started_utc="s",
            finished_utc="f",
            elapsed_s=1.0,
            backend_result={"status": "success"},
            metrics={"area_iou_mean": 0.99, "area_iou_min": 0.99},
        )
        ranked = rank_results([metric_only, full], objective="quality_win")
        self.assertEqual(ranked[0][0].variant_id, "full")
        self.assertEqual(promotion_decision(metric_only).tier, "blocked")
        self.assertEqual(
            promotion_decision(metric_only).state,
            "metric_only_candidate",
        )
        self.assertIn(
            "missing_required_metrics",
            promotion_decision(metric_only).blockers,
        )

    def test_backend_status_metric_only_candidate_can_seed_exploration(self) -> None:
        metric_only = ExperimentResult(
            run_id="r",
            case_id="c",
            variant_id="metric-only",
            mode="ensemble",
            status="pass",
            exit_code=0,
            started_utc="s",
            finished_utc="f",
            elapsed_s=1.0,
            backend_result={"status": "success"},
            metrics={
                "validation_mode": "backend-status",
                "backend": {"area_iou_mean": 0.99, "area_iou_min": 0.97},
            },
        )

        decision = promotion_decision(metric_only)

        self.assertFalse(decision.promotable)
        self.assertTrue(decision.parent_selectable)
        self.assertEqual(decision.state, "metric_only_candidate")
        self.assertEqual(decision.blocking_for_parent_selection, ())

    def test_shape_program_missing_render_qa_can_seed_exploration(self) -> None:
        shape_program = ExperimentResult(
            run_id="r",
            case_id="c",
            variant_id="shape-program",
            mode="shape_program",
            status="pass",
            exit_code=0,
            started_utc="s",
            finished_utc="f",
            elapsed_s=1.0,
            backend_result={
                "status": "degraded",
                "backend_name": "shape_program",
                "metric_result": {
                    "extras": {
                        "render_qa": {
                            "status": "missing",
                            "missing_required_metrics": True,
                        }
                    }
                },
                "artifacts": {"shape_program": "program.json"},
            },
            metrics={
                "render": {
                    "qa": {
                        "status": "missing",
                        "missing_required_metrics": True,
                    }
                }
            },
            artifacts={"shape_program": Path("program.json")},
        )

        decision = promotion_decision(shape_program)

        self.assertFalse(decision.promotable)
        self.assertTrue(decision.parent_selectable)
        self.assertEqual(decision.state, "shape_program_missing_render_qa")
        self.assertIn("missing_required_metrics", decision.blockers)
        self.assertNotIn("catastrophic_view_failure", decision.blockers)
        self.assertEqual(decision.blocking_for_parent_selection, ())

    def test_topology_failure_blocks_promotion(self) -> None:
        bad_topology = _result(
            "bad-topology",
            0.9,
            0.9,
            0.9,
            0.9,
            metrics={"topology": {"score": 0.4}},
        )

        decision = promotion_decision(bad_topology)

        self.assertFalse(decision.promotable)
        self.assertTrue(decision.parent_selectable)
        self.assertEqual(decision.state, "blocked_topology")
        self.assertEqual(decision.blocking_for_parent_selection, ())

    def test_proxy_render_namespace_violation_is_flagged(self) -> None:
        result = _result(
            "gaussian",
            0.0,
            0.0,
            0.0,
            0.0,
            metrics={
                "backend": {"area_iou_min": 0.97, "area_iou_mean": 0.98},
                "render": {
                    "min_view_iou": 0.0,
                    "per_view": {
                        "front": {"area_iou": 0.0},
                        "side": {"area_iou": 0.0},
                        "top": {"area_iou": 0.0},
                    },
                },
            },
        )

        decision = promotion_decision(result)

        self.assertIn("proxy_render_namespace_violation", decision.blockers)

    def test_proxy_render_disagreement_blocks_promotion(self) -> None:
        result = _result(
            "gaussian-disagreement",
            0.75,
            0.92,
            0.74,
            0.61,
            metrics={
                "backend": {"area_iou_min": 0.98, "area_iou_mean": 0.99},
                "render": {
                    "min_view_iou": 0.61,
                    "boundary_iou_min": 0.04,
                    "per_view": {
                        "front": {"area_iou": 0.92, "boundary_iou": 0.4},
                        "side": {"area_iou": 0.74, "boundary_iou": 0.2},
                        "top": {"area_iou": 0.61, "boundary_iou": 0.04},
                    },
                },
            },
        )

        decision = promotion_decision(result)

        self.assertFalse(decision.promotable)
        self.assertEqual(decision.state, "blocked_proxy_render_disagreement")
        self.assertIn("proxy_render_disagreement", decision.blockers)
        self.assertFalse(decision.parent_selectable)

    def test_high_quality_proxy_render_disagreement_can_seed_exploration(self) -> None:
        result = _result(
            "boundary-repair-parent",
            0.955,
            0.977,
            0.983,
            0.905,
            metrics={
                "backend": {"area_iou_min": 0.999, "area_iou_mean": 0.999},
                "render": {
                    "min_view_iou": 0.905,
                    "boundary_iou_min": 0.201,
                    "per_view": {
                        "front": {"area_iou": 0.977, "boundary_iou": 0.783},
                        "side": {"area_iou": 0.983, "boundary_iou": 0.792},
                        "top": {"area_iou": 0.905, "boundary_iou": 0.201},
                    },
                },
            },
        )

        decision = promotion_decision(result)

        self.assertFalse(decision.promotable)
        self.assertTrue(decision.parent_selectable)
        self.assertEqual(decision.state, "blocked_proxy_render_disagreement")
        self.assertIn("proxy_render_disagreement", decision.blockers)
        self.assertEqual(decision.blocking_for_parent_selection, ())

    def test_research_only_requires_review_across_objectives(self) -> None:
        result = _result(
            "research",
            0.95,
            0.95,
            0.95,
            0.95,
            backend_result={"status": "research_only"},
        )
        decision = promotion_decision(result)
        self.assertEqual(decision.tier, "research_only")
        self.assertFalse(decision.promotable)
        for objective in OBJECTIVES:
            score = score_result(result, objective=objective)
            terms = {term["name"]: term for term in score["terms"]}
            self.assertIn("research_only_candidate", terms)
            self.assertLess(terms["research_only_candidate"]["weighted"], 0.0)

    def test_diagnostic_only_candidate_cannot_promote(self) -> None:
        result = _result(
            "diagnostic",
            0.96,
            0.96,
            0.96,
            0.96,
            metrics={"variant": {"diagnostic_only": True}},
        )

        decision = promotion_decision(result)
        score = score_result(result, objective="reliability_first")

        self.assertFalse(decision.promotable)
        self.assertEqual(decision.tier, "blocked")
        self.assertEqual(decision.state, "diagnostic_only")
        self.assertIn("diagnostic_only_candidate", decision.blockers)
        self.assertEqual(score["total"], 0.0)


if __name__ == "__main__":
    unittest.main()
