"""Tests for refinement objective scoring."""

from __future__ import annotations

import unittest

from refinement_lab.contracts import ExperimentResult
from refinement_lab.parameter_search import (
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
        self.assertIn(
            "missing_required_metrics",
            promotion_decision(metric_only).blockers,
        )

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


if __name__ == "__main__":
    unittest.main()
