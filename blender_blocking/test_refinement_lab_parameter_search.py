"""Tests for refinement objective scoring."""

from __future__ import annotations

import unittest

from refinement_lab.contracts import ExperimentResult
from refinement_lab.parameter_search import rank_results, score_result


def _result(name: str, avg: float, front: float, side: float, top: float) -> ExperimentResult:
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
        metrics={
            "average_iou": avg,
            "front_iou": front,
            "side_iou": side,
            "top_iou": top,
            "topology_score": 1.0,
            "editability_score": 0.5,
        },
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


if __name__ == "__main__":
    unittest.main()
