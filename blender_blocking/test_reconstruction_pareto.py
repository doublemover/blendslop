"""Tests for reconstruction Pareto frontier reports."""

from __future__ import annotations

import unittest

from reconstruction.ensemble import EnsembleRunResult
from reconstruction.pareto import pareto_report_from_candidates
from reconstruction.types import CandidateMetrics, CandidateResult


def _candidate(
    candidate_id: str,
    *,
    min_iou: float,
    boundary: float,
    editability: float,
    topology: float,
    elapsed_s: float = 1.0,
    complexity: float = 0.2,
) -> CandidateResult:
    return CandidateResult(
        candidate_id=candidate_id,
        backend_name=candidate_id,
        status="success",
        metric_result=CandidateMetrics(
            area_iou_min=min_iou,
            area_iou_mean=min_iou,
            boundary_iou_mean=boundary,
            editability_score=editability,
            topology_score=topology,
            elapsed_s=elapsed_s,
            complexity_penalty=complexity,
        ),
    )


class ReconstructionParetoTests(unittest.TestCase):
    def test_pareto_frontier_keeps_real_tradeoffs_and_marks_dominated(self) -> None:
        fidelity = _candidate(
            "fidelity",
            min_iou=0.94,
            boundary=0.9,
            editability=0.35,
            topology=0.75,
        )
        editable = _candidate(
            "editable",
            min_iou=0.82,
            boundary=0.8,
            editability=0.95,
            topology=0.92,
        )
        dominated = _candidate(
            "dominated",
            min_iou=0.72,
            boundary=0.7,
            editability=0.3,
            topology=0.6,
            elapsed_s=2.0,
            complexity=0.7,
        )

        report = pareto_report_from_candidates(
            (fidelity, editable, dominated),
            selected_id="dominated",
            policy="balanced",
        )
        payload = report.to_dict()
        rows = {row["candidate_id"]: row for row in payload["candidates"]}  # type: ignore[index]

        self.assertEqual(set(report.frontier_ids), {"fidelity", "editable"})
        self.assertTrue(rows["fidelity"]["frontier"])
        self.assertTrue(rows["editable"]["frontier"])
        self.assertFalse(rows["dominated"]["frontier"])
        self.assertIn("fidelity", rows["dominated"]["dominated_by"])
        self.assertIn("selected candidate", payload["notes"][0])  # type: ignore[index]

    def test_ensemble_result_serializes_pareto_report(self) -> None:
        selected = _candidate(
            "fidelity",
            min_iou=0.94,
            boundary=0.9,
            editability=0.35,
            topology=0.75,
        )
        report = pareto_report_from_candidates((selected,), selected_id="fidelity")
        result = EnsembleRunResult(
            selected=selected,
            candidates=(selected,),
            scores=(),
            policy="best_score",
            pareto_report=report.to_dict(),
        )
        payload = result.to_dict()

        self.assertEqual(payload["pareto_report"]["frontier_ids"], ["fidelity"])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
