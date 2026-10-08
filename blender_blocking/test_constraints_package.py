"""Pure Python tests for human correction constraints."""

from __future__ import annotations

import json
import unittest

import numpy as np

from constraints import (
    ConstraintSet,
    DimensionConstraint,
    ScribbleConstraint,
    SymmetryConstraint,
    apply_constraints_to_mask,
    apply_constraints_to_objective_scores,
    apply_dimension_constraints_to_bounds,
    dumps_constraint_set,
    loads_constraint_set,
)
from reconstruction.candidate_scoring import score_candidate
from reconstruction.types import CandidateMetrics, CandidateResult


class ConstraintPackageTests(unittest.TestCase):
    def test_constraint_json_round_trip_is_deterministic(self) -> None:
        constraints = ConstraintSet.from_constraints(
            [
                ScribbleConstraint(
                    view="front",
                    kind="foreground",
                    points_px=((2.0, 3.0),),
                    brush_radius_px=1.0,
                    confidence=1.0,
                ),
                DimensionConstraint(name="height", value_u=4.0, tolerance_u=0.1),
            ]
        )

        encoded_a = dumps_constraint_set(constraints)
        encoded_b = dumps_constraint_set(loads_constraint_set(encoded_a))

        self.assertEqual(encoded_a, encoded_b)
        self.assertEqual(json.loads(encoded_a)["version"], 1)

    def test_foreground_and_background_scribbles_override_mask(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        constraints = ConstraintSet.from_constraints(
            [
                ScribbleConstraint(
                    view="front",
                    kind="foreground",
                    points_px=((3.0, 3.0),),
                    brush_radius_px=1.0,
                    confidence=1.0,
                ),
                ScribbleConstraint(
                    view="front",
                    kind="background",
                    points_px=((6.0, 6.0),),
                    brush_radius_px=0.0,
                    confidence=0.5,
                ),
            ]
        )

        result = apply_constraints_to_mask(mask, constraints, "front")

        self.assertTrue(result.mask[3, 3])
        self.assertFalse(result.mask[6, 6])
        self.assertGreater(result.report["forced_foreground_pixels"], 0)
        self.assertGreater(result.report["forced_background_pixels"], 0)

    def test_dimension_constraints_resize_bounds(self) -> None:
        constraints = ConstraintSet.from_constraints(
            [
                DimensionConstraint(name="width", value_u=6.0, tolerance_u=0.1),
                DimensionConstraint(name="height", value_u=10.0, tolerance_u=0.1),
            ]
        )

        bounds_min, bounds_max, satisfaction = apply_dimension_constraints_to_bounds(
            (-1.0, -1.0, 0.0),
            (1.0, 1.0, 2.0),
            constraints,
        )

        self.assertAlmostEqual(bounds_max[0] - bounds_min[0], 6.0)
        self.assertAlmostEqual(bounds_max[2] - bounds_min[2], 10.0)
        self.assertEqual(len(satisfaction), 2)

    def test_soft_symmetry_scores_do_not_hard_fail(self) -> None:
        constraints = ConstraintSet.from_constraints(
            [SymmetryConstraint(plane="xz", confidence=0.8)]
        )
        candidate = {"symmetry_scores": {"xz": 0.25}}

        result = apply_constraints_to_objective_scores(
            {"score": 1.0},
            constraints,
            candidate=candidate,
            fail_on_unsatisfied_hard_constraints=True,
        )

        self.assertFalse(result.hard_failed)
        self.assertGreater(result.scores["score"], 1.0)

    def test_constraint_penalty_report_feeds_candidate_scoring(self) -> None:
        constraints = ConstraintSet.from_constraints(
            [DimensionConstraint(name="height", value_u=4.0, tolerance_u=0.1)]
        )
        objective = apply_constraints_to_objective_scores(
            {"score": 1.0},
            constraints,
            candidate={"dimensions": {"height": 2.0}},
            constraint_weight=2.0,
        )
        metrics = CandidateMetrics(
            constraint_report=objective.report,
            per_view={
                "front": {
                    "area_iou": 1.0,
                    "boundary_iou": 1.0,
                    "soft_iou": 1.0,
                    "signed_distance_loss": 0.0,
                    "required": True,
                    "pass": True,
                    "reason": "",
                }
            },
        )
        result = CandidateResult(
            candidate_id="candidate",
            backend_name="backend",
            status="success",
            metric_result=metrics,
        )

        score = score_candidate(result)
        terms = {term.name: term for term in score.terms}

        self.assertGreater(objective.report["constraint_penalty"], 0.0)
        self.assertGreater(metrics.constraint_penalty, 0.0)
        self.assertLess(terms["constraint_penalty"].weighted, 0.0)


if __name__ == "__main__":
    unittest.main()
