"""Focused tests for exterior fields, changed-part reuse and residual growth."""
from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from blender_blocking.placement.resfit_objective import (
    ResFitLossWeights, ResFitObjectiveEvaluator, default_constraint_penalty,
    evaluate_resfit_objective, surface_residual,
)
from blender_blocking.placement.resfit.residual_proposals import (
    generate_residual_proposals, refine_residual_parts,
)
from blender_blocking.placement.resfit_optimizer import CoordinateDescentConfig, OptimizationBudget
from blender_blocking.primitives.analytic_primitives import EllipsoidPrimitive, AnisotropicGaussianPrimitive


class CountingEllipsoid(EllipsoidPrimitive):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sdf_lengths = []
        self.sample_calls = 0

    def sdf_batch(self, points):
        self.sdf_lengths.append(len(points))
        return super().sdf_batch(points)

    def sample_surface(self, count):
        self.sample_calls += 1
        return super().sample_surface(count)


def geometry_weights():
    return ResFitLossWeights(primitive_count=0.0, overlap_penalty=0.0,
                             constraint_penalty=0.0)


class TestExteriorObjective(unittest.TestCase):
    def test_signed_union_does_not_reward_buried_surface(self):
        inner = EllipsoidPrimitive(radii=(1, 1, 1))
        outer = EllipsoidPrimitive(radii=(2, 2, 2))
        points = inner.sample_surface(32)
        self.assertAlmostEqual(surface_residual([inner], points), 0.0, places=12)
        self.assertAlmostEqual(surface_residual([inner, outer], points), 1.0, places=12)

    def test_reverse_samples_exclude_completely_buried_part(self):
        outer = EllipsoidPrimitive(radii=(2, 2, 2))
        inner = EllipsoidPrimitive(radii=(0.5, 0.5, 0.5))
        points = outer.sample_surface(64)
        objective = ResFitObjectiveEvaluator(points, geometry_weights())
        alone = objective([outer])
        combined = objective([outer, inner])
        self.assertAlmostEqual(combined.terms["surface_exterior_samples"], 0.0, places=12)
        self.assertAlmostEqual(combined.total, alone.total, places=12)

    def test_changed_part_reuses_unchanged_target_row_and_samples(self):
        first = CountingEllipsoid(center=(-2, 0, 0))
        second = CountingEllipsoid(center=(2, 0, 0))
        points = np.concatenate((first.sample_surface(31), second.sample_surface(31)))
        first.sdf_lengths.clear(); second.sdf_lengths.clear()
        first.sample_calls = second.sample_calls = 0
        objective = ResFitObjectiveEvaluator(points, geometry_weights(), occupied_points=points[:10])
        objective([first, second])
        first_calls, second_calls = list(first.sdf_lengths), list(second.sdf_lengths)
        objective([first, second])
        self.assertEqual(first.sdf_lengths, first_calls)
        self.assertEqual(second.sdf_lengths, second_calls)
        second.center[0] += 0.25
        reused = objective([first, second])
        self.assertEqual(first.sdf_lengths.count(len(points)), 1)
        self.assertEqual(first.sample_calls, 1)
        self.assertEqual(second.sdf_lengths.count(len(points)), 2)
        fresh = evaluate_resfit_objective([first, second], points, geometry_weights(), points[:10])
        self.assertEqual(reused.terms, fresh.terms)
        self.assertAlmostEqual(reused.total, fresh.total, places=12)

    def test_covariance_and_rotation_changes_invalidate_geometry_rows(self):
        ellipsoid = EllipsoidPrimitive(radii=(1, 2, 3))
        gaussian = AnisotropicGaussianPrimitive(center=(4, 0, 0), covariance=np.eye(3))
        points = ellipsoid.sample_surface(40)
        objective = ResFitObjectiveEvaluator(points, geometry_weights())
        objective([ellipsoid, gaussian])
        ellipsoid.rotation = np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1)))
        gaussian.covariance[0, 0] = 2.0
        changed = objective([ellipsoid, gaussian])
        fresh = evaluate_resfit_objective([ellipsoid, gaussian], points, geometry_weights())
        self.assertEqual(changed.terms, fresh.terms)

    def test_target_is_frozen_and_cache_is_bounded(self):
        primitive = EllipsoidPrimitive()
        points = primitive.sample_surface(64)
        objective = ResFitObjectiveEvaluator(points, geometry_weights(), max_cached_parts=2)
        original = objective([primitive]).total
        points[:] = 100.0
        self.assertEqual(objective([primitive]).total, original)
        for index in range(5):
            primitive.center[0] = float(index)
            objective([primitive])
        self.assertLessEqual(len(objective._parts), 2)
        self.assertLessEqual(len(objective._sample_fields), 4)

    def test_inactive_terms_do_not_run_numeric_work_or_hooks(self):
        part = CountingEllipsoid()
        weights = ResFitLossWeights(surface_residual=0, primitive_count=0,
                                    overlap_penalty=0, constraint_penalty=0)
        def forbidden(_):
            raise AssertionError("inactive hook must not execute")
        objective = ResFitObjectiveEvaluator(np.zeros((4, 3)), weights,
            occupied_points=np.zeros((2, 3)), silhouette_hook=forbidden,
            topology_penalty_hook=forbidden, constraint_penalty_hook=forbidden,
            uncertainty_penalty_hook=forbidden)
        self.assertEqual(objective([part]).total, 0)
        self.assertEqual(part.sdf_lengths, [])
        self.assertEqual(part.sample_calls, 0)

    def test_batch_returns_scored_prefix_and_reuses_unchanged_parts(self):
        first = CountingEllipsoid(center=(-2, 0, 0))
        second = CountingEllipsoid(center=(2, 0, 0))
        points = np.concatenate((first.sample_surface(31), second.sample_surface(31)))
        first.sdf_lengths.clear(); first.sample_calls = 0
        objective = ResFitObjectiveEvaluator(points, geometry_weights())
        objective([first, second])
        plus, minus = deepcopy(second), deepcopy(second)
        plus.center[0] += 0.2
        minus.center[0] -= 0.2
        budget = OptimizationBudget(max_objective_evaluations=1)
        results = list(objective.evaluate_batch(([first, plus], [first, minus]), budget=budget))
        self.assertEqual(len(results), 1)
        self.assertEqual(budget.objective_evaluations, 1)
        self.assertEqual(first.sdf_lengths.count(len(points)), 1)
        self.assertEqual(first.sample_calls, 1)
        self.assertAlmostEqual(results[0].total, objective([first, plus]).total)

    def test_constraint_accepts_covariance_method_and_matrix(self):
        self.assertEqual(default_constraint_penalty([
            EllipsoidPrimitive(), AnisotropicGaussianPrimitive(covariance=np.eye(3)),
        ]), 0.0)

    def test_invalid_sdf_is_rejected(self):
        class BadPart:
            def sdf_batch(self, points):
                return np.full(len(points), np.nan)
        with self.assertRaisesRegex(ValueError, "finite"):
            ResFitObjectiveEvaluator(np.zeros((2, 3)))([BadPart()])


class TestResidualProposals(unittest.TestCase):
    def setUp(self):
        self.first = EllipsoidPrimitive(center=(0, 0, 0))
        self.second = EllipsoidPrimitive(center=(4, 0, 0))
        self.points = np.concatenate((self.first.sample_surface(64), self.second.sample_surface(64)))

    def test_separate_missing_regions_are_not_averaged(self):
        third = EllipsoidPrimitive(center=(-4, 0, 0))
        points = np.concatenate((self.points, third.sample_surface(64)))
        proposals = generate_residual_proposals([self.first], points, "ellipsoid", max_parts=3)
        additions = [proposal for proposal in proposals if proposal.label.startswith("add_")]
        self.assertEqual(len(additions), 2)
        self.assertTrue(all(abs(proposal.primitives[-1].center[0]) > 3 for proposal in additions))

    def test_part_cap_still_allows_replacement_and_reallocation(self):
        buried = EllipsoidPrimitive(radii=(0.2, 0.2, 0.2))
        proposals = generate_residual_proposals([self.first, buried], self.points,
                                                "ellipsoid", max_parts=2)
        self.assertTrue(any(proposal.label.startswith("replace_low_") for proposal in proposals))
        self.assertTrue(any(proposal.label.startswith("reallocate_") for proposal in proposals))
        self.assertTrue(all(len(proposal.primitives) <= 2 for proposal in proposals))

    def test_growth_is_accepted_only_on_complete_objective_improvement(self):
        objective = ResFitObjectiveEvaluator(self.points, geometry_weights())
        initial = objective([self.first])
        budget = OptimizationBudget(max_objective_evaluations=4)
        result = refine_residual_parts([self.first], self.points, "ellipsoid", objective,
            CoordinateDescentConfig(iterations=0), budget=budget, initial_result=initial,
            max_parts=2, refinement_steps=0)
        self.assertLess(result.final_result.total, initial.total)
        self.assertEqual(len(result.primitives), 2)
        self.assertEqual(result.accepted_proposals, 1)
        self.assertLessEqual(budget.objective_evaluations, 4)
        self.assertAlmostEqual(objective(result.primitives).total, result.final_result.total, places=12)
        np.testing.assert_array_equal(self.first.center, np.zeros(3))

    def test_failed_proposals_preserve_previous_state(self):
        evaluator = ResFitObjectiveEvaluator(self.points, geometry_weights())
        initial = evaluator([self.first])
        def fails(parts):
            raise RuntimeError("fixture failure")
        budget = OptimizationBudget(max_objective_evaluations=2)
        result = refine_residual_parts([self.first], self.points, "ellipsoid", fails,
            CoordinateDescentConfig(iterations=0), budget=budget, initial_result=initial,
            max_parts=2, refinement_steps=0)
        self.assertEqual(result.final_result, initial)
        self.assertEqual(len(result.primitives), 1)
        self.assertEqual(result.accepted_proposals, 0)
        self.assertLessEqual(budget.objective_evaluations, 2)
        self.assertTrue(all(item["status"] == "failed" for item in result.proposals))

    def test_configured_silhouette_term_can_reject_geometric_improvement(self):
        weights = ResFitLossWeights(primitive_count=0.0, overlap_penalty=0.0,
                                   constraint_penalty=0.0, silhouette=1.0)
        def silhouette(parts):
            same = len(parts) == 1 and np.array_equal(parts[0].center, self.first.center)
            return {"fixture": 0.0 if same else 100.0}
        objective = ResFitObjectiveEvaluator(self.points, weights, silhouette_hook=silhouette)
        initial = objective([self.first])
        result = refine_residual_parts([self.first], self.points, "ellipsoid", objective,
            CoordinateDescentConfig(iterations=0), budget=OptimizationBudget(max_objective_evaluations=4),
            initial_result=initial, max_parts=2, refinement_steps=0)
        self.assertEqual(result.final_result, initial)
        self.assertEqual(result.accepted_proposals, 0)

    def test_later_failure_does_not_discard_completed_improvement(self):
        evaluator = ResFitObjectiveEvaluator(self.points, geometry_weights())
        initial = evaluator([self.first])
        calls = 0
        def first_only(parts):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("later fixture failure")
            return evaluator(parts)
        result = refine_residual_parts([self.first], self.points, "ellipsoid", first_only,
            CoordinateDescentConfig(iterations=0), budget=OptimizationBudget(max_objective_evaluations=4),
            initial_result=initial, max_parts=2, refinement_steps=0)
        self.assertLess(result.final_result.total, initial.total)
        self.assertEqual(result.accepted_proposals, 1)
        self.assertTrue(any(item["status"] == "failed" for item in result.proposals))

    def test_no_proposal_for_exact_fit_or_exhausted_budget(self):
        points = self.first.sample_surface(64)
        self.assertEqual(generate_residual_proposals([self.first], points, "ellipsoid"), ())
        objective = ResFitObjectiveEvaluator(points, geometry_weights())
        initial = objective([self.first])
        result = refine_residual_parts([self.first], points, "ellipsoid", objective,
            CoordinateDescentConfig(iterations=0), budget=OptimizationBudget(max_objective_evaluations=0),
            initial_result=initial)
        self.assertEqual(result.objective_evaluations, 0)
        self.assertEqual(result.proposals, ())


if __name__ == "__main__":
    unittest.main()
