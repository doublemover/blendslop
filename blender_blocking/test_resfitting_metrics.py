"""Tests for ResFit metrics (pure Python)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from placement.resfit_objective import ResFitObjectiveResult
from placement.resfit_optimizer import CoordinateDescentConfig, coordinate_descent_optimize
from placement.resfit.status import apply_resfit_quality_floors, resfit_candidate_status
from placement.resfitting import ResidualFitter
from primitives.superfrustum import SuperFrustum


class TestResfittingMetrics(unittest.TestCase):
    def test_sdf_batch_matches_scalar(self) -> None:
        rng = np.random.default_rng(1234)
        points = rng.normal(size=(10, 3))
        sf = SuperFrustum(
            position=(0.1, -0.2, 0.3),
            orientation=(0.2, 0.4),
            radius_bottom=1.2,
            radius_top=0.8,
            height=2.5,
        )
        batch = sf.sdf_batch(points)
        scalar = np.array([sf.sdf(p) for p in points])
        np.testing.assert_allclose(batch, scalar, rtol=1e-6, atol=1e-6)

    def test_gradient_batch_matches_scalar(self) -> None:
        sf = SuperFrustum(
            position=(0.1, -0.2, 0.3),
            orientation=(0.2, 0.4),
            radius_bottom=1.2,
            radius_top=0.8,
            height=2.5,
        )
        point = np.array([[0.3, -0.4, 0.9]])
        grads_batch = sf.gradient_batch(point)
        grads_scalar = sf.gradient(point[0])

        np.testing.assert_allclose(
            grads_batch["position"][0], grads_scalar["position"], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            grads_batch["orientation"][0],
            grads_scalar["orientation"],
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertAlmostEqual(
            grads_batch["radius_bottom"][0], grads_scalar["radius_bottom"], places=5
        )
        self.assertAlmostEqual(
            grads_batch["radius_top"][0], grads_scalar["radius_top"], places=5
        )
        self.assertAlmostEqual(
            grads_batch["height"][0], grads_scalar["height"], places=5
        )

    def test_residual_error_shapes(self) -> None:
        fitter = ResidualFitter()
        target_points = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        primitive = SuperFrustum(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0),
            radius_bottom=1.0,
            radius_top=1.0,
            height=2.0,
        )
        total_error, per_point_errors = fitter.compute_residual_error(
            [primitive], target_points
        )
        self.assertEqual(len(per_point_errors), len(target_points))
        self.assertGreaterEqual(total_error, 0.0)

    def test_residual_error_matches_scalar(self) -> None:
        rng = np.random.default_rng(2025)
        target_points = rng.normal(size=(20, 3))
        primitives = [
            SuperFrustum(
                position=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0),
                radius_bottom=1.0,
                radius_top=1.0,
                height=2.0,
            ),
            SuperFrustum(
                position=(0.5, -0.2, 0.1),
                orientation=(0.1, 0.2),
                radius_bottom=0.9,
                radius_top=0.6,
                height=1.5,
            ),
        ]
        fitter = ResidualFitter()
        total_error, per_point_errors = fitter.compute_residual_error(
            primitives, target_points
        )

        manual = []
        for p in target_points:
            min_sdf = min(abs(sf.sdf(p)) for sf in primitives)
            manual.append(min_sdf)
        manual = np.array(manual)

        self.assertAlmostEqual(total_error, float(np.mean(manual)), places=6)
        np.testing.assert_allclose(per_point_errors, manual, rtol=1e-6, atol=1e-6)

    def test_initialize_from_empty_voxels(self) -> None:
        fitter = ResidualFitter()
        empty_grid = np.zeros((4, 4, 4), dtype=np.float32)
        primitives = fitter.initialize_from_voxels(empty_grid, num_initial=3)
        self.assertEqual(primitives, [])

    def test_fit_history_captures_objective_progress(self) -> None:
        rng = np.random.default_rng(2026)
        theta = rng.uniform(0.0, 2 * np.pi, size=320)
        z = rng.uniform(-1.0, 1.0, size=320)
        target_points = np.column_stack(
            [
                1.4 * np.cos(theta),
                1.4 * np.sin(theta),
                z,
            ]
        )
        initial_primitives = [
            SuperFrustum(
                position=(2.5, -2.1, 0.0),
                orientation=(0.1, 0.2),
                radius_bottom=0.2,
                radius_top=0.2,
                height=0.5,
            )
        ]

        fitter = ResidualFitter(
            max_primitives=3,
            max_iterations=3,
            learning_rate=0.01,
            optimization_steps=12,
            error_threshold=1e-5,
        )
        fitter.fit(target_points, initial_primitives=initial_primitives, verbose=False)
        history = fitter.get_history()

        self.assertIsNotNone(history["final_error"])
        self.assertGreater(history["iterations"], 0)
        self.assertEqual(history["iterations"], len(history["errors"]))
        self.assertGreater(history["num_primitives"], 0)
        self.assertGreater(history["final_error"], 0.0)
        self.assertEqual(history["num_primitives"], len(fitter.primitives))

        start_error = float(history["errors"][0])
        end_error = float(history["errors"][-1])
        self.assertGreaterEqual(start_error, 0.0)
        self.assertGreaterEqual(end_error, 0.0)
        self.assertLessEqual(
            end_error,
            start_error * 1.10 if start_error > 0 else end_error,
            "Objective history should not degrade severely",
        )

        improvement = (start_error - end_error) / start_error if start_error > 0 else 0.0
        self.assertGreaterEqual(improvement, -0.10)
        self.assertLessEqual(improvement, 1.0)

    def test_optimize_vectorized_matches_scalar(self) -> None:
        rng = np.random.default_rng(2026)
        target_points = rng.normal(size=(40, 3))
        base_primitives = [
            SuperFrustum(
                position=(0.1, -0.2, 0.3),
                orientation=(0.2, 0.4),
                radius_bottom=1.1,
                radius_top=0.9,
                height=2.2,
            ),
            SuperFrustum(
                position=(-0.15, 0.25, -0.05),
                orientation=(0.3, 0.5),
                radius_bottom=0.9,
                radius_top=0.7,
                height=1.6,
            ),
        ]

        def copy_primitives(primitives: list[SuperFrustum]) -> list[SuperFrustum]:
            return [SuperFrustum.from_dict(sf.to_dict()) for sf in primitives]

        fitter = ResidualFitter(learning_rate=0.02, optimization_steps=1)
        vec_primitives = copy_primitives(base_primitives)
        scalar_primitives = copy_primitives(base_primitives)

        fitter.optimize_primitives(
            vec_primitives, target_points, steps=1, use_vectorized=True
        )
        fitter.optimize_primitives(
            scalar_primitives, target_points, steps=1, use_vectorized=False
        )

        for vec, scalar in zip(vec_primitives, scalar_primitives):
            np.testing.assert_allclose(
                vec.position, scalar.position, rtol=1e-4, atol=1e-5
            )
            np.testing.assert_allclose(
                vec.orientation, scalar.orientation, rtol=1e-4, atol=1e-5
            )
            np.testing.assert_allclose(
                vec.radius_bottom, scalar.radius_bottom, rtol=1e-4, atol=1e-5
            )
            np.testing.assert_allclose(
                vec.radius_top, scalar.radius_top, rtol=1e-4, atol=1e-5
            )
            np.testing.assert_allclose(vec.height, scalar.height, rtol=1e-4, atol=1e-5)

    def test_coordinate_descent_allows_initialization_only(self) -> None:
        primitive = SuperFrustum(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0),
            radius_bottom=1.0,
            radius_top=1.0,
            height=2.0,
        )

        result = coordinate_descent_optimize(
            [primitive],
            lambda _: ResFitObjectiveResult(total=1.0, terms={"constant": 1.0}),
            CoordinateDescentConfig(iterations=0),
        )

        self.assertEqual(result.termination_reason, "zero_iterations")
        self.assertEqual(result.history, ())
        self.assertEqual(result.objective_evaluations, 1)

    def test_coordinate_descent_respects_objective_evaluation_budget(self) -> None:
        primitive = SuperFrustum(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0),
            radius_bottom=1.0,
            radius_top=1.0,
            height=2.0,
        )

        result = coordinate_descent_optimize(
            [primitive],
            lambda _: ResFitObjectiveResult(total=1.0, terms={"constant": 1.0}),
            CoordinateDescentConfig(
                iterations=50,
                max_objective_evaluations=3,
            ),
        )

        self.assertEqual(result.termination_reason, "objective_evaluation_budget")
        self.assertLessEqual(result.objective_evaluations, 3)

    def test_resfit_budget_limited_improvement_can_still_succeed(self) -> None:
        result = SimpleNamespace(
            primitives=(object(),),
            warnings=(),
            optimization_termination_reason="elapsed_time_budget",
            final_loss=SimpleNamespace(total=0.25),
        )

        status, degraded, errors, warnings = resfit_candidate_status(
            config={},
            result=result,
            improved=0.1,
            profile_rows_present=True,
            profile_init_warning="",
        )

        self.assertEqual(status, "success")
        self.assertFalse(degraded)
        self.assertEqual(errors, ())
        self.assertIn("budget-limited primitive fit accepted", "\n".join(warnings))

    def test_resfit_budget_limited_can_remain_degraded_when_strict(self) -> None:
        result = SimpleNamespace(
            primitives=(object(),),
            warnings=(),
            optimization_termination_reason="elapsed_time_budget",
            final_loss=SimpleNamespace(total=0.25),
        )

        status, degraded, _errors, _warnings = resfit_candidate_status(
            config={"require_optimizer_completion": True},
            result=result,
            improved=0.1,
            profile_rows_present=True,
            profile_init_warning="",
        )

        self.assertEqual(status, "degraded")
        self.assertTrue(degraded)

    def test_primitive_quality_floors_fail_low_backend_iou(self) -> None:
        metric = SimpleNamespace(
            area_iou_min=0.12,
            topology_score=0.95,
            extras={"topology": {"watertight": True, "boundary_edges": 0}},
        )

        status, degraded, errors, warnings = apply_resfit_quality_floors(
            config={},
            status="success",
            degraded=False,
            errors=(),
            warnings=(),
            metric=metric,
        )

        self.assertEqual(status, "failed")
        self.assertFalse(degraded)
        self.assertIn("backend min IoU", "\n".join(errors))
        self.assertEqual(warnings, ())

    def test_primitive_quality_floors_degrade_bad_topology(self) -> None:
        metric = SimpleNamespace(
            area_iou_min=0.8,
            topology_score=0.55,
            extras={"topology": {"watertight": False, "boundary_edges": 12}},
        )

        status, degraded, errors, warnings = apply_resfit_quality_floors(
            config={},
            status="success",
            degraded=False,
            errors=(),
            warnings=(),
            metric=metric,
        )

        self.assertEqual(status, "degraded")
        self.assertTrue(degraded)
        self.assertEqual(errors, ())
        self.assertIn("topology score", "\n".join(warnings))


if __name__ == "__main__":
    unittest.main()
