"""Tests for ResFit metrics (pure Python)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from placement.resfit_objective import ResFitObjectiveResult
from placement.resfit_optimizer import CoordinateDescentConfig, coordinate_descent_optimize
from placement.resfit.backend_adapter import _family_pipeline_config
from placement.resfit.config import ResFitPipelineConfig
from placement.resfit.initialization import _candidate_families
from placement.resfit.metrics import build_resfit_candidate_metrics
from placement.resfit.status import apply_resfit_quality_floors, resfit_candidate_status
from placement.resfitting import ResidualFitter
from metrics.topology import mesh_topology_report
from primitives.analytic_primitives import EllipsoidPrimitive, SuperquadricPrimitive
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

    def test_candidate_families_deduplicates_configured_families(self) -> None:
        self.assertEqual(
            _candidate_families(
                {
                    "primitive_families": (
                        "superquadric",
                        "superquadric",
                        "superfrustum",
                        "ellipsoid",
                    )
                }
            ),
            ("superquadric", "superfrustum", "ellipsoid"),
        )
        self.assertEqual(
            _candidate_families({"primitive_families": "ellipsoid, superfrustum"}),
            ("ellipsoid", "superfrustum"),
        )

    def test_candidate_family_override_stays_single_family(self) -> None:
        self.assertEqual(
            _candidate_families(
                {
                    "primitive_family": "ellipsoid",
                    "primitive_families": ("superquadric", "superfrustum"),
                }
            ),
            ("ellipsoid",),
        )

    def test_family_pipeline_config_shares_runtime_and_eval_budget(self) -> None:
        base = ResFitPipelineConfig(
            primitive_family="superquadric",
            optimizer=CoordinateDescentConfig(
                iterations=4,
                max_elapsed_s=9.0,
                max_objective_evaluations=90,
            ),
        )

        family_config = _family_pipeline_config(
            base,
            family="superfrustum",
            family_count=3,
            share_budget=True,
        )

        self.assertEqual(family_config.primitive_family, "superfrustum")
        self.assertEqual(family_config.optimizer.max_elapsed_s, 3.0)
        self.assertEqual(family_config.optimizer.max_objective_evaluations, 30)
        self.assertEqual(base.primitive_family, "superquadric")

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

    def test_primitive_quality_floors_degrade_low_backend_iou_by_default(self) -> None:
        metric = SimpleNamespace(
            area_iou_min=0.31,
            topology_score=0.95,
            extras={
                "topology": {"watertight": True, "boundary_edges": 0},
                "backend_quality_source": "objective_proxy",
                "primitive_count": 4,
            },
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
        self.assertIn("renderable artifacts", "\n".join(warnings))
        self.assertIn("objective proxy min IoU 0.310", "\n".join(warnings))

    def test_primitive_quality_floors_fail_low_backend_iou_when_strict(self) -> None:
        metric = SimpleNamespace(
            area_iou_min=0.12,
            topology_score=0.95,
            extras={"topology": {"watertight": True, "boundary_edges": 0}},
        )

        status, degraded, errors, warnings = apply_resfit_quality_floors(
            config={"fail_on_backend_iou_floor": True},
            status="success",
            degraded=False,
            errors=(),
            warnings=(),
            metric=metric,
        )

        self.assertEqual(status, "failed")
        self.assertFalse(degraded)
        self.assertIn("min IoU 0.120 is below 0.350", "\n".join(errors))
        self.assertEqual(warnings, ())

    def test_resfit_metrics_use_required_view_support_for_backend_floor(self) -> None:
        result = SimpleNamespace(
            initial_loss=SimpleNamespace(total=2.0, terms={}),
            final_loss=SimpleNamespace(
                total=1.0,
                terms={"surface_residual": 4.0, "silhouette": 4.0},
            ),
            optimization_termination_reason="",
            primitives=(object(),),
            attempts=(),
            family_attempts=(),
            selected_attempt="",
            objective_evaluations=1,
            optimizer_elapsed_s=0.1,
        )

        metric, _summary = build_resfit_candidate_metrics(
            elapsed_s=0.1,
            result=result,
            primitive_family="ellipsoid",
            mesh_metadata={},
            topology_payload={"topology_score": 0.95, "watertight": True},
            uncertainty_signal={},
            profile_rows=(
                {"view": "front", "width_world": 1.0, "z_world": 0.0, "confidence": 0.40},
                {"view": "side", "width_world": 1.0, "z_world": 0.0, "confidence": 0.38},
                {"view": "top", "width_world": 1.0, "z_world": 0.0, "confidence": 0.39},
            ),
            pipeline_config=ResFitPipelineConfig(),
            history_records=({"accepted_moves": 1, "rejected_moves": 0},),
            surface_meta={},
            occupied_meta={},
            signal_summary={},
            initial_primitives=(),
            max_runtime_s=None,
            max_objective_evaluations=None,
        )

        self.assertAlmostEqual(metric.area_iou_min, 0.38)
        self.assertAlmostEqual(metric.area_iou_mean, 0.39)
        self.assertAlmostEqual(metric.boundary_iou_mean, 0.39)
        self.assertEqual(metric.extras["backend_quality_source"], "profile_per_view")
        self.assertAlmostEqual(metric.extras["objective_proxy_iou"], 0.2)

    def test_primitive_quality_floors_remove_accepted_warning_on_failure(self) -> None:
        metric = SimpleNamespace(
            area_iou_min=0.004,
            topology_score=0.95,
            extras={"topology": {"watertight": True, "boundary_edges": 0}},
        )

        status, _degraded, errors, warnings = apply_resfit_quality_floors(
            config={"fail_on_backend_iou_floor": True},
            status="success",
            degraded=False,
            errors=(),
            warnings=(
                "optimization stopped by elapsed_time_budget",
                "budget-limited primitive fit accepted: objective improved and valid primitives were emitted",
            ),
            metric=metric,
        )

        self.assertEqual(status, "failed")
        self.assertIn("min IoU 0.004 is below 0.350", "\n".join(errors))
        self.assertNotIn("budget-limited primitive fit accepted", "\n".join(warnings))

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

    def test_primitive_quality_floors_fail_strict_noop_optimization(self) -> None:
        metric = SimpleNamespace(
            area_iou_min=0.8,
            topology_score=0.95,
            extras={
                "topology": {"watertight": True, "boundary_edges": 0},
                "objective": {
                    "improved": False,
                    "all_attempts_noop": True,
                    "accepted_move_count": 0,
                },
            },
        )

        status, degraded, errors, warnings = apply_resfit_quality_floors(
            config={"fail_on_noop_optimization": True},
            status="success",
            degraded=False,
            errors=(),
            warnings=(),
            metric=metric,
        )

        self.assertEqual(status, "failed")
        self.assertFalse(degraded)
        self.assertIn("primitive optimizer made no accepted", "\n".join(errors))
        self.assertIn("primitive objective did not improve", "\n".join(warnings))

    def test_analytic_ellipsoid_mesh_is_watertight(self) -> None:
        mesh = EllipsoidPrimitive().to_mesh_data(resolution=24)
        report = mesh_topology_report(mesh.vertices, mesh.faces)

        self.assertTrue(report.watertight)
        self.assertEqual(report.boundary_edges, 0)
        self.assertEqual(report.non_manifold_edges, 0)
        self.assertEqual(report.connected_components, 1)

    def test_analytic_superquadric_mesh_is_watertight(self) -> None:
        mesh = SuperquadricPrimitive(epsilon1=0.28, epsilon2=0.28).to_mesh_data(
            resolution=24
        )
        report = mesh_topology_report(mesh.vertices, mesh.faces)

        self.assertTrue(report.watertight)
        self.assertEqual(report.boundary_edges, 0)
        self.assertEqual(report.non_manifold_edges, 0)
        self.assertEqual(report.connected_components, 1)


if __name__ == "__main__":
    unittest.main()
