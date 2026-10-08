"""Focused geometry and optimizer checks for family-specific local updates."""
from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from placement.resfit_objective import ResFitObjectiveResult
from placement.resfit_optimizer import (CoordinateDescentConfig, ParameterBounds,
                                       coordinate_descent_optimize, discover_parameters,
                                       finite_difference_gradient)
from placement.resfit_parameters import (apply_parameter_increment, parameter_scale,
                                        parameter_value, set_parameter_value,
                                        parameter_jacobian, pullback_render_gradients)
from primitives.analytic_primitives import (AnisotropicGaussianPrimitive,
                                          EllipsoidPrimitive, SuperquadricPrimitive)
from primitives.superfrustum import SuperFrustum


class TestResfitParameterAdapters(unittest.TestCase):
    def test_discovery_covers_family_geometry(self):
        parts = [EllipsoidPrimitive(), SuperquadricPrimitive(), SuperFrustum(),
                 AnisotropicGaussianPrimitive()]
        refs = discover_parameters(parts)
        for index in (0, 1):
            self.assertEqual([axis for part, attr, axis in refs
                              if part == index and attr == "rotation"], [0, 1, 2])
        self.assertEqual([axis for part, attr, axis in refs
                          if part == 2 and attr == "orientation"], [0, 1])
        self.assertEqual([axis for part, attr, axis in refs
                          if part == 3 and attr == "covariance_cholesky"], list(range(6)))
        self.assertFalse(any(part == 3 and attr == "rotation" for part, attr, _ in refs))

    def test_positive_sizes_are_relative_and_bounded(self):
        for scale in (0.01, 1.0, 100.0):
            part = EllipsoidPrimitive(radii=(scale, 2 * scale, 3 * scale))
            apply_parameter_increment([part], (0, "radii", 1), 0.2, ParameterBounds())
            self.assertAlmostEqual(part.radii[1] / (2 * scale), np.exp(0.2))
            apply_parameter_increment([part], (0, "radii", 1), -1e4, ParameterBounds())
            self.assertGreaterEqual(part.radii[1], ParameterBounds().min_radius)
        part = SuperFrustum(radius_bottom=0.0, radius_top=0.0, height=0.1)
        apply_parameter_increment([part], (0, "radius_top", None), 0.1, ParameterBounds())
        self.assertGreater(part.radius_top, 0.0)

    def test_translation_scales_with_part_size(self):
        for size in (0.01, 1.0, 100.0):
            part = EllipsoidPrimitive(radii=(size, size, size))
            self.assertAlmostEqual(parameter_scale(part, "center", 0), size)
            apply_parameter_increment([part], (0, "center", 0), 0.1, ParameterBounds())
            self.assertAlmostEqual(part.center[0], 0.1 * size)

    def test_local_rotation_preserves_frame(self):
        part = EllipsoidPrimitive(rotation=np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))))
        axis_before = part.rotation[:, 0].copy()
        apply_parameter_increment([part], (0, "rotation", 0), 0.3, ParameterBounds())
        np.testing.assert_allclose(part.rotation[:, 0], axis_before, atol=1e-14)
        np.testing.assert_allclose(part.rotation.T @ part.rotation, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(part.rotation), 1.0)
        for axis in range(3):
            value = parameter_value(part, "rotation", axis)
            set_parameter_value(part, "rotation", axis, value, ParameterBounds())
        np.testing.assert_allclose(part.rotation.T @ part.rotation, np.eye(3), atol=1e-14)

    def test_frustum_tangents_move_both_poles(self):
        for phi in (0.0, np.pi):
            directions = []
            for axis in (0, 1):
                part = SuperFrustum(orientation=(0.0, phi))
                apply_parameter_increment([part], (0, "orientation", axis), 0.2, ParameterBounds())
                direction = part.get_axis_vector()
                self.assertAlmostEqual(np.linalg.norm(direction), 1.0)
                self.assertAlmostEqual(np.linalg.norm(direction[:2]), np.sin(0.2))
                directions.append(direction[:2])
            self.assertAlmostEqual(np.dot(*directions), 0.0, places=12)

    def test_cholesky_steps_keep_covariance_positive(self):
        factor = np.array(((0.01, 0., 0.), (0.005, 0.02, 0.), (0.003, -0.004, 0.03)))
        seed = AnisotropicGaussianPrimitive(covariance=factor @ factor.T)
        for axis in range(6):
            for delta in (-100.0, -0.2, 0.2, 100.0):
                part = deepcopy(seed)
                apply_parameter_increment([part], (0, "covariance_cholesky", axis), delta,
                                          ParameterBounds())
                self.assertTrue(np.isfinite(part.covariance).all())
                np.testing.assert_allclose(part.covariance, part.covariance.T, atol=1e-12)
                self.assertGreater(np.linalg.eigvalsh(part.covariance).min(), 0.0)
                np.linalg.cholesky(part.covariance)

    def test_optimizer_can_fit_rotation_orientation_and_covariance(self):
        angle = 0.5
        rotation = np.array(((np.cos(angle), -np.sin(angle), 0.),
                             (np.sin(angle), np.cos(angle), 0.), (0., 0., 1.)))
        parts = [EllipsoidPrimitive(radii=(2., 1., 0.5)), SuperFrustum(),
                 AnisotropicGaussianPrimitive()]
        cov = np.array(((1., 0.25, 0.), (0.25, 1., 0.), (0., 0., 1.)))
        target_axis = np.array((0., -np.sin(angle), np.cos(angle)))
        def objective(primitives):
            total = (np.sum((primitives[0].rotation - rotation) ** 2)
                     + np.sum((primitives[1].get_axis_vector() - target_axis) ** 2)
                     + np.sum((primitives[2].covariance - cov) ** 2))
            return ResFitObjectiveResult(total=float(total), terms={})
        result = coordinate_descent_optimize(parts, objective,
            CoordinateDescentConfig(iterations=5, initial_step=0.25,
                                    max_objective_evaluations=300))
        self.assertLess(result.best_loss, result.initial_result.total * 0.2)
        self.assertLessEqual(result.objective_evaluations, 300)
        self.assertEqual(result.best_loss, result.final_result.total)
        self.assertAlmostEqual(objective(result.primitives).total, result.best_loss, places=12)
        np.testing.assert_allclose(parts[0].rotation, np.eye(3))
        np.testing.assert_allclose(parts[2].covariance, np.eye(3))

    def test_budget_keeps_partial_rotation_improvement(self):
        part = EllipsoidPrimitive(radii=(2., 1., 0.5))
        refs = discover_parameters([part])
        target = deepcopy(part)
        apply_parameter_increment([target], (0, "rotation", 0), 0.1, ParameterBounds())
        def objective(primitives):
            return ResFitObjectiveResult(total=float(np.sum((primitives[0].rotation
                                                             - target.rotation) ** 2)), terms={})
        limit = 1 + 2 * refs.index((0, "rotation", 0)) + 1
        result = coordinate_descent_optimize([part], objective,
            CoordinateDescentConfig(iterations=2, max_objective_evaluations=limit))
        self.assertEqual(result.objective_evaluations, limit)
        self.assertLess(result.best_loss, 1e-20)
        self.assertAlmostEqual(objective(result.primitives).total, result.best_loss)

    def test_analytic_jacobians_match_local_finite_differences(self):
        seed = EllipsoidPrimitive(radii=(0.6, 1.2, 2.0), center=(0.2, -0.4, 0.1))
        for axis, angle in ((0, 0.3), (1, -0.2), (2, 0.5)):
            apply_parameter_increment([seed], (0, "rotation", axis), angle, ParameterBounds())
        factor = np.array(((0.8, 0., 0.), (0.2, 1.1, 0.), (-0.1, 0.3, 0.5)))
        gaussian = AnisotropicGaussianPrimitive(center=(-0.2, 0.3, -0.1),
            covariance=factor @ factor.T, opacity=0.7, confidence=0.8)
        superquadric = SuperquadricPrimitive(radii=(0.5, 0.8, 1.2), rotation=seed.rotation)
        for seed in (seed, gaussian, superquadric):
            for ref in discover_parameters([seed]):
                epsilon = 1e-6
                plus, minus = deepcopy(seed), deepcopy(seed)
                apply_parameter_increment([plus], ref, epsilon, ParameterBounds())
                apply_parameter_increment([minus], ref, -epsilon, ParameterBounds())
                jacobian = parameter_jacobian(seed, ref[1], ref[2])
                covariance = lambda p: (p.covariance() if callable(getattr(p, "covariance", None))
                    else p.covariance if hasattr(p, "covariance")
                    else p.rotation @ np.diag(p.radii ** 2) @ p.rotation.T)
                opacity = lambda p: np.clip(getattr(p, "opacity", getattr(p, "density", 1.0))
                                            * p.confidence, 0.0, 1.0)
                np.testing.assert_allclose(jacobian.center,
                    (plus.center - minus.center) / (2 * epsilon), atol=1e-8, rtol=1e-7)
                np.testing.assert_allclose(jacobian.covariance,
                    (covariance(plus) - covariance(minus)) / (2 * epsilon), atol=1e-8, rtol=1e-7)
                self.assertAlmostEqual(jacobian.opacity,
                    (opacity(plus) - opacity(minus)) / (2 * epsilon), places=8)

    def test_pullback_matches_scalar_local_derivative(self):
        factor = np.array(((1., 0., 0.), (0.2, 0.7, 0.), (-0.1, 0.3, 0.8)))
        parts = [EllipsoidPrimitive(radii=(0.5, 0.9, 1.2)),
                 AnisotropicGaussianPrimitive(covariance=factor @ factor.T,
                                               opacity=0.6, confidence=0.7)]
        rng = np.random.default_rng(42)
        centers = rng.normal(size=(2, 3))
        covariances = rng.normal(size=(2, 3, 3))
        opacities = rng.normal(size=2)
        def scalar(items):
            total = 0.0
            for index, p in enumerate(items):
                cov = p.covariance() if callable(p.covariance) else p.covariance
                alpha = np.clip(getattr(p, "opacity", getattr(p, "density", 1.0))
                                * p.confidence, 0.0, 1.0)
                total += (np.dot(centers[index], p.center)
                          + np.sum(covariances[index] * cov) + opacities[index] * alpha)
            return total
        gradients = pullback_render_gradients(parts, centers, covariances, opacities)
        for ref, gradient in gradients.items():
            plus, minus = deepcopy(parts), deepcopy(parts)
            apply_parameter_increment(plus, ref, 1e-6, ParameterBounds())
            apply_parameter_increment(minus, ref, -1e-6, ParameterBounds())
            self.assertAlmostEqual(gradient, (scalar(plus) - scalar(minus)) / 2e-6, places=7)

    def test_opacity_jacobian_keeps_feasible_inward_boundary_direction(self):
        part = AnisotropicGaussianPrimitive(opacity=1.0)
        self.assertEqual(parameter_jacobian(part, "opacity").opacity, 1.0)
        part.confidence = 2.0
        self.assertEqual(parameter_jacobian(part, "opacity").opacity, 0.0)

    def test_gradient_matches_dimensionless_translation_and_size_steps(self):
        part = EllipsoidPrimitive(center=(0.5, 0., 0.), radii=(2., 2., 2.))
        def objective(primitives):
            p = primitives[0]
            return ResFitObjectiveResult(total=float(p.center[0] ** 2 + p.radii[0] ** 2), terms={})
        gradient = finite_difference_gradient([part], objective, epsilon=1e-6)
        self.assertAlmostEqual(gradient[(0, "center", 0)], 2.0, places=4)
        self.assertAlmostEqual(gradient[(0, "radii", 0)], 8.0, places=4)
        np.testing.assert_array_equal(part.center, (0.5, 0., 0.))
        np.testing.assert_array_equal(part.radii, (2., 2., 2.))


if __name__ == "__main__":
    unittest.main()
