"""Bounded numerical and admission checks for the live analytic silhouette path."""
from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch
import unittest

import numpy as np

from placement.resfit_optimizer import (ParameterBounds, CoordinateDescentConfig,
                                        coordinate_descent_optimize)
from placement.resfit_objective import ResFitObjectiveResult
from placement.resfit_parameters import apply_parameter_increment, pullback_render_gradients
from primitives.analytic_primitives import AnisotropicGaussianPrimitive, EllipsoidPrimitive
from primitives.soft_silhouette import (OrthographicCamera, ProjectedSilhouetteCache,
                                        render_projected_soft_silhouette)
from reconstruction.differentiable.config import _normalize_differentiable_config
from reconstruction.differentiable.contracts import (CameraSpec, LossResult, LossWeights,
                                                     ReconstructionTarget, RenderableScene)
from reconstruction.differentiable.cpu_soft import CpuSoftSilhouetteBackend
from reconstruction.differentiable.optimization import (run_differentiable_optimization,
                                                        _optimizer_objective_total)
from reconstruction.differentiable.soft_objective import (SoftMaskTarget, SoftSilhouetteObjective,
                                                         soft_mask_loss_and_gradient)
from reconstruction.differentiable.target_adapter import renderable_from_primitive


class TestAnalyticSilhouetteRefinement(unittest.TestCase):
    def test_soft_objective_pixel_derivative(self):
        rng = np.random.default_rng(53)
        target = SoftMaskTarget.from_mask(rng.uniform(size=(5, 7)))
        predicted = rng.uniform(0.05, 0.95, size=(5, 7))
        _, gradients, _ = soft_mask_loss_and_gradient(predicted, target,
            l2_weight=0.7, iou_weight=1.3, distance_weight=0.4)
        for index in np.ndindex(predicted.shape):
            plus, minus = predicted.copy(), predicted.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            scalar = lambda values: soft_mask_loss_and_gradient(values, target,
                l2_weight=0.7, iou_weight=1.3, distance_weight=0.4)[0]
            self.assertAlmostEqual(gradients[index], (scalar(plus) - scalar(minus)) / 2e-6,
                                   places=8)

    def test_analytic_local_parameter_gradients(self):
        factor = np.array(((0.6, 0., 0.), (0.17, 0.8, 0.), (-0.11, 0.07, 0.5)))
        parts = [EllipsoidPrimitive(center=(0.1, -0.2, 0.05), radii=(0.4, 0.6, 0.8)),
                 AnisotropicGaussianPrimitive(center=(-0.3, 0.1, 0.2), covariance=factor @ factor.T,
                                               opacity=0.7, confidence=0.8)]
        for axis, angle in ((0, 0.2), (1, -0.3), (2, 0.4)):
            apply_parameter_increment(parts, (0, 'rotation', axis), angle, ParameterBounds())
        cameras = [OrthographicCamera.from_view(view, image_size=(24, 24))
                   for view in ('front', 'side', 'top')]
        cache = ProjectedSilhouetteCache(cameras, softness=12.0)
        cache.render(parts)
        rng = np.random.default_rng(29)
        mask_gradients = {camera.name: rng.normal(size=(24, 24)) for camera in cameras}
        gradients = pullback_render_gradients(parts, *cache.backward(mask_gradients))
        def scalar(items):
            return sum(np.sum(mask * mask_gradients[name]) for name, mask in cache.render(items).items())
        for ref, gradient in gradients.items():
            plus, minus = deepcopy(parts), deepcopy(parts)
            apply_parameter_increment(plus, ref, 1e-6, ParameterBounds())
            apply_parameter_increment(minus, ref, -1e-6, ParameterBounds())
            numerical = (scalar(plus) - scalar(minus)) / 2e-6
            np.testing.assert_allclose(gradient, numerical, atol=2e-7, rtol=2e-6,
                                       err_msg=str(ref))

    def test_spectral_floor_gradient_and_opaque_overlap(self):
        camera = OrthographicCamera.from_view('top', image_size=(9, 9))
        parts = [AnisotropicGaussianPrimitive(covariance=np.diag((0.02, 0.3, 0.5)), opacity=1.0),
                 AnisotropicGaussianPrimitive(covariance=np.diag((0.1, 0.1, 0.1)), opacity=1.0)]
        cache = ProjectedSilhouetteCache([camera], softness=60.0, min_variance=0.1)
        mask = cache.render(parts)['top']
        self.assertEqual(mask[4, 4], 1.0)
        gradient = np.arange(81, dtype=float).reshape(9, 9) / 81
        centers, covariances, opacities = cache.backward({'top': gradient})
        self.assertTrue(all(np.isfinite(values).all() for values in (centers, covariances, opacities)))
        # The first principal variance is below its floor and therefore has zero derivative.
        self.assertAlmostEqual(covariances[0, 0, 0], 0.0)
        for axis in (0, 1):
            plus, minus = deepcopy(parts), deepcopy(parts)
            plus[0].covariance[axis, axis] += 1e-6
            minus[0].covariance[axis, axis] -= 1e-6
            numerical = (np.sum(cache.render(plus)['top'] * gradient)
                         - np.sum(cache.render(minus)['top'] * gradient)) / 2e-6
            self.assertAlmostEqual(covariances[0, axis, axis], numerical, places=7)

    def test_changed_part_reuse_and_forward_agreement(self):
        parts = [EllipsoidPrimitive(center=(-0.3, 0., 0.), radii=(0.4, 0.6, 0.5)),
                 EllipsoidPrimitive(center=(0.4, 0., 0.), radii=(0.6, 0.4, 0.7))]
        cameras = [OrthographicCamera.from_view('front', image_size=(20, 20)),
                   OrthographicCamera.from_view('top', image_size=(20, 20))]
        cache = ProjectedSilhouetteCache(cameras)
        cache.render(parts)
        self.assertEqual(cache.recomputed_components, 4)
        cache.render(deepcopy(parts))
        self.assertEqual(cache.recomputed_components, 4)
        parts[1].center[0] += 0.1
        masks = cache.render(parts)
        self.assertEqual(cache.recomputed_components, 6)
        self.assertEqual(cache.reused_components, 6)
        for camera in cameras:
            np.testing.assert_allclose(masks[camera.name],
                render_projected_soft_silhouette(parts, camera), atol=1e-15, rtol=0.0)

    def test_view_weights_follow_actual_loss_rules(self):
        masks = {'front': np.zeros((4, 4)), 'top': np.ones((4, 4))}
        predictions = {name: np.full((4, 4), 0.3) for name in masks}
        objective = SoftSilhouetteObjective(masks, LossWeights(), {'front': 4.0, 'top': -1.0})
        self.assertEqual(objective.view_weights, {'front': 1.0, 'top': 0.0})
        _, gradients, _ = objective.evaluate(predictions)
        np.testing.assert_array_equal(gradients['top'], np.zeros((4, 4)))
        fallback = SoftSilhouetteObjective(masks, LossWeights(), {'front': 0.0, 'top': 0.0})
        self.assertEqual(fallback.view_weights, {'front': 0.5, 'top': 0.5})

    def test_paired_footprints_reuse_immutable_base(self):
        base = [EllipsoidPrimitive(center=(-0.2, 0., 0.), radii=(0.4, 0.6, 0.5)),
                AnisotropicGaussianPrimitive(center=(0.3, 0., 0.),
                    covariance=np.diag((0.3, 0.2, 0.4)), opacity=0.8)]
        cameras = [OrthographicCamera.from_view(view, image_size=(20, 20))
                   for view in ('front', 'top')]
        cache = ProjectedSilhouetteCache(cameras)
        initial_masks = cache.render(base)
        initial_gradients = cache.backward({camera.name: np.ones((20, 20)) for camera in cameras})
        generation = cache.generation
        proposals = [deepcopy(base), deepcopy(base)]
        for proposal, direction in zip(proposals, (1.0, -1.0)):
            apply_parameter_increment(proposal, (1, 'covariance_cholesky', 1),
                                      direction * 0.1, ParameterBounds())
        recomputed = cache.recomputed_components
        outputs = cache.render_coordinate_batch(base, proposals)
        self.assertEqual(cache.recomputed_components - recomputed, 4)
        self.assertEqual(cache.generation, generation)
        for proposal, masks in zip(proposals, outputs):
            for camera in cameras:
                np.testing.assert_allclose(masks[camera.name],
                    render_projected_soft_silhouette(proposal, camera), atol=1e-14, rtol=0.0)
        for expected, actual in zip(initial_gradients,
                cache.backward({camera.name: np.ones((20, 20)) for camera in cameras})):
            np.testing.assert_array_equal(actual, expected)
        for name, mask in cache.render(base).items():
            np.testing.assert_array_equal(mask, initial_masks[name])
        np.testing.assert_array_equal(base[1].covariance, np.diag((0.3, 0.2, 0.4)))
        with self.assertRaises(ValueError):
            cache.render_coordinate_batch(base, [base, base, base])

    def test_coordinate_batch_admits_one_winner_before_next_base(self):
        class BatchObjective:
            def __init__(self):
                self.pairs = []
            def __call__(self, parts):
                return ResFitObjectiveResult(float(np.sum((parts[0].center - (0.1, 0.1, 0.)) ** 2)), {})
            def evaluate_batch(self, candidates, *, budget):
                self.pairs.append([parts[0].center.copy() for parts in candidates])
                for candidate in candidates:
                    if budget.reason() is not None:
                        return
                    yield budget.evaluate(self, candidate)
        objective = BatchObjective()
        part = EllipsoidPrimitive()
        result = coordinate_descent_optimize([part], objective,
            CoordinateDescentConfig(iterations=1, initial_step=0.1, max_objective_evaluations=5))
        np.testing.assert_allclose(result.primitives[0].center, (0.1, 0.1, 0.))
        np.testing.assert_array_equal(objective.pairs[0], ((0.1, 0., 0.), (-0.1, 0., 0.)))
        np.testing.assert_array_equal(objective.pairs[1], ((0.1, 0.1, 0.), (0.1, -0.1, 0.)))
        self.assertEqual(result.objective_evaluations, 5)
        self.assertAlmostEqual(result.best_loss, objective(result.primitives).total)
        np.testing.assert_array_equal(part.center, (0., 0., 0.))

    def test_coordinate_batch_retains_first_winner_on_count_and_deadline(self):
        class PartialObjective:
            def __init__(self, expire):
                self.expire = expire
                self.scored = 0
            def __call__(self, parts):
                self.scored += 1
                return ResFitObjectiveResult(float((parts[0].center[0] - 0.1) ** 2), {})
            def evaluate_batch(self, candidates, *, budget):
                for candidate in candidates:
                    if budget.reason() is not None:
                        return
                    yield budget.evaluate(self, candidate)
                    if self.expire:
                        # Deterministically expire after the completed first score.
                        budget.started_at -= 11.0
        for expire, limit, expected in ((False, 2, 'objective_evaluation_budget'),
                                         (True, 10, 'elapsed_time_budget')):
            objective = PartialObjective(expire)
            result = coordinate_descent_optimize([EllipsoidPrimitive()], objective,
                CoordinateDescentConfig(iterations=1, initial_step=0.1,
                                        max_objective_evaluations=limit, max_elapsed_s=10.0))
            self.assertEqual(result.termination_reason, expected)
            self.assertEqual(result.objective_evaluations, 2)
            self.assertEqual(objective.scored, 2)
            self.assertAlmostEqual(result.primitives[0].center[0], 0.1)
            self.assertLess(result.best_loss, result.initial_result.total)

    def test_live_nonanalytic_fallback_uses_paired_batches(self):
        fixture = self._live_fixture(optimization_steps=1, loss_weights={
            'silhouette_l2': 0.0, 'soft_iou': 0.0, 'signed_distance': 0.0})
        _, _, _, _, _, parsed, initial_loss = fixture
        result = self._run_fixture(fixture)
        summary = result['optimization_summary']
        self.assertEqual(summary['analytic_gradient_steps'], 0)
        self.assertGreater(summary['coordinate_fallbacks'], 0)
        self.assertGreater(summary['paired_coordinate_batches'], 0)
        self.assertLess(_optimizer_objective_total(result['loss'], parsed),
                        _optimizer_objective_total(initial_loss, parsed))
        self.assertLessEqual(summary['objective_evaluations'], 50)

    def _live_fixture(self, **config):
        cameras = (CameraSpec(image_size=(32, 32)),)
        seed = EllipsoidPrimitive(center=(0.35, 0., 0.), radii=(0.45, 0.6, 0.7))
        target_part = deepcopy(seed)
        target_part.center[0] = 0.0
        target = ReconstructionTarget(silhouettes={'front':
            render_projected_soft_silhouette([target_part], cameras[0].to_orthographic_camera()) >= 0.5})
        renderer = CpuSoftSilhouetteBackend()
        batch = renderer.render(RenderableScene(primitives=(renderable_from_primitive(seed),)), cameras)
        parsed, errors, _ = _normalize_differentiable_config({
            'optimization_steps': 4, 'optimization_initial_step': 0.1,
            'max_objective_evaluations': 50, 'max_runtime_s': 10.0, **config})
        self.assertEqual(errors, ())
        loss = renderer.loss(batch, target, parsed['loss_weights'])
        return seed, cameras, target, renderer, batch, parsed, loss

    def _run_fixture(self, fixture):
        seed, cameras, target, renderer, batch, parsed, loss = fixture
        return run_differentiable_optimization(backend_choice='cpu_soft_silhouette',
            parsed_config=parsed, primitives=(seed,), renderer=renderer, cameras=cameras,
            target_record=target, target_view_weights={}, initial_render_batch=batch,
            initial_loss=loss, request_timeout_s=None)

    def test_live_refinement_improves_actual_score_without_mesh_proxies(self):
        fixture = self._live_fixture()
        seed, _, _, _, _, parsed, initial_loss = fixture
        with patch.object(EllipsoidPrimitive, 'to_mesh_data', side_effect=AssertionError('mesh hot path')):
            result = self._run_fixture(fixture)
        self.assertLess(_optimizer_objective_total(result['loss'], parsed),
                        _optimizer_objective_total(initial_loss, parsed))
        self.assertEqual(result['optimization_summary']['method'], 'analytic_projected_silhouette')
        self.assertGreater(result['optimization_summary']['analytic_gradient_steps'], 0)
        self.assertLessEqual(result['optimization_summary']['objective_evaluations'], 50)
        np.testing.assert_allclose(seed.center, (0.35, 0., 0.))

    def test_soft_improvement_cannot_override_geometric_rejection(self):
        fixture = self._live_fixture(max_objective_evaluations=8)
        seed, _, _, renderer, _, _, loss = fixture
        with patch.object(renderer, 'loss', return_value=LossResult(
            total=100.0, terms={'boundary_iou_loss': 1.0, 'signed_distance_loss': 1.0,
                              'area_iou': 1.0, 'soft_iou': 1.0}, per_view={})):
            result = self._run_fixture(fixture)
        self.assertIs(result['loss'], loss)
        np.testing.assert_array_equal(result['optimized_primitives'][0].center, seed.center)
        self.assertEqual(result['optimization_summary']['accepted_moves'], 0)

    def test_failed_trial_and_exhausted_budget_keep_initial_winner(self):
        fixture = self._live_fixture(max_objective_evaluations=4)
        seed, _, _, renderer, _, _, loss = fixture
        with patch.object(renderer, 'loss', side_effect=ValueError('invalid trial')):
            result = self._run_fixture(fixture)
        self.assertIs(result['loss'], loss)
        self.assertTrue(result['optimization_summary']['failed_proposals'])
        self.assertEqual(result['optimization_summary']['objective_evaluations'], 4)
        np.testing.assert_array_equal(result['optimized_primitives'][0].center, seed.center)
        budget_one = self._run_fixture(self._live_fixture(max_objective_evaluations=1))
        self.assertEqual(budget_one['optimization_summary']['objective_evaluations'], 1)
        self.assertEqual(budget_one['optimization_summary']['reason'], 'objective_evaluation_budget')

    def test_canonical_gaussian_import_in_live_cpu_adapter(self):
        # Same numeric Gaussian class, imported under the worker's package namespace.
        from blender_blocking.primitives.analytic_primitives import AnisotropicGaussianPrimitive as CanonicalGaussian
        part = CanonicalGaussian(covariance=np.diag((0.2, 0.3, 0.4)), opacity=0.7)
        camera = CameraSpec(image_size=(16, 16))
        cache = ProjectedSilhouetteCache([camera.to_orthographic_camera()])
        direct = cache.render([part])['front']
        renderer = CpuSoftSilhouetteBackend()
        batch = renderer.render(RenderableScene(primitives=(renderable_from_primitive(part),)), [camera])
        np.testing.assert_allclose(direct, batch.silhouettes['front'], atol=1e-15, rtol=0.0)
        fixture = self._live_fixture(max_objective_evaluations=2)
        _, cameras, target, renderer, _, parsed, _ = fixture
        batch = renderer.render(RenderableScene(primitives=(renderable_from_primitive(part),)), cameras)
        loss = renderer.loss(batch, target, parsed['loss_weights'])
        result = run_differentiable_optimization(backend_choice='cpu_soft_silhouette',
            parsed_config=parsed, primitives=(part,), renderer=renderer, cameras=cameras,
            target_record=target, target_view_weights={}, initial_render_batch=batch,
            initial_loss=loss, request_timeout_s=None)
        self.assertTrue(result['optimization_summary']['enabled'])
        np.linalg.cholesky(result['optimized_primitives'][0].covariance)


if __name__ == '__main__':
    unittest.main()
