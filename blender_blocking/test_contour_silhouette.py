"""Subpixel contour continuity and bounded nonellipse family pullbacks."""
from copy import deepcopy
import sys
import unittest
from unittest.mock import patch

import numpy as np

from primitives.analytic_primitives import EllipsoidPrimitive, SuperquadricPrimitive
from primitives.superfrustum import SuperFrustum
from primitives.polygon_extrusion import PolygonExtrusionPrimitive
from primitives.contour_silhouette import contour_footprint, projection_resolution
from primitives.shape_aware_silhouette import ShapeAwareSilhouetteCache
from primitives.soft_silhouette import OrthographicCamera, render_projected_soft_silhouette
from placement.resfit_optimizer import ParameterBounds, OptimizationBudget
from placement.resfit_parameters import apply_parameter_increment, discover_primitive_parameters
from reconstruction.differentiable.config import _normalize_differentiable_config
from reconstruction.differentiable.contracts import CameraSpec, ReconstructionTarget, RenderableScene
from reconstruction.differentiable.cpu_soft import CpuSoftSilhouetteBackend
from reconstruction.differentiable.optimization import run_differentiable_optimization, _optimizer_objective_total
from reconstruction.differentiable.target_adapter import renderable_from_primitive

try:
    import shapely
    HAS_SHAPELY = hasattr(shapely, 'contains_xy')
except ImportError:
    HAS_SHAPELY = False


@unittest.skipUnless(HAS_SHAPELY, 'optional Shapely 2.x contour operator unavailable')
class ContourSilhouetteTests(unittest.TestCase):
    def frustum(self):
        return SuperFrustum(position=(.08, -.02, .07), orientation=(.3, .25),
                            radius_bottom=.4, radius_top=.2, height=.9)

    def cameras(self):
        return [OrthographicCamera.from_view(view, image_size=(16, 16))
                for view in ('front', 'top')]

    def test_subpixel_motion_changes_continuous_contour_without_integer_raster(self):
        part = self.frustum()
        camera = self.cameras()[0]
        before = contour_footprint(part, camera)
        shifted = deepcopy(part)
        shifted.position[0] += 1e-4
        after = contour_footprint(shifted, camera)
        self.assertGreater(float(np.max(np.abs(after-before))), 1e-6)
        self.assertLess(float(np.max(np.abs(after-before))), .01)

    def test_frustum_chord_policy_declares_real_bound_and_cap(self):
        part = self.frustum()
        camera = OrthographicCamera.from_view('top', image_size=(512, 512))
        count, report = projection_resolution(part, camera)
        self.assertGreater(count, 16)
        self.assertLessEqual(report['geometric_error_bound_px'], .25)
        part.radius_bottom = 1000
        count, report = projection_resolution(part, camera)
        self.assertEqual(count, 128)
        self.assertFalse(report['within_requested_chord_error'])
        _, generic = projection_resolution(SuperquadricPrimitive(epsilon1=.2), camera)
        self.assertIsNone(generic['geometric_error_bound_px'])
        invalid = self.frustum()
        invalid.height = 0
        with self.assertRaisesRegex(ValueError, 'positive height'):
            contour_footprint(invalid, self.cameras()[0], resolution=16)

    def test_perforated_and_concave_union_does_not_fill_real_holes(self):
        outer = [[-.8, -.8], [.8, -.8], [.8, .8], [-.8, .8]]
        hole = [[-.3, -.3], [-.3, .3], [.3, .3], [.3, -.3]]
        part = PolygonExtrusionPrimitive(outer, (hole,), height=.2)
        camera = OrthographicCamera.from_view('top', image_size=(25, 25))
        mask, report = contour_footprint(part, camera, return_metadata=True)
        self.assertEqual(report['loops'], 2)
        self.assertLess(mask[12, 12], .01)
        self.assertGreater(mask[12, 17], .9)
        self.assertEqual(report['union_precision_grid_px'], 1e-9)
        self.assertFalse(report['final_opaque_admission'])

    def test_long_thin_holes_survive_declared_microhole_precision(self):
        outer = [[-.8, -.8], [.8, -.8], [.8, .8], [-.8, .8]]
        hole = [[-.6, -1e-5], [-.6, 1e-5], [.6, 1e-5], [.6, -1e-5]]
        part = PolygonExtrusionPrimitive(outer, (hole,), height=.2)
        camera = OrthographicCamera.from_view('top', image_size=(25, 25))
        _, report = contour_footprint(part, camera, return_metadata=True)
        self.assertEqual(report['loops'], 2)
        self.assertEqual(report['unresolved_small_holes'], 0)
        np.testing.assert_array_equal(part.outer, np.asarray(outer))

    def test_mixed_forward_matches_direct_and_reuses_only_unchanged_parts(self):
        parts = [self.frustum(), EllipsoidPrimitive(center=(.3, 0, 0), radii=(.2, .3, .4))]
        cache = ShapeAwareSilhouetteCache(self.cameras())
        first = cache.render(iter(parts))
        cache.render(deepcopy(parts))
        self.assertEqual(cache.recomputed_components, 2)
        parts[0].position[0] += .02
        masks = cache.render(parts)
        self.assertEqual(cache.recomputed_components, 4)
        for camera in self.cameras():
            np.testing.assert_allclose(masks[camera.name],
                                      render_projected_soft_silhouette(parts, camera), atol=1e-15)
        self.assertFalse(np.array_equal(first['front'], masks['front']))

    def test_frustum_and_ellipse_pullbacks_match_independent_local_differences(self):
        parts = [self.frustum(), EllipsoidPrimitive(center=(.3, 0, 0), radii=(.2, .3, .4))]
        cache = ShapeAwareSilhouetteCache(self.cameras())
        cache.render(parts)
        rng = np.random.default_rng(56)
        gradients = {c.name: rng.normal(size=c.image_size[::-1]) for c in self.cameras()}
        actual = cache.parameter_gradients(gradients, ParameterBounds(), maximum_controls=32)
        def scalar(items):
            return sum(np.sum(render_projected_soft_silhouette(items, c)*gradients[c.name])
                       for c in self.cameras())
        for ref, derivative in actual.items():
            plus, minus = deepcopy(parts), deepcopy(parts)
            apply_parameter_increment(plus, ref, 1e-5, ParameterBounds())
            apply_parameter_increment(minus, ref, -1e-5, ParameterBounds())
            numerical = (scalar(plus)-scalar(minus))/2e-5
            np.testing.assert_allclose(derivative, numerical, atol=2e-4, rtol=3e-4,
                                       err_msg=str(ref))
        self.assertEqual(set(actual), set(discover_primitive_parameters(parts)))

    def test_superquadric_exponents_have_family_gradients_not_ellipse_zeroes(self):
        part = SuperquadricPrimitive(center=(.07, 0, -.04), radii=(.4, .6, .5),
                                     epsilon1=.7, epsilon2=1.3)
        from scipy.spatial.transform import Rotation
        part.rotation = Rotation.from_euler('xyz', [.2, -.3, .4]).as_matrix()
        cache = ShapeAwareSilhouetteCache(self.cameras()[:1])
        cache.render([part])
        yy, xx = np.indices((16, 16))
        gradients = {'front': (xx+2*yy)/48}
        actual = cache.parameter_gradients(gradients, ParameterBounds(), maximum_controls=32)
        self.assertGreater(abs(actual[(0, 'epsilon1', None)]), .01)
        self.assertGreater(abs(actual[(0, 'epsilon2', None)]), .01)
        self.assertTrue(all(np.isfinite(list(actual.values()))))
        self.assertEqual(set(actual), set(discover_primitive_parameters([part])))

    def test_nonconvex_superquadric_is_not_replaced_by_a_convex_hull(self):
        part = SuperquadricPrimitive(radii=(.4, .4, .4), epsilon1=3., epsilon2=3.)
        camera = OrthographicCamera.from_view('front', image_size=(101, 101))
        mask, report = contour_footprint(part, camera, return_metadata=True)
        self.assertEqual(report['projection_representation'], 'mesh_triangle_union')
        self.assertLess(mask[44, 56], .5)
        self.assertGreater(mask[50, 50], .5)

    def test_axis_aligned_convex_superquadric_front_is_independent_of_hidden_exponent(self):
        part = SuperquadricPrimitive(center=(.07, 0, -.04), radii=(.4, .6, .5),
                                     epsilon1=.7, epsilon2=1.3)
        camera = self.cameras()[0]
        baseline, report = contour_footprint(part, camera, return_metadata=True)
        self.assertEqual(report['projection_representation'], 'convex_primitive_vertex_hull')
        for exponent in (.6, 1.29987, 1.30013, 1.8):
            changed = deepcopy(part)
            changed.epsilon2 = exponent
            np.testing.assert_allclose(contour_footprint(changed, camera), baseline, atol=1e-12)

    def test_ellipse_only_covariance_interface_rejects_nonellipse_contract(self):
        cache = ShapeAwareSilhouetteCache(self.cameras())
        cache.render([self.frustum()])
        with self.assertRaisesRegex(ValueError, 'family parameter pullbacks'):
            cache.backward({c.name: np.ones(c.image_size[::-1]) for c in self.cameras()})

    def test_derivative_pairs_share_budget_and_reserve_actual_proposal_score(self):
        cache = ShapeAwareSilhouetteCache(self.cameras()[:1])
        cache.render([self.frustum()])
        gradients = {'front': np.ones((16, 16))}
        budget = OptimizationBudget(max_objective_evaluations=7)
        result = cache.parameter_gradients(gradients, ParameterBounds(), budget=budget)
        self.assertEqual(len(result), 3)
        self.assertEqual(budget.objective_evaluations, 6)
        self.assertEqual(cache.derivative_evaluations, 6)
        self.assertEqual(budget.remaining_evaluations(), 1)
        none = cache.parameter_gradients(gradients, ParameterBounds(), budget=budget)
        self.assertEqual(none, {})
        self.assertEqual(budget.objective_evaluations, 6)

    def test_rotating_prefix_visits_all_controls_and_failures_are_reported(self):
        cache = ShapeAwareSilhouetteCache(self.cameras()[:1])
        cache.render([self.frustum()])
        gradients = {'front': np.ones((16, 16))}
        visited = set()
        for _ in range(4):
            visited.update(cache.parameter_gradients(gradients, ParameterBounds(), maximum_controls=2))
        self.assertEqual(visited, set(discover_primitive_parameters(cache.parts)))
        with patch('primitives.shape_aware_silhouette.contour_footprint', side_effect=ValueError('invalid contour')):
            self.assertEqual(cache.parameter_gradients(gradients, ParameterBounds(), maximum_controls=1), {})
        self.assertTrue(cache.derivative_failures)

    def test_missing_optional_geometry_dependency_does_not_change_ellipse_path(self):
        camera = self.cameras()[0]
        with patch.dict(sys.modules, {'shapely': None}):
            with self.assertRaisesRegex(RuntimeError, 'Shapely 2.x'):
                contour_footprint(self.frustum(), camera)
            mask = render_projected_soft_silhouette([EllipsoidPrimitive()], camera)
            self.assertTrue(np.isfinite(mask).all())

    def test_live_nonellipse_optimizer_keeps_scored_winner_and_shared_allowance(self):
        camera = CameraSpec(image_size=(16, 16))
        seed = self.frustum()
        target_part = deepcopy(seed)
        target_part.position[0] -= .15
        target = ReconstructionTarget(silhouettes={'front':
            render_projected_soft_silhouette([target_part], camera.to_orthographic_camera()) >= .5})
        renderer = CpuSoftSilhouetteBackend()
        batch = renderer.render(RenderableScene(primitives=(renderable_from_primitive(seed),)), (camera,))
        parsed, errors, _ = _normalize_differentiable_config({
            'optimization_steps': 2, 'max_objective_evaluations': 32, 'max_runtime_s': 8})
        self.assertFalse(errors)
        self.assertTrue(batch.metadata['contour_approximation_reports'])
        initial = renderer.loss(batch, target, parsed['loss_weights'])
        result = run_differentiable_optimization(
            backend_choice='cpu_soft_silhouette', parsed_config=parsed, primitives=(seed,),
            renderer=renderer, cameras=(camera,), target_record=target, target_view_weights={},
            initial_render_batch=batch, initial_loss=initial, request_timeout_s=None)
        summary = result['optimization_summary']
        self.assertEqual(summary['method'], 'mixed_family_contour_silhouette')
        self.assertGreater(summary['contour_derivative_evaluations'], 0)
        self.assertFalse(summary['contour_derivative_failures'])
        self.assertLessEqual(summary['objective_evaluations'], 32)
        self.assertLessEqual(_optimizer_objective_total(result['loss'], parsed),
                             _optimizer_objective_total(initial, parsed))
        np.testing.assert_array_equal(seed.position, [.08, -.02, .07])


if __name__ == '__main__':
    unittest.main()
