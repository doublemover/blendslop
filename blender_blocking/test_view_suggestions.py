"""Read-only next-view ranking and explicit generalized planning calibration."""
from dataclasses import replace
import itertools
import unittest
from unittest.mock import patch

import numpy as np

from reconstruction.evidence_identity import target_evidence_hash
from reconstruction.grouped_solids import concatenate
from reconstruction.native_geometry import GeometryArrays
from reconstruction.planning_camera import PlanningOrthographicCamera
from reconstruction.projected_metrics import projected_mesh_masks
from reconstruction.types import CandidateMetrics, CandidateResult
from reconstruction.view_suggestions import (
    CaptureDirection, filtered_candidate_projection, suggest_additional_view,
)
from test_quality_geometry import target_for_masks

try:
    import shapely
    HAS_SHAPELY = hasattr(shapely, 'union_all')
except ImportError:
    HAS_SHAPELY = False


def cube(center, radius=.18):
    points = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                       [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    faces = [[0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
             [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return GeometryArrays.capture(points*radius+center, faces)


def parity_hypotheses():
    positive, negative = [], []
    for signs in itertools.product((-1, 1), repeat=3):
        (positive if np.prod(signs) > 0 else negative).append(cube(np.array(signs)*.45))
    return concatenate(positive), concatenate(negative)


def planning_fixture():
    first, second = parity_hypotheses()
    target = target_for_masks({view: np.zeros((32, 32), bool) for view in ('front', 'side', 'top')})
    masks = projected_mesh_masks(target, first.vertices, first.faces)
    second_masks = projected_mesh_masks(target, second.vertices, second.faces)
    for view in masks:
        np.testing.assert_array_equal(masks[view], second_masks[view])
    target = target_for_masks(masks)
    results = {}
    for name, data in [('first', first), ('second', second)]:
        # Actual canonical masks above are identical. Eligibility is a synthetic
        # source-contract fixture, not a native camera reconstruction campaign.
        metrics = CandidateMetrics(per_view={view: {'area_iou': 1., 'passed': True,
                                                     'required': True} for view in masks},
                                   extras={'input_evidence_hash': target_evidence_hash(target)})
        results[name] = CandidateResult(candidate_id=name, backend_name='numeric_fixture',
                                        status='success', metric_result=metrics, geometry=data)
    return target, results


class PlanningCameraTests(unittest.TestCase):
    def test_arbitrary_frame_pixel_depth_roundtrip_and_proper_poles(self):
        rng = np.random.default_rng(68)
        for azimuth, elevation in [(0, 0), (35, 27), (-70, -18), (0, 90), (0, -90)]:
            camera = PlanningOrthographicCamera.from_direction(
                'frame', azimuth_deg=azimuth, elevation_deg=elevation, origin=(10, -2, .3),
                viewport=(-.7, 2.1, -1.2, 3.8), resolution=(9, 13))
            points = rng.normal(size=(31, 3))+camera.origin
            pixels, depth = camera.project_world(points)
            np.testing.assert_allclose(camera.backproject(pixels, depth), points, atol=2e-14)
            self.assertAlmostEqual(np.linalg.det(camera.frame), 1., places=13)
            origins, directions = camera.pixel_rays(depth=3.)
            self.assertEqual(origins.shape, (117, 3))
            np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-14)
            np.testing.assert_allclose(camera.project_world(origins)[1], 3., atol=2e-14)

    def test_front_cell_centers_equal_existing_canonical_projection(self):
        from reconstruction.projection_contract import project_vertices
        target = target_for_masks({'front': np.ones((2, 3), bool)})
        camera = PlanningOrthographicCamera.from_direction(
            'front', azimuth_deg=0, elevation_deg=0, origin=(0, 0, 0),
            viewport=(-1, 1, -1, 1), resolution=(3, 2))
        points = np.array([[-2/3, 0, .5], [0, .1, -.5], [2/3, -.7, .5]])
        np.testing.assert_allclose(camera.project_world(points)[0],
                                   project_vertices(target, target.constraints[0], points), atol=1e-15)
        np.testing.assert_allclose(camera.project_world(points)[0], [[0, 0], [1, 1], [2, 0]], atol=1e-15)

    def test_planning_camera_does_not_silently_admit_oblique_training_input(self):
        from reconstruction.projection_contract import validate_view_calibration
        camera = PlanningOrthographicCamera.from_direction(
            'oblique', azimuth_deg=45, elevation_deg=35, origin=(0, 0, 0))
        with self.assertRaisesRegex(ValueError, 'unsupported calibrated view'):
            validate_view_calibration({'oblique': camera.to_dict()})
        self.assertIn('no real observation added', camera.to_dict()['scope'])

    def test_invalid_frame_and_resolution_are_rejected(self):
        for right in ((-1, 0, 0), (2, 0, 0), (np.nan, 0, 0)):
            with self.assertRaises(ValueError):
                PlanningOrthographicCamera('bad', (0, 0, 0), right, (0, 1, 0), (0, 0, 1), (-1, 1, -1, 1))
        with self.assertRaises(ValueError):
            PlanningOrthographicCamera.from_direction('bad', azimuth_deg=0, elevation_deg=91, origin=(0, 0, 0))
        with self.assertRaises(ValueError):
            PlanningOrthographicCamera.from_direction('bad', azimuth_deg=0, elevation_deg=0, origin=(0, 0, 0), resolution=(3.5, 4))


@unittest.skipUnless(HAS_SHAPELY, 'optional Shapely 2.x planning projection unavailable')
class ViewSuggestionTests(unittest.TestCase):
    def directions(self):
        return [CaptureDirection('oblique', 45, 35), CaptureDirection('other', -45, 20)]

    def test_equal_three_canonical_masks_still_yield_informative_oblique_request(self):
        target, results = planning_fixture()
        identity = target_evidence_hash(target)
        hashes = [result.geometry.content_hash for result in results.values()]
        report = suggest_additional_view(target, results, capture_available=True, directions=self.directions(), resolution=(16, 16))
        self.assertEqual(report['status'], 'capture_suggested')
        self.assertGreater(report['request']['ranking_score'], .08)
        self.assertIsNotNone(report['request']['disagreement_region_px'])
        self.assertTrue(report['planning_only'])
        self.assertEqual(report['observations_added'], 0)
        self.assertFalse(report['automatic_selection_changed'])
        self.assertFalse(report['request']['oblique_input_supported_by_current_fitters'])
        self.assertEqual(target_evidence_hash(target), identity)
        self.assertEqual([result.geometry.content_hash for result in results.values()], hashes)

    def test_unavailable_real_capture_only_returns_planning_diagnostics(self):
        target, results = planning_fixture()
        report = suggest_additional_view(target, results, directions=self.directions(), resolution=(16, 16))
        self.assertEqual(report['status'], 'capture_unavailable_diagnostic_only')
        self.assertIsNone(report['request'])
        self.assertTrue(report['ranked_directions'])
        self.assertFalse(report['real_capture_available'])

    def test_pixel_area_operator_keeps_subpixel_geometry_without_confidence(self):
        data = cube([0, 0, 0], radius=.01)
        camera = PlanningOrthographicCamera.from_direction(
            'tiny', azimuth_deg=0, elevation_deg=0, origin=(0, 0, 0), resolution=(16, 16))
        mask = filtered_candidate_projection(data, camera)
        # Projected width = .02/2*16 = .16 pixels; total cell area .0256.
        self.assertAlmostEqual(float(mask.sum()), .16**2, places=12)
        self.assertGreater(float(mask.max()), 0.)
        self.assertLess(float(mask.max()), .1)

    def test_observed_opposite_and_reserved_cameras_are_excluded(self):
        target, results = planning_fixture()
        directions = [CaptureDirection('front', 0, 0), CaptureDirection('back', 180, 0),
                      CaptureDirection('side', 90, 0), CaptureDirection('top', 0, 90),
                      CaptureDirection('reserved', 45, 35)]
        report = suggest_additional_view(target, results, capture_available=True, directions=directions,
                                         held_out_directions=(directions[-1],), resolution=(16, 16))
        self.assertEqual(report['status'], 'no_accessible_new_direction')
        self.assertIsNone(report['request'])
        self.assertEqual(len(report['excluded_directions']), 5)
        self.assertEqual(report['excluded_directions'][-1]['reason'], 'reserved_evaluation_camera')

    def test_access_and_visibility_weights_control_ranking_without_new_evidence(self):
        target, results = planning_fixture()
        directions = [CaptureDirection('unavailable', 45, 35, access_weight=0),
                      CaptureDirection('hidden', -45, 35, visible_mask=np.zeros((16, 16), bool)),
                      CaptureDirection('open', 135, 35, access_weight=.5)]
        report = suggest_additional_view(target, results, capture_available=True, directions=directions, resolution=(16, 16), minimum_disagreement=.01)
        self.assertEqual(report['request']['direction'], 'open')
        row = report['ranked_directions'][0]
        self.assertAlmostEqual(row['ranking_score'], row['mean_pairwise_disagreement']*.5)
        self.assertEqual(report['observations_added'], 0)

    def test_stale_failed_missing_or_low_quality_hypotheses_do_not_trigger_capture(self):
        target, results = planning_fixture()
        original = results['second']
        variants = [replace(original, status='failed'),
                    replace(original, metric_result=replace(original.metric_result, extras={'input_evidence_hash': 'stale'})),
                    replace(original, metric_result=CandidateMetrics(per_view={'front': {'area_iou': 1., 'passed': True}},
                                                                      extras=original.metric_result.extras)),
                    replace(original, metric_result=replace(original.metric_result,
                        per_view={view: {'area_iou': .8, 'passed': True} for view in ('front', 'side', 'top')})),
                    replace(original, metric_result=replace(original.metric_result,
                        per_view={view: {'area_iou': 1., 'passed': False} for view in ('front', 'side', 'top')}))]
        for variant in variants:
            report = suggest_additional_view(target, capture_available=True, results={'first': results['first'], 'second': variant}, directions=self.directions())
            self.assertEqual(report['status'], 'insufficient_retained_hypotheses')
            self.assertIsNone(report['request'])
            self.assertTrue(report['excluded_candidates'])

    def test_duplicate_or_reindexed_identical_surfaces_do_not_manufacture_ambiguity(self):
        target, results = planning_fixture()
        original = results['first']
        duplicate = replace(original, candidate_id='duplicate')
        report = suggest_additional_view(target, capture_available=True, results={'first': original, 'duplicate': duplicate})
        self.assertEqual(report['status'], 'insufficient_retained_hypotheses')
        data = original.geometry
        permutation = np.arange(len(data.vertices))[::-1]
        inverse = np.argsort(permutation)
        reindexed = GeometryArrays.capture(data.vertices[permutation], inverse[data.faces])
        duplicate = replace(duplicate, geometry=reindexed)
        report = suggest_additional_view(target, capture_available=True, results={'first': original, 'duplicate': duplicate}, directions=self.directions(), resolution=(16, 16))
        self.assertEqual(report['status'], 'no_material_projected_disagreement')
        self.assertIsNone(report['request'])

    def test_expired_allowance_does_not_start_projection_or_alter_evidence(self):
        target, results = planning_fixture()
        clock = itertools.count(0, 10)
        with patch('reconstruction.view_suggestions.time.perf_counter', side_effect=lambda: next(clock)), patch(
                'reconstruction.view_suggestions.filtered_candidate_projection', side_effect=AssertionError('unexpected render')):
            report = suggest_additional_view(target, results, capture_available=True, directions=self.directions())
        self.assertEqual(report['status'], 'partial_planning_allowance')
        self.assertIsNone(report['request'])
        self.assertEqual(report['observations_added'], 0)

    def test_invalid_visibility_and_direction_allowances_are_rejected(self):
        target, results = planning_fixture()
        with self.assertRaises(ValueError):
            suggest_additional_view(target, results, capture_available=True, directions=self.directions()*7)
        with self.assertRaises(ValueError):
            suggest_additional_view(target, results, capture_available=True, directions=[CaptureDirection('nan-mask', 45, 35,
                                       visible_mask=np.full((16, 16), np.nan))], resolution=(16, 16))
        with self.assertRaises(ValueError):
            CaptureDirection('invalid', 0, 0, access_weight=2)


if __name__ == '__main__':
    unittest.main()
