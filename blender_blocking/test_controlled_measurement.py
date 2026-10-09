"""Focused pure checks for controlled linear-alpha acquisition contracts."""
from copy import deepcopy
import unittest
import numpy as np
from evaluation.controlled_measurement import (PROTOCOL, coverage_and_mask,
    measurement_signature, compare_controlled_measurements)
from evaluation.silhouette_eval import SilhouetteGateConfig


def contract():
    return {'protocol': PROTOCOL, 'hard_mask_threshold': .5,
        'camera': {'projection': 'ORTHO', 'resolution': [16, 16], 'clip_start': .1, 'clip_end': 100,
                   'matrix_world': np.eye(4).tolist(), 'ortho_scale': 2.5},
        'renderer': {'engine': 'BLENDER_EEVEE', 'blender_version': [5, 2, 2]},
        'sampling': {'taa_render_samples': 64, 'filter_size': 1.5},
        'pixel_convention': 'top-left; pixel centers i+.5',
        'surface_policy': 'opaque target meshes', 'encoding': 'linear alpha'}


class TestControlledMeasurement(unittest.TestCase):
    def test_half_coverage_tie_and_fractional_precision(self):
        raw = np.array([[0., .49, .5, .51, 1.]])
        coverage, mask = coverage_and_mask(raw)
        np.testing.assert_array_equal(mask, [[False, False, True, True, True]])
        np.testing.assert_allclose(coverage, raw, atol=1e-7)

    def test_invalid_coverage_does_not_become_evidence(self):
        for raw in ([], [0., 1.], [[np.nan]], [[-.01]], [[1.01]]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                coverage_and_mask(raw)

    def test_matching_measurements_use_unchanged_strict_thresholds(self):
        coverage = np.zeros((16, 16)); coverage[4:12, 4:12] = 1.
        record = {'contract': contract(), 'coverage': coverage}
        gates = SilhouetteGateConfig(min_area_iou=.7, min_boundary_iou=.8,
                                     max_signed_distance_loss=.05)
        result = compare_controlled_measurements(record, record, view='front', config=gates)
        self.assertTrue(result['passed'])
        self.assertEqual(result['thresholds'], {'min_area_iou': .7,
            'min_boundary_iou': .8, 'max_signed_distance_loss': .05})
        self.assertEqual(result['coverage_l1'], 0.)

    def test_changed_camera_sampling_encoding_or_threshold_is_refused(self):
        reference = {'contract': contract(), 'coverage': np.eye(16)}
        for key, value in (('camera', {}), ('sampling', {}), ('encoding', 'display RGB'),
                           ('hard_mask_threshold', .49)):
            changed = deepcopy(reference); changed['contract'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare_controlled_measurements(reference, changed, view='front', config=SilhouetteGateConfig())

    def test_incomplete_protocol_is_refused(self):
        for value in ({}, {'protocol': PROTOCOL, 'hard_mask_threshold': .5}):
            with self.assertRaises(ValueError): measurement_signature(value)

    def test_matching_but_incomplete_acquisition_is_refused(self):
        for key in ('camera', 'sampling', 'renderer'):
            invalid = contract(); invalid[key] = {}
            with self.subTest(key=key), self.assertRaises((ValueError, KeyError)):
                measurement_signature(invalid)

    def test_resolution_mismatch_is_refused(self):
        reference = {'contract': contract(), 'coverage': np.eye(16)}
        candidate = {'contract': contract(), 'coverage': np.eye(17)}
        with self.assertRaises(ValueError):
            compare_controlled_measurements(reference, candidate, view='front', config=SilhouetteGateConfig())
