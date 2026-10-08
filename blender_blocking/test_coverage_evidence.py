"""Photometric calibration, subpixel phase and censored profile regressions."""
import pickle
import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from reconstruction.coverage_evidence import (coverage_from_grayscale, coverage_interval,
                                              coverage_bbox, constraint_coverage)
from reconstruction.profile_evidence import measure_profile_row
from reconstruction.projection_contract import bounds_from_calibrated_masks, calibrated_profile
from reconstruction.types import Bounds2D
from test_quality_geometry import target_for_masks


class CoverageEvidenceTests(unittest.TestCase):
    def test_declared_nonlinear_transfer_restores_half_coverage(self):
        linear = np.linspace(0, 1, 1025)
        encoded = .77 * np.sqrt(linear)
        fractions = np.array([[0., .25, .5, .75, 1.]])
        image = .77 * np.sqrt(1 - fractions)
        np.testing.assert_allclose(coverage_from_grayscale(image, linear, encoded), fractions, atol=1e-6)
        self.assertGreater(float(coverage_from_grayscale(np.array([[.5]]), linear, encoded)[0,0]), .5)

    def test_transfer_refuses_guessed_or_incompatible_encoding(self):
        linear = np.linspace(0, 1, 65)
        for encoded in (linear[::-1], linear * .4, np.full(65, .7)):
            with self.assertRaises(ValueError):
                coverage_from_grayscale(np.array([[.8]]), linear, encoded)

    def test_half_coverage_edges_follow_fractional_pixel_phase(self):
        row = np.array([0., .25, 1., 1., .75, 0.])
        left, right = coverage_interval(row)
        self.assertAlmostEqual(left, 1.5 + .25 / .75)
        self.assertAlmostEqual(right, 4.5 + .25 / .75)
        record = measure_profile_row(row >= .8, np.ones(6, bool), row)
        self.assertAlmostEqual(record['exact_width_px'], right-left)
        self.assertEqual(record['edge_model'], 'declared_linear_half_coverage_crossing')

    def test_unknown_coverage_cannot_create_an_exact_edge_or_move_a_profile(self):
        row = np.array([0., .25, 1., 1., .75, 0.])
        valid = np.ones(6, bool); valid[4:] = False
        first = measure_profile_row(row >= .8, valid, row)
        changed = row.copy(); changed[~valid] = np.nan
        second = measure_profile_row(row >= .8, valid, changed)
        self.assertEqual(first, second)
        self.assertIsNone(first['right_edge_px'])
        self.assertIsNone(first['exact_width_px'])

    def test_complete_bounds_refuse_empty_or_clipped_coverage(self):
        for coverage in (np.zeros((8,8)), np.ones((8,8))):
            with self.assertRaises(ValueError): coverage_bbox(coverage)

    def test_coverage_restores_cap_phase_and_keeps_hard_gates_unchanged(self):
        mask = np.zeros((20,20), bool); mask[3:17,5:15] = True
        coverage = np.zeros((20,20)); coverage[2:18,5:15] = 1.
        coverage[2,5:15] = .65; coverage[17,5:15] = .65
        records = {v:{'world_bounds':(-1.,1.,-1.,1.)} for v in ('front','side')}
        boxes = {v:Bounds2D(5,3,15,17) for v in records}
        masks = {v:mask.copy() for v in records}
        bounds = bounds_from_calibrated_masks(masks, boxes, records, coverage_masks={v:coverage for v in records})
        target = target_for_masks(masks)
        target = replace(target, bounds=bounds, constraints=tuple(replace(c,coverage_mask=coverage) for c in target.constraints))
        profile = calibrated_profile(SimpleNamespace(target=target, config={'num_samples':17}))
        np.testing.assert_allclose(profile.rx, .5)
        cap = profile.meta['row_evidence']['front'][0]
        self.assertIsNone(cap['exact_radius_world'])
        self.assertLess(cap['filtered_width_px'], 10.)
        self.assertEqual(cap['completion_reason'], 'partial_terminal_coverage_flat_cap_prior')
        self.assertGreater(bounds.max_z, .7)
        np.testing.assert_array_equal(target.constraints[0].mask, mask)
        self.assertEqual(profile.meta['coverage_views'], ['front','side'])
        restored = pickle.loads(pickle.dumps(target))
        np.testing.assert_array_equal(restored.constraints[0].coverage_mask, coverage)

    def test_builder_preserves_coverage_artifacts_and_unknown_invariance(self):
        import tempfile
        from pathlib import Path
        from reconstruction.target_builder import build_target_from_images
        image = np.full((20, 20, 3), 255, np.uint8)
        image[3:17, 5:15] = 0
        coverage = np.zeros((20, 20))
        coverage[3:17, 5:15] = 1.
        valid = np.ones((20, 20), bool)
        valid[:, :2] = False
        unknown_changed = coverage.copy()
        unknown_changed[~valid] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            first = build_target_from_images(
                {'front': image}, coverage_masks={'front': coverage},
                valid_evidence_masks={'front': valid}, artifact_root=directory)
            saved = np.load(first.artifact_paths['front_coverage'])
            np.testing.assert_array_equal(saved, coverage)
            second = build_target_from_images(
                {'front': image}, coverage_masks={'front': unknown_changed},
                valid_evidence_masks={'front': valid})
            self.assertEqual(first.target.extras['coverage_sha256'],
                             second.target.extras['coverage_sha256'])
            self.assertFalse(np.shares_memory(
                first.target.constraints[0].coverage_mask, coverage))
            np.testing.assert_array_equal(first.masks['front'], second.masks['front'])
        with self.assertRaisesRegex(ValueError, 'absent view'):
            build_target_from_images({'front': image}, coverage_masks={'side': coverage})

    def test_known_empty_coverage_row_remains_zero_instead_of_completion(self):
        mask = np.zeros((20, 20), bool)
        mask[2:18, 5:15] = True
        mask[10] = False
        target = target_for_masks({'front': mask, 'side': mask})
        target = replace(target, bounds=type(target.bounds).from_min_max(
            (-.5, -.5, -.8), (.5, .5, .8)), constraints=tuple(
                replace(c, coverage_mask=mask.astype(float)) for c in target.constraints))
        profile = calibrated_profile(SimpleNamespace(target=target, config={'num_samples': 17}))
        self.assertEqual(profile.rx[8], 0.)
        self.assertEqual(profile.meta['row_evidence']['front'][8]['exact_radius_world'], 0.)

    def test_coverage_identity_ignores_unknowns_and_retains_observed_precision(self):
        from reconstruction.evidence_identity import target_evidence_hash
        mask = np.zeros((8, 8), bool)
        mask[2:6, 2:6] = True
        coverage = mask.astype(float)
        valid = np.ones(mask.shape, bool)
        valid[:, 0] = False
        target = target_for_masks({'front': mask})
        first = replace(target, constraints=(replace(
            target.constraints[0], coverage_mask=coverage, valid_mask=valid),))
        changed = coverage.copy()
        changed[~valid] = np.nan
        second = replace(first, constraints=(replace(first.constraints[0], coverage_mask=changed),))
        self.assertEqual(target_evidence_hash(first), target_evidence_hash(second))
        changed[3, 3] = 1 - 1e-10
        third = replace(first, constraints=(replace(first.constraints[0], coverage_mask=changed),))
        self.assertNotEqual(target_evidence_hash(first), target_evidence_hash(third))
        self.assertNotEqual(target_evidence_hash(target), target_evidence_hash(first))

    def test_partial_view_keeps_coverage_extents_from_other_complete_views(self):
        from config import BlockingConfig
        from reconstruction.target_builder import build_target_from_images
        image = np.full((20, 20, 3), 255, np.uint8)
        image[3:17, 5:15] = 0
        coverage = np.zeros((20, 20))
        coverage[2:18, 5:15] = 1.
        coverage[2, 5:15] = coverage[17, 5:15] = .65
        valid = np.ones((20, 20), bool)
        valid[:, :3] = False
        config = BlockingConfig()
        config.reconstruction.view_calibration = {
            view: {'world_bounds': (-1., 1., -1., 1.)}
            for view in ('front', 'side', 'top')}
        built = build_target_from_images(
            {view: image for view in config.reconstruction.view_calibration},
            config=config, coverage_masks={'front': coverage, 'side': coverage},
            valid_evidence_masks={'top': valid})
        self.assertGreater(built.target.bounds.max_z, .7)
        self.assertIn('partial_evidence_search_bounds', built.warnings[0])
        self.assertIsNone(built.target.constraints[-1].coverage_mask)

    def test_coverage_requires_physical_shape_and_known_value_bounds(self):
        c = target_for_masks({'front':np.ones((8,8),bool)}).constraints[0]
        for value in (np.ones((9,8)), np.full((8,8),1.1), np.full((8,8),np.nan)):
            with self.assertRaises(ValueError): constraint_coverage(replace(c,coverage_mask=value), np.ones((8,8),bool))
        valid=np.zeros((8,8),bool)
        np.testing.assert_array_equal(constraint_coverage(replace(c,coverage_mask=np.full((8,8),np.nan)),valid),np.zeros((8,8)))
