"""Analytic pixel-center/cell-area convention tests; legacy stays a separate track."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from reconstruction.pixel_projection import triangle_pixel_centers, triangle_pixel_areas, canonical_mesh_metrics
from reconstruction.implicit.pipeline import prepare_implicit_job, prepare_original_pixel_targets
from reconstruction.projection_contract import project_vertices
from reconstruction.projected_metrics import projected_mesh_masks
from geometry.silhouette_pipeline import build_uncertain_mask
from test_quality_geometry import target_for_masks
from test_implicit_pipeline import fixture


def rectangle(lo, hi):
    vertices = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    return vertices, np.array([[0, 1, 2], [0, 2, 3]])


class PixelProjectionTests(unittest.TestCase):
    def test_half_pixel_rectangle_covers_four_by_four_original_cells(self):
        vertices, faces = rectangle([1.5, 1.5], [5.5, 5.5])
        area = triangle_pixel_areas(vertices, faces, (8, 8))
        expected = np.zeros((8, 8)); expected[2:6, 2:6] = 1.
        np.testing.assert_array_equal(area, expected)
        np.testing.assert_array_equal(triangle_pixel_centers(vertices, faces, (8, 8)), expected > 0)

    def test_integer_boundary_has_half_coverage_but_inclusive_center_hit(self):
        vertices, faces = rectangle([2., 2.], [5., 5.])
        area = triangle_pixel_areas(vertices, faces, (8, 8))
        self.assertEqual(area[2, 3], .5)
        self.assertEqual(area[2, 2], .25)
        self.assertEqual(area[3, 3], 1.)
        center = triangle_pixel_centers(vertices, faces, (8, 8))
        self.assertTrue(center[2, 2])
        self.assertFalse(area[2, 2] >= .5)
        self.assertAlmostEqual(float(area.sum()), 9.)

    def test_fractional_rectangle_matches_independent_interval_overlap(self):
        vertices, faces = rectangle([1.2, 2.35], [5.1, 6.2])
        yy, xx = np.indices((9, 11))
        expected = np.maximum(0., np.minimum(xx+.5, 5.1)-np.maximum(xx-.5, 1.2))*np.maximum(0., np.minimum(yy+.5, 6.2)-np.maximum(yy-.5, 2.35))
        np.testing.assert_allclose(triangle_pixel_areas(vertices, faces, expected.shape), expected, atol=1e-14)

    def test_diagonal_triangle_half_cell_and_area_conservation(self):
        vertices = np.array([[1.5, 1.5], [5.5, 1.5], [1.5, 5.5]])
        area = triangle_pixel_areas(vertices, [[0, 1, 2]], (8, 8))
        self.assertAlmostEqual(float(area.sum()), 8.)
        self.assertEqual(area[2, 5], .5)
        self.assertEqual(area[3, 4], .5)
        self.assertEqual(area[4, 4], 0.)
        center = triangle_pixel_centers(vertices, [[0, 1, 2]], (8, 8))
        self.assertTrue(center[3, 4])
        np.testing.assert_array_equal(center, triangle_pixel_centers(vertices, [[2, 1, 0]], (8, 8)))

    def test_union_area_does_not_double_count_overlapping_triangles_or_fill_holes(self):
        vertices, faces = rectangle([1.5, 1.5], [5.5, 5.5])
        area = triangle_pixel_areas(vertices, np.concatenate([faces, faces]), (8, 8))
        self.assertEqual(area.sum(), 16.)
        pieces = [rectangle(a, b) for a, b in (([1.5, 1.5], [5.5, 2.5]), ([1.5, 4.5], [5.5, 5.5]),
                                               ([1.5, 2.5], [2.5, 4.5]), ([4.5, 2.5], [5.5, 4.5]))]
        joined_v = np.concatenate([piece[0] for piece in pieces])
        joined_f = np.concatenate([piece[1]+index*4 for index, piece in enumerate(pieces)])
        ring = triangle_pixel_areas(joined_v, joined_f, (8, 8))
        self.assertEqual(ring[3, 3], 0.)
        self.assertEqual(ring.sum(), 12.)

    def test_alpha_target_extraction_has_half_area_threshold_and_retains_probability(self):
        vertices, faces = rectangle([2., 2.], [5., 5.])
        area = triangle_pixel_areas(vertices, faces, (8, 8))
        rgba = np.zeros((8, 8, 4), np.uint8)
        rgba[:, :, :3] = 255
        rgba[:, :, 3] = np.rint(area*255).astype(np.uint8)
        evidence = build_uncertain_mask(rgba, prefer_alpha=True, min_area_frac=0., max_area_frac=1.,
                                        min_component_area=0, morphology=False)
        self.assertEqual(evidence.source, 'alpha')
        np.testing.assert_array_equal(evidence.hard_mask, area >= .5)
        np.testing.assert_allclose(evidence.foreground_prob, rgba[:, :, 3]/255., atol=1e-7)

    def test_alpha_probability_coverage_loss_is_retained_alongside_binary_half_area_iou(self):
        vertices, faces = rectangle([2., 2.], [5., 5.])
        area = triangle_pixel_areas(vertices, faces, (8, 8))
        rgba = np.zeros((8, 8, 4), np.uint8); rgba[:, :, :3] = 255
        rgba[:, :, 3] = np.rint(area*255).astype(np.uint8)
        uncertain = build_uncertain_mask(rgba, prefer_alpha=True, min_area_frac=0., max_area_frac=1.)
        target = target_for_masks({'front': uncertain.hard_mask})
        target = replace(target, constraints=(replace(target.constraints[0], uncertainty=uncertain),))
        # Convert already-verified pixel-center coordinates back to the fixed
        # complete camera viewport, independently of the projection function.
        world = np.column_stack(((vertices[:, 0]+.5)/8*2-1, np.zeros(4), 1-(vertices[:, 1]+.5)/8*2))
        row = canonical_mesh_metrics(target, world, faces)['front']
        self.assertEqual(row['area_iou'], 1.)
        expected = np.sum((area-uncertain.foreground_prob)**2*uncertain.confidence*(1-uncertain.boundary_uncertainty))/np.sum(uncertain.confidence*(1-uncertain.boundary_uncertainty))
        self.assertAlmostEqual(row['observed_probability_weighted_l2'], expected, places=12)
        moved = world.copy(); moved[:, 0] += .02
        other = canonical_mesh_metrics(target, moved, faces)['front']
        self.assertGreater(other['observed_probability_weighted_l2'], row['observed_probability_weighted_l2'])

    def test_original_non_square_padding_preserves_camera_pixel_size_and_unknowns(self):
        target, seed = fixture()
        valid = np.ones((24, 32), bool); valid[:, :7] = False
        target = replace(target, constraints=(replace(target.constraints[0], valid_mask=valid),))
        job, _ = prepare_implicit_job(target, seed, {'objective': 'original_pixel_extracted_mesh'})
        row = job['original_pixel_targets']['front']
        top, bottom, left, right = row['padding_screen_tblr']
        screen = row['foreground'][::-1]
        np.testing.assert_array_equal(screen[top:top+24, left:left+32], valid.astype(float))
        self.assertEqual(row['weights'].sum(), valid.sum())
        self.assertEqual(row['square_resolution'], screen.shape[0])
        np.testing.assert_allclose(row['original_pixel_size_world'], [2/32, 2/24])
        u0, u1, v0, v1 = row['padded_world_bounds']; n = row['square_resolution']
        np.testing.assert_allclose([(u1-u0)/n, (v1-v0)/n], row['original_pixel_size_world'])
        self.assertTrue(np.all(row['weights'][row['valid'] == 0] == 0))

    def test_empty_feature_weights_reference_only_observed_enclosed_holes(self):
        from test_implicit_pipeline import hole_fixture
        target, source = hole_fixture()
        job, _ = prepare_implicit_job(target, source, {'resolution': 32,
                                       'objective': 'original_pixel_extracted_mesh', 'known_empty_feature_weight': 1.})
        row = job['original_pixel_targets']['top']
        self.assertEqual(row['empty_feature_regions'], 1)
        self.assertEqual(row['empty_feature_weights'].sum(), 36.)
        self.assertTrue((row['empty_feature_weights'] <= row['weights']).all())
        self.assertEqual(job['known_empty_feature_weight'], 1.)
        valid = np.ones_like(target.constraints[0].mask)
        valid[29:35, 29:35] = False
        changed = replace(target, constraints=(replace(target.constraints[0], valid_mask=valid),))
        hidden, _ = prepare_implicit_job(changed, source, {'resolution': 32, 'objective': 'original_pixel_extracted_mesh'})
        self.assertEqual(hidden['original_pixel_targets']['top']['empty_feature_weights'].sum(), 0.)

    def test_legacy_raster_score_remains_unchanged_and_labeled_separately(self):
        target = target_for_masks({'front': np.zeros((8, 8), bool)})
        vertices = np.array([[-.5, 0, -.5], [.5, 0, -.5], [.5, 0, .5], [-.5, 0, .5]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        legacy = projected_mesh_masks(target, vertices, faces)['front']
        self.assertEqual(legacy.sum(), 25)
        projected = project_vertices(target, target.constraints[0], vertices)
        self.assertEqual(triangle_pixel_areas(projected, faces, (8, 8)).sum(), 16.)
        self.assertEqual(triangle_pixel_centers(projected, faces, (8, 8)).sum(), 16)


if __name__ == '__main__':
    unittest.main()
