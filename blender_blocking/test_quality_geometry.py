"""Cross-contract regressions for geometry proposals and incomplete observations."""
import unittest
from dataclasses import replace
import numpy as np
from reconstruction.types import ReconstructionTarget, ViewConstraint, Bounds2D, Bounds3D, OrthographicCameraSpec


def target_for_masks(masks):
    constraints = tuple(ViewConstraint(view, mask, OrthographicCameraSpec(view, {'front':'Y','side':'X','top':'Z'}[view],
        bounds=Bounds2D(-1., -1., 1., 1.))) for view, mask in masks.items())
    return ReconstructionTarget(constraints, bounds=Bounds3D.from_min_max((-1.,)*3, (1.,)*3))


class QualityGeometryTests(unittest.TestCase):
    def test_interval_union_preserves_known_gap(self):
        from placement.resfit.profile_intervals import union_intervals, interval_disagreement
        intervals = [(-.8, -.3), (.3, .8)]
        self.assertEqual(union_intervals(intervals), intervals)
        self.assertGreater(interval_disagreement(intervals, [(-.8, .8)], 2.), 0.)

    def test_unknown_pixels_do_not_penalize_projection(self):
        from reconstruction.visibility import evaluate_visible_pair
        ref = np.zeros((16, 16), bool); ref[4:12, 4:12] = True
        valid = np.ones_like(ref); valid[:, 12:] = False
        c = replace(target_for_masks({'front': ref}).constraints[0], valid_mask=valid)
        pred = ref.copy(); pred[:, 12:] = True
        row = evaluate_visible_pair(ref, pred, c)
        self.assertEqual(row['area_iou'], 1.)
        self.assertTrue(row['passed'])

    def test_unknown_view_is_excluded_from_aggregation(self):
        from reconstruction.visibility import evaluate_visible_pair
        from reconstruction.types import CandidateMetrics
        ref = np.zeros((8, 8), bool)
        c = replace(target_for_masks({'front': ref}).constraints[0], valid_mask=np.zeros_like(ref))
        row = evaluate_visible_pair(ref, ref, c)
        metrics = CandidateMetrics(per_view={'front': row, 'side': {'area_iou': .8, 'boundary_iou': .5}})
        self.assertEqual(metrics.area_iou_mean, .8)
        self.assertFalse(row['required'])

    def test_partial_view_unknown_world_rays_are_allowed(self):
        from reconstruction.visibility import point_support
        mask = np.zeros((8, 8), bool)
        target = target_for_masks({'front': mask})
        allowed, observed = point_support(target, np.array([[5., 0., 0.], [0., 0., 0.]]))
        np.testing.assert_array_equal(allowed, [True, False])
        np.testing.assert_array_equal(observed, [0, 1])

    def test_hierarchy_preserves_subcell_silhouette_strip(self):
        from reconstruction.adaptive_geometry import hierarchical_hull
        mask = np.zeros((64, 64), bool); mask[:, 31:32] = True
        target = target_for_masks({'front': mask})
        grid = hierarchical_hull(target, 8, conservative=True)
        self.assertGreater(np.count_nonzero(grid.to_dense()), 0)
        self.assertGreater(grid.adaptive_report['subdivided'], 0)

    def test_adaptive_sections_keep_off_center_neck(self):
        from geometry.profile_models import EllipticalProfileU
        from reconstruction.adaptive_geometry import adaptive_slices
        t = np.linspace(0., 1., 41); radii = np.ones(41); radii[19:22] = .15
        p = EllipticalProfileU(t, radii, radii, 2., cx=t*.2, cy=-t*.1)
        slices = adaptive_slices(p, minimum=4, maximum=24)
        self.assertLess(min(s.rx for s in slices), .2)
        self.assertAlmostEqual(slices[-1].cx, .2)
        self.assertGreater(len(slices), 4)

    def test_boxy_family_uses_actual_shape(self):
        from primitives.analytic_primitives import SuperquadricPrimitive, EllipsoidPrimitive
        from primitives.soft_silhouette import render_projected_soft_silhouette, OrthographicCamera
        camera = OrthographicCamera.from_view('front', image_size=(32, 32))
        args = {'radii': np.array([.7, .4, .7])}
        boxy = render_projected_soft_silhouette([SuperquadricPrimitive(**args, epsilon1=.15, epsilon2=.15)], camera)
        ellipse = render_projected_soft_silhouette([EllipsoidPrimitive(**args)], camera)
        self.assertGreater(np.count_nonzero(boxy>.5), np.count_nonzero(ellipse>.5))

    def test_program_dedupe_retains_parameter_change(self):
        from primitives.shape_program import ShapeNode, ShapeProgram
        from primitives.program_search import _program_signature
        p = ShapeProgram('shape-program-v1', 'p', (ShapeNode('a', 'add', 'box', {'width_world': 1.}),))
        q = replace(p, root_nodes=(replace(p.root_nodes[0], parameters={'width_world': 1.1}),))
        self.assertNotEqual(_program_signature(p), _program_signature(q))
        self.assertEqual(_program_signature(p), _program_signature(replace(p, program_id='other')))

    def test_cuboid_merges_retain_point_correspondence(self):
        from reconstruction.spatial_regions import cuboid_levels
        points = np.random.default_rng(4).normal(size=(96, 3))
        levels = cuboid_levels(points, fine_parts=6, levels=(4, 2))
        self.assertEqual([len(x) for x in levels], [6, 4, 2])
        for level in levels:
            self.assertEqual(sum(len(p) for p in level), len(points))

    def test_signed_subtraction_retains_cavity(self):
        from reconstruction.program_proposals import signed_csg_field
        result = signed_csg_field([np.array([-1., -1., 1.])], [np.array([-1., 1., 1.])])
        np.testing.assert_array_equal(result, [1., -1., 1.])

    def test_generated_cap_winding_preserves_primitive_coordinates(self):
        from primitives.analytic_primitives import EllipsoidPrimitive
        from reconstruction.grouped_solids import oriented_generated_mesh, solid_guard
        mesh = EllipsoidPrimitive(radii=np.array([.5, .3, .4])).to_mesh_data(16)
        data = oriented_generated_mesh(mesh)
        np.testing.assert_array_equal(data.vertices, np.asarray(mesh.vertices))
        self.assertTrue(solid_guard(data)['valid_solid'])
        self.assertGreater(solid_guard(data)['signed_volume'], 0.)

    def test_dvx_import_is_lazy_and_execution_guarded(self):
        from reconstruction.differentiable.dvx_adapter import tiny_gradient_check
        with self.assertRaises(PermissionError): tiny_gradient_check()

    def test_default_quality_config_is_unchanged(self):
        from reconstruction.quality_config import quality_config
        cfg = {'max_objective_evaluations': 256, 'adaptive_hull': False}
        self.assertEqual(quality_config(cfg), cfg)
        quality = quality_config({**cfg, 'quality_preset': 'quality'})
        self.assertTrue(quality['adaptive_hull'])
        self.assertEqual(quality['max_objective_evaluations'], 768)


if __name__ == '__main__': unittest.main()
