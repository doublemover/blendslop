"""Bounded closed-source fields and topology-changing residual coordinates."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from reconstruction.implicit.field_model import NarrowBandField
from reconstruction.implicit.seed_field import signed_seed_field, triangle_distance_and_winding
from reconstruction.native_geometry import GeometryArrays
from reconstruction.grouped_solids import solid_guard
from volume import Bounds3D, DenseVolumeGrid, extract_mesh
from test_view_suggestions import cube


def sampled_box(half_extents=(.6, .6, .1), *, n=16):
    axes = -.8+(np.arange(n)+.5)*1.6/n
    points = np.stack(np.meshgrid(axes, axes, axes, indexing='ij'), axis=-1)
    q = np.abs(points)-half_extents
    phi = np.linalg.norm(np.maximum(q, 0), axis=-1)+np.minimum(q.max(axis=-1), 0)
    return phi.astype(np.float32).transpose(2, 1, 0), np.full(3, 1.6/n), np.full(3, axes[0])


def model_for(phi=None, **kwargs):
    seed, spacing, origin = sampled_box() if phi is None else (phi, np.full(3, .1), np.full(3, -.75))
    return NarrowBandField.create(seed, spacing, origin_xyz=origin,
                                band_width=.2, maximum_displacement=.18, **kwargs)


class ImplicitFieldTests(unittest.TestCase):
    def test_actual_screened_cube_distance_matches_analytic_box_at_fixed_cell_centers(self):
        source = cube([.1, -.1, .05], radius=.4)
        result = signed_seed_field(source, [-.8]*3, [.8]*3, resolution=16)
        spacing, origin = result['voxel_size_xyz'], result['origin_xyz']
        indices = np.stack(np.meshgrid(*[np.arange(16)]*3, indexing='ij'), axis=-1)
        points = origin+indices*spacing
        q = np.abs(points-[.1, -.1, .05])-.4
        expected = np.linalg.norm(np.maximum(q, 0), axis=-1)+np.minimum(q.max(axis=-1), 0)
        np.testing.assert_allclose(result['field_xyz'], expected, atol=4e-8)
        self.assertEqual(result['source_geometry_hash'], source.content_hash)
        self.assertTrue(result['exact_within_part_boundary']['passed'])
        self.assertLess(result['maximum_sampled_winding_error'], 1e-12)

    def test_triangle_sampler_face_edge_corner_distances_and_winding(self):
        source = cube([0, 0, 0], radius=.5)
        points = np.array([[0, 0, 0], [.7, 0, 0], [.7, .8, 0], [.7, .8, .9], [.5, 0, 0]])
        distance, winding = triangle_distance_and_winding(points, source.vertices, source.faces)
        np.testing.assert_allclose(distance, [.5, .2, np.hypot(.2, .3), np.linalg.norm([.2, .3, .4]), 0], atol=1e-14)
        self.assertAlmostEqual(abs(winding[0]), 1., places=13)
        np.testing.assert_allclose(winding[1:4], 0, atol=1e-14)

    def test_open_seed_and_clipped_fixed_domain_fail_without_geometry_changes(self):
        source = cube([0, 0, 0], radius=.4)
        opened = GeometryArrays.capture(source.vertices, source.faces[:-1])
        with self.assertRaisesRegex(ValueError, 'closed'):
            signed_seed_field(opened, [-.8]*3, [.8]*3)
        with self.assertRaisesRegex(ValueError, 'strictly contained'):
            signed_seed_field(source, [-.4]*3, [.4]*3)
        self.assertTrue(solid_guard(source)['valid_solid'])

    def test_residual_changes_only_active_cells_and_is_bounded_in_world_distance(self):
        model = model_for()
        rng = np.random.default_rng(79)
        parameters = rng.normal(size=len(model.active_flat))*4
        field = model.decode(parameters)
        difference = field-model.seed_zyx
        self.assertLessEqual(float(np.abs(difference).max()), model.maximum_displacement+1e-12)
        inactive = np.ones(model.seed_zyx.size, bool)
        inactive[model.active_flat] = False
        np.testing.assert_array_equal(field.ravel()[inactive], model.seed_zyx.ravel()[inactive])
        self.assertTrue((field[model.protected_empty_zyx] > 0).all())
        np.testing.assert_array_equal(model.decode(np.zeros_like(parameters)), model.seed_zyx)
        self.assertFalse(model.report()['native_qualification'])
        self.assertIn('dense base field', model.report()['storage_scope'])

    def test_confirmed_empty_ray_can_form_a_through_hole_without_unbounded_motion(self):
        seed, spacing, origin = sampled_box()
        known_empty = np.zeros(seed.shape, bool)
        known_empty[:, 7:9, 7:9] = True
        model = model_for(known_empty_zyx=known_empty)
        field = model.decode(np.zeros(len(model.active_flat)))
        self.assertTrue((field[known_empty] > 0).all())
        self.assertLessEqual(float(np.abs(field-seed).max()), .18+1e-8)
        bounds = Bounds3D(-.8, .8, -.8, .8, -.8, .8)
        initial = extract_mesh(DenseVolumeGrid(seed.transpose(2, 1, 0), bounds, value_type='signed_distance', default_value=1.))
        final = extract_mesh(DenseVolumeGrid(field.transpose(2, 1, 0), bounds, value_type='signed_distance', default_value=1.))
        self.assertEqual(initial.topology['euler_characteristic'], 2)
        self.assertEqual(final.topology['euler_characteristic'], 0)
        self.assertTrue(solid_guard(GeometryArrays.capture(final.vertices, final.faces))['valid_solid'])
        from blender_blocking.evaluation.triangle_contacts import within_part_boundary_guard
        self.assertTrue(within_part_boundary_guard(final.vertices, final.faces, timeout_s=2.)['passed'])
        # A retained field is authoritative for exact numerical extraction
        # replay; serialized latent reconstruction is a different tolerance.
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'retained-field.npz'
            np.savez_compressed(path, field_zyx=field)
            with np.load(path, allow_pickle=False) as artifact:
                replay = extract_mesh(DenseVolumeGrid(artifact['field_zyx'].transpose(2, 1, 0), bounds,
                                                       value_type='signed_distance', default_value=1.))
        np.testing.assert_array_equal(replay.vertices, final.vertices)
        np.testing.assert_array_equal(replay.faces, final.faces)

    def test_local_field_can_join_a_thin_bridge_but_confirmed_empty_gap_blocks_it(self):
        axes = -.8+(np.arange(16)+.5)*.1
        points = np.stack(np.meshgrid(axes, axes, axes, indexing='ij'), axis=-1)
        def box_field(center):
            q = np.abs(points-center)-[.2, .45, .1]
            return np.linalg.norm(np.maximum(q, 0), axis=-1)+np.minimum(q.max(axis=-1), 0)
        seed = np.minimum(box_field([-.35, 0, 0]), box_field([.35, 0, 0])).transpose(2, 1, 0)
        model = model_for(seed)
        xyz = points.transpose(2, 1, 0, 3).reshape(-1, 3)[model.active_flat]
        bridge = (np.abs(xyz[:, 0]) < .18) & (np.abs(xyz[:, 1]) < .16) & (np.abs(xyz[:, 2]) < .075)
        parameters = np.zeros(len(model.active_flat))
        parameters[bridge] = np.arctanh(-.17/.18)
        joined = model.decode(parameters)
        empty = np.zeros(seed.shape, bool)
        empty[:, :, 7:9] = True
        protected = model_for(seed, known_empty_zyx=empty).decode(parameters)
        bounds = Bounds3D(-.8, .8, -.8, .8, -.8, .8)
        def mesh(field):
            return extract_mesh(DenseVolumeGrid(field.transpose(2, 1, 0), bounds,
                                                 value_type='signed_distance', default_value=1.))
        self.assertEqual(mesh(seed).topology['connected_components'], 2)
        self.assertEqual(mesh(joined).topology['connected_components'], 1)
        self.assertEqual(mesh(protected).topology['connected_components'], 2)
        self.assertTrue((protected[empty] > 0).all())
        # This proves representation capacity using an analytic field fixture,
        # not that a bridge was inferred or that multipart source admission exists.

    def test_unobserved_cavity_is_not_invented_by_zero_residual_coordinates(self):
        model = model_for()
        field = model.decode(np.zeros(len(model.active_flat)))
        np.testing.assert_array_equal(field, model.seed_zyx)
        self.assertLess(field[8, 8, 8], 0.)
        self.assertFalse(model.report()['native_qualification'])

    def test_incompatible_empty_constraints_and_origin_identity_fail_closed(self):
        phi, spacing, origin = sampled_box(half_extents=(.6, .6, .6))
        known = np.zeros_like(phi, bool)
        known[8, 8, 8] = True
        with self.assertRaisesRegex(ValueError, 'conflicts outside'):
            model_for(phi, known_empty_zyx=known)
        first = model_for()
        second = NarrowBandField.create(first.seed_zyx, spacing, origin_xyz=origin+1,
                                        band_width=.2, maximum_displacement=.18)
        self.assertNotEqual(first.seed_field_hash, second.seed_field_hash)
        with self.assertRaises(ValueError):
            model_for(known_empty_zyx=np.full(first.seed_zyx.shape, np.nan))


if __name__ == '__main__':
    unittest.main()
