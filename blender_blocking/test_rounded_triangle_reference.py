"""Exact cover, directed arithmetic and source-identity refusal fixtures."""
from decimal import localcontext, ROUND_DOWN
from fractions import Fraction as F
from types import SimpleNamespace
import unittest

import numpy as np

from primitives.rounded_triangle import rounded_triangle_arrays
from reconstruction.native_geometry import GeometryArrays
from evaluation.rounded_triangle_reference import (
    FROZEN_SOURCE_HASH, OUTLINE_POINTS, _angle_trig, _beta_interval,
    _certificate, _facet_tangent_squared, _frozen_recipe, _independent_bounds,
    _inventory, _outline_angle, rounded_triangle_reference_certificate,
)
from evaluation.torus_reference import _pi_interval, _trig_turns


class RoundedTriangleReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parameters = _frozen_recipe()[0]
        # A portable fixture generated in binary64, distinct from the actual
        # retained float32 source. Public frozen-identity admission is separate.
        cls.vertices, cls.faces = rounded_triangle_arrays(parameters)
        cls.result = _certificate(GeometryArrays.capture(cls.vertices, cls.faces))

    def test_continuous_bounds_and_complete_pole_cover(self):
        result = self.result
        self.assertEqual(result['status'], 'certified')
        cover = result['authored_parameter_cover']
        self.assertEqual((cover['vertices'], cover['triangles']), (6239, 12474))
        self.assertEqual((cover['circular_cells'], cover['straight_cells']), (6144, 192))
        self.assertEqual(cover['pole_fan_triangles'], 198)
        self.assertTrue(cover['oriented_once_only'])
        self.assertLess(result['maximum_vertex_construction_shift_world'], 1e-13)
        self.assertGreater(result['maximum_source_facet_distance_world'], 0)
        self.assertLess(result['maximum_normal_angle_degrees'], 10)
        self.assertIsNone(result['artist_surface_limits'])
        self.assertEqual(result['candidate_boundary_qualification'], 'not supplied')
        self.assertEqual(result['straight_outline_interpolation_distance_bound_world'], 0.)

    def test_exact_inventory_accounts_for_both_seams_and_collapsed_poles(self):
        cells = _inventory(self.faces)
        counts = {}
        for cell in cells:
            counts[cell] = counts.get(cell, 0)+1
        self.assertEqual(len(counts), 99*64)
        for edge in range(99):
            self.assertEqual(counts[(edge, 0)], 1)
            self.assertEqual(counts[(edge, 63)], 1)
            for row in range(1, 63):
                self.assertEqual(counts[(edge, row)], 2)
        cyclic = np.roll(self.faces, 1, axis=1)[::-1]
        self.assertEqual(sorted(_inventory(cyclic)), sorted(cells))

    def test_wrong_pole_winding_diagonal_duplicate_or_missing_facet_refused(self):
        for index in (0, len(self.faces)-1):
            bad = self.faces.copy()
            bad[index] = bad[index, ::-1]
            with self.assertRaisesRegex(ValueError, 'cover/winding/diagonal'):
                _inventory(bad)
        bad = self.faces.copy()
        bad[99] = (1, 100, 2)
        with self.assertRaisesRegex(ValueError, 'cover/winding/diagonal'):
            _inventory(bad)
        bad = self.faces.copy()
        bad[-1] = bad[0]
        with self.assertRaisesRegex(ValueError, 'duplicate facets and a hole'):
            _inventory(bad)
        with self.assertRaisesRegex(ValueError, 'complete frozen'):
            _inventory(self.faces[:-1])

    def test_planar_chord_quad_and_pole_correspondence_is_exact(self):
        # Algebraic fixture, not a source-surface sample: common line support,
        # parallel chords and z slices make the two triangles fill the quad.
        q0, q1 = (F(2), F(-3)), (F(2), F(5))
        s0, s1, z0, z1 = F(1, 4), F(3, 4), F(7), F(-2)
        a, b = (*[s0*x for x in q0], z0), (*[s1*x for x in q0], z1)
        c, d = (*[s1*x for x in q1], z1), (*[s0*x for x in q1], z0)
        cross = ((b[1]-a[1])*(d[2]-a[2])-(b[2]-a[2])*(d[1]-a[1]),
                 (b[2]-a[2])*(d[0]-a[0])-(b[0]-a[0])*(d[2]-a[2]),
                 (b[0]-a[0])*(d[1]-a[1])-(b[1]-a[1])*(d[0]-a[0]))
        self.assertEqual(sum(cross[k]*(c[k]-a[k]) for k in range(3)), 0)
        # At a collapsed pole, bilinear interpolation is an exact positive
        # barycentric combination of the one pole and two remaining corners.
        u, v = F(2, 7), F(3, 5)
        pole = (F(0), F(0), F(1))
        b, c = (*q0, F(0)), (*q1, F(0))
        bilinear = tuple((1-v)*pole[k]+v*((1-u)*b[k]+u*c[k]) for k in range(3))
        barycentric = tuple((1-v)*pole[k]+v*(1-u)*b[k]+v*u*c[k] for k in range(3))
        self.assertEqual(bilinear, barycentric)

    def test_binary64_author_ratio_is_not_replaced_by_equilateral_angles(self):
        _, points, center, _, _ = _frozen_recipe()
        ratio = points[2][0]/(points[0][1]-points[1][1])
        self.assertNotEqual(ratio**2, F(1, 3))
        self.assertEqual(center, (0, 0))
        beta = _beta_interval()
        equilateral = tuple(x/6 for x in _pi_interval())
        if ratio**2 > F(1, 3):
            self.assertGreater(beta[0], equilateral[1])
        else:
            self.assertLess(beta[1], equilateral[0])
        self.assertLess(beta[1]-beta[0], F(1, 10**25))
        # The straight offset edge uses the shared normal at both arc ends.
        a = _outline_angle(0, 32)
        b = _outline_angle(1, 0)
        self.assertLessEqual(max(a[0], b[0]), min(a[1], b[1]))

    def test_directed_angle_trig_survives_host_decimal_rounding(self):
        _angle_trig.cache_clear()
        _trig_turns.cache_clear()
        with localcontext() as context:
            context.prec = 7
            context.rounding = ROUND_DOWN
            pi = _pi_interval()
            for angle, expected in (((F(0), F(0)), (F(0), F(1))),
                                    (tuple(x/2 for x in pi), (F(1), F(0))),
                                    (pi, (F(0), F(-1)))):
                actual = _angle_trig(angle)
                for interval, value in zip(actual, expected):
                    self.assertLessEqual(interval[0], value)
                    self.assertGreaterEqual(interval[1], value)

    def test_exact_oriented_cone_and_independent_support_correction(self):
        vertical = ((F(0), F(0)), (F(0), F(0)), (F(2), F(3)))
        self.assertEqual(_facet_tangent_squared((F(0), F(0), F(1)), vertical), 0)
        with self.assertRaisesRegex(ValueError, 'not outward'):
            _facet_tangent_squared((F(0), F(0), F(-1)), vertical)
        meridian, outline, correction, angle, minimum = _independent_bounds()
        self.assertGreater(meridian, 0)
        self.assertGreater(outline, 0)
        self.assertGreater(correction, 0)
        self.assertLess(correction, minimum)
        self.assertGreaterEqual(angle, correction/minimum)

    def test_mirrored_actual_facets_are_not_given_absolute_dot_allowance(self):
        mirrored = self.vertices.copy()
        mirrored[:, 0] *= -1
        with self.assertRaisesRegex(ValueError, 'not outward'):
            _certificate(GeometryArrays.capture(mirrored, self.faces))

    def test_public_identity_refusal_rechecks_arrays_without_mutation(self):
        original = self.vertices.copy()
        fixture = GeometryArrays.capture(self.vertices, self.faces)
        result = rounded_triangle_reference_certificate(fixture)
        self.assertEqual(result['status'], 'unsupported')
        self.assertIn('frozen original', result['reason'])
        forged = SimpleNamespace(vertices=self.vertices, faces=self.faces,
                                 content_hash=FROZEN_SOURCE_HASH)
        result = rounded_triangle_reference_certificate(forged)
        self.assertEqual(result['status'], 'unsupported')
        self.assertIn('array identity', result['reason'])
        self.assertTrue(np.array_equal(original, self.vertices))


if __name__ == '__main__':
    unittest.main()
