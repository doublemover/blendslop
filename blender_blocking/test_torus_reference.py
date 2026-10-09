"""Portable torus proof-core fixtures and strict frozen-source refusal checks."""
import math
from decimal import localcontext, ROUND_DOWN
from fractions import Fraction as F
from types import SimpleNamespace
import unittest

import numpy as np
from reconstruction.native_geometry import GeometryArrays
from evaluation.torus_reference import (
    FROZEN_SOURCE_HASH, _certificate, _pi_interval, _trig_turns,
    torus_reference_certificate,
)


def lattice():
    n, m = 128, 24
    vertices = np.array([[(.7+.22*math.cos(2*math.pi*j/m))*math.cos(2*math.pi*i/n),
                          (.7+.22*math.cos(2*math.pi*j/m))*math.sin(2*math.pi*i/n),
                          .22*math.sin(2*math.pi*j/m)] for i in range(n) for j in range(m)])
    faces = []
    for i in range(n):
        for j in range(m):
            a, b = i*m+j, ((i+1)%n)*m+j
            c, d = ((i+1)%n)*m+(j+1)%m, i*m+(j+1)%m
            faces.extend(((a, b, c), (a, c, d)))
    return vertices, np.array(faces)


class TorusReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vertices, cls.faces = lattice()
        cls.result = _certificate(GeometryArrays.capture(cls.vertices, cls.faces))

    def test_full_oriented_periodic_cover_and_independent_bounds(self):
        result = self.result
        self.assertEqual(result['status'], 'certified')
        self.assertEqual(result['authored_parameter_lattice']['triangles'], 6144)
        self.assertTrue(result['authored_parameter_lattice']['periodic_seams_verified'])
        self.assertLess(result['maximum_vertex_construction_shift_world'], 1e-14)
        expected = math.pi**2*(.92/128**2+.44/(128*24)+.22/24**2)/2
        self.assertGreaterEqual(result['maximum_source_facet_distance_world'], expected)
        self.assertGreater(result['maximum_normal_angle_degrees'], 0.)
        self.assertLess(result['maximum_normal_angle_degrees'], 180.)
        self.assertIsNone(result['artist_surface_limits'])
        self.assertEqual(result['candidate_boundary_qualification'], 'not supplied')

    def test_barycentric_correspondence_crosses_both_periodic_seams(self):
        for i, j in ((0, 0), (0, 23), (127, 0), (127, 23), (37, 11)):
            face = self.faces[2*(i*24+j)]
            weights = np.array([.23, .31, .46])
            theta = 2*math.pi*(i+weights[1]+weights[2])/128
            phi = 2*math.pi*(j+weights[2])/24
            analytic = np.array([(.7+.22*math.cos(phi))*math.cos(theta),
                                 (.7+.22*math.cos(phi))*math.sin(theta), .22*math.sin(phi)])
            point = weights@self.vertices[face]
            self.assertLessEqual(np.linalg.norm(point-analytic),
                                 self.result['maximum_source_facet_distance_world'])

    def test_duplicate_facet_and_missing_periodic_seam_are_refused(self):
        bad = self.faces.copy()
        bad[-1] = bad[0]
        with self.assertRaisesRegex(ValueError, 'duplicate facets and a hole'):
            _certificate(GeometryArrays.capture(self.vertices, bad))
        with self.assertRaisesRegex(ValueError, 'complete frozen'):
            _certificate(GeometryArrays.capture(self.vertices, self.faces[:-1]))

    def test_reversed_seam_winding_and_changed_diagonal_are_refused(self):
        bad = self.faces.copy()
        bad[-1] = bad[-1, ::-1]
        with self.assertRaisesRegex(ValueError, 'cover/winding'):
            _certificate(GeometryArrays.capture(self.vertices, bad))
        bad = self.faces.copy()
        bad[0] = [0, 24, 1]
        with self.assertRaisesRegex(ValueError, 'cover/winding'):
            _certificate(GeometryArrays.capture(self.vertices, bad))

    def test_actual_geometry_must_preserve_a_positive_oriented_normal_cone(self):
        mirrored = self.vertices.copy()
        mirrored[:, 0] *= -1
        with self.assertRaisesRegex(ValueError, 'not outward'):
            _certificate(GeometryArrays.capture(mirrored, self.faces))

    def test_public_api_refuses_changed_source_without_a_tolerance(self):
        original = GeometryArrays.capture(self.vertices, self.faces)
        result = torus_reference_certificate(original)
        self.assertEqual(result['status'], 'unsupported')
        self.assertIn('frozen original', result['reason'])
        changed = self.vertices.copy()
        changed[0, 0] += 1e-9
        self.assertEqual(torus_reference_certificate(GeometryArrays.capture(changed, self.faces))['status'], 'unsupported')

    def test_forged_frozen_hash_cannot_bypass_recomputed_array_identity(self):
        result = torus_reference_certificate(SimpleNamespace(
            vertices=self.vertices, faces=self.faces, content_hash=FROZEN_SOURCE_HASH))
        self.assertEqual(result['status'], 'unsupported')
        self.assertIn('array identity', result['reason'])

    def test_directed_trig_bounds_do_not_depend_on_global_decimal_rounding(self):
        _trig_turns.cache_clear()
        with localcontext() as context:
            context.prec = 7
            context.rounding = ROUND_DOWN
            for turn, sin_value, cos_value in ((F(0), F(0), F(1)), (F(1, 4), F(1), F(0)),
                                             (F(1, 2), F(0), F(-1)), (F(3, 4), F(-1), F(0))):
                sine, cosine = _trig_turns(turn)
                self.assertLessEqual(sine[0], sin_value)
                self.assertGreaterEqual(sine[1], sin_value)
                self.assertLessEqual(cosine[0], cos_value)
                self.assertGreaterEqual(cosine[1], cos_value)
            sine, cosine = _trig_turns(F(1, 8))
            for interval in (sine, cosine):
                self.assertLessEqual(interval[0]**2, F(1, 2))
                self.assertGreaterEqual(interval[1]**2, F(1, 2))
        low, high = _pi_interval()
        self.assertLessEqual(low, F('3.141592653589793238462643383279502884197169399375105820974944592'))
        self.assertGreaterEqual(high, F('3.141592653589793238462643383279502884197169399375105820974944592')+F(1, 10**63))


if __name__=='__main__':
    unittest.main()
