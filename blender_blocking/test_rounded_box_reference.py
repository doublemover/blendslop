"""Meaningful portable patch-cover/correspondence and frozen-source refusals."""
from fractions import Fraction as F
from itertools import combinations, product
from types import SimpleNamespace
import unittest
import numpy as np
from evaluation.rounded_box_reference import (
    _proof, _cover, _norm_range, _sqrt, RADIUS, IDEAL_CORE,
    FROZEN_SOURCE_HASH, rounded_box_reference_certificate,
)
from reconstruction.native_geometry import GeometryArrays


def coarse_fixture(core=IDEAL_CORE):
    """One chord interval per rounded edge; exact26patch sphere-topology mesh."""
    vertices=[];lookup={};faces=[]
    def vertex(axis, signs):
        point=tuple(signs[i]*(core[i]+(RADIUS if i==axis else 0)) for i in range(3))
        if point not in lookup:lookup[point]=len(vertices);vertices.append(point)
        return lookup[point]
    def face(ids):
        a,b,c=(np.array([float(x) for x in vertices[i]]) for i in ids)
        if np.dot(np.cross(b-a,c-a),a+b+c)<0:ids=(ids[0],ids[2],ids[1])
        faces.append(ids)
    for axis in range(3):
        other=[i for i in range(3) if i!=axis]
        for sign in (-1,1):
            ids=[]
            for first,second in ((-1,-1),(1,-1),(1,1),(-1,1)):
                signs=[0,0,0];signs[axis]=sign;signs[other[0]]=first;signs[other[1]]=second
                ids.append(vertex(axis,signs))
            face((ids[0],ids[1],ids[2]));face((ids[0],ids[2],ids[3]))
    for axes in combinations(range(3),2):
        other=next(i for i in range(3) if i not in axes)
        for pair in product((-1,1),repeat=2):
            ids=[]
            for axis,end in ((axes[0],-1),(axes[0],1),(axes[1],1),(axes[1],-1)):
                signs=[0,0,0];signs[axes[0]]=pair[0];signs[axes[1]]=pair[1];signs[other]=end
                ids.append(vertex(axis,signs))
            face((ids[0],ids[1],ids[2]));face((ids[0],ids[2],ids[3]))
    for signs in product((-1,1),repeat=3):face(tuple(vertex(axis,signs) for axis in range(3)))
    return np.array(vertices,float),np.array(faces,np.int64)


def proof(v,f,core=IDEAL_CORE,**options):
    return _proof(v,f,core=core,ideal_core=IDEAL_CORE,radius=RADIUS,edge_intervals=1,**options)


class RoundedBoxReferenceTests(unittest.TestCase):
    def test_complete_portable26patch_cover_and_positive_normals(self):
        v,f=coarse_fixture();r=proof(v,f)
        self.assertEqual((len(v),len(f)),(24,44));self.assertEqual(r['complete_analytic_patches'],26)
        counts={kind:sum(p['kind']==kind for p in r['patch_cover']) for kind in
                ('planar_rectangle','cylindrical_strip','spherical_octant')}
        self.assertEqual(counts,{'planar_rectangle':6,'cylindrical_strip':12,'spherical_octant':8})
        self.assertTrue(all(p['complete_nonoverlapping_cover'] for p in r['patch_cover']))
        self.assertTrue(r['positive_actual_normal_cones']);self.assertLess(r['maximum_normal_angle_degrees'],90)
        self.assertFalse(r['sampled']);self.assertIsNone(r['artist_limits']);self.assertFalse(r['aggregate_accepted'])

    def test_continuous_corner_interior_bound_not_just_exact_surface_vertices(self):
        q=[(RADIUS,F(0),F(0)),(F(0),RADIUS,F(0)),(F(0),F(0),RADIUS)]
        lower,upper=_norm_range(q)
        self.assertEqual(upper,RADIUS**2);self.assertEqual(lower,RADIUS**2/3)
        v,f=coarse_fixture();r=proof(v,f)
        actual_centroid_deficit=float(RADIUS)*(1-1/np.sqrt(3))
        self.assertGreaterEqual(r['maximum_source_facet_distance_world'],actual_centroid_deficit)
        self.assertLess(r['maximum_source_facet_distance_world'],actual_centroid_deficit+1e-12)

    def test_exact_cover_rejects_equal_total_area_with_overlap_and_missing_region(self):
        a,b,c,d=(F(0),F(0)),(F(1),F(0)),(F(1),F(1)),(F(0),F(1))
        polygon=[a,b,c,d]
        self.assertTrue(_cover([(a,b,c),(a,c,d)],polygon)['complete_nonoverlapping_cover'])
        # Both inputs have the correct total area1 but cover one half twice.
        with self.assertRaisesRegex(ValueError,'overlap|duplicate|boundary'):_cover([(a,b,c),(a,b,c)],polygon)
        with self.assertRaisesRegex(ValueError,'boundary'):_cover([(a,b,c)],polygon)

    def test_missing_source_face_and_duplicate_source_face_refused(self):
        v,f=coarse_fixture()
        with self.assertRaisesRegex(ValueError,'closed') :proof(v,f[:-1])
        with self.assertRaisesRegex(ValueError,'closed') :proof(v,np.vstack((f,f[0])))

    def test_closed_reversed_source_is_not_outward(self):
        v,f=coarse_fixture()
        with self.assertRaisesRegex(ValueError,'outward'):proof(v,f[:,[0,2,1]])

    def test_two_ring_inventory_not_an_angular_vertex_cloud(self):
        v,f=coarse_fixture()
        with self.assertRaisesRegex(ValueError,'angular lattice'):
            _proof(v,f,core=IDEAL_CORE,ideal_core=IDEAL_CORE,radius=RADIUS,edge_intervals=2)

    def test_construction_core_shift_is_separate_and_enclosed(self):
        core=tuple(c+F('0.00000003') for c in IDEAL_CORE)
        v,f=coarse_fixture(core);r=proof(v,f,core=core)
        expected=3e-8*np.sqrt(3)
        self.assertGreaterEqual(r['native_to_ideal_core_shift_world'],expected)
        self.assertLess(r['native_to_ideal_core_shift_world'],expected+1e-20)
        self.assertGreaterEqual(r['construction_distance_bound_world'],expected)
        self.assertGreaterEqual(r['affine_construction_shift_bound_world'],expected)
        self.assertTrue(r['radial_bound_includes_vertex_construction'])
        self.assertGreaterEqual(r['maximum_vertex_radial_construction_error_world'],0)
        self.assertGreater(r['maximum_source_facet_distance_world'],r['radial_facet_distance_bound_world'])

    def test_directed_arithmetic_verified_with_rational_square(self):
        for x in (F(0),F(2),F(1,7),F(1,10**30)):
            self.assertLessEqual(_sqrt(x,False)**2,x);self.assertGreaterEqual(_sqrt(x,True)**2,x)
        with self.assertRaises(ValueError):_sqrt(F(-1),True)

    def test_public_refuses_different_source_and_forged_identity(self):
        v,f=coarse_fixture();arrays=GeometryArrays.capture(v,f)
        r=rounded_box_reference_certificate(arrays)
        self.assertEqual(r['status'],'unsupported');self.assertIn('original frozen',r['reason'])
        forged=SimpleNamespace(vertices=v,faces=f,content_hash=FROZEN_SOURCE_HASH)
        r=rounded_box_reference_certificate(forged)
        self.assertEqual(r['status'],'unsupported');self.assertIsNone(r['artist_limits'])
        self.assertFalse(r['candidate_boundary_qualified'])


if __name__=='__main__':unittest.main()
