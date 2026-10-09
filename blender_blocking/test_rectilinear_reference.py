"""Complete rectilinear facet cover, orientation and numerical source bounds."""
import unittest
import numpy as np
from reconstruction.native_geometry import GeometryArrays
from evaluation.rectilinear_reference import rectilinear_reference_certificate


class RectilinearReferenceTests(unittest.TestCase):
    def plate(self):
        vertices=np.array([[-.8,-.04,-.6],[.8,-.04,-.6],[.8,.04,-.6],[-.8,.04,-.6],
                           [-.8,-.04,.6],[.8,-.04,.6],[.8,.04,.6],[-.8,.04,.6]])
        faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],
                        [0,1,5],[0,5,4],[1,2,6],[1,6,5],
                        [2,3,7],[2,7,6],[3,0,4],[3,4,7]])
        return vertices,faces

    def certify(self,vertices,faces):
        return rectilinear_reference_certificate('thin_plate',GeometryArrays.capture(vertices,faces))

    def volume(self,vertices,faces):
        p=vertices[faces]
        return np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])).sum()/6

    def test_exact_plate_all_filled_facets_have_zero_discretization_distance(self):
        vertices,faces=self.plate();result=self.certify(vertices,faces)
        self.assertEqual(result['status'],'certified')
        self.assertEqual(result['source_triangles_proved'],12)
        self.assertEqual(result['boundary_patches_proved'],6)
        self.assertEqual(result['maximum_source_facet_distance_world'],0.)
        self.assertEqual(result['normal_tangent_upper_bound'],0.)
        self.assertEqual(result['reference_geometry_hash'],GeometryArrays.capture(vertices,faces).content_hash)
        self.assertIsNone(result['artist_surface_limits'])

    def test_overlap_and_gap_cannot_pass_even_with_same_vertices_area_and_volume(self):
        vertices,faces=self.plate();bad=faces.copy();bad[3]=bad[2]
        self.assertAlmostEqual(self.volume(vertices,faces),self.volume(vertices,bad),places=14)
        self.assertEqual(set(np.unique(faces)),set(np.unique(bad)))
        result=self.certify(vertices,bad)
        self.assertEqual(result['status'],'unsupported')
        self.assertIn('positive-area overlap',result['reason'])

    def test_missing_face_refuses_incomplete_boundary_patch(self):
        vertices,faces=self.plate();result=self.certify(vertices,np.delete(faces,3,axis=0))
        self.assertEqual(result['status'],'unsupported')
        self.assertIn('complete nonoverlapping',result['reason'])

    def test_wrong_orientation_and_internal_plane_are_refused(self):
        vertices,faces=self.plate();bad=faces.copy();bad[0]=bad[0,::-1]
        self.assertIn('outward',self.certify(vertices,bad)['reason'])
        wrong=vertices.copy();wrong[0,0]=0.
        result=self.certify(wrong,faces)
        self.assertEqual(result['status'],'unsupported')
        self.assertIn('lattice association',result['reason'])

    def test_rounding_bound_covers_continuous_facets_and_outward_normal_cone(self):
        vertices,faces=self.plate();rounded=vertices.astype(np.float32).astype(float)
        rounded[0]+=[1e-8,-1e-8,1e-8]
        result=self.certify(rounded,faces)
        self.assertEqual(result['status'],'certified')
        self.assertGreater(result['maximum_source_facet_distance_world'],0.)
        self.assertGreater(result['normal_tangent_upper_bound'],0.)
        self.assertGreaterEqual(result['maximum_source_facet_distance_world'],np.linalg.norm(rounded-vertices,axis=1).max())
        weights=np.array([.07,.39,.54])
        for face in faces:
            shift=np.linalg.norm(weights@(rounded[face]-vertices[face]))
            self.assertLessEqual(shift,result['maximum_source_facet_distance_world'])

    def test_actual_arrays_cannot_use_a_stale_or_forged_source_hash(self):
        from types import SimpleNamespace
        vertices,faces=self.plate()
        result=rectilinear_reference_certificate('thin_plate',
            SimpleNamespace(vertices=vertices,faces=faces,content_hash='not-the-actual-geometry'))
        self.assertEqual(result['status'],'unsupported')
        self.assertIn('identity',result['reason'])

    def test_large_coordinate_drift_and_unknown_families_stay_unsupported(self):
        vertices,faces=self.plate();vertices[0,0]+=1e-4
        self.assertEqual(self.certify(vertices,faces)['status'],'unsupported')
        result=rectilinear_reference_certificate('rounded_box',GeometryArrays.capture(*self.plate()))
        self.assertEqual(result['status'],'unsupported')
        self.assertIn('unsupported',result['reason'])


if __name__=='__main__':unittest.main()
