"""Projected opaque union carriers without native rendering or qualification."""
import importlib.util
import unittest
import numpy as np
from reconstruction.differentiable.projected_mesh_rays import projected_boundary_carriers


@unittest.skipUnless(importlib.util.find_spec('shapely'),'optional Shapely geometry runtime')
class ProjectedMeshRayTests(unittest.TestCase):
    def test_overlap_resolves_to_one_opaque_union_with_intersection_carriers(self):
        vertices=np.array([[-.7,-.4],[.5,-.4],[0.,.6],[-.5,.1],[.7,.1],[.2,-.7]])
        faces=np.array([[0,1,2],[3,4,5]])
        points,edges,carriers,report=projected_boundary_carriers(vertices,faces)
        self.assertEqual(report['components'],1)
        self.assertTrue(any(row[0]=='intersection' for row in carriers))
        self.assertLess(report['maximum_carrier_reconstruction_error'],1e-14)
        self.assertEqual(len(points),len(edges))
        self.assertEqual(set(edges[:,0]),set(edges[:,1]))

    def test_hole_winding_remains_negative_inside_the_outer_loop(self):
        vertices=np.array([[-.8,-.8],[.8,-.8],[.8,.8],[-.8,.8],[-.3,-.3],[.3,-.3],[.3,.3],[-.3,.3]])
        faces=np.array([[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
        points,edges,carriers,report=projected_boundary_carriers(vertices,faces)
        self.assertEqual(report['loops'],2)
        area=sum(points[a,0]*points[b,1]-points[a,1]*points[b,0] for a,b in edges)*.5
        self.assertAlmostEqual(area,2.56-.36)
        self.assertTrue(all(row[0]=='vertex' for row in carriers))

    def test_degenerate_projection_and_face_overflow_fail_closed(self):
        vertices=np.array([[0.,0.],[.2,0.],[.4,0.]])
        with self.assertRaises(ValueError):projected_boundary_carriers(vertices,np.array([[0,1,2]]))
        with self.assertRaises(ValueError):projected_boundary_carriers(vertices,np.tile([[0,1,2]],(2049,1)))

    def test_real_projected_gap_is_not_welded(self):
        gap=2.**-35
        vertices=np.array([[-.5,0.],[0.,0.],[0.,.5],[gap,0.],[.5,0.],[gap,.5]])
        points,edges,carriers,report=projected_boundary_carriers(vertices,np.array([[0,1,2],[3,4,5]]))
        self.assertEqual(report['components'],2)
        self.assertTrue(any(np.array_equal(point,[gap,0.]) for point in points))

    def test_quality_mode_keeps_explicit_volume_ray_alternative(self):
        from reconstruction.quality_config import quality_config
        self.assertEqual(quality_config({'quality_preset':'quality'})['dvx_objective'],'observed_projected_rays')
        self.assertEqual(quality_config({'quality_preset':'quality','dvx_objective':'observed_rays'})['dvx_objective'],'observed_rays')


if __name__=='__main__':unittest.main()
