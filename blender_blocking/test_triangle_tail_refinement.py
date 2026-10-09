"""Finite-mesh triangle pose/depth controls, observability and tail retention."""
from copy import deepcopy
import unittest

import numpy as np

from primitives.rounded_triangle import RoundedTrianglePrimitive
from primitives.shape_program import ShapeNode, ShapeProgram
from reconstruction.adaptive_family import _hull_polygon, _model_camera
from reconstruction.triangle_tail_refinement import (
    triangle_pose_controls, triangle_pose_update, triangle_pose_boundary,
    triangle_pose_signed_distance, refine_triangle_pose,
)
from test_arch_tail_refinement import coverage


def triangle_fixture(size=256):
    params={'height_world':.48,'front_fraction':.55,'corner_radius_world':.16,
            'scale_xy':[1.,1.],'corner_segments':32,'dome_segments':64,
            'rotation':np.eye(3).tolist(),'x':0.,'y':.02,'z':.02}
    wire=ShapeProgram('1','triangle',(ShapeNode('same','add','rounded_triangle',parameters=params),),
        metadata={'adaptive_detail':{'prior_ob145':'already viewed'}}).to_dict()
    base={'projection':'ORTHO','ortho_scale':2.5,'resolution':[size,size],
          'pixel_aspect':[1.,1.],'shift_x':0.,'shift_y':0.}
    top={**base,'matrix_world':np.eye(4).tolist()}
    front={**base,'matrix_world':[[1.,0.,0.,0.],[0.,0.,-1.,-4.],[0.,1.,0.,0.],[0.,0.,0.,1.]]}
    return wire,{'top':top,'front':front}


class TrianglePoseTailTests(unittest.TestCase):
    def fit(self, wire, observations, **kwargs):
        return refine_triangle_pose(wire,observations,
            parameter_bounds={'center_y_world':[-.04,.05],'center_z_world':[-.04,.05],'front_fraction':[.4,.65]},
            heldout_views=('ob145',),prior_view_exposure={'ob145':'previous actual admission exposed'},
            baseline_geometry_hash='c'*64,baseline_program_sha256='d'*64,**kwargs)

    def test_projected_boundary_is_actual_full_template_for_asymmetric_depth(self):
        wire,cameras=triangle_fixture();p=wire['root_nodes'][0]['parameters']
        mesh=RoundedTrianglePrimitive.from_program_parameters(p).to_mesh_data()
        self.assertEqual(len(mesh.vertices),6239);self.assertEqual(len(mesh.faces),12474)
        for camera in cameras.values():
            matrix,_=_model_camera(camera)
            expected=_hull_polygon(mesh.vertices,matrix)
            observed=triangle_pose_boundary(wire,camera)
            self.assertLess(expected.symmetric_difference(observed).area,1e-12)
            np.testing.assert_allclose(expected.bounds,observed.bounds,rtol=0.,atol=1e-12)

    def test_three_updates_keep_outline_thickness_orientation_and_exact_restoration(self):
        wire,_=triangle_fixture();original=deepcopy(wire);p=wire['root_nodes'][0]['parameters']
        changed=triangle_pose_update(wire,{'center_y_world':-.01,'center_z_world':.005,'front_fraction':.48}).to_dict()
        q=changed['root_nodes'][0]['parameters']
        for key in set(p)-{'y','z','front_fraction'}:self.assertEqual(q[key],p[key])
        before=RoundedTrianglePrimitive.from_program_parameters(p).to_mesh_data()
        restored=triangle_pose_update(changed,triangle_pose_controls(wire)).to_dict()
        after=RoundedTrianglePrimitive.from_program_parameters(restored['root_nodes'][0]['parameters']).to_mesh_data()
        self.assertTrue(np.array_equal(before.vertices,after.vertices));self.assertEqual(before.faces,after.faces)
        self.assertEqual(wire,original)
        self.assertEqual(triangle_pose_update(wire,triangle_pose_controls(wire)).to_dict(),wire)
        for change in ({'front_fraction':True},{'front_fraction':1.},{'center_z_world':float('inf')},{'height_world':.5}):
            with self.assertRaises(ValueError):triangle_pose_update(wire,change)

    def test_complementary_observed_front_and_top_identify_three_controls(self):
        wire,cameras=triangle_fixture()
        truth=triangle_pose_update(wire,{'center_y_world':.003,'center_z_world':-.005,'front_fraction':.5}).to_dict()
        observations={view:{'coverage':coverage(truth,camera,triangle_pose_signed_distance),'camera':camera}
                      for view,camera in cameras.items()}
        result=self.fit(wire,observations);detail=result.metadata['triangle_pose_tail']
        self.assertEqual(detail['local_rank'],3);self.assertTrue(detail['observed_tail_admitted'])
        controls=triangle_pose_controls(result.to_dict())
        self.assertAlmostEqual(controls['center_y_world'],.003,delta=.005)
        self.assertAlmostEqual(controls['center_z_world'],-.005,delta=.005)
        self.assertAlmostEqual(controls['front_fraction'],.5,delta=.01)
        for key,value in detail['selected_observed_tail'].items():
            self.assertLessEqual(value,detail['baseline_observed_tail'][key]+1e-12)
        self.assertLessEqual(detail['residual_calls'],96)
        self.assertFalse(detail['heldout_fit_or_roi_used']);self.assertFalse(detail['global_uniqueness_established'])
        self.assertEqual(result.metadata['historical_adaptive_stages'][0]['record'],wire['metadata']['adaptive_detail'])
        self.assertEqual(detail['surface_tail'],'unrun')

    def test_top_alone_is_rank_deficient_and_cannot_promote_unseen_depth(self):
        wire,cameras=triangle_fixture();camera=cameras['top']
        truth=triangle_pose_update(wire,{'center_y_world':.003}).to_dict()
        result=self.fit(wire,{'top':{'coverage':coverage(truth,camera,triangle_pose_signed_distance),'camera':camera}})
        detail=result.metadata['triangle_pose_tail']
        self.assertEqual(detail['local_rank'],1);self.assertFalse(detail['observed_tail_admitted'])
        self.assertEqual(result.root_nodes[0].parameters,wire['root_nodes'][0]['parameters'])
        with self.assertRaisesRegex(ValueError,'reserved'):
            self.fit(wire,{'ob145':{}})


if __name__ == '__main__': unittest.main()
