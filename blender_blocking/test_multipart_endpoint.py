"""Actual union boundaries and isolated observed multipart endpoint updates."""
import unittest
import numpy as np
from reconstruction.multipart_endpoint import boxes_from_program, projected_union_signed_distance, refine_short_far_endpoint
from primitives.shape_program import ShapeNode, ShapeProgram


class MultipartEndpointTests(unittest.TestCase):
    def fixture(self, far=.3):
        boxes = [(np.array([-1., -.45, -.5]), np.array([1., .45, .1])),
                 (np.array([-.8, -.12, .04]), np.array([-.25, .325, 1.35])),
                 (np.array([.3, -.3, .04]), np.array([.8, far, .8]))]
        nodes = []
        for i, (lower, upper) in enumerate(boxes):
            nodes.append(ShapeNode('node'+str(i), 'add', 'box', parameters={
                **dict(zip(('x','y','z'), ((lower+upper)/2).tolist())),
                **dict(zip(('width_world','depth_world','height_world'), (upper-lower).tolist()))}))
        wire = ShapeProgram('1', 'saved', tuple(nodes), metadata={'identifiability': {
            'original_short_far_interval': [-.2, .45]}, 'held_out_view': 'oblique_145_40'}).to_dict()
        az, el = np.deg2rad(-15.), np.deg2rad(20.)
        back = np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)])
        right = np.array([-np.sin(az),np.cos(az),0.]); up = np.cross(back,right)
        matrix = np.eye(4); matrix[:3,:3] = np.column_stack((right,up,back))
        camera = {'matrix_world':matrix.tolist(),'ortho_scale':3.2}
        return wire, boxes, camera

    def coverage(self, boxes, camera, size=96):
        yy,xx = np.mgrid[:size,:size]; scale=camera['ortho_scale']
        uv=np.column_stack(((xx.ravel()+.5-size/2)*scale/size,(size/2-yy.ravel()-.5)*scale/size))
        distance=projected_union_signed_distance(uv,boxes,camera)
        return np.clip(.5-distance.reshape(size,size)/(scale/size),0.,1.).astype(np.float32)

    def test_overlap_seams_are_not_union_boundaries(self):
        camera={'matrix_world':np.eye(4).tolist(),'ortho_scale':4.}
        boxes=[(np.array([-1.,-1.,0.]),np.array([.2,1.,1.])),
               (np.array([-.2,-1.,0.]),np.array([1.,1.,1.]))]
        distances=projected_union_signed_distance(np.array([[0.,0.],[.2,0.],[1.2,0.]]),boxes,camera)
        np.testing.assert_allclose(distances,[-1.,-.8,.2],atol=1e-12)

    def test_new_view_refines_only_far_endpoint_and_preserves_heldout_metadata(self):
        wire,boxes,camera=self.fixture(); boxes[2][1][1]=.11
        coverage=self.coverage(boxes,camera)
        updated=refine_short_far_endpoint(wire,coverage,camera).to_dict()
        self.assertEqual(updated['root_nodes'][:2],wire['root_nodes'][:2])
        old=wire['root_nodes'][2]['parameters']; new=updated['root_nodes'][2]['parameters']
        self.assertEqual({k:v for k,v in old.items() if k not in ('y','depth_world')},
                         {k:v for k,v in new.items() if k not in ('y','depth_world')})
        self.assertAlmostEqual(new['y']-new['depth_world']/2,old['y']-old['depth_world']/2,places=12)
        self.assertAlmostEqual(new['y']+new['depth_world']/2,.11,delta=.03)
        self.assertEqual(updated['metadata']['held_out_view'],'oblique_145_40')
        evidence=updated['metadata']['endpoint_refinement']
        self.assertLessEqual(evidence['residual_calls'],64)
        self.assertEqual(evidence['identifiability'],'locally_identified')
        self.assertFalse(evidence['heldout_oblique145_fit_used'])

    def test_occluded_endpoint_retains_interval_and_underconstrained_state(self):
        wire,boxes,camera=self.fixture()
        boxes[1][1][1]=.45
        p=wire['root_nodes'][1]['parameters']
        p['y']=(boxes[1][0][1]+.45)/2
        p['depth_world']=.45-boxes[1][0][1]
        boxes[2][1][1]=.11
        updated=refine_short_far_endpoint(wire,self.coverage(boxes,camera),camera)
        evidence=updated.metadata['endpoint_refinement']
        self.assertEqual(evidence['identifiability'],'underconstrained')
        self.assertEqual(evidence['original_interval'],[-.2,.45])
        self.assertEqual(updated.to_dict()['root_nodes'][:2],wire['root_nodes'][:2])

    def test_invalid_budget_and_holdout_fail_before_observation(self):
        for count,seconds in [(True,1.),(64.,1.),(65,1.),(7,1.),(64,1.1),(64,float('nan'))]:
            with self.subTest(count=count,seconds=seconds):
                with self.assertRaises(ValueError):
                    refine_short_far_endpoint({},None,{},max_evaluations=count,max_elapsed_s=seconds)
        with self.assertRaisesRegex(ValueError,'reserved'):
            refine_short_far_endpoint({},None,{},view='oblique_145_40')

    def test_interval_bound_keeps_local_uncertainty(self):
        wire,boxes,camera=self.fixture(); boxes[2][1][1]=.55
        updated=refine_short_far_endpoint(wire,self.coverage(boxes,camera),camera)
        evidence=updated.metadata['endpoint_refinement']
        self.assertEqual(evidence['original_interval'],[-.2,.45])
        self.assertTrue(evidence['interval_bound_active'])
        self.assertEqual(evidence['identifiability'],'underconstrained')


if __name__=='__main__':
    unittest.main()
