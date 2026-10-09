"""Observed multipart decomposition, held-out isolation and camera contracts."""
import unittest
import numpy as np

from reconstruction.multipart_family import (
    FIT_VIEWS, boxes_from_endpoints, fitted_multipart_program, observed_box_seed,
    polygon_signed_distance, projected_box_polygon, retained_multipart_program,
)
from test_frozen_family import cameras


class MultipartFamilyTests(unittest.TestCase):
    def observations(self, size=128, identifiable=True, short_far=.12):
        records=cameras(scale=4.)
        theta,phi=np.deg2rad(35.),np.deg2rad(28.)
        right=np.array([np.cos(theta),-np.sin(theta),0.])
        back=np.array([np.sin(theta)*np.cos(phi),np.cos(theta)*np.cos(phi),np.sin(phi)])
        up=np.cross(back,right)
        matrix=np.eye(4); matrix[:3,:3]=np.column_stack((right,up,back))
        records['oblique_35_28']={'matrix_world':matrix.tolist(),'ortho_scale':4.}
        if not identifiable: records['oblique_35_28']=records['front'].copy()
        boxes=[(np.array([-1.2,-.45,-.55]),np.array([1.2,.45,.2])),
               (np.array([-.94,-.15,.12]),np.array([-.28,.45,1.7])),
               (np.array([.38,-.42,.12]),np.array([.9,short_far,.95]))]
        y,x=np.mgrid[:size,:size]
        points=np.column_stack(((x.ravel()+.5-size/2)*4/size,(size/2-y.ravel()-.5)*4/size))
        coverages={}
        for view in FIT_VIEWS:
            distance=np.min([polygon_signed_distance(points,projected_box_polygon(a,b,records[view])) for a,b in boxes],axis=0)
            coverages[view]=np.clip(.5-distance.reshape(size,size)/(4/size),0.,1.)
        return {v:c>=.5 for v,c in coverages.items()},coverages,records,boxes

    def test_camera_projection_handles_noncanonical_basis_and_translation(self):
        _,_,records,boxes=self.observations(size=64)
        original=projected_box_polygon(*boxes[0],records['oblique_35_28'])
        shifted={**records['oblique_35_28']}; matrix=np.array(shifted['matrix_world'])
        displacement=np.array([.3,-.2,.7]); matrix[:3,3]+=displacement
        shifted['matrix_world']=matrix.tolist()
        actual=projected_box_polygon(*boxes[0],shifted)
        np.testing.assert_allclose(actual-original,np.broadcast_to(-displacement @ matrix[:3,:2],actual.shape),atol=1e-12)
        front=projected_box_polygon(*boxes[0],records['front'])
        np.testing.assert_allclose(front.min(axis=0),[-1.2,-.55],atol=1e-12)
        np.testing.assert_allclose(front.max(axis=0),[1.2,.2],atol=1e-12)

    def test_seed_recovers_exposed_plateaus_and_keeps_hidden_interval(self):
        _,coverage,records,boxes=self.observations()
        seed=observed_box_seed(coverage,records)
        np.testing.assert_allclose(seed['base_x'],[-1.2,1.2],atol=.02)
        self.assertAlmostEqual(seed['base_top'],.2,delta=.02)
        self.assertAlmostEqual(seed['tall_top'],1.7,delta=.02)
        self.assertAlmostEqual(seed['short_top'],.95,delta=.02)
        self.assertLess(seed['short_far_interval'][0],boxes[2][1][1])
        self.assertGreater(seed['short_far_interval'][1],boxes[2][1][1])
        parts=boxes_from_endpoints(seed,seed['initial'])
        self.assertEqual(len(parts),3)
        self.assertTrue(all((b>a).all() for a,b in parts))

    def test_oblique_fit_recovers_hidden_depth_without_reading_holdout(self):
        masks,coverage,records,boxes=self.observations()
        class Unreadable:
            def __array__(self,*args,**kwargs): raise AssertionError('held-out pixels read')
        masks['oblique_145_40']=Unreadable(); coverage['oblique_145_40']=Unreadable()
        records['oblique_145_40']={'do_not_serialize':Unreadable()}
        program=fitted_multipart_program(masks,records,coverage_masks=coverage)
        self.assertEqual(program.node_count(),3)
        self.assertEqual(program.metadata['fit_views'],list(FIT_VIEWS))
        self.assertLessEqual(program.metadata['support_evaluations'],160)
        self.assertEqual(program.metadata['identifiability']['status'],'locally_identified')
        params=program.root_nodes[2].parameters
        far=params['y']+params['depth_world']/2
        self.assertAlmostEqual(far,boxes[2][1][1],delta=.03)
        self.assertNotIn('reference_geometry',program.metadata)

    def test_cropped_oblique_recovers_visible_endpoint_without_border_closure(self):
        masks,coverage,records,boxes=self.observations()
        record=records['oblique_35_28']; matrix=np.array(record['matrix_world'])
        matrix[:3,3]+=matrix[:3,1]*1.15; record['matrix_world']=matrix.tolist()
        size=128; yy,xx=np.mgrid[:size,:size]
        points=np.column_stack(((xx.ravel()+.5-size/2)*4/size,(size/2-yy.ravel()-.5)*4/size))
        distances=np.min([polygon_signed_distance(points,projected_box_polygon(a,b,record)) for a,b in boxes],axis=0)
        coverage['oblique_35_28']=np.clip(.5-distances.reshape(size,size)/(4/size),0.,1.)
        masks['oblique_35_28']=coverage['oblique_35_28']>=.5
        self.assertTrue(masks['oblique_35_28'][-1].any())
        program=fitted_multipart_program(masks,records,coverage_masks=coverage)
        self.assertEqual(program.metadata['censored_views'],['oblique_35_28'])
        self.assertIn('no frame-edge closure',program.metadata['censored_contour_scope'])
        self.assertEqual(program.metadata['identifiability']['status'],'locally_identified')
        params=program.root_nodes[2].parameters
        self.assertAlmostEqual(params['y']+params['depth_world']/2,.12,delta=.03)

    def test_endpoint_at_observed_interval_bound_retains_uncertainty(self):
        masks,coverage,records,_=self.observations(short_far=.45)
        program=fitted_multipart_program(masks,records,coverage_masks=coverage)
        evidence=program.metadata['identifiability']
        self.assertEqual(evidence['status'],'underconstrained')
        self.assertTrue(evidence['short_far_interval_bound_active'])
        self.assertIn('retain original interval',evidence['uncertainty_reason'])
        self.assertGreater(evidence['original_short_far_interval'][1]-evidence['original_short_far_interval'][0],.4)

    def test_duplicate_oblique_cannot_qualify_hidden_endpoint(self):
        masks,coverage,records,_=self.observations(size=96,identifiable=False)
        program=fitted_multipart_program(masks,records,coverage_masks=coverage)
        evidence=program.metadata['identifiability']
        self.assertEqual(evidence['status'],'underconstrained')
        self.assertLess(evidence['local_rank'],4)
        self.assertGreater(evidence['original_short_far_interval'][1]-evidence['original_short_far_interval'][0],.4)

    def test_saved_three_leaf_replay_preserves_precision_and_rejects_unbounded_nodes(self):
        from primitives.shape_program import ShapeNode,ShapeProgram
        precision=.12345678912345678
        nodes=tuple(ShapeNode('n'+str(i),'add','box',parameters={'x':precision+i,'width_world':1.}) for i in range(3))
        wire=ShapeProgram('1','saved',nodes,metadata={'held_out_view':'oblique_145_40'}).to_dict()
        program=retained_multipart_program(wire)
        self.assertEqual(program.root_nodes[0].parameters['x'],precision)
        self.assertEqual(program.to_dict(),wire)
        with self.assertRaisesRegex(ValueError,'exactly three'):
            retained_multipart_program({**wire,'root_nodes':wire['root_nodes']*2})
        changed={**wire,'root_nodes':[dict(n) for n in wire['root_nodes']]}
        changed['root_nodes'][0]['operation']='subtract'
        with self.assertRaisesRegex(ValueError,'additive box leaves'):
            retained_multipart_program(changed)

    def test_invalid_budget_and_missing_step_scope_fail_before_fit(self):
        for count,seconds in [(True,3.),(160.,3.),(161,3.),(15,3.),(160,3.1),(160,float('nan'))]:
            with self.subTest(count=count,seconds=seconds):
                with self.assertRaises(ValueError):
                    fitted_multipart_program({}, {}, coverage_masks={},max_evaluations=count,max_elapsed_s=seconds)
        masks,coverage,records,_=self.observations(size=64)
        front=np.zeros((64,64)); front[12:54,10:54]=1.
        with self.assertRaisesRegex(ValueError,'exactly two observed raised plateaus'):
            observed_box_seed({**coverage,'front':front},records)


if __name__=='__main__':
    unittest.main()
