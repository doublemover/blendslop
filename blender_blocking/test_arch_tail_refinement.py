"""Focused roof-only physical controls and conservative observed-tail fitting."""
from copy import deepcopy
import unittest

import numpy as np

from primitives.shape_program import ShapeNode, ShapeProgram
from reconstruction.adaptive_arch_exterior import arch_exterior_mesh, arch_exterior_signed_distance
from reconstruction.arch_tail_refinement import arch_roof_controls, arch_roof_update, refine_arch_roof


def arch_fixture(size=256):
    params = {'outer':[[-.9,-.9],[-.9,.9],[.9,.9],[.9,-.9],
                       [.5,-.9],[.5,.35],[-.5,.35],[-.5,-.9]],
              'holes':[], 'width_world':1.8, 'depth_world':1.8, 'height_world':.4,
              'rotation':[[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]], 'x':0., 'y':0., 'z':0.}
    wire = ShapeProgram('1','arch',(ShapeNode('same','add','polygon_extrusion',parameters=params),),
                        metadata={'adaptive_arch_exterior':{'old_ob145':'already inspected'}}).to_dict()
    camera = {'projection':'ORTHO', 'matrix_world':[[1.,0.,0.,0.],[0.,0.,-1.,-4.],
               [0.,1.,0.,0.],[0.,0.,0.,1.]], 'ortho_scale':2.5,
              'resolution':[size,size], 'pixel_aspect':[1.,1.], 'shift_x':0., 'shift_y':0.}
    return wire, camera


def coverage(wire, camera, distance):
    size=camera['resolution'][0]; step=camera['ortho_scale']/size
    yy,xx=np.mgrid[:size,:size]
    xy=np.column_stack(((xx.ravel()+.5-size/2)*step,(size/2-yy.ravel()-.5)*step))
    return np.clip(.5-distance(xy,wire,camera).reshape(size,size)/step,0.,1.)


class ArchRoofTailTests(unittest.TestCase):
    def fit(self, wire, observations, **kwargs):
        return refine_arch_roof(wire, observations,
            parameter_bounds={'notch_height_from_bottom_world':[1.1,1.4]},
            heldout_views=('ob145',), prior_view_exposure={'ob145':'prior selection exposed'},
            baseline_geometry_hash='a'*64, baseline_program_sha256='b'*64, **kwargs)

    def test_roof_update_changes_only_inner_roof_and_restores_exact_mesh(self):
        wire,_=arch_fixture(); original=deepcopy(wire)
        old=arch_exterior_mesh(wire)
        changed=arch_roof_update(wire,{'notch_height_from_bottom_world':1.3}).to_dict()
        p=changed['root_nodes'][0]['parameters']; before=wire['root_nodes'][0]['parameters']
        for key in set(before)-{'outer'}: self.assertEqual(p[key],before[key])
        a,b=np.array(before['outer']),np.array(p['outer'])
        self.assertTrue(np.array_equal(a[[0,1,2,3,4,7]],b[[0,1,2,3,4,7]]))
        self.assertAlmostEqual(b[5,1],.4); self.assertAlmostEqual(b[6,1],.4)
        restored=arch_roof_update(changed,arch_roof_controls(wire)).to_dict()
        mesh=arch_exterior_mesh(restored)
        self.assertTrue(np.array_equal(old.vertices,mesh.vertices));self.assertEqual(old.faces,mesh.faces)
        self.assertEqual(wire,original)
        self.assertEqual(arch_roof_update(wire,arch_roof_controls(wire)).to_dict(),wire)

    def test_observed_roof_fit_is_locally_identified_and_all_tail_scores_nonworse(self):
        wire,camera=arch_fixture()
        truth=arch_roof_update(wire,{'notch_height_from_bottom_world':1.3}).to_dict()
        alpha=coverage(truth,camera,arch_exterior_signed_distance)
        result=self.fit(wire,{'front':{'coverage':alpha,'camera':camera}})
        detail=result.metadata['arch_roof_tail']
        self.assertTrue(detail['observed_tail_admitted']);self.assertEqual(detail['local_rank'],1)
        self.assertAlmostEqual(arch_roof_controls(result.to_dict())['notch_height_from_bottom_world'],1.3,delta=.006)
        for key,value in detail['selected_observed_tail'].items():
            self.assertLessEqual(value,detail['baseline_observed_tail'][key]+1e-12)
        self.assertLessEqual(detail['residual_calls'],96)
        self.assertFalse(detail['heldout_fit_or_roi_used']);self.assertFalse(detail['global_uniqueness_established'])
        history=result.metadata['historical_adaptive_stages'][0]
        self.assertEqual(history['record'],wire['metadata']['adaptive_arch_exterior'])
        self.assertEqual(history['baseline_geometry_hash'],'a'*64)
        self.assertFalse(history['observations_apply_to_current_geometry'])
        self.assertEqual(detail['surface_tail'],'unrun')

    def test_cropped_outer_edge_cannot_identify_unobserved_roof(self):
        wire,camera=arch_fixture(96);camera['ortho_scale']=.16
        camera['matrix_world'][0][3]=.9;camera['matrix_world'][2][3]=.7
        alpha=coverage(wire,camera,arch_exterior_signed_distance)
        result=self.fit(wire,{'detail':{'coverage':alpha,'camera':camera}})
        detail=result.metadata['arch_roof_tail']
        self.assertTrue(detail['observations']['detail']['frame_censored'])
        self.assertFalse(detail['observations']['detail']['frame_border_closure_added'])
        self.assertEqual(detail['local_rank'],0);self.assertFalse(detail['observed_tail_admitted'])
        self.assertEqual(result.root_nodes[0].parameters,wire['root_nodes'][0]['parameters'])

    def test_reserved_views_and_invalid_bounds_coverage_or_budget_are_refused(self):
        wire,camera=arch_fixture(64)
        with self.assertRaisesRegex(ValueError,'reserved'):
            self.fit(wire,{'ob145':{}})
        for calls,seconds in ((True,1.),(96.,1.),(97,1.),(8,0.),(8,float('nan'))):
            with self.assertRaises(ValueError):self.fit(wire,{},max_evaluations=calls,max_elapsed_s=seconds)
        for value in (True,float('nan'),0.,2.):
            with self.assertRaises(ValueError):arch_roof_update(wire,{'notch_height_from_bottom_world':value})
        bad=np.zeros((64,64));bad[30,30]=float('nan')
        with self.assertRaisesRegex(ValueError,'linear alpha'):
            self.fit(wire,{'front':{'coverage':bad,'camera':camera}})


    def test_saved_checker_refuses_expanded_native_scope(self):
        import importlib.util
        from pathlib import Path
        spec=importlib.util.spec_from_file_location('tail_check_fixture',Path(__file__).resolve().parents[1]/'scripts/run_arch_triangle_tail_check.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        scope={'work_seconds':85,'join_seconds':5,'threads':2,'memory_limit_bytes':8*1024**3,
            'native_frames':7,'source_acquisitions':0,'candidate_alpha_frames':5,
            'retained_alpha_frames':5,'logical_frames':12,
            'candidate_neutral_frames':2,'native_fits':0,'raw_comparisons':2,
            'raw_count_per_direction':4096,'raw_seed':61007,'semantic_transactions':2,
            'qualification_children':2,'qualification_timeout_seconds':15,
            'resolution':[512,512],'heldout_views':['oblique_145_40']}
        plan={'protocol':runner.PROTOCOL,'scope':scope,'cases':[{'family':x} for x in runner.FAMILIES]}
        runner.validate_scope(plan)
        for key,value in [('source_acquisitions',1),('qualification_timeout_seconds',16),
                          ('native_frames',8),('native_fits',1),('raw_count_per_direction',8192)]:
            bad=deepcopy(plan);bad['scope'][key]=value
            with self.assertRaises(ValueError):runner.validate_scope(bad)

    def test_saved_checker_requires_exact_declared_body_change_and_observed_admission(self):
        import importlib.util
        from pathlib import Path
        spec=importlib.util.spec_from_file_location('tail_check_body_fixture',Path(__file__).resolve().parents[1]/'scripts/run_arch_triangle_tail_check.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        wire,_=arch_fixture();proposed=arch_roof_update(wire,{'notch_height_from_bottom_world':1.3}).to_dict()
        proposed['metadata']['arch_roof_tail']={'observed_tail_admitted':True,'identifiability':'locally_identified',
            'interval_bound_active':False,'heldout_fit_or_roi_used':False,'heldout_views':['oblique_145_40'],
            'baseline_program_file_sha256':'a'*64,'selected_controls':{'notch_height_from_bottom_world':1.3},
            'free_controls':['notch_height_from_bottom_world'],'local_rank':1,
            'selected_observed_tail':{'mean_absolute_source_pixels':.1},
            'baseline_observed_tail':{'mean_absolute_source_pixels':.2}}
        runner.require_recipe_update(wire,proposed,'concave_arch')
        for field,value in [('local_rank',None),('observed_tail_admitted',False),('heldout_fit_or_roi_used',True)]:
            bad=deepcopy(proposed);bad['metadata']['arch_roof_tail'][field]=value
            with self.assertRaises(ValueError):runner.require_recipe_update(wire,bad,'concave_arch')
        bad=deepcopy(proposed);bad['root_nodes'][0]['parameters']['height_world']=.41
        with self.assertRaisesRegex(ValueError,'unobserved'):runner.require_recipe_update(wire,bad,'concave_arch')

if __name__ == '__main__': unittest.main()
