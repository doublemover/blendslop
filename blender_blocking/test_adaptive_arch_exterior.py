"""Separate exterior arch controls, concave projection and real receipt binding."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
import numpy as np
from primitives.shape_program import ShapeNode, ShapeProgram
from reconstruction.native_geometry import GeometryArrays
from reconstruction.adaptive_arch_exterior import (
    arch_exterior_controls, arch_notch_dimensions, arch_exterior_update,
    arch_exterior_mesh, arch_exterior_boundary, arch_exterior_signed_distance,
    arch_exterior_candidate_sensitivity, refine_arch_exterior,
)


def runner_module():
    path=Path(__file__).resolve().parents[1]/'scripts/run_adaptive_arch_exterior.py'
    spec=importlib.util.spec_from_file_location('arch_exterior_fixture',path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    return runner


class AdaptiveArchExteriorTests(unittest.TestCase):
    def fixture(self, size=192):
        p={'outer':[[-.9,-.9],[-.9,.9],[.9,.9],[.9,-.9],[.5,-.9],[.5,.4],[-.5,.4],[-.5,-.9]],
           'holes':[],'width_world':1.8,'depth_world':1.8,'height_world':.4,
           'rotation':[[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]], 'x':0.,'y':0.,'z':0.,'preserved_style':'flat'}
        wire=ShapeProgram('1','arch',(ShapeNode('same-source','add','polygon_extrusion',parameters=p),),
                          metadata={'old_ob145':'previously inspected'}).to_dict()
        matrix=[[ -.5735766887664795,-.3845685124397278,.7232682108879089,4.530552864074707],
                [.8191518783569336,-.26927801966667175,.5064380764961243,3.1723272800445557],
                [9.926917954317105e-8,.882947564125061,.46947160363197327,2.940770149230957],
                [0.,0.,0.,1.]]
        camera={'projection':'ORTHO','matrix_world':matrix,'ortho_scale':2.67,
                'resolution':[size,size],'pixel_aspect':[1.,1.],'shift_x':0.,'shift_y':0.}
        return wire,camera

    def coverage(self, wire, camera):
        size=camera['resolution'][0];scale=camera['ortho_scale']
        yy,xx=np.mgrid[:size,:size]
        xy=np.column_stack(((xx.ravel()+.5-size/2)*scale/size,(size/2-yy.ravel()-.5)*scale/size))
        distance=arch_exterior_signed_distance(xy,wire,camera)
        return np.clip(.5-distance.reshape(size,size)/(scale/size),0.,1.)

    def fit(self, wire, camera, coverage, **kwargs):
        controls=arch_exterior_controls(wire)
        return refine_arch_exterior(wire,{'fit':{'coverage':coverage,'camera':camera}},
            parameter_bounds={k:[v*.9,v*1.1] for k,v in controls.items()},
            heldout_views=('ob145',),prior_view_exposure={'ob145':'old protocol already inspected pixels'},**kwargs)

    def test_dimension_convention_preserves_notch_moves_absolute_roof_and_pose(self):
        wire,_=self.fixture();old=wire['root_nodes'][0]['parameters']
        edited=arch_exterior_update(wire,{'outer_width_world':1.9,'outer_height_world':1.92,
                                         'extrusion_depth_world':.42}).to_dict()
        new=edited['root_nodes'][0]['parameters'];outline=np.asarray(new['outer'])
        self.assertEqual(arch_notch_dimensions(edited),arch_notch_dimensions(wire))
        self.assertAlmostEqual(outline[5,1],.34)
        self.assertAlmostEqual(outline[0,1],-.96)
        self.assertEqual((new['width_world'],new['depth_world'],new['height_world']),(1.9,1.92,.42))
        for key in ('rotation','x','y','z','preserved_style'):self.assertEqual(old[key],new[key])
        self.assertEqual(edited['metadata'],wire['metadata'])
        self.assertEqual(arch_exterior_update(wire,arch_exterior_controls(wire)).to_dict(),wire)
        for value in arch_exterior_controls(edited).values():self.assertIs(type(value),float)

    def test_illegal_notch_crossing_pose_or_control_is_rejected(self):
        wire,_=self.fixture()
        for change in ({'outer_width_world':.99},{'outer_height_world':1.29},
                       {'opening_width_world':1.1},{'extrusion_depth_world':True},
                       {'extrusion_depth_world':float('nan')}):
            with self.assertRaises(ValueError):arch_exterior_update(wire,change)
        bad=deepcopy(wire);bad['root_nodes'][0]['parameters']['outer'][1][0]-=.01
        with self.assertRaisesRegex(ValueError,'rectangular'):arch_exterior_controls(bad)

    def test_actual_discrete_extrusion_keeps_notch_and_discards_internal_seams(self):
        wire,camera=self.fixture();camera['matrix_world']=[[1.,0.,0.,0.],[0.,0.,-1.,-4.],[0.,1.,0.,0.],[0.,0.,0.,1.]]
        shape,meta=arch_exterior_boundary(wire,camera,return_metadata=True)
        self.assertLess(shape.area,shape.convex_hull.area)
        self.assertEqual(meta['projection_representation'],'mesh_triangle_union')
        distance=arch_exterior_signed_distance(np.array([[0.,0.],[.7,0.],[0.,.7]]),wire,camera)
        self.assertGreater(distance[0],0.);self.assertLess(distance[1],0.);self.assertLess(distance[2],0.)
        before=arch_exterior_mesh(wire);after=arch_exterior_mesh(arch_exterior_update(wire,{'extrusion_depth_world':.41}).to_dict())
        self.assertEqual((len(before.vertices),len(before.faces)),(16,20))
        self.assertEqual(before.faces,after.faces)

    def test_candidate_ob35_has_three_controls_while_front_has_two(self):
        wire,camera=self.fixture();front=deepcopy(camera)
        front['matrix_world']=[[1.,0.,0.,0.],[0.,0.,-1.,-4.],[0.,1.,0.,0.],[0.,0.,0.,1.]]
        ob=arch_exterior_candidate_sensitivity(wire,{'ob35':camera},heldout_views=('ob145',))
        self.assertEqual(ob['local_rank'],3)
        self.assertFalse(ob['observations']['ob35']['candidate_frame_censored'])
        self.assertEqual(ob['source_observations'],0)
        self.assertEqual(arch_exterior_candidate_sensitivity(wire,{'front':front})['local_rank'],2)
        with self.assertRaisesRegex(ValueError,'held-outs'):
            arch_exterior_candidate_sensitivity(wire,{'ob145':camera},heldout_views=('ob145',))

    def test_bounded_observed_three_control_update_preserves_inner_dimensions(self):
        wire,camera=self.fixture()
        old_detail={'protocol':'bounded_discrete_family_detail_v1',
                    'selected_controls':{'opening_width_world':1.,'cavity_roof_height_world':1.3},
                    'observations':{'front':{'coverage_sha256':'unchanged old observation'}}}
        wire['metadata']['adaptive_detail']=deepcopy(old_detail)
        truth=arch_exterior_update(wire,{'outer_width_world':1.81,'outer_height_world':1.806,'extrusion_depth_world':.415}).to_dict()
        result=self.fit(wire,camera,self.coverage(truth,camera),max_evaluations=96,max_elapsed_s=1.,
                        baseline_geometry_hash='a'*64,baseline_program_sha256='b'*64)
        detail=result.metadata['adaptive_arch_exterior']
        for key,value in arch_exterior_controls(truth).items():
            self.assertAlmostEqual(arch_exterior_controls(result.to_dict())[key],value,delta=.012)
        self.assertEqual(detail['local_rank'],3);self.assertEqual(detail['identifiability'],'locally_identified')
        self.assertLessEqual(detail['residual_calls'],96)
        self.assertEqual(arch_notch_dimensions(wire),arch_notch_dimensions(result.to_dict()))
        self.assertFalse(detail['global_uniqueness_established']);self.assertFalse(detail['heldout_fit_or_roi_used'])
        self.assertIn('absolute notch roof',detail['parameter_convention']['outer_height_coupling'])
        self.assertNotIn('adaptive_detail',result.metadata)
        history=result.metadata['historical_adaptive_stages'][0]
        self.assertEqual(history['record'],old_detail)
        self.assertEqual(history['baseline_geometry_hash'],'a'*64)
        self.assertEqual(history['baseline_program_file_sha256'],'b'*64)
        self.assertFalse(history['observations_apply_to_current_geometry'])
        self.assertEqual(wire['metadata']['adaptive_detail'],old_detail)
        self.assertEqual(detail['baseline_stage_provenance']['geometry_hash'],'a'*64)
        with self.assertRaisesRegex(ValueError,'original baseline'):
            self.fit(wire,camera,self.coverage(truth,camera))

    def test_cropped_single_edge_stays_censored_and_underconstrained(self):
        wire,camera=self.fixture(96);camera['matrix_world']=[[1.,0.,0.,.9],[0.,0.,-1.,-4.],[0.,1.,0.,0.],[0.,0.,0.,1.]]
        camera['ortho_scale']=.12
        truth=arch_exterior_update(wire,{'outer_width_world':1.81}).to_dict()
        result=self.fit(wire,camera,self.coverage(truth,camera),max_evaluations=96,max_elapsed_s=1.)
        detail=result.metadata['adaptive_arch_exterior']
        self.assertTrue(detail['observations']['fit']['frame_censored'])
        self.assertFalse(detail['observations']['fit']['complete_contour_claim'])
        self.assertEqual(detail['local_rank'],1);self.assertEqual(detail['identifiability'],'underconstrained')

    def test_heldout_observation_and_invalid_budget_never_reach_solver(self):
        wire,camera=self.fixture();controls=arch_exterior_controls(wire)
        kwargs={'parameter_bounds':{k:[v*.9,v*1.1] for k,v in controls.items()},
                'heldout_views':('ob145',),'prior_view_exposure':{'ob145':'exposed'}}
        for n,t in ((True,1.),(96.,1.),(193,1.),(8,0.),(8,float('nan'))):
            with self.assertRaises(ValueError):refine_arch_exterior(wire,{},max_evaluations=n,max_elapsed_s=t,**kwargs)
        with self.assertRaisesRegex(ValueError,'reserved'):
            refine_arch_exterior(wire,{'ob145':{}},**kwargs)

    def test_native_depth_response_checks_fixed_xy_and_faces_separately(self):
        wire,_=self.fixture();runner=runner_module()
        def arrays(wire):
            mesh=arch_exterior_mesh(wire)
            faces=np.array([(f[0],f[i],f[i+1]) for f in mesh.faces for i in range(1,len(f)-1)])
            return GeometryArrays.capture(mesh.vertices,faces)
        before=arrays(wire);edited=arrays(arch_exterior_update(wire,{'extrusion_depth_world':.42}).to_dict())
        pose=np.eye(4);pose[:3,:3]=wire['root_nodes'][0]['parameters']['rotation']
        self.assertTrue(runner.depth_response(before,edited,pose)['passed'])
        bad=arrays(arch_exterior_update(wire,{'extrusion_depth_world':.42,'outer_width_world':1.81}).to_dict())
        self.assertFalse(runner.depth_response(before,bad,pose)['passed'])

    def test_refined_owner_binding_rejects_changed_recipe_and_raw_or_owned_bytes(self):
        runner=runner_module();wire,_=self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            owner=Path(temp).resolve();directory=owner/'concave_arch/refined';directory.mkdir(parents=True)
            mesh=arch_exterior_mesh(wire);faces=np.array([(f[0],f[i],f[i+1]) for f in mesh.faces for i in range(1,len(f)-1)])
            arrays=GeometryArrays.capture(mesh.vertices,faces)
            np.savez(directory/'evaluated-exact.npz',vertices=arrays.vertices,faces=arrays.faces)
            (directory/'evaluated.obj').write_text('owned obj fixture')
            (directory/'program.json').write_text(json.dumps(wire))
            raw={'candidate_geometry_hash':arrays.content_hash,'reference_geometry_hash':'source',
                 'sample_count_per_direction':4096,'seed':61007}
            receipt={'status':'measured','cases':{'concave_arch':{'status':'measured','program':wire,
                'refined_geometry_hash':arrays.content_hash,'refined_raw_surface':raw}}}
            (owner/'results.json').write_text(json.dumps(receipt))
            paths=['results.json','concave_arch/refined/program.json','concave_arch/refined/evaluated-exact.npz','concave_arch/refined/evaluated.obj']
            manifest={'state':'succeeded','run_root':str(owner),'run_id':'owned','owner_token':'token',
                      'artifacts':[{'path':p,'sha256':runner.sha(owner/p)} for p in paths]}
            (owner/'run-ownership.json').write_text(json.dumps(manifest))
            (owner/'run-lease.json').write_text(json.dumps({'status':'released','run_id':'owned','owner_token':'token'}))
            item={'baseline_directory':str(directory),'baseline_receipt':str(owner/'results.json'),
                  'baseline_geometry_hash':arrays.content_hash,'baseline_raw_surface':raw,
                  'source_geometry_hash':'source','program_sha256':runner.sha(directory/'program.json')}
            self.assertFalse(runner.verify_refined_baseline(item,wire)['historical_adoption'])
            bad=deepcopy(item);bad['baseline_raw_surface']['seed']=9
            with self.assertRaises(ValueError):runner.verify_refined_baseline(bad,wire)
            (directory/'evaluated.obj').write_text('changed old owned obj')
            with self.assertRaisesRegex(ValueError,'artifact bytes changed'):runner.verify_refined_baseline(item,wire)


if __name__=='__main__':unittest.main()
