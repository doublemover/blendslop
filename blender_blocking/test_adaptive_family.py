"""Discrete adaptive family contours, fixed controls and held-out isolation."""
from copy import deepcopy
from pathlib import Path
import unittest
import json
import numpy as np
from primitives.shape_program import ShapeNode, ShapeProgram
from reconstruction.adaptive_family import (
    adaptive_crop_request, family_control_values, family_program_update,
    projected_family_signed_distance, refine_family_detail,
)


class AdaptiveFamilyTests(unittest.TestCase):
    def fixture(self, family):
        p = {'x': .03, 'y': -.02, 'z': .04, 'rotation': np.eye(3).tolist()}
        primitive = {'capsule': 'capsule', 'tapered_frustum': 'frustum',
                     'torus': 'torus', 'concave_arch': 'polygon_extrusion'}[family]
        if family == 'capsule':
            p.update(radius_world=.4, segment_height_world=.9, width_world=.8, depth_world=.8, height_world=2*.4+.9)
        elif family == 'tapered_frustum':
            p.update(radius_bottom=.7, radius_top=.3, width_world=1.4, depth_world=1.4, height_world=1.6)
        elif family == 'torus':
            p.update(major_radius=.7, minor_radius=.2)
        else:
            p.update(outer=[[-.9,-.8],[-.9,.8],[.9,.8],[.9,-.8],
                            [.45,-.8],[.45,.3],[-.45,.3],[-.45,-.8]],
                     holes=[], width_world=1.8, depth_world=1.6, height_world=.2)
        p['source_style'] = 'preserved'
        wire = ShapeProgram('1', 'retained-'+family,
                            (ShapeNode('original', 'add', primitive, parameters=p),),
                            metadata={'original_heldout_history': 'previously inspected'}).to_dict()
        matrix = np.eye(4)
        if family in ('capsule', 'tapered_frustum'):
            matrix[:3,:3] = np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
        camera = {'projection': 'ORTHO', 'matrix_world': matrix.tolist(), 'ortho_scale': 2.4,
                  'resolution': [96,96], 'pixel_aspect': [1.,1.], 'shift_x': 0., 'shift_y': 0.}
        return wire, camera

    def coverage(self, wire, family, camera):
        size = camera['resolution'][0]
        yy,xx = np.mgrid[:size,:size]
        scale = camera['ortho_scale']
        uv = np.column_stack(((xx.ravel()+.5-size/2)*scale/size,
                              (size/2-yy.ravel()-.5)*scale/size))
        distance = projected_family_signed_distance(uv, wire, family, camera)
        return np.clip(.5-distance.reshape(size,size)/(scale/size),0.,1.)

    def fit(self, wire, family, source, camera, bounds):
        return refine_family_detail(wire,family,{'detail': {'coverage': source,'camera': camera}},
            parameter_bounds=bounds,heldout_views=('validation',),
            prior_view_exposure={'validation':'old pixels inspected before this local protocol'},
            max_evaluations=128,max_elapsed_s=3.).to_dict()

    def test_all_four_models_recover_observed_controls_without_pose_changes(self):
        changes = {'capsule': {'radius_world': .42},
                   'tapered_frustum': {'radius_top': .32},
                   'torus': {'major_radius': .72,'minor_radius': .19},
                   'concave_arch': {'opening_width_world': .94,'cavity_roof_height_world': 1.13}}
        for family, values in changes.items():
            with self.subTest(family=family):
                wire,camera = self.fixture(family)
                truth = family_program_update(wire,family,values).to_dict()
                controls = family_control_values(wire,family)
                bounds = {name:[controls[name]*.85,controls[name]*1.15] for name in values}
                updated = self.fit(wire,family,self.coverage(truth,family,camera),camera,bounds)
                actual = family_control_values(updated,family)
                for name,value in values.items(): self.assertAlmostEqual(actual[name],value,delta=.006)
                for name in ('x','y','z','rotation','source_style'):
                    self.assertEqual(updated['root_nodes'][0]['parameters'][name],wire['root_nodes'][0]['parameters'][name])
                detail=updated['metadata']['adaptive_detail']
                self.assertLessEqual(detail['residual_calls'],128)
                self.assertFalse(detail['global_uniqueness_established'])
                self.assertFalse(detail['heldout_fit_or_roi_used'])
                self.assertEqual(detail['prior_view_exposure']['validation'],
                    'old pixels inspected before this local protocol')

    def test_partial_ring_arc_has_no_complete_boundary_or_unique_two_radius_claim(self):
        wire,camera=self.fixture('torus')
        matrix=np.eye(4);matrix[:3,3]=[.55,0.,0.]
        camera={**camera,'matrix_world':matrix.tolist(),'ortho_scale':.35}
        truth=family_program_update(wire,'torus',{'minor_radius':.21}).to_dict()
        updated=self.fit(wire,'torus',self.coverage(truth,'torus',camera),camera,
                         {'major_radius':[.65,.75],'minor_radius':[.16,.24]})
        detail=updated['metadata']['adaptive_detail']
        self.assertTrue(detail['observations']['detail']['frame_censored'])
        self.assertFalse(detail['observations']['detail']['complete_contour_claim'])
        self.assertEqual(detail['identifiability'],'underconstrained')
        self.assertEqual(detail['local_rank'],1)

    def test_arch_update_fixes_exterior_and_has_no_internal_mesh_seam_distance(self):
        wire,camera=self.fixture('concave_arch')
        updated=family_program_update(wire,'concave_arch',{'opening_width_world':1.,
                                                        'cavity_roof_height_world':1.2}).to_dict()
        old=np.asarray(wire['root_nodes'][0]['parameters']['outer'])
        new=np.asarray(updated['root_nodes'][0]['parameters']['outer'])
        np.testing.assert_array_equal(old[:4],new[:4])
        self.assertEqual(updated['root_nodes'][0]['parameters']['height_world'],.2)
        distances=projected_family_signed_distance(np.array([[.03,-.02],[.03,.68],[.73,-.02]]),updated,'concave_arch',camera)
        self.assertGreater(distances[0],0.)
        self.assertLess(distances[1],0.)
        self.assertLess(distances[2],0.)

    def test_dimension_aliases_follow_semantic_controls_and_restore_exactly(self):
        for family in ('capsule','tapered_frustum','torus','concave_arch'):
            wire,_=self.fixture(family);controls=family_control_values(wire,family)
            key=next(iter(controls));edited=family_program_update(wire,family,{key:controls[key]*1.05}).to_dict()
            restored=family_program_update(edited,family,controls).to_dict()
            self.assertEqual(restored['root_nodes'],wire['root_nodes'])
            self.assertEqual(restored['metadata'],wire['metadata'])
        capsule,_=self.fixture('capsule')
        p=family_program_update(capsule,'capsule',{'radius_world':.45}).root_nodes[0].parameters
        self.assertEqual((p['width_world'],p['depth_world'],p['height_world']),(.9,.9,1.8))

    def test_heldout_pixels_are_rejected_before_fit_or_roi_selection(self):
        wire,camera=self.fixture('torus');image=self.coverage(wire,'torus',camera)
        with self.assertRaisesRegex(ValueError,'reserved'):
            refine_family_detail(wire,'torus',{'validation':{'coverage':image,'camera':camera}},
                parameter_bounds={'minor_radius':[.16,.24]},heldout_views=('validation',),
                prior_view_exposure={'validation':'exposed'})
        with self.assertRaisesRegex(ValueError,'reserved'):
            adaptive_crop_request(image,image,camera,fit_view='validation',
                                  heldout_views=('validation',),prior_view_exposure={'validation':'exposed'},pixel_span=32)
        with self.assertRaisesRegex(ValueError,'history'):
            adaptive_crop_request(image,image,camera,fit_view='top',heldout_views=('validation',),
                                  prior_view_exposure={},pixel_span=32)

    def test_invalid_intervals_budget_and_ring_basis_fail_explicitly(self):
        wire,camera=self.fixture('torus');image=self.coverage(wire,'torus',camera)
        kwargs={'parameter_bounds':{'minor_radius':[.16,.24]},'heldout_views':(), 'prior_view_exposure':{}}
        for n,t in [(True,1.),(96.,1.),(193,1.),(7,1.),(96,0.),(96,float('nan'))]:
            with self.assertRaises(ValueError):
                refine_family_detail(wire,'torus',{},max_evaluations=n,max_elapsed_s=t,**kwargs)
        for bounds in ({'minor_radius':[.21,.24]},{'unknown':[.1,.4]}, {'minor_radius':[-.1,.4]}):
            with self.assertRaises(ValueError):
                refine_family_detail(wire,'torus',{'detail':{'coverage':image,'camera':camera}},
                    parameter_bounds=bounds,heldout_views=(),prior_view_exposure={})
        invalid=deepcopy(camera);invalid['matrix_world']=[[1.,0.,0.,0.],[0.,0.,-1.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]]
        with self.assertRaisesRegex(ValueError,'local axis'):
            projected_family_signed_distance(np.array([[0.,0.]]),wire,'torus',invalid)

    def test_fixed_images_do_not_request_manufactured_new_truth(self):
        wire,camera=self.fixture('torus');image=self.coverage(wire,'torus',camera)
        result=adaptive_crop_request(image,image,camera,fit_view='top',
            heldout_views=('validation',),prior_view_exposure={'validation':'previously inspected'})
        self.assertEqual(result['status'],'unsupported')
        self.assertIn('fixed images',result['reason'])
        result=adaptive_crop_request(image,image,camera,fit_view='top',
            heldout_views=('validation',),prior_view_exposure={'validation':'previously inspected'},
            source_geometry_available=True,pixel_span=32)
        self.assertEqual(result['status'],'unneeded')

    def triangle_fixture(self):
        p={'scale_xy':[.985,.99],'corner_radius_world':.165,
           'height_world':.48,'front_fraction':.505,'corner_segments':32,'dome_segments':64,
           'x':.003,'y':-.002,'z':.005,'rotation':np.eye(3).tolist()}
        wire=ShapeProgram('1','triangle-retained',(ShapeNode('triangle','add','rounded_triangle',parameters=p),),
                          metadata={'baseline_oblique_history':'previously inspected'}).to_dict()
        camera={'projection':'ORTHO','matrix_world':np.eye(4).tolist(),'ortho_scale':2.8,
                'resolution':[128,128],'pixel_aspect':[1.,1.]}
        return wire,camera

    def test_triangle_three_control_fit_retains_depth_pose_and_declared_tessellation(self):
        family='rounded_triangle_dot';wire,camera=self.triangle_fixture()
        truth=family_program_update(wire,family,{'scale_x':1.,'scale_y':1.,'corner_radius_world':.16}).to_dict()
        controls=family_control_values(wire,family)
        bounds={key:[value*.9,value*1.1] for key,value in controls.items()}
        updated=self.fit(wire,family,self.coverage(truth,family,camera),camera,bounds)
        actual=family_control_values(updated,family)
        self.assertAlmostEqual(actual['scale_x'],1.,delta=.003)
        self.assertAlmostEqual(actual['scale_y'],1.,delta=.003)
        self.assertAlmostEqual(actual['corner_radius_world'],.16,delta=.003)
        old=wire['root_nodes'][0]['parameters'];new=updated['root_nodes'][0]['parameters']
        self.assertEqual({k:v for k,v in old.items() if k not in ('scale_xy','corner_radius_world')},
                         {k:v for k,v in new.items() if k not in ('scale_xy','corner_radius_world')})
        detail=updated['metadata']['adaptive_detail']
        self.assertEqual(detail['local_rank'],3)
        self.assertEqual(detail['identifiability'],'locally_identified')
        self.assertEqual(detail['tessellation'],{'corner_segments':32,'dome_segments':64,'fixed_from_recipe':True})
        self.assertIsNone(detail['segments'])
        self.assertFalse(detail['global_uniqueness_established'])
        self.assertIs(type(new['corner_radius_world']),float)
        self.assertTrue(all(type(x) is float for x in new['scale_xy']))

    def test_triangle_scale_aliases_cannot_be_shadowed_and_depth_is_not_a_free_control(self):
        wire,camera=self.triangle_fixture()
        with self.assertRaisesRegex(ValueError,'undeclared'):
            family_program_update(wire,'rounded_triangle_dot',{'height_world':.5})
        wire['root_nodes'][0]['parameters']['width_world']=2.
        with self.assertRaisesRegex(ValueError,'overrides'):
            family_control_values(wire,'rounded_triangle_dot')

    def test_triangle_cached_template_matches_direct_declared_mesh_and_is_bounded(self):
        from primitives.rounded_triangle import RoundedTrianglePrimitive
        from reconstruction.adaptive_family import _triangle_local_vertices,_triangle_vertex_basis
        wire,_=self.triangle_fixture();original=wire['root_nodes'][0]['parameters']
        for radius,sx,sy in [(.143,.9,1.1),(.19,1.03,.98),(.16,1.,1.)]:
            params={**original,'corner_radius_world':radius,'scale_xy':[sx,sy]}
            direct=RoundedTrianglePrimitive.from_program_parameters(params,world=False).to_mesh_data().vertices
            cached=_triangle_local_vertices(json.dumps(params))
            np.testing.assert_allclose(cached,direct,rtol=0,atol=5e-14)
        for front in np.linspace(.4,.6,10):
            _triangle_local_vertices(json.dumps({**original,'front_fraction':float(front)}))
        self.assertLessEqual(_triangle_vertex_basis.cache_info().currsize,8)

    def test_triangle_projection_uses_recipe_tessellation_without_raising_resolution(self):
        from primitives.rounded_triangle import RoundedTrianglePrimitive
        wire,camera=self.triangle_fixture()
        before=RoundedTrianglePrimitive.from_program_parameters(wire['root_nodes'][0]['parameters'],world=False).to_mesh_data()
        updated=family_program_update(wire,'rounded_triangle_dot',{'scale_x':1.01,'corner_radius_world':.17}).to_dict()
        after=RoundedTrianglePrimitive.from_program_parameters(updated['root_nodes'][0]['parameters'],world=False).to_mesh_data()
        self.assertEqual((len(before.vertices),len(before.faces)),(len(after.vertices),len(after.faces)))
        self.assertEqual(len(before.vertices),6239)
        distance=projected_family_signed_distance(np.array([[0.,0.],[3.,3.]]),updated,'rounded_triangle_dot',camera,segments=48)
        self.assertLess(distance[0],0.)
        self.assertGreater(distance[1],0.)

    def test_numpy_control_values_have_builtin_json_semantic_response(self):
        wire,_=self.fixture('capsule')
        updated=family_program_update(wire,'capsule',{'segment_height_world':np.float64(.93)})
        p=updated.root_nodes[0].parameters
        self.assertIs(type(p['segment_height_world']),float)
        expected=.05*p['segment_height_world']
        response={'checks':{'axial_response':abs(.0465-expected)<=1e-5}}
        decoded=json.loads(json.dumps({'program':updated.to_dict(),'response':response},allow_nan=False))
        self.assertTrue(decoded['response']['checks']['axial_response'])
        self.assertAlmostEqual(decoded['program']['root_nodes'][0]['parameters']['segment_height_world'],.93)

    def test_native_receipt_transport_handles_numpy_bool_without_accepting_nan(self):
        import importlib.util
        path=Path(__file__).resolve().parents[1]/'scripts/run_adaptive_family_detail.py'
        spec=importlib.util.spec_from_file_location('adaptive_receipt_fixture',path)
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        encoded=runner.encoded_json({'response':{'passed':np.bool_(True),'ratio':np.float64(1.05)}})
        self.assertEqual(json.loads(encoded),{'response':{'passed':True,'ratio':1.05}})
        with self.assertRaises(ValueError):runner.encoded_json({'metric':np.float64(float('nan'))})

    def test_failure_case_transport_preserves_primary_error(self):
        import importlib.util
        path=Path(__file__).resolve().parents[1]/'scripts/run_adaptive_family_detail.py'
        spec=importlib.util.spec_from_file_location('adaptive_receipt_failure_fixture',path)
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        receipt=runner.failure_receipt(RuntimeError('primary geometry failure'),'frozen',
                                       {'capsule':{'invalid_value':object()}},6)
        decoded=json.loads(runner.encoded_json(receipt))
        self.assertEqual(decoded['reason'],'RuntimeError: primary geometry failure')
        self.assertEqual(decoded['completed_case_keys'],['capsule'])
        self.assertIn('unsupported receipt value',decoded['case_transport_error'])
        self.assertEqual(decoded['rendered_frames'],6)

    def test_budget_stop_retains_best_recipe_and_unqualified_sensitivity(self):
        wire,camera=self.fixture('torus')
        truth=family_program_update(wire,'torus',{'major_radius':.72,'minor_radius':.19}).to_dict()
        updated=refine_family_detail(wire,'torus',{'detail':{'coverage':self.coverage(truth,'torus',camera),'camera':camera}},
            parameter_bounds={'major_radius':[.6,.8],'minor_radius':[.15,.25]},
            heldout_views=(),prior_view_exposure={},max_evaluations=8)
        detail=updated.metadata['adaptive_detail']
        self.assertLessEqual(detail['residual_calls'],8)
        self.assertEqual(detail['termination'],'evaluation_allowance')
        self.assertEqual(detail['identifiability'],'underconstrained')
        self.assertIsNone(detail['artist_surface_limits'])


if __name__=='__main__':
    unittest.main()
