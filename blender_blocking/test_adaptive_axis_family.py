"""Standalone axis-family control, discrete support and view-rank contracts."""
import unittest
from copy import deepcopy
import math
import numpy as np
from primitives.shape_program import ShapeNode, ShapeProgram
from reconstruction.adaptive_axis_family import (
    axis_family_control_values, axis_family_program_update, axis_family_local_vertices,
    axis_projected_family_boundary, axis_projected_signed_distance, axis_candidate_view_sensitivity,
)


def fixture(family):
    primitive = {'sphere':'ellipsoid','anisotropic_ellipsoid':'ellipsoid',
                 'cylinder':'cylinder','thin_plate':'box'}[family]
    dims = {'sphere':(1.6,1.599,1.598),'anisotropic_ellipsoid':(1.8,1.1,1.4),
            'cylinder':(1.3,1.3,1.6),'thin_plate':(1.6,.08,1.2)}[family]
    p = dict(zip(('width_world','depth_world','height_world'),dims))
    p.update(x=.03,y=-.02,z=.04,rotation=np.eye(3).tolist(),source_style='preserved')
    if family=='cylinder':p.update(radius_bottom=.65,radius_top=.65)
    return ShapeProgram('1','observed-'+family,(ShapeNode('retained','add',primitive,parameters=p),),
                        metadata={'prior_history':'old views exposed'}).to_dict()


def camera(view='front'):
    matrix = np.eye(4)
    if view=='front':matrix[:3,:3] = [[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]]
    elif view=='oblique':
        az, el = math.radians(35), math.radians(28)
        normal = np.array([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)])
        right = np.array([-math.sin(az),math.cos(az),0.])
        matrix[:3,:3] = np.column_stack((right,np.cross(normal,right),normal))
    return {'projection':'ORTHO','matrix_world':matrix.tolist(),'ortho_scale':2.6,
            'resolution':[512,512],'pixel_aspect':[1.,1.],'shift_x':0.,'shift_y':0.}


class AdaptiveAxisFamilyTests(unittest.TestCase):
    def test_coupled_sphere_preserves_retained_axis_ratios_pose_and_style(self):
        wire=fixture('sphere');old=wire['root_nodes'][0]['parameters']
        radius=axis_family_control_values(wire,'sphere')['radius_world']
        updated=axis_family_program_update(wire,'sphere',{'radius_world':np.float64(radius*1.05)}).to_dict()
        p=updated['root_nodes'][0]['parameters']
        for key in ('width_world','depth_world','height_world'):
            self.assertAlmostEqual(p[key]/old[key],1.05,places=14)
        for key in ('x','y','z','rotation','source_style'):self.assertEqual(p[key],old[key])
        self.assertEqual(updated['metadata'],wire['metadata'])
        self.assertEqual(axis_family_program_update(wire,'sphere',{'radius_world':radius}).to_dict(),wire)
        with self.assertRaises(ValueError):axis_family_program_update(wire,'sphere',{'depth_world':1.7})

    def test_cylinder_radius_updates_both_caps_and_diameter_aliases(self):
        wire=fixture('cylinder')
        p=axis_family_program_update(wire,'cylinder',{'radius_world':.68}).root_nodes[0].parameters
        self.assertEqual(p['radius_bottom'],p['radius_top'])
        self.assertEqual(p['width_world'],1.36);self.assertEqual(p['depth_world'],1.36)
        self.assertEqual(p['height_world'],1.6)
        bad=deepcopy(wire);bad['root_nodes'][0]['parameters']['radius_top']=.64
        with self.assertRaisesRegex(ValueError,'equal cap'):axis_family_control_values(bad,'cylinder')

    def test_dimension_updates_preserve_every_unselected_recipe_parameter(self):
        for family in ('anisotropic_ellipsoid','thin_plate'):
            wire=fixture(family);p=wire['root_nodes'][0]['parameters']
            result=axis_family_program_update(wire,family,{'depth_world':p['depth_world']*1.05}).to_dict()
            new=result['root_nodes'][0]['parameters']
            for key in p:
                if key!='depth_world':self.assertEqual(new[key],p[key])
            self.assertIs(type(new['depth_world']),float)

    def test_existing_uv_mesh_and_cylinder_end_ring_support_are_retained(self):
        from primitives.analytic_primitives import EllipsoidPrimitive
        from reconstruction.adaptive_family import _hull_polygon,_model_camera
        wire=fixture('anisotropic_ellipsoid');p=wire['root_nodes'][0]['parameters']
        direct=EllipsoidPrimitive(radii=[p[k]/2 for k in ('width_world','depth_world','height_world')]).to_mesh_data(96).vertices
        modeled=axis_family_local_vertices(wire,'anisotropic_ellipsoid',segments=96)
        self.assertEqual(len(modeled),4514)
        np.testing.assert_allclose(modeled,direct,rtol=0,atol=2e-16)
        wire=fixture('cylinder');p=wire['root_nodes'][0]['parameters'];c=camera('oblique');matrix,_=_model_camera(c)
        angle=np.arange(96)*2*np.pi/96
        rings=np.concatenate([np.column_stack((.65*np.cos(angle),.65*np.sin(angle),np.full(96,z))) for z in (-.8,.8)])
        rings+=np.array([p[k] for k in ('x','y','z')])
        expected=_hull_polygon(rings,matrix)
        self.assertLess(axis_projected_family_boundary(wire,'cylinder',c).hausdorff_distance(expected),1e-14)
        self.assertEqual(len(axis_family_local_vertices(fixture('thin_plate'),'thin_plate')),8)

    def test_front_has_one_or_two_controls_and_complement_exposes_depth(self):
        expected={'sphere':1,'cylinder':2,'anisotropic_ellipsoid':2,'thin_plate':2}
        for family,rank in expected.items():
            with self.subTest(family=family):
                result=axis_candidate_view_sensitivity(fixture(family),family,{'front':camera()})
                self.assertEqual(result['local_rank'],rank)
                self.assertEqual(result['source_observations'],0)
                self.assertIsNone(result['artist_surface_limits'])
        for family in ('anisotropic_ellipsoid','thin_plate'):
            result=axis_candidate_view_sensitivity(fixture(family),family,{'ob35':camera('oblique')})
            self.assertEqual(result['local_rank'],3)
            self.assertGreater(result['normalized_interval_pixel_sensitivity']['depth_world'],.1)
            self.assertFalse(result['global_uniqueness_established'])

    def test_censored_single_plate_edge_cannot_invent_missing_dimensions(self):
        wire=fixture('thin_plate');p=wire['root_nodes'][0]['parameters']
        c=camera();matrix=np.array(c['matrix_world']);matrix[:3,3]=[p['x']+.8,p['y'],p['z']]
        c={**c,'matrix_world':matrix.tolist(),'ortho_scale':.2}
        result=axis_candidate_view_sensitivity(wire,'thin_plate',{'partial':c})
        self.assertEqual(result['local_rank'],1)
        record=result['observations']['partial']
        self.assertTrue(record['candidate_frame_censored'])
        self.assertFalse(record['frame_border_closure_added'])
        self.assertFalse(record['source_complete_contour_claim'])

    def test_heldout_view_choice_and_bad_control_camera_contracts_are_refused(self):
        wire=fixture('thin_plate')
        with self.assertRaises(ValueError):axis_candidate_view_sensitivity(wire,'thin_plate',{'held':camera()},heldout_views=('held',))
        for value in (True,0.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):axis_family_program_update(wire,'thin_plate',{'depth_world':value})
        with self.assertRaises(ValueError):axis_family_local_vertices(wire,'thin_plate',segments=True)
        with self.assertRaises(ValueError):axis_projected_family_boundary(wire,'thin_plate',{**camera(),'shift_x':.1})
        bad=deepcopy(wire);bad['root_nodes'][0]['parameters']['rotation'][0][0]=-1
        with self.assertRaises(ValueError):axis_family_control_values(bad,'thin_plate')

    def coverage(self, wire, family, c):
        from reconstruction.adaptive_family import projected_family_signed_distance
        size=c['resolution'][0];yy,xx=np.mgrid[:size,:size];scale=c['ortho_scale']
        points=np.column_stack(((xx.ravel()+.5-size/2)*scale/size,(size/2-yy.ravel()-.5)*scale/size))
        d=projected_family_signed_distance(points,wire,family,c)
        return np.clip(.5-d.reshape(size,size)/(scale/size),0.,1.)

    def test_integrated_bounded_updater_recovers_all_four_models_with_fixed_pose(self):
        from reconstruction.adaptive_family import family_control_values,family_program_update,refine_family_detail
        changes={'sphere':{'radius_world':.82},'cylinder':{'radius_world':.67,'height_world':1.62},
                 'anisotropic_ellipsoid':{'width_world':1.82,'depth_world':1.12,'height_world':1.38},
                 'thin_plate':{'width_world':1.63,'depth_world':.088,'height_world':1.23}}
        for family,values in changes.items():
            with self.subTest(family=family):
                wire=fixture(family);c=camera('oblique' if family in ('anisotropic_ellipsoid','thin_plate') else 'front')
                c={**c,'resolution':[192,192]}
                truth=family_program_update(wire,family,values).to_dict()
                old=family_control_values(wire,family)
                updated=refine_family_detail(wire,family,{'fit':{'coverage':self.coverage(truth,family,c),'camera':c}},
                    parameter_bounds={key:[old[key]*.8,old[key]*1.2] for key in values},
                    heldout_views=('held',),prior_view_exposure={'held':'old view exposure explicitly retained'},
                    max_evaluations=128,max_elapsed_s=3.).to_dict()
                actual=family_control_values(updated,family)
                for key,value in values.items():self.assertAlmostEqual(actual[key],value,delta=.004)
                for key in ('x','y','z','rotation','source_style'):
                    self.assertEqual(updated['root_nodes'][0]['parameters'][key],wire['root_nodes'][0]['parameters'][key])
                detail=updated['metadata']['adaptive_detail']
                self.assertEqual(detail['local_rank'],len(values))
                self.assertEqual(detail['identifiability'],'locally_identified')
                self.assertFalse(detail['heldout_fit_or_roi_used'])
                self.assertFalse(detail['global_uniqueness_established'])
                self.assertIsNone(detail['artist_surface_limits'])
                self.assertTrue(detail['tessellation']['fixed_for_update'])
                self.assertLessEqual(detail['residual_calls'],128)

    def test_integrated_censored_contour_keeps_rank_and_missing_dimensions_unqualified(self):
        from reconstruction.adaptive_family import family_program_update,refine_family_detail
        wire=fixture('thin_plate');p=wire['root_nodes'][0]['parameters'];c=camera()
        matrix=np.array(c['matrix_world']);matrix[:3,3]=[p['x']+.8,p['y'],p['z']]
        c={**c,'matrix_world':matrix.tolist(),'ortho_scale':.2,'resolution':[96,96]}
        truth=family_program_update(wire,'thin_plate',{'width_world':1.62}).to_dict()
        updated=refine_family_detail(wire,'thin_plate',{'partial':{'coverage':self.coverage(truth,'thin_plate',c),'camera':c}},
            parameter_bounds={'width_world':[1.5,1.7],'depth_world':[.07,.09],'height_world':[1.1,1.3]},
            heldout_views=('held',),prior_view_exposure={'held':'untouched by this synthetic fitting fixture'},
            max_evaluations=128,max_elapsed_s=3.)
        detail=updated.metadata['adaptive_detail']
        self.assertEqual(detail['local_rank'],1)
        self.assertEqual(detail['identifiability'],'underconstrained')
        self.assertTrue(detail['observations']['partial']['frame_censored'])
        self.assertFalse(detail['observations']['partial']['complete_contour_claim'])

    def test_external_hull_signed_distance_ignores_internal_mesh_seams(self):
        wire=fixture('thin_plate');p=wire['root_nodes'][0]['parameters']
        result=axis_projected_signed_distance(np.array([[p['x'],p['z']],[p['x']+2.,p['z']]]),wire,'thin_plate',camera())
        self.assertLess(result[0],0.);self.assertGreater(result[1],0.)


if __name__=='__main__':unittest.main()
