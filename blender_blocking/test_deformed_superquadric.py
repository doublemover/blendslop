"""Bounded deformation geometry checks; no native reconstruction score claims."""
from copy import deepcopy
from dataclasses import replace
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

from primitives.analytic_primitives import SuperquadricPrimitive
from primitives.deformed_superquadric import DeformedSuperquadricPrimitive as Deformed
from primitives.shape_program import ShapeNode, ShapeProgram
from reconstruction.program_transforms import reflect_parameters, rotation_matrix, position_vector
from reconstruction.proposal_screening import whole_program_geometry
from reconstruction.differentiable.target_adapter import primitive_from_renderable, renderable_from_primitive
from placement.resfit_parameters import discover_primitive_parameters, apply_parameter_increment, set_parameter_value
from placement.resfit_optimizer import ParameterBounds


class DeformedSuperquadricTests(unittest.TestCase):
    def part(self, **kwargs):
        params = dict(radii=(.6, .4, .9), center=(.2, -.1, .3),
                      rotation=Rotation.from_rotvec([.2, -.3, .1]).as_matrix(),
                      epsilon1=.7, epsilon2=1.3, taper_x=.2, taper_y=-.15, bend_angle=.5)
        params.update(kwargs)
        return Deformed(**params)

    def parameters(self):
        return dict(width_world=1.2, depth_world=.8, height_world=1.8,
                    location_x=.2, location_y=-.1, location_z=.3,
                    rotation_row_major=Rotation.from_rotvec([.2, -.3, .1]).as_matrix().ravel().tolist(),
                    epsilon1=.7, epsilon2=1.3, taper_x=.2, taper_y=-.15, bend_angle=.5)

    def program(self, params=None):
        return ShapeProgram('shape-program-v1', 'deform', (ShapeNode('part', 'add', 'deformed_superquadric',
                            self.parameters() if params is None else params),))

    def triangles(self, mesh):
        return np.array([(f[0], f[i], f[i+1]) for f in mesh.faces for i in range(1, len(f)-1)])

    def test_rigid_limit_preserves_base_surface_field_and_mesh(self):
        part=self.part(taper_x=0., taper_y=0., bend_angle=0.)
        params=part.to_dict(); base=SuperquadricPrimitive.from_dict(params)
        np.testing.assert_allclose(part.sample_surface(173), base.sample_surface(173), atol=0.)
        a,b=part.to_mesh_data(16),base.to_mesh_data(16)
        np.testing.assert_allclose(a.vertices,b.vertices,atol=0.)
        self.assertEqual(a.faces,b.faces)
        query=np.random.default_rng(3).normal(size=(75,3))
        np.testing.assert_allclose(part.sdf_batch(query),base.sdf_batch(query),atol=1e-14)

    def test_inverse_round_trip_at_signed_and_tiny_bends(self):
        rng=np.random.default_rng(9)
        for angle in (0., .5, -.5, 1e-10, -1e-10):
            part=self.part(bend_angle=angle)
            local=rng.uniform(-1.,1.,(200,3))*part.radii
            np.testing.assert_allclose(part.inverse_local(part.forward_local(local)), local,atol=2e-15)

    def test_rotated_surface_and_generated_vertices_share_field_zero_set(self):
        for angle in (-.5,.5):
            part=self.part(bend_angle=angle)
            np.testing.assert_allclose(part.sdf_batch(part.sample_surface(237)),0.,atol=3e-14)
            mesh=part.to_mesh_data(16)
            np.testing.assert_allclose(part.sdf_batch(mesh.vertices),0.,atol=3e-14)
            self.assertLess(part.sdf_batch(part.center[None,:])[0],0.)
            self.assertGreater(part.sdf_batch(np.array([[10.,10.,10.]]))[0],0.)

    def test_jacobian_matches_formula_and_conservative_positive_bound(self):
        for angle in (-.5,.5):
            part=self.part(bend_angle=angle); report=part.deformation_report()
            for p in np.random.default_rng(4).uniform(-.8,.8,(30,3))*part.radii:
                h=1e-6; jac=np.column_stack([(part.forward_local(p+np.eye(3)[i]*h)-
                                            part.forward_local(p-np.eye(3)[i]*h))/(2*h) for i in range(3)])
                z=p[2]/part.radii[2];fx=1+part.taper_x*z;fy=1+part.taper_y*z
                expected=fx*fy*(1-part.bend_angle*fx*p[0]/part.radii[2])
                self.assertAlmostEqual(np.linalg.det(jac),expected,places=8)
                self.assertGreaterEqual(expected,report['jacobian_determinant_lower_bound'])
                self.assertLessEqual(expected,report['jacobian_determinant_upper_bound'])
            self.assertGreater(report['jacobian_determinant_lower_bound'],0.)
            self.assertFalse(report['native_qualification'])

    def test_invalid_shapes_pose_and_fold_controls_fail_closed(self):
        for params in (dict(taper_x=.351),dict(taper_y=np.nan),dict(bend_angle=1.),
                       dict(radii=(0.,1.,1.)),dict(radii=(-1.,1.,1.)),
                       dict(rotation=np.diag([-1.,1.,1.])),dict(rotation=np.ones((3,3))),
                       dict(deformation_fit_enabled=1)):
            with self.subTest(params=params),self.assertRaises(ValueError):self.part(**params)
        part=self.part()
        with self.assertRaises(ValueError):part.forward_local([[np.inf,0,0]])
        with self.assertRaises(NotImplementedError):part.profile_width_at_world_z(0.)

    def test_serialized_renderable_replays_actual_vertices(self):
        part=self.part(); record=renderable_from_primitive(part)
        self.assertEqual(record.primitive_type,'deformed_superquadric')
        replay=primitive_from_renderable(record)
        np.testing.assert_allclose(part.to_mesh_data(16).vertices,replay.to_mesh_data(16).vertices,atol=1e-15)
        self.assertEqual(part.deformation_report()['field_semantics'],replay.deformation_report()['field_semantics'])

    def test_program_local_compiler_payload_and_numeric_screening_agree(self):
        params=self.parameters();world=Deformed.from_program_parameters(params)
        local=Deformed.from_program_parameters(params,world=False).to_mesh_data(16)
        expected=local.vertices@rotation_matrix(params).T+position_vector(params)
        mesh=world.to_mesh_data(16)
        np.testing.assert_allclose(expected,mesh.vertices,atol=1e-15)
        actual=whole_program_geometry(self.program(),resolution=16)
        np.testing.assert_allclose(actual.vertices,expected,atol=1e-15)
        np.testing.assert_array_equal(actual.faces,self.triangles(mesh))

    def test_compiler_dispatch_keeps_deformed_local_mesh_and_one_world_pose(self):
        from primitives.shape_program_compiler import _compile_node
        class FakeObject(dict):pass
        obj=FakeObject();captured={}
        mesh=SimpleNamespace(from_pydata=lambda v,e,f:captured.update(vertices=v,faces=f),update=lambda:None)
        fake=SimpleNamespace(data=SimpleNamespace(meshes=SimpleNamespace(new=lambda name:mesh),
                                                  objects=SimpleNamespace(new=lambda name,data:obj)))
        def matrix(values):
            captured['rotation']=values
            return SimpleNamespace(to_euler=lambda:(.1,.2,.3))
        params=self.parameters()
        with patch('primitives.shape_program_compiler.bpy',fake),patch.dict(sys.modules,{'mathutils':SimpleNamespace(Matrix=matrix)}):
            result,warnings=_compile_node(self.program().root_nodes[0],lathe_segments=16,
                                          bevel_modifier=False,weighted_normals=False)
        self.assertIs(result,obj);self.assertFalse(warnings)
        np.testing.assert_allclose(captured['vertices'],Deformed.from_program_parameters(params,world=False).to_mesh_data(16).vertices)
        np.testing.assert_allclose(captured['rotation'],rotation_matrix(params))
        np.testing.assert_allclose(obj.location,position_vector(params))
        self.assertEqual(obj['blendslop_deformation_contract'],self.part().deformation_report()['deformation'])

    def test_compilation_validation_rejects_unsafe_deformation(self):
        from primitives.shape_program import validate_compilable_program
        self.assertFalse(validate_compilable_program(self.program()))
        params=self.parameters();params['bend_angle']=2.
        self.assertTrue(any('invalid deformed' in error for error in validate_compilable_program(self.program(params))))

    def test_reflection_preserves_signed_field_in_proper_pose(self):
        params=self.parameters();original=Deformed.from_program_parameters(params)
        query=np.random.default_rng(8).normal(size=(100,3))
        for axis in range(3):
            reflected=reflect_parameters(params,axis=axis,plane=.3)
            part=Deformed.from_program_parameters(reflected);mirror=query.copy();mirror[:,axis]=.6-mirror[:,axis]
            np.testing.assert_allclose(original.sdf_batch(query),part.sdf_batch(mirror),atol=2e-14)
            self.assertAlmostEqual(np.linalg.det(part.rotation),1.)

    def test_fit_controls_require_release_and_invalid_coupled_edit_is_atomic(self):
        part=self.part();refs=discover_primitive_parameters([part])
        self.assertFalse(any(r[1] in ('taper_x','taper_y','bend_angle') for r in refs))
        with self.assertRaises(ValueError):apply_parameter_increment([part],(0,'bend_angle',None),.02,ParameterBounds())
        part.deformation_fit_enabled=True
        self.assertTrue(all((0,k,None) in discover_primitive_parameters([part]) for k in ('taper_x','taper_y','bend_angle')))
        apply_parameter_increment([part],(0,'bend_angle',None),.02,ParameterBounds())
        self.assertAlmostEqual(part.bend_angle,.52)
        before=part.to_dict()
        with self.assertRaises(ValueError):set_parameter_value(part,'radii',0,4.,ParameterBounds())
        self.assertEqual(before,part.to_dict())
        with self.assertRaises(ValueError):set_parameter_value(part,'taper_x',None,.9,ParameterBounds())
        self.assertEqual(before,part.to_dict())

    def test_program_search_releases_controls_explicitly_and_skips_folded_trials(self):
        from reconstruction.backends.shape_program.geometry_search import parameter_variants
        coarse=parameter_variants(self.program(),limit=40)
        self.assertFalse(any(p.metadata['refinement_control']['kind'] in ('taper_x','taper_y','bend_angle') for p in coarse))
        fine=parameter_variants(self.program(),limit=40,release_deformations=True)
        self.assertTrue(any(p.metadata['refinement_control']['kind']=='bend_angle' for p in fine))
        params=self.parameters();params['bend_angle']=Deformed.from_program_parameters(params).maximum_bend_angle()
        for program in parameter_variants(self.program(params),limit=40,release_deformations=True):
            Deformed.from_program_parameters(program.root_nodes[0].parameters).validate_deformation()

    def test_actual_mesh_contact_and_solid_guards_remain_separate(self):
        from evaluation.solid_validity import solid_validity_report
        from evaluation.triangle_contacts import within_part_boundary_guard
        part=self.part();mesh=part.to_mesh_data(12);faces=self.triangles(mesh)
        report=solid_validity_report(mesh.vertices,faces)
        self.assertEqual(report['topology']['boundary_edges'],0)
        self.assertEqual(report['inconsistent_winding_edges'],0)
        self.assertFalse(report['single_solid_eligible'])
        boundary=within_part_boundary_guard(mesh.vertices,faces,timeout_s=15.)
        self.assertTrue(boundary['passed'],boundary)

    def test_contour_uses_actual_triangles_and_changes_with_bend(self):
        from primitives.contour_silhouette import contour_footprint
        from primitives.soft_silhouette import OrthographicCamera
        camera=OrthographicCamera.from_view('front',image_size=(16,16))
        mask,report=contour_footprint(self.part(),camera,resolution=16,return_metadata=True)
        rigid=contour_footprint(self.part(bend_angle=0.),camera,resolution=16)
        self.assertTrue(np.isfinite(mask).all());self.assertGreater(np.max(np.abs(mask-rigid)),1e-4)
        self.assertFalse(report['final_opaque_admission'])
        self.assertEqual(report['projection_representation'],'mesh_triangle_union')

    def test_cache_tracks_deformation_and_released_pullbacks_match_independent_differences(self):
        from primitives.shape_aware_silhouette import ShapeAwareSilhouetteCache
        from primitives.soft_silhouette import OrthographicCamera,render_projected_soft_silhouette
        part=self.part(deformation_fit_enabled=True)
        camera=OrthographicCamera.from_view('front',image_size=(12,12))
        cache=ShapeAwareSilhouetteCache([camera]);first=cache.render([part])['front'].copy()
        changed=deepcopy(part);changed.bend_angle+=.01
        moved=cache.render([changed])['front']
        self.assertEqual(cache.recomputed_components,2)
        self.assertGreater(np.max(np.abs(first-moved)),1e-5)
        gradient=np.random.default_rng(11).normal(size=(12,12))
        actual=cache.parameter_gradients({'front':gradient},ParameterBounds(),maximum_controls=32)
        self.assertFalse(cache.derivative_failures)
        for key in ('taper_x','taper_y','bend_angle'):
            ref=(0,key,None);self.assertIn(ref,actual)
            plus,minus=deepcopy(changed),deepcopy(changed)
            apply_parameter_increment([plus],ref,1e-4,ParameterBounds())
            apply_parameter_increment([minus],ref,-1e-4,ParameterBounds())
            numerical=np.sum((render_projected_soft_silhouette([plus],camera)-
                              render_projected_soft_silhouette([minus],camera))*gradient)/2e-4
            np.testing.assert_allclose(actual[ref],numerical,atol=1e-10,rtol=1e-8)


if __name__=='__main__':unittest.main()
