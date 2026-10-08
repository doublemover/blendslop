"""Monotone local-frame sweep contracts, independent of Blender rendering."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from primitives.generalized_sweep import GeneralizedSweepPrimitive
from primitives.shape_program import ShapeNode,ShapeProgram,validate_compilable_program
from reconstruction.native_geometry import GeometryArrays
from reconstruction.grouped_solids import solid_guard
from test_geometry_consistency import triangles
from test_quality_geometry import target_for_masks


KNOTS=np.array([[-.5,0.,0.,.45,.4],[0.,.15,-.1,.25,.3],[.5,.05,.1,.35,.45]])


class GeneralizedSweepTests(unittest.TestCase):
    def part(self,exponent=1.):
        return GeneralizedSweepPrimitive(KNOTS,center=(.3,-.2,.1),radii=(.7,.4,1.),
            rotation=Rotation.from_rotvec([.3,-.2,.1]).as_matrix(),section_exponent=exponent)

    def program(self):
        return ShapeProgram('shape-program-v1','sweep',(ShapeNode('a','add','generalized_sweep',{
            'section_knots_normalized':KNOTS.tolist(),'width_world':1.4,'depth_world':.8,'height_world':2.,
            'x':.3,'y':-.2,'z':.1,'rotation':self.part().rotation.tolist()}),))

    def test_field_and_closed_outward_mesh_agree_at_all_knots_and_caps(self):
        for exponent in (.15,.5,1.,2.):
            part=self.part(exponent);mesh=part.to_mesh_data(24)
            np.testing.assert_allclose(part.sdf_batch(mesh.vertices),0.,atol=2e-14)
            guard=solid_guard(GeometryArrays.capture(mesh.vertices,triangles(mesh)))
            self.assertTrue(guard['valid_solid'],guard)
            points=np.array([[0.,0.,0.],[0.,0.,-2.],[2.,0.,0.]])@part.rotation.T+part.center
            self.assertLess(part.sdf_batch(points)[0],0.)
            self.assertGreater(part.sdf_batch(points)[1],0.)
            self.assertGreater(part.sdf_batch(points)[2],0.)

    def test_world_local_pose_and_serialization_are_exact(self):
        part=self.part(.5);clone=GeneralizedSweepPrimitive.from_dict(part.to_dict())
        np.testing.assert_array_equal(part.to_mesh_data(16).vertices,clone.to_mesh_data(16).vertices)
        query=np.random.default_rng(4).normal(size=(40,3))
        local=GeneralizedSweepPrimitive(KNOTS,radii=part.radii,section_exponent=.5)
        np.testing.assert_allclose(part.sdf_batch(query@part.rotation.T+part.center),local.sdf_batch(query),atol=1e-14)
        self.assertIn('not Euclidean',part.to_dict()['field_semantics'])

    def test_mesh_chord_error_decreases_without_claiming_exact_distance(self):
        part=self.part(.5);errors=[]
        for resolution in (12,48):
            mesh=part.to_mesh_data(resolution);faces=triangles(mesh)
            centers=mesh.vertices[faces].mean(axis=1)
            errors.append(float(np.max(np.abs(part.sdf_batch(centers)))))
        self.assertLess(errors[1],errors[0])
        self.assertLess(errors[1],.005)

    def test_rejects_axial_folds_invalid_sections_and_improper_frames(self):
        for bad in (KNOTS[[0,2,1]],KNOTS[:1],np.array([[-.5,0.,0.,0.,.4],[.5,0.,0.,.4,.4]])):
            with self.assertRaises(ValueError):GeneralizedSweepPrimitive(bad)
        with self.assertRaises(ValueError):GeneralizedSweepPrimitive(KNOTS,rotation=np.diag([-1.,1.,1.]))
        with self.assertRaises(ValueError):self.part(3.)
        with self.assertRaises(ValueError):GeneralizedSweepPrimitive(KNOTS,rotation=np.eye(3)*2.)

    def test_parameter_adapters_preserve_positive_sections_and_fixed_knots(self):
        from placement.resfit_parameters import discover_primitive_parameters,apply_parameter_increment,parameter_value
        from placement.resfit_optimizer import ParameterBounds
        part=self.part();heights=part.section_knots_normalized[:,0].copy()
        refs=discover_primitive_parameters([part])
        self.assertEqual(sum(ref[1]=='section_knots_normalized' for ref in refs),12)
        for ref in refs:
            apply_parameter_increment([part],ref,-.05,ParameterBounds())
            self.assertTrue(np.isfinite(parameter_value(part,ref[1],ref[2])))
        part.validate();np.testing.assert_array_equal(part.section_knots_normalized[:,0],heights)
        self.assertTrue(np.all(part.section_knots_normalized[:,3:]>0.))
        self.assertTrue(solid_guard(GeometryArrays.capture(part.to_mesh_data(24).vertices,triangles(part.to_mesh_data(24))))['valid_solid'])

    def test_numeric_program_screen_uses_identical_sweep_generator(self):
        from reconstruction.proposal_screening import whole_program_geometry
        program=self.program();data=whole_program_geometry(program,resolution=16)
        part=GeneralizedSweepPrimitive.from_program_parameters(program.root_nodes[0].parameters)
        mesh=part.to_mesh_data(16)
        np.testing.assert_array_equal(data.vertices,mesh.vertices)
        np.testing.assert_array_equal(data.faces,triangles(mesh))
        self.assertFalse(validate_compilable_program(program))

    def test_invalid_sweep_declines_before_compiler_scene_mutation(self):
        from primitives.shape_program_compiler import compile_shape_program
        program=self.program();node=program.root_nodes[0]
        bad=replace(program,root_nodes=(replace(node,parameters={**node.parameters,'section_knots_normalized':KNOTS[:1].tolist()}),))
        with patch('primitives.shape_program_compiler._ensure_collection',side_effect=AssertionError('scene mutation')):
            with self.assertRaisesRegex(ValueError,'invalid generalized sweep'):
                compile_shape_program(bad)

    def test_program_controls_include_local_bend_taper_exponent_and_pose(self):
        from reconstruction.backends.shape_program.geometry_search import parameter_variants
        program=self.program()
        variants=parameter_variants(program,limit=14)+parameter_variants(program,limit=14,start_control=14)
        kinds={row.metadata['refinement_control']['kind'] for row in variants}
        self.assertEqual(kinds,{'size','translation','rotation','sweep_bend','sweep_taper','section_exponent'})
        for proposal in variants:self.assertFalse(validate_compilable_program(proposal))
        bend=next(row for row in variants if row.metadata['refinement_control']['kind']=='sweep_bend')
        knots=np.asarray(bend.root_nodes[0].parameters['section_knots_normalized'])
        np.testing.assert_array_equal(knots[[0,-1]],KNOTS[[0,-1]])
        self.assertFalse(np.array_equal(knots[1],KNOTS[1]))

    def test_completed_profile_proposals_preserve_evidence_and_decline_unobserved_axes(self):
        from reconstruction.sweep_proposals import generalized_sweep_programs
        empty=np.zeros((32,32),bool);mask=empty.copy();mask[3:29,8:24]=True
        target=target_for_masks({'front':mask,'side':mask})
        seed=ShapeProgram('shape-program-v1','seed',())
        proposals=generalized_sweep_programs(target,seed)
        self.assertEqual(len(proposals),3)
        for proposal in proposals:
            self.assertFalse(validate_compilable_program(proposal))
            self.assertIn('censored_completion',proposal.metadata['evidence_completion'])
        self.assertEqual(generalized_sweep_programs(target_for_masks({'front':mask}),seed),[])
        unknown=replace(target,constraints=tuple(replace(c,valid_mask=np.zeros_like(mask)) for c in target.constraints))
        self.assertEqual(generalized_sweep_programs(unknown,seed),[])

    def test_legacy_program_builder_cannot_restore_censored_rows_as_exact_curve(self):
        from reconstruction.backends.shape_program.builder import build_shape_program_from_target
        from reconstruction.types import ProfileBand
        mask=np.zeros((32,32),bool);mask[3:29,8:24]=True
        target=target_for_masks({'front':mask,'side':mask})
        valid=np.ones_like(mask);valid[:,20:]=False
        target=replace(target,constraints=tuple(replace(c,valid_mask=valid) for c in target.constraints),
            profile_bands={'front':tuple(ProfileBand(t=t,width_px=16,center_x=16,intervals=((8.,24.),)) for t in (.1,.5,.9))},
            extras={'view_calibration':{name:{'world_bounds':[-1.,1.,-1.,1.]} for name in ('front','side')}})
        program,_=build_shape_program_from_target(target,config={'root_strategy':'profile_lathe','residual_policy':'ignore'},program_id='censored')
        self.assertEqual(program.root_nodes[0].primitive_type,'rounded_box')
        self.assertNotIn('profile_curve',program.root_nodes[0].parameters)
        self.assertIn('censored-only',program.root_nodes[0].parameters['profile_seed_status'])


if __name__=='__main__':unittest.main()
