"""Independent CPU conditioning/evidence math; no pinned Torch/DVX execution."""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import tempfile,json
import unittest
import numpy as np
from reconstruction.differentiable.mesh_conditioning import trilinear_cage,DifferentialCoordinates,deformation_reference_terms
from reconstruction.differentiable.ray_evidence import prepare_ray_targets,prepare_coverage_target,ray_reference_loss,union_coverage_reference
from reconstruction.evidence_identity import target_evidence_hash
from reconstruction.differentiable.seed_selection import select_existing_seed
from reconstruction.types import CandidateResult,CandidateMetrics,CandidateRequest,CandidateBudget
from reconstruction.native_geometry import GeometryArrays
from primitives.analytic_primitives import EllipsoidPrimitive
from test_geometry_consistency import triangles
from test_quality_geometry import target_for_masks


class ConditioningMathTests(unittest.TestCase):
    def test_cage_partition_and_exact_affine_displacement(self):
        vertices=np.array([[-.7,.2,.3],[.8,-.6,.4],[0.,0.,0.],[-1.,1.,1.]])
        cage=trilinear_cage(vertices)
        np.testing.assert_allclose(cage.weights.sum(axis=1),1.)
        np.testing.assert_array_equal(cage.decode(np.zeros((64,3))),vertices)
        matrix=np.array([[.1,.02,0.],[0.,-.05,.01],[.03,0.,.07]])
        translation=np.array([.02,-.04,.03])
        controls=cage.control_positions@matrix.T+translation
        np.testing.assert_allclose(cage.decode(controls),vertices+vertices@matrix.T+translation,atol=1e-15)

    def test_cage_pullback_matches_independent_numeric_derivative(self):
        vertices=np.random.default_rng(7).uniform(-.8,.8,(11,3));cage=trilinear_cage(vertices)
        gradient=np.random.default_rng(8).normal(size=vertices.shape);analytic=cage.pullback(gradient)
        controls=np.zeros((64,3));epsilon=1e-6
        for i,axis in ((0,0),(17,1),(42,2),(63,0)):
            controls[i,axis]=epsilon;plus=float(np.sum(cage.decode(controls)*gradient))
            controls[i,axis]=-epsilon;minus=float(np.sum(cage.decode(controls)*gradient))
            controls[i,axis]=0.
            self.assertAlmostEqual((plus-minus)/(2*epsilon),analytic[i,axis],places=9)

    def test_differential_seed_and_transposed_pullback(self):
        vertices=np.array([[0.,0.,0.],[.6,0.,0.],[0.,.7,0.],[0.,0.,.8]])
        faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
        operator=DifferentialCoordinates(vertices,faces,strength=3.)
        np.testing.assert_allclose(operator.decode(operator.seed_coordinates),vertices,atol=1e-15)
        gradient=np.random.default_rng(4).normal(size=vertices.shape)
        pulled=operator.pullback(gradient);epsilon=1e-6
        for i,axis in ((0,0),(2,1),(3,2)):
            delta=np.zeros_like(vertices);delta[i,axis]=epsilon
            numeric=np.sum((operator.decode(operator.seed_coordinates+delta)-operator.decode(operator.seed_coordinates-delta))*gradient)/(2*epsilon)
            self.assertAlmostEqual(numeric,pulled[i,axis],places=9)
        identity=DifferentialCoordinates(vertices,faces,strength=0.)
        np.testing.assert_array_equal(identity.decode(vertices),vertices)

    def test_local_protection_is_scale_invariant_and_detects_collapse(self):
        vertices=np.array([[0.,0.,0.],[.6,0.,0.],[0.,.7,0.],[0.,0.,.8]])
        faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
        unchanged=deformation_reference_terms(vertices+[2.,-1.,.5],vertices,faces)
        self.assertAlmostEqual(unchanged['edge_distortion'],0.,places=14)
        collapsed=vertices.copy();collapsed[2]=vertices[0]
        first=deformation_reference_terms(collapsed,vertices,faces)
        second=deformation_reference_terms(collapsed*100.,vertices*100.,faces)
        self.assertGreater(first['area_barrier'],0.)
        for key in first:self.assertAlmostEqual(first[key],second[key],places=12)


class RayEvidenceTests(unittest.TestCase):
    def test_ray_contract_keeps_depth_ambiguity_explicit(self):
        mask=np.zeros((8,8),bool);mask[2:6,2:6]=True
        target=target_for_masks({'front':mask})
        rays=prepare_ray_targets(target,np.zeros(3),1.,8,quadrature=1)
        grid=np.zeros((8,8,8));grid[2:6,1,2:6]=1.
        moved=np.zeros_like(grid);moved[2:6,6,2:6]=1.
        self.assertEqual(ray_reference_loss(grid,rays),0.)
        self.assertEqual(ray_reference_loss(moved,rays),0.)
        self.assertGreater(np.mean((grid-moved)**2),0.)

    def test_screen_vertical_flip_matches_world_increasing_grid_rows(self):
        mask=np.zeros((8,8),bool);mask[0,0]=True
        rays=prepare_ray_targets(target_for_masks({'front':mask}),np.zeros(3),1.,8,quadrature=1)
        expected=np.zeros((8,8));expected[7,0]=1.
        np.testing.assert_array_equal(rays['front']['foreground'],expected)

    def test_unknown_coverage_values_are_ignored_and_not_empty(self):
        mask=np.zeros((8,8),bool);mask[2:6,2:6]=True
        valid=np.ones_like(mask);valid[:,4:]=False
        target=target_for_masks({'front':mask});c=replace(target.constraints[0],valid_mask=valid)
        first=replace(target,constraints=(c,));changed=mask.copy();changed[~valid]=True
        second=replace(target,constraints=(replace(c,mask=changed),))
        a=prepare_coverage_target(first,np.zeros(3),1.,8,quadrature=2)
        b=prepare_coverage_target(second,np.zeros(3),1.,8,quadrature=2)
        np.testing.assert_array_equal(a['target_zyx'],b['target_zyx'])
        np.testing.assert_array_equal(a['target_valid_zyx'],b['target_valid_zyx'])
        self.assertTrue(np.any(a['target_valid_zyx']==0.))
        self.assertIn('pending',a['voxelizer_footprint_match'])

    def test_per_part_union_coverage_does_not_sum_overlap_winding(self):
        a=np.ones((4,4,4));b=np.ones_like(a)
        np.testing.assert_array_equal(union_coverage_reference([a,b]),a)
        np.testing.assert_array_equal(union_coverage_reference([a*.5,b*.5]),np.full_like(a,.75))


class RetainedSeedTests(unittest.TestCase):
    def target(self):
        mask=np.zeros((16,16),bool);mask[3:13,3:13]=True
        return target_for_masks({'front':mask,'side':mask,'top':mask})
    def result(self,target):
        mesh=EllipsoidPrimitive(radii=(.6,.5,.7)).to_mesh_data(8)
        data=GeometryArrays.capture(mesh.vertices,triangles(mesh))
        return CandidateResult('seed','profile_loft','success',geometry=data,
            metric_result=CandidateMetrics(per_view={v:{'area_iou':.8,'boundary_iou':.5,'passed':True} for v in target.views()},
                extras={'input_evidence_hash':target_evidence_hash(target)}))

    def test_seed_identity_binds_camera_masks_and_geometry(self):
        target=self.target();result=self.result(target)
        data,report=select_existing_seed(target,{'profile':result})
        self.assertIs(data,result.geometry)
        self.assertEqual(report['seed_content_hash'],data.content_hash)
        constraint=target.constraints[0];changed=constraint.mask.copy();changed[0,0]=True
        other=replace(target,constraints=(replace(constraint,mask=changed),*target.constraints[1:]))
        self.assertIsNone(select_existing_seed(other,{'profile':result})[0])
        self.assertIsNone(select_existing_seed(target,{'profile':result},maximum_vertices=4)[0])

    def test_unknown_foreground_does_not_change_evidence_identity(self):
        target=self.target();c=target.constraints[0];valid=np.ones_like(c.mask);valid[:2]=False
        first=replace(target,constraints=(replace(c,valid_mask=valid),))
        changed=c.mask.copy();changed[~valid]=True
        second=replace(target,constraints=(replace(c,valid_mask=valid,mask=changed),))
        self.assertEqual(target_evidence_hash(first),target_evidence_hash(second))

    def test_globally_thin_retained_seed_declines_biased_ray_updates(self):
        from blender_blocking.reconstruction.differentiable.dvx_adapter import run_candidate
        target=self.target();seed=self.result(target)
        mesh=EllipsoidPrimitive(radii=(.6,.008,.7)).to_mesh_data(8)
        seed=replace(seed,geometry=GeometryArrays.capture(mesh.vertices,triangles(mesh)))
        request=CandidateRequest('thin','differentiable_refine',target,
            config={'dvx_execution_approved':True,'dvx_resolution':32,'dvx_objective':'observed_rays',
                    'seed_results':{'profile':seed}},budget=CandidateBudget(timeout_s=5.))
        with patch('blender_blocking.reconstruction.differentiable.dvx_adapter.dependency_state',return_value={'available':True}):
            result=run_candidate(request)
        self.assertEqual(result.status,'skipped')
        self.assertTrue(result.metric_result.extras['thin_ray_screen'][32]['global_envelope_below_two_voxels'])
        self.assertEqual(result.metric_result.extras['seed_content_hash'],seed.geometry.content_hash)
        self.assertIn('not valid projected coverage',result.warnings[0])

    def test_projected_ray_mode_can_keep_a_thin_seed_without_volume_ray_guard(self):
        from blender_blocking.reconstruction.differentiable.dvx_adapter import run_candidate
        from reconstruction.process_executor import JobOutcome
        target=self.target();seed=self.result(target)
        mesh=EllipsoidPrimitive(radii=(.6,.008,.7)).to_mesh_data(8)
        seed=replace(seed,geometry=GeometryArrays.capture(mesh.vertices,triangles(mesh)))
        class Executor:
            def map(self,jobs,timeout_s=None):
                payload=jobs[0][1];self.payload=payload
                return [JobOutcome('success',{'vertices':payload['vertices'],'faces':payload['faces'],'history':[.2],
                    'fixed_transform':{'center':payload['center'].tolist(),'scale':payload['scale']},
                    'requested_steps':1,'optimizer_updates':0,'objective_evaluations':1,'best_evaluation':1,
                    'stop_reason':'requested_steps','final_update_evaluated':True,'partial':False})]
        executor=Executor()
        request=CandidateRequest('thin-projected','differentiable_refine',target,
            config={'dvx_execution_approved':True,'dvx_resolution':16,'dvx_objective':'observed_projected_rays',
                    'seed_results':{'profile':seed}},context=SimpleNamespace(process_executor=executor),
            budget=CandidateBudget(timeout_s=5.))
        with patch('blender_blocking.reconstruction.differentiable.dvx_adapter.dependency_state',
                return_value={'available':True,'projected_mesh_rays_available':True}):
            result=run_candidate(request)
        self.assertEqual(result.geometry.content_hash,seed.geometry.content_hash)
        self.assertEqual(executor.payload['objective'],'observed_projected_rays')
        self.assertTrue(executor.payload['voxel_target_unused_by_ray_objective'])

    def test_noop_retained_seed_has_exact_artifact_replay_and_separate_artist_source(self):
        from blender_blocking.reconstruction.differentiable.dvx_adapter import run_candidate
        from blender_blocking.reconstruction.process_executor import JobOutcome
        from blender_blocking.reconstruction.differentiable.dvx_artifacts import replay_dvx_artifacts
        target=self.target();seed=self.result(target)
        class Executor:
            def map(self,jobs,timeout_s=None):
                payload=jobs[0][1]
                self.payload=payload
                return [JobOutcome('success',{'vertices':payload['vertices'],'faces':payload['faces'],'history':[.2],
                    'fixed_transform':{'center':payload['center'].tolist(),'scale':payload['scale']},
                    'requested_steps':1,'optimizer_updates':0,'objective_evaluations':1,'best_evaluation':1,
                    'stop_reason':'requested_steps','final_update_evaluated':True,'partial':False})]
        pool=Executor()
        with tempfile.TemporaryDirectory() as directory:
            artist=Path(directory)/'artist.json';artist.write_text('{"program":"fixture source"}')
            seed=replace(seed,primitive_path=artist)
            request=CandidateRequest('conditioned','differentiable_refine',target,
                config={'dvx_execution_approved':True,'dvx_resolution':16,'dvx_objective':'observed_rays',
                        'dvx_parameterization':'cage','seed_results':{'profile':seed}},
                context=SimpleNamespace(process_executor=pool),artifact_root=Path(directory)/'output',
                budget=CandidateBudget(timeout_s=5.))
            with patch('blender_blocking.reconstruction.differentiable.dvx_adapter.dependency_state',return_value={'available':True}), patch(
                    'blender_blocking.reconstruction.differentiable.ray_evidence.prepare_coverage_target',
                    side_effect=AssertionError('ray objective must not prepare unused solid-hull occupancy')):
                result=run_candidate(request)
            self.assertEqual(result.geometry.content_hash,seed.geometry.content_hash)
            self.assertEqual(pool.payload['objective'],'observed_rays')
            self.assertEqual(pool.payload['parameterization'],'cage')
            self.assertTrue(pool.payload['voxel_target_unused_by_ray_objective'])
            self.assertEqual(float(np.sum(pool.payload['target_valid_zyx'])),0.)
            payload=json.loads(result.primitive_path.read_text());self.assertEqual(payload['primitives'],[])
            replay=replay_dvx_artifacts(payload,())
            self.assertEqual(replay['seed'].content_hash,seed.geometry.content_hash)
            self.assertFalse(replay['receipt']['seed_parameters_reproduce_final_deformation'])
            self.assertTrue(Path(payload['metadata']['artist_source_path']).is_file())
            Path(payload['metadata']['artist_source_path']).write_text('changed')
            with self.assertRaisesRegex(ValueError,'artist source identity'):
                replay_dvx_artifacts(payload,())


if __name__=='__main__':unittest.main()
