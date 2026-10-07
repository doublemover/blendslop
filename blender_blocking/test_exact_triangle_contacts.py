"""Exact contact/admission regressions for demonstrated saved-artifact failures."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest
import numpy as np
from blender_blocking.evaluation.triangle_contacts import triangle_relation, verify_reported_pairs, within_part_boundary_guard


class ExactTriangleContactTests(unittest.TestCase):
    def setUp(self):
        self.left=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])

    def test_exact_separation_survives_a_gap_smaller_than_diagnostic_tolerance(self):
        right=self.left+[0.,0.,2.**-45]
        self.assertEqual(triangle_relation(self.left,right)['relation'],'disjoint')
        self.assertEqual(triangle_relation(self.left,self.left+.25)['relation'],'disjoint')

    def test_crossing_coplanar_overlap_duplicate_and_contact_dimension(self):
        cross=np.array([[.5,.5,-1],[.5,.5,1],[.5,1.5,0.]])
        self.assertEqual(triangle_relation(self.left,cross)['relation'],'proper_crossing')
        self.assertEqual(triangle_relation(self.left,self.left+[.1,.1,0])['relation'],'coplanar_area_overlap')
        self.assertEqual(triangle_relation(self.left,self.left[::-1])['relation'],'duplicate_triangle')
        point=triangle_relation(self.left,[[2.,0.,0.],[3.,0.,0.],[2.,-1.,0.]])
        edge=triangle_relation(self.left,[[0.,0.,0.],[2.,0.,0.],[1.,-1.,0.]])
        self.assertEqual((point['relation'],point['intersection_dimension']),('boundary_contact',0))
        self.assertEqual((edge['relation'],edge['intersection_dimension']),('boundary_contact',1))

    def test_unverified_remainder_and_degenerate_pairs_cannot_qualify(self):
        vertices=np.concatenate([self.left,self.left+[0.,0.,1.]])
        faces=np.array([[0,1,2],[3,4,5]])
        partial=verify_reported_pairs(vertices,faces,[[0,1]],max_pairs=0)
        self.assertFalse(partial['complete']);self.assertEqual(partial['unverified_pairs'],1)
        degenerate=triangle_relation(self.left,[[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
        self.assertFalse(degenerate['certified'])
        with self.assertRaises(ValueError):triangle_relation(self.left,self.left*np.nan)

    def test_multipart_overlap_is_separate_from_within_part_crossing(self):
        right=np.array([[0.,0.,0.],[.5,.5,-1.],[.5,.5,1.]])
        vertices=np.concatenate([self.left,right]);faces=np.array([[0,1,2],[3,4,5]])
        verified=verify_reported_pairs(vertices,faces,[[0,1]])
        self.assertEqual(verified['counts'],{'multipart_overlap':1})
        self.assertTrue(within_part_boundary_guard(vertices,faces)['passed'])
        # Shared vertex 0: the same physical crossing is now within one part.
        connected=np.array([[0,1,2],[0,4,5]])
        guarded=within_part_boundary_guard(vertices,connected)
        self.assertFalse(guarded['passed'])
        self.assertEqual(guarded['first_blocking_pair']['relation'],'proper_crossing')

    def test_expected_adjacency_passes_but_exhausted_allowance_is_unqualified(self):
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
        faces=np.array([[0,1,2],[0,2,3]])
        self.assertTrue(within_part_boundary_guard(vertices,faces)['passed'])
        self.assertFalse(within_part_boundary_guard(vertices,faces,timeout_s=0.)['passed'])

    def test_shared_edge_area_overlap_and_extended_shared_vertex_contact_are_rejected(self):
        vertices=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.],[.5,.5,0.],[.5,0.,0.],[.1,-1.,0.]])
        edge_overlap=within_part_boundary_guard(vertices,np.array([[0,1,2],[0,1,3]]))
        self.assertFalse(edge_overlap['passed'])
        self.assertEqual(edge_overlap['first_blocking_pair']['relation'],'coplanar_area_overlap')
        extended_contact=within_part_boundary_guard(vertices,np.array([[0,1,2],[0,4,5]]))
        self.assertFalse(extended_contact['passed'])
        self.assertEqual(extended_contact['first_blocking_pair']['relation'],'boundary_contact')

    def test_exact_relations_are_stable_under_binary_scaling_and_winding(self):
        right=np.array([[.5,.5,-1],[.5,.5,1],[.5,1.5,0.]])
        for scale in (2.**-20,2.**20):
            for a,b in ((self.left,right),(self.left[::-1],right[::-1])):
                self.assertEqual(triangle_relation(a*scale+10,b*scale+10)['relation'],'proper_crossing')

    def test_dvx_rejects_folded_proposal_but_preserves_exact_proposal_evidence(self):
        from blender_blocking.reconstruction.differentiable.dvx_adapter import run_candidate
        from blender_blocking.reconstruction.process_executor import JobOutcome
        from blender_blocking.reconstruction.types import CandidateRequest,CandidateBudget,Bounds3D,ReconstructionTarget
        class Executor:
            def map(self,jobs,timeout_s=None):
                payload=jobs[0][1];vertices=payload['vertices'].copy();vertices[0]=vertices[100]
                return [JobOutcome('success',{'vertices':vertices,'faces':payload['faces'],'history':[1.,.5],
                    'fixed_transform':{'center':payload['center'].tolist(),'scale':payload['scale']},
                    'requested_steps':1,'optimizer_updates':1,'objective_evaluations':2,'best_evaluation':2,
                    'stop_reason':'requested_steps','final_update_evaluated':True,'partial':False})]
        points=np.random.default_rng(123).uniform(-.4,.4,(32,3))
        target=ReconstructionTarget(bounds=Bounds3D.from_min_max((-1.,)*3,(1.,)*3))
        score=lambda i:{'front':{'area_iou':i,'boundary_iou':i,'passed':True}}
        with tempfile.TemporaryDirectory() as root:
            request=CandidateRequest('folded-dvx','differentiable_refine',target,artifact_root=Path(root),
                budget=CandidateBudget(timeout_s=10.),config={'dvx_execution_approved':True,'dvx_resolution':16},
                context=SimpleNamespace(process_executor=Executor()))
            with patch('blender_blocking.reconstruction.differentiable.dvx_adapter.dependency_state',return_value={'available':True}), \
                 patch('reconstruction.point_cloud.target_surface_points',return_value=(points,{})), \
                 patch('reconstruction.visibility.point_support',side_effect=lambda target,p:(np.ones(len(p)),np.ones(len(p)))), \
                 patch('reconstruction.projected_metrics.projected_mesh_metrics',side_effect=[score(.5),score(.6)]):
                result=run_candidate(request)
            self.assertTrue(result.succeeded,result.errors)
            self.assertFalse(result.metric_result.extras['full_geometry_admitted'])
            payload=json.loads(result.primitive_path.read_text());metadata=payload['metadata']
            self.assertEqual(metadata['retained_content_hash'],metadata['seed_content_hash'])
            self.assertNotEqual(metadata['proposed_content_hash'],metadata['seed_content_hash'])
            self.assertFalse(metadata['proposed_within_part_boundary_guard']['passed'])
            with np.load(metadata['deformation_state_path'],allow_pickle=False) as state:
                self.assertFalse(np.array_equal(state['proposed_vertices'],state['retained_vertices']))


if __name__=='__main__':unittest.main()
