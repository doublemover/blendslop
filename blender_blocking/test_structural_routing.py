"""Routing and local-feature contracts with deterministic fake executor outcomes."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from reconstruction.types import CandidateRequest, CandidateResult, CandidateMetrics
from reconstruction.feature_evidence import known_empty_feature_guard, hole_capable_request
from reconstruction.geometry_selection import prefer_candidate, missing_observed_points
from reconstruction.measured_selection import routing_run
from test_quality_geometry import target_for_masks


class FakePool:
    max_workers=2
    root=Path('/tmp/blendslop-routing-fixture')
    def __init__(self,results):self.results=results;self.submitted=[]
    def submit(self,kind,payload,**kwargs):
        request=payload[0] if isinstance(payload,tuple) else payload
        self.submitted.append(request);return request.candidate_id
    def result(self,job):
        return SimpleNamespace(status='success',value=self.results[job],queue_s=0.,elapsed_s=.01,total_wall_s=.01,stop_reason='fixture')


def scored(request,*,score=.995,faces=12):
    from reconstruction.native_geometry import GeometryArrays
    geometry=GeometryArrays.capture(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),np.tile([[0,1,2]],(faces,1)))
    return CandidateResult(request.candidate_id,request.backend_name,'success',geometry=geometry,
        metric_result=CandidateMetrics(per_view={'front':{'area_iou':score,'boundary_iou':score,'passed':True,'required':True}},
            extras={'selection_evidence':'fresh_blender_render'}))


class StructuralRoutingTests(unittest.TestCase):
    def target(self):
        mask=np.zeros((24,24),bool);mask[4:20,4:20]=True
        return target_for_masks({'front':mask})
    def requests(self,*,structural=True):
        return [CandidateRequest(name,name,self.target(),config={'routing_policy':'structure_v1'} if structural else {})
                for name in ('profile_loft','visual_hull_voxel','silhouette_intersection','shape_program')]

    def test_explicit_implicit_residual_waits_for_all_prior_parallel_source_results(self):
        requests = self.requests()[:2]
        residual = CandidateRequest('residual', 'implicit_residual', self.target(),
                                    config={'routing_policy': 'structure_v1', 'execution_approved': True})
        requests.append(residual)
        pool = FakePool({r.candidate_id: scored(r) for r in requests})
        results, _, _ = routing_run(requests, executor=pool, max_render_candidates=4)
        submitted = next(r for r in pool.submitted if r.backend_name == 'implicit_residual')
        self.assertEqual(set(submitted.config['seed_results']), {'profile_loft', 'visual_hull_voxel'})
        self.assertEqual(pool.submitted[-1].backend_name, 'implicit_residual')

    def test_high_scoring_hull_does_not_preempt_complementary_structure(self):
        requests=self.requests();pool=FakePool({r.candidate_id:scored(r) for r in requests})
        with patch('reconstruction.program_proposals.whole_primitive_programs',return_value=[]):
            results,best,ledger=routing_run(requests,executor=pool,max_render_candidates=3)
        self.assertEqual([r.backend_name for r in pool.submitted],['profile_loft','visual_hull_voxel','shape_program'])
        self.assertEqual(ledger['rendered_candidates'],3)
        self.assertEqual(next(r for r in results if r.backend_name=='silhouette_intersection').status,'skipped')

    def test_construction_failure_releases_only_its_render_reservation(self):
        requests=self.requests();results={r.candidate_id:scored(r) for r in requests}
        results['profile_loft']=CandidateResult('profile_loft','profile_loft','failed',errors=('contour construction fixture failed',))
        pool=FakePool(results)
        with patch('reconstruction.program_proposals.whole_primitive_programs',return_value=[]):
            _,_,ledger=routing_run(requests,executor=pool,max_render_candidates=3)
        self.assertEqual(ledger['submitted_candidates'],4)
        self.assertEqual(ledger['rendered_candidates'],3)
        self.assertEqual([r.backend_name for r in pool.submitted][-1],'silhouette_intersection')

    def test_legacy_early_stop_is_still_explicitly_available(self):
        requests=self.requests(structural=False);pool=FakePool({r.candidate_id:scored(r) for r in requests})
        _,_,ledger=routing_run(requests,executor=pool,max_render_candidates=3)
        self.assertEqual(ledger['submitted_candidates'],2)
        self.assertEqual(ledger['routing_policy'],'silhouette_only')

    def test_hole_eligibility_uses_effective_representation(self):
        request=CandidateRequest('p','profile_loft',self.target())
        self.assertFalse(hole_capable_request(request))
        self.assertTrue(hole_capable_request(replace(request,config={'contour_sections':True})))
        self.assertTrue(hole_capable_request(replace(request,config={'quality_preset':'quality'})))

    def test_known_hole_cannot_hide_in_whole_image_iou(self):
        mask=np.zeros((32,32),bool);mask[2:30,2:30]=True;mask[14:18,14:18]=False
        constraint=target_for_masks({'front':mask}).constraints[0]
        filled=mask.copy();filled[14:18,14:18]=True
        self.assertFalse(known_empty_feature_guard(constraint,filled)['passed'])
        self.assertTrue(known_empty_feature_guard(constraint,mask)['passed'])
        valid=np.ones_like(mask);valid[14:18,14:18]=False
        self.assertTrue(known_empty_feature_guard(replace(constraint,valid_mask=valid),filled)['passed'])

    def test_compact_prior_has_an_observed_evidence_guard(self):
        request=self.requests()[0];large=scored(request,score=.995,faces=100)
        compact=scored(request,score=.992,faces=12)
        self.assertTrue(prefer_candidate(compact,large,policy='structure_v1'))
        self.assertFalse(prefer_candidate(compact,large,policy='silhouette_only'))
        self.assertFalse(prefer_candidate(scored(request,score=.97,faces=12),large,policy='structure_v1'))

    def test_hull_bulges_are_not_missing_observed_material(self):
        target=self.target();reference=target.constraints[0].mask
        points=np.array([[0.,0.,0.],[.9,0.,.9]])
        np.testing.assert_array_equal(missing_observed_points(target,{'front':reference.copy()},points),[False,False])
        prediction=np.zeros_like(reference)
        np.testing.assert_array_equal(missing_observed_points(target,{'front':prediction},points),[True,False])

    def test_screened_geometry_preserves_a_matrix_pose(self):
        from primitives.shape_program import ShapeNode,ShapeProgram
        from scipy.spatial.transform import Rotation
        from reconstruction.proposal_screening import whole_program_geometry
        frame=Rotation.from_rotvec([.3,-.2,.1]).as_matrix()
        program=ShapeProgram('shape-program-v1','p',(ShapeNode('a','add','box',{'width_world':2.,'depth_world':1.,'height_world':3.,'rotation':frame.tolist()}),))
        data=whole_program_geometry(program)
        local=data.vertices@frame
        np.testing.assert_allclose(local.min(0),[-1.,-.5,-1.5],atol=1e-14)
        np.testing.assert_allclose(local.max(0),[1.,.5,1.5],atol=1e-14)


if __name__=='__main__':unittest.main()
