"""Spatial reliability transport and gradients, separate from hard admission."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import tempfile
from pathlib import Path
import numpy as np
from geometry.silhouette_types import UncertainMask
from reconstruction.pixel_evidence import observed_pixel_evidence,resize_pixel_evidence
from reconstruction.evidence_identity import target_evidence_hash
from test_quality_geometry import target_for_masks


class PixelEvidenceTests(unittest.TestCase):
    def constraint(self):
        mask=np.zeros((8,10),bool);mask[2:6,3:7]=True
        uncertainty=UncertainMask(mask.astype(float),mask,np.ones(mask.shape),np.zeros(mask.shape),'fixture',.5,{})
        return replace(target_for_masks({'front':mask}).constraints[0],uncertainty=uncertainty)

    def test_unknown_map_values_and_nan_are_not_observations_or_identity(self):
        c=self.constraint();valid=np.ones_like(c.mask);valid[:,5:]=False;c=replace(c,valid_mask=valid)
        changed={}
        for name in ('foreground_prob','confidence','boundary_uncertainty'):
            values=getattr(c.uncertainty,name).copy();values[~valid]=np.nan;changed[name]=values
        other=replace(c,uncertainty=replace(c.uncertainty,**changed))
        one=observed_pixel_evidence(c);two=observed_pixel_evidence(other)
        for name in ('foreground','confidence','boundary_reliability','valid'):
            np.testing.assert_array_equal(getattr(one,name),getattr(two,name))
            self.assertFalse(getattr(one,name).flags.writeable)
        target=target_for_masks({'front':c.mask})
        self.assertEqual(target_evidence_hash(replace(target,constraints=(c,))),target_evidence_hash(replace(target,constraints=(other,))))

    def test_sub_float32_observed_map_change_still_invalidates_evidence_identity(self):
        c=self.constraint();confidence=c.uncertainty.confidence.copy();confidence[0,0]=1.-1e-10
        other=replace(c,uncertainty=replace(c.uncertainty,confidence=confidence))
        target=target_for_masks({'front':c.mask})
        self.assertNotEqual(target_evidence_hash(replace(target,constraints=(c,))),target_evidence_hash(replace(target,constraints=(other,))))

    def test_endpoint_center_and_cell_edge_cameras_cannot_share_identity(self):
        c=self.constraint();target=replace(target_for_masks({'front':c.mask}),constraints=(c,))
        legacy=replace(target,constraints=(replace(c,camera=replace(c.camera,bounds=None)),))
        self.assertNotEqual(target_evidence_hash(target),target_evidence_hash(legacy))

    def test_shape_or_invalid_observed_probability_fails_without_implicit_resize(self):
        c=self.constraint()
        with self.assertRaises(ValueError):observed_pixel_evidence(replace(c,uncertainty=replace(c.uncertainty,confidence=np.ones((4,5)))))
        probability=c.uncertainty.foreground_prob.copy();probability[0,0]=1.01
        with self.assertRaises(ValueError):observed_pixel_evidence(replace(c,uncertainty=replace(c.uncertainty,foreground_prob=probability)))

    def test_coarse_reference_preserves_weighted_evidence_instead_of_unknown_zeros(self):
        c=self.constraint();valid=np.zeros_like(c.mask);valid[2:4,3:5]=True
        c=replace(c,valid_mask=valid);evidence=observed_pixel_evidence(c)
        coarse=resize_pixel_evidence(evidence,(1,1))
        self.assertEqual(float(coarse['foreground'][0,0]),1.)
        self.assertAlmostEqual(float(coarse['observed_fraction'][0,0]),4/80)
        self.assertAlmostEqual(float(coarse['reliability_weight'][0,0]),4/80)

    def test_weighted_soft_pullback_matches_numeric_derivative_and_zero_weights(self):
        from reconstruction.differentiable.soft_objective import SoftMaskTarget,soft_mask_loss_and_gradient
        rng=np.random.default_rng(6);reference=rng.uniform(0.,1.,(5,7));prediction=rng.uniform(.1,.9,(5,7))
        reliability=rng.uniform(0.,1.,reference.shape);reliability[1,2]=0.
        target=SoftMaskTarget.from_mask(reference,pixel_weights=reliability)
        loss,gradient,_=soft_mask_loss_and_gradient(prediction,target)
        epsilon=1e-6
        for row,column in ((0,0),(2,4),(4,6)):
            plus=prediction.copy();minus=prediction.copy();plus[row,column]+=epsilon;minus[row,column]-=epsilon
            derivative=(soft_mask_loss_and_gradient(plus,target)[0]-soft_mask_loss_and_gradient(minus,target)[0])/(2*epsilon)
            self.assertAlmostEqual(derivative,gradient[row,column],places=8)
        changed=prediction.copy();changed[1,2]=100.
        self.assertEqual(loss,soft_mask_loss_and_gradient(changed,target)[0]);self.assertEqual(gradient[1,2],0.)

    def test_ray_observed_fraction_is_distinct_from_confidence_weight(self):
        from reconstruction.differentiable.ray_evidence import prepare_ray_targets
        c=self.constraint();confidence=np.full(c.mask.shape,.4);boundary=np.full(c.mask.shape,.5)
        c=replace(c,uncertainty=replace(c.uncertainty,confidence=confidence,boundary_uncertainty=boundary))
        target=replace(target_for_masks({'front':c.mask}),constraints=(c,))
        ray=prepare_ray_targets(target,np.zeros(3),1.,4)['front']
        np.testing.assert_allclose(ray['valid'],1.)
        np.testing.assert_allclose(ray['weights'],.2,atol=1e-15)

    def test_input_validity_restricts_retained_maps_and_uncertainty_summaries(self):
        from reconstruction.target_builder import build_target_from_images
        from reconstruction.target_signals import collect_uncertainty_signal
        c=self.constraint();image=np.full((*c.mask.shape,3),255,np.uint8);image[c.mask]=0
        valid=np.ones_like(c.mask);valid[:2]=False
        result=build_target_from_images({'front':image},valid_evidence_masks={'front':valid})
        uncertain=result.uncertainties['front']
        self.assertTrue(np.all(uncertain.confidence[~valid]==0.))
        self.assertTrue(np.all(uncertain.foreground_prob[~valid]==0.))
        self.assertTrue(np.all(uncertain.boundary_uncertainty[~valid]==1.))
        signal=collect_uncertainty_signal(result.target.constraints)
        self.assertAlmostEqual(signal['overall_confidence_mean'],float(uncertain.confidence[valid].mean()))

    def test_spatial_zero_confidence_does_not_hide_a_known_hole_from_admission(self):
        from reconstruction.feature_evidence import known_empty_feature_guard
        c=self.constraint();mask=np.zeros((16,16),bool);mask[1:15,1:15]=True;mask[6:10,6:10]=False
        c=replace(c,mask=mask,uncertainty=replace(c.uncertainty,confidence=np.zeros(mask.shape)))
        prediction=mask.copy();prediction[6:10,6:10]=True
        self.assertFalse(known_empty_feature_guard(c,prediction)['passed'])

    def test_cpu_adapter_transports_original_maps_and_coarse_weighted_moments(self):
        from reconstruction.differentiable.candidate_adapter import run_refinement_candidate
        from reconstruction.differentiable.optimization import run_differentiable_optimization
        from reconstruction.types import CandidateRequest,CandidateBudget
        from primitives.analytic_primitives import EllipsoidPrimitive
        c=self.constraint();target=replace(target_for_masks({'front':c.mask}),constraints=(c,))
        captured=[]
        def inspect(**kwargs):
            captured.append(kwargs['target_record'])
            return run_differentiable_optimization(**kwargs)
        with tempfile.TemporaryDirectory() as root:
            request=CandidateRequest('pixel-fixture','differentiable_refine',target,config={
                'backend':'cpu_soft_silhouette','primitive_count':1,'include_bounds_proxy':False,
                'optimization_steps':1,'max_objective_evaluations':8,'optimization_resolution':32,
                'pixel_evidence_mode':'pixel_reliability_v1','calibrate_silhouette_bounds':False},
                budget=CandidateBudget(timeout_s=3.),artifact_root=Path(root))
            points=EllipsoidPrimitive(radii=(.5,.4,.6)).sample_surface(64)
            with patch('reconstruction.point_cloud.target_surface_points',return_value=(points,{'source':'fixture','count':64})), patch(
                    'reconstruction.differentiable.candidate_adapter.run_differentiable_optimization',side_effect=inspect):
                result=run_refinement_candidate(request)
            self.assertTrue(result.succeeded,result.errors)
            self.assertTrue(captured)
            record=captured[0];self.assertEqual(record.pixel_weights['front'].shape,record.silhouettes['front'].shape)
            self.assertEqual(record.pixel_weights['front'].shape,(32,32))
            self.assertEqual(record.probability_masks['front'].shape,record.pixel_weights['front'].shape)
            saved=np.load(result.artifacts['pixel_evidence_npz'])
            np.testing.assert_array_equal(saved['front__foreground'],c.uncertainty.foreground_prob)
            np.testing.assert_array_equal(saved['front__confidence'],c.uncertainty.confidence)
            saved.close()


if __name__=='__main__':unittest.main()
