"""Opaque fit and export invariants; these fixtures are not reconstruction scores."""
from dataclasses import replace
from unittest.mock import patch
import tempfile,json
from pathlib import Path
import unittest
import numpy as np
from primitives.analytic_primitives import AnisotropicGaussianPrimitive,EllipsoidPrimitive
from primitives.ellipsoid_proxy.fitting import (OpaqueProxyObjective,fit_proxy,hard_evidence,
    admissible,unique_foreground,missing_region_seed)
from reconstruction.mesh_io import combine_primitive_meshes,mesh_arrays_from_object
from reconstruction.projected_metrics import projected_mesh_masks
from test_quality_geometry import target_for_masks


class FittedProxyTests(unittest.TestCase):
    def target(self,parts):
        empty=np.zeros((32,32),bool)
        target=target_for_masks({name:empty for name in ('front','side','top')})
        return target_for_masks(projected_mesh_masks(target,*mesh_arrays_from_object(combine_primitive_meshes(parts,resolution=20))))

    def test_confidence_and_opacity_cannot_improve_opaque_geometry_loss(self):
        a=AnisotropicGaussianPrimitive(covariance=np.diag([.25,.16,.36]),opacity=.1,confidence=.2)
        target=self.target([a]);objective=OpaqueProxyObjective(target,resolution=24)
        first=objective([a]).total
        b=AnisotropicGaussianPrimitive(center=a.center,covariance=a.covariance,opacity=1.,confidence=1.)
        self.assertEqual(first,objective([b]).total)
        self.assertAlmostEqual(first,np.dot(objective.residual_vector([a]),objective.residual_vector([a])))

    def test_unknown_foreground_is_ignored_even_in_fractional_coarse_cells(self):
        part=EllipsoidPrimitive(radii=(.6,.5,.7));target=self.target([part]);c=target.constraints[0]
        valid=np.ones_like(c.mask);valid[:,17:]=False
        one=replace(target,constraints=(replace(c,valid_mask=valid),))
        changed=c.mask.copy();changed[~valid]=~changed[~valid]
        two=replace(target,constraints=(replace(c,mask=changed,valid_mask=valid),))
        a=OpaqueProxyObjective(one,resolution=13);b=OpaqueProxyObjective(two,resolution=13)
        np.testing.assert_array_equal(a.residual_vector([part]),b.residual_vector([part]))

    def test_analytic_loss_priority_gradient_matches_local_adapters(self):
        from copy import deepcopy
        from placement.resfit_parameters import apply_parameter_increment
        from placement.resfit_optimizer import ParameterBounds
        truth=EllipsoidPrimitive(radii=(.6,.5,.7));target=self.target([truth])
        part=AnisotropicGaussianPrimitive(center=(-.1,.03,0.),covariance=np.diag([.3,.16,.36]),opacity=.2,confidence=.3)
        objective=OpaqueProxyObjective(target,resolution=24);gradients=objective.parameter_gradients([part])
        for ref in ((0,'center',0),(0,'covariance_cholesky',0),(0,'covariance_cholesky',3)):
            plus=deepcopy([part]);minus=deepcopy([part]);epsilon=1e-6
            apply_parameter_increment(plus,ref,epsilon,ParameterBounds())
            apply_parameter_increment(minus,ref,-epsilon,ParameterBounds())
            numeric=(objective(plus).total-objective(minus).total)/(2*epsilon)
            self.assertAlmostEqual(numeric,gradients[ref],places=6)

    def test_coupled_geometry_moves_are_scored_but_admission_keeps_safe_best(self):
        truth=EllipsoidPrimitive(radii=(.55,.45,.6));target=self.target([truth])
        seed=EllipsoidPrimitive(center=(-.2,0.,0.),radii=truth.radii)
        parts,report=fit_proxy([seed],target,np.zeros((0,3)),family='ellipsoid',
            config={'proxy_fit_evaluations':80,'proxy_fit_iterations':1},timeout_s=3.)
        stage=report['fit_history'][0]
        self.assertLess(stage['surrogate_best'],stage['surrogate_initial'])
        self.assertTrue(stage['parameter_visits'])
        self.assertFalse(any(row[1] in {'opacity','density','confidence'} for row in stage['parameter_visits']))
        retained=hard_evidence(target,parts);initial=hard_evidence(target,[seed])
        self.assertGreaterEqual(retained['minimum'],initial['minimum'])
        self.assertTrue(all(retained['empty'][v]<=initial['empty'][v] for v in initial['empty']))

    def test_exact_mesh_improvement_admitted_but_hole_fill_rejected(self):
        truth=EllipsoidPrimitive(radii=(.6,.5,.7));target=self.target([truth])
        seed=EllipsoidPrimitive(center=(.2,0.,0.),radii=truth.radii)
        self.assertTrue(admissible(hard_evidence(target,[truth]),hard_evidence(target,[seed])))
        mask=np.ones((32,32),bool);mask[12:20,12:20]=False
        hole_target=target_for_masks({'front':mask})
        candidate=hard_evidence(hole_target,[EllipsoidPrimitive(radii=(1.,1.,1.))])
        self.assertFalse(admissible(candidate,candidate,allow_equal=True))

    def test_duplicate_component_has_no_unique_foreground_and_is_pruned(self):
        part=EllipsoidPrimitive(radii=(.6,.5,.7));target=self.target([part])
        np.testing.assert_array_equal(unique_foreground(target,[part,part]),[0,0])
        parts,report=fit_proxy([part,part],target,np.zeros((0,3)),family='ellipsoid',
            config={'proxy_fit_evaluations':1,'proxy_fit_iterations':1},timeout_s=3.)
        self.assertEqual(len(parts),1)
        self.assertTrue(any(row['stage']=='unique_coverage_prune' for row in report['fit_history']))

    def test_missing_component_seed_is_backed_by_real_observed_region(self):
        left=EllipsoidPrimitive(center=(-.55,0.,0.),radii=(.2,.3,.3))
        right=EllipsoidPrimitive(center=(.55,0.,0.),radii=(.2,.3,.3))
        target=self.target([left,right]);current=hard_evidence(target,[left])
        points=right.sample_surface(128)
        proposal=missing_region_seed(target,current,points,family='ellipsoid')
        self.assertIsNotNone(proposal)
        self.assertGreater(proposal.center[0],.3)
        self.assertIsNone(missing_region_seed(target,current,left.sample_surface(128),family='ellipsoid'))

    def test_gaussian_eigenframe_does_not_invert_mesh_winding(self):
        from reconstruction.grouped_solids import solid_guard
        from reconstruction.native_geometry import GeometryArrays
        from test_geometry_consistency import triangles
        for covariance in (np.diag([.5,.1,.3]),np.array([[.4,.05,0.],[.05,.2,.03],[0.,.03,.1]])):
            gaussian=AnisotropicGaussianPrimitive(covariance=covariance)
            ellipsoid=gaussian.to_ellipsoid()
            self.assertGreater(np.linalg.det(ellipsoid.rotation),0.)
            mesh=gaussian.to_mesh_data(12)
            self.assertTrue(solid_guard(GeometryArrays.capture(mesh.vertices,triangles(mesh)))['valid_solid'])

    def test_fitted_variant_refuses_editable_isosurface_mismatch(self):
        from primitives.ellipsoid_proxy.config import validate_gaussian_ellipsoid_config
        errors=validate_gaussian_ellipsoid_config({'proxy_variant':'fitted_opaque_union_v1','editable_proxy_sigma':2.})
        self.assertTrue(any('isosurface' in row for row in errors))
        self.assertFalse(validate_gaussian_ellipsoid_config({'proxy_variant':'initializer_only','editable_proxy_sigma':2.}))

    def test_backend_writes_actual_fit_receipt_and_matching_editable_geometry(self):
        from primitives.ellipsoid_proxy.backend_adapter import run_gaussian_ellipsoid_proxy
        from reconstruction.types import CandidateRequest,CandidateBudget
        part=EllipsoidPrimitive(radii=(.6,.5,.7));target=self.target([part])
        with tempfile.TemporaryDirectory() as root:
            request=CandidateRequest('fit-fixture','gaussian_ellipsoid_proxy',target,
                config={'family':'gaussian','primitive_count':1,'include_bounds_proxy':False,
                    'proxy_variant':'fitted_opaque_union_v1','proxy_fit_evaluations':1,
                    'target_point_count':128,'distillation_resolution':8},
                artifact_root=Path(root),budget=CandidateBudget(timeout_s=3.))
            with patch('primitives.ellipsoid_proxy.backend_adapter.target_surface_points',
                    return_value=(part.sample_surface(128),{'source':'unit_fixture','count':128})):
                result=run_gaussian_ellipsoid_proxy(request)
            self.assertTrue(result.succeeded,result.errors)
            self.assertTrue(result.metric_result.extras['geometry_fit']['optimization_performed'])
            payload=json.loads(result.primitive_path.read_text())
            self.assertEqual(payload['metadata']['geometry_fit']['variant'],'fitted_opaque_union_v1')
            nodes=result.metric_result.extras['editable_proxy']['shape_program']['root_nodes']
            for primitive,node in zip(result.payload,nodes):
                physical=primitive.to_ellipsoid()
                np.testing.assert_allclose([node['parameters']['radius_'+a+'_world'] for a in 'xyz'],physical.radii)
                np.testing.assert_allclose(node['parameters']['rotation_row_major'],physical.rotation.ravel())

    def test_quality_chooses_fitted_variant_only_without_explicit_alternative(self):
        from reconstruction.quality_config import quality_config
        self.assertEqual(quality_config({'quality_preset':'quality'})['proxy_variant'],'fitted_opaque_union_v1')
        self.assertEqual(quality_config({'quality_preset':'quality','proxy_variant':'initializer_only'})['proxy_variant'],'initializer_only')


if __name__=='__main__':unittest.main()
