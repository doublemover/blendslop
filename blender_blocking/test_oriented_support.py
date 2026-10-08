"""Support-fit algebra and bounded fixture solves; no reconstruction campaign."""
from dataclasses import replace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from reconstruction.oriented_support import SupportEvidence, primitive_support, support_evidence, support_residual, fit_whole_support
from test_quality_geometry import target_for_masks


class OrientedSupportTests(unittest.TestCase):
    def evidence(self,family,center,dimensions,rotation):
        directions=[]
        for axes in ((0,2),(1,2),(0,1)):
            for angle in np.linspace(0.,2*np.pi,32,endpoint=False):
                u=np.zeros(3);u[list(axes)]=[np.cos(angle),np.sin(angle)];directions.append(u)
        directions=np.asarray(directions)
        values=primitive_support(family,directions,center,dimensions,rotation)
        return SupportEvidence(directions,values,np.zeros(len(values)),np.zeros(len(values),bool),np.ones(len(values)),3.)

    def test_box_support_matches_all_eight_transformed_vertices(self):
        frame=Rotation.from_rotvec([.3,-.2,.1]).as_matrix();radii=np.array([.7,.4,1.])
        evidence=self.evidence('box',[.2,-.3,.4],radii,frame)
        corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*radii@frame.T+[.2,-.3,.4]
        np.testing.assert_allclose(evidence.values,np.max(corners@evidence.directions.T,axis=0),atol=1e-14)

    def test_cylinder_and_frustum_special_case_agree(self):
        frame=Rotation.from_rotvec([.3,-.2,.1]).as_matrix()
        evidence=self.evidence('cylinder',[.2,-.3,.4],[.4,1.],frame)
        np.testing.assert_allclose(evidence.values,primitive_support('frustum',evidence.directions,[.2,-.3,.4],[.4,.4,1.],frame))
        self.assertAlmostEqual(primitive_support('cylinder',np.array([[0.,0.,1.]]),[0.,0.,0.],[.4,1.],np.eye(3))[0],1.)

    def test_capsule_and_triangle_support_match_independent_tessellated_geometry(self):
        from primitives.capsule import CapsulePrimitive
        from primitives.rounded_triangle import RoundedTrianglePrimitive
        frame=Rotation.from_rotvec([.3,-.2,.1]).as_matrix()
        center=np.array([.2,-.3,.4])
        parts=(("capsule",[.4,.6],CapsulePrimitive(radius=.4,segment_height=1.2,rotation=frame,center=center).to_mesh_data(128)),
               ("rounded_triangle",[1.2,.8,.13,.42,.18],RoundedTrianglePrimitive(scale_xy=[1.2,.8],corner_radius=.13,thickness=.6,front_fraction=.7,rotation=frame,center=center).to_mesh_data(64)))
        for family,dims,mesh in parts:
            evidence=self.evidence(family,center,dims,frame)
            sampled=np.max(mesh.vertices@evidence.directions.T,axis=0)
            self.assertTrue(np.all(evidence.values >= sampled - 1e-12))
            np.testing.assert_allclose(evidence.values,sampled,atol=.0004,rtol=0.)

    def test_bounded_triangle_support_fit_improves_independent_mesh_observations(self):
        from primitives.rounded_triangle import RoundedTrianglePrimitive
        dims=np.array([1.2,.8,.13,.42,.18]);center=np.array([.2,-.3,.4]);frame=Rotation.from_rotvec([.3,-.2,.1]).as_matrix()
        evidence=self.evidence("rounded_triangle",center,dims,frame)
        mesh=RoundedTrianglePrimitive(scale_xy=dims[:2],corner_radius=dims[2],thickness=.6,front_fraction=.7,rotation=frame,center=center).to_mesh_data(64)
        evidence=replace(evidence,values=np.max(mesh.vertices@evidence.directions.T,axis=0),tolerances=np.full(len(evidence.values),.0004))
        initial=primitive_support("rounded_triangle",evidence.directions,center+[.05,-.03,.02],dims*1.05,frame)
        before=float(np.sum(support_residual(evidence,initial)**2))
        result=fit_whole_support("rounded_triangle",evidence,center=center+[.05,-.03,.02],dimensions=dims*1.05,rotation=frame,max_evaluations=160,max_elapsed_s=2.)
        self.assertLess(result["support_squared_residual"],before*.01)
        self.assertLessEqual(result["support_evaluations"],160)

    def test_coupled_box_fixture_recovers_observed_supports(self):
        frame=Rotation.from_rotvec([.3,-.2,.1]).as_matrix();dims=np.array([.7,.4,1.]);center=np.array([.2,-.3,.4])
        evidence=self.evidence('box',center,dims,frame)
        result=fit_whole_support('box',evidence,center=center+[.05,-.03,.02],dimensions=dims*1.05,
                                 rotation=frame@Rotation.from_rotvec([.02,-.02,.01]).as_matrix(),max_evaluations=160,max_elapsed_s=2.)
        self.assertLess(result['support_squared_residual'],1e-12)
        self.assertLessEqual(result['support_evaluations'],160)

    def test_coupled_ellipsoid_fixture_recovers_observed_supports(self):
        frame=Rotation.from_rotvec([.3,-.2,.1]).as_matrix();dims=np.array([.7,.4,1.]);center=np.array([.2,-.3,.4])
        evidence=self.evidence('ellipsoid',center,dims,frame)
        result=fit_whole_support('ellipsoid',evidence,center=center+[.05,-.03,.02],dimensions=dims*1.05,
                                 rotation=frame,max_evaluations=160,max_elapsed_s=2.)
        self.assertLess(result['support_squared_residual'],1e-12)

    def test_allowance_retains_completed_seed(self):
        evidence=self.evidence('box',[0.,0.,0.],[1.,.7,.5],np.eye(3))
        result=fit_whole_support('box',evidence,center=[.1,0.,0.],dimensions=[1.,.7,.5],rotation=np.eye(3),max_evaluations=1)
        self.assertEqual(result['support_evaluations'],1)
        self.assertEqual(result['termination'],'support_evaluation_allowance')
        np.testing.assert_array_equal(result['center'],[.1,0.,0.])

    def test_unknown_support_is_one_sided_and_foreground_invariant(self):
        mask=np.zeros((24,24),bool);mask[6:18,6:18]=True
        target=target_for_masks({'front':mask})
        valid=np.ones_like(mask);valid[:,18:]=False
        c=replace(target.constraints[0],valid_mask=valid)
        a=support_evidence(replace(target,constraints=(c,)))
        changed=mask.copy();changed[~valid]=True
        b=support_evidence(replace(target,constraints=(replace(c,mask=changed),)))
        np.testing.assert_array_equal(a.values,b.values);np.testing.assert_array_equal(a.censored,b.censored)
        index=np.flatnonzero(a.censored)[0]
        higher=a.values.copy();higher[index]+=1.
        self.assertEqual(support_residual(a,higher)[index],0.)
        lower=a.values.copy();lower[index]-=1.
        self.assertLess(support_residual(a,lower)[index],0.)

    def test_support_values_and_objective_commute_with_units(self):
        frame=Rotation.from_rotvec([.3,-.2,.1]).as_matrix()
        evidence=self.evidence('cylinder',[.2,-.3,.4],[.4,1.],frame)
        prediction=primitive_support('cylinder',evidence.directions,[.3,-.3,.4],[.4,1.],frame)
        scaled=replace(evidence,values=evidence.values*100.,tolerances=evidence.tolerances*100.,length_scale=evidence.length_scale*100.)
        np.testing.assert_allclose(support_residual(evidence,prediction),support_residual(scaled,prediction*100.),atol=1e-14)

    def test_known_hole_precludes_convex_whole_proposal(self):
        from reconstruction.program_proposals import whole_primitive_programs
        from primitives.shape_program import ShapeProgram
        mask=np.ones((20,20),bool);mask[8:12,8:12]=False
        self.assertEqual(whole_primitive_programs(target_for_masks({'front':mask}),ShapeProgram('shape-program-v1','p',())),[])


if __name__=='__main__':unittest.main()
