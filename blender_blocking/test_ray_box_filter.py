"""Exact camera pixel-cell area targets, with an independent overlap oracle."""
from dataclasses import replace
import unittest
import numpy as np
from reconstruction.differentiable.ray_evidence import prepare_ray_targets
from reconstruction.projection_contract import pixel_cell_viewport,project_vertices
from test_quality_geometry import target_for_masks


class RayBoxFilterTests(unittest.TestCase):
    def test_exact_area_preserves_a_thin_strip_missed_by_two_point_quadrature(self):
        mask=np.zeros((32,32),bool);mask[:,12]=True
        target=target_for_masks({'front':mask})
        exact=prepare_ray_targets(target,np.zeros(3),1.,8)['front']
        legacy=prepare_ray_targets(target,np.zeros(3),1.,8,filter_mode='subcell_quadrature')['front']
        self.assertEqual(float(exact['foreground'].sum()),2.)
        self.assertGreater(np.max(np.abs(exact['foreground']-legacy['foreground'])),0.)
        np.testing.assert_array_equal(exact['valid'],np.ones((8,8)))

    def test_fractional_partial_cells_match_independent_world_overlap_integral(self):
        rng=np.random.default_rng(12);mask=rng.random((11,13))>.5;valid=rng.random(mask.shape)>.3
        target=target_for_masks({'front':mask});c=replace(target.constraints[0],valid_mask=valid)
        target=replace(target,constraints=(c,));n=7;scale=1.2;center=np.array([.13,0.,-.07])
        actual=prepare_ray_targets(target,center,scale,n)['front']
        edges=[center[axis]-scale+np.arange(n+1)/n*2*scale for axis in (0,2)]
        h,w=mask.shape;xedges=np.linspace(-1.,1.,w+1);yedges=np.linspace(1.,-1.,h+1)
        known=np.zeros((n,n));foreground=np.zeros_like(known);area=(2*scale/n)**2
        for row in range(n):
            for col in range(n):
                for y in range(h):
                    dy=max(0.,min(edges[1][row+1],yedges[y])-max(edges[1][row],yedges[y+1]))
                    for x in range(w):
                        dx=max(0.,min(edges[0][col+1],xedges[x+1])-max(edges[0][col],xedges[x]))
                        known[row,col]+=dx*dy*valid[y,x]
                        foreground[row,col]+=dx*dy*(valid[y,x]&mask[y,x])
        np.testing.assert_allclose(actual['valid'],known/area,atol=2e-15)
        np.testing.assert_allclose(actual['foreground'],np.divide(foreground,known,out=np.zeros_like(known),where=known>0.),atol=1e-13,rtol=1e-12)

    def test_unknown_foreground_and_camera_external_regions_make_no_assertion(self):
        mask=np.zeros((9,13),bool);valid=np.ones_like(mask);valid[:,6:]=False
        target=target_for_masks({'front':mask});c=replace(target.constraints[0],valid_mask=valid)
        target=replace(target,constraints=(c,));changed=mask.copy();changed[~valid]=True
        other=replace(target,constraints=(replace(c,mask=changed),))
        first=prepare_ray_targets(target,np.zeros(3),2.,7)['front']
        second=prepare_ray_targets(other,np.zeros(3),2.,7)['front']
        np.testing.assert_array_equal(first['foreground'],second['foreground'])
        np.testing.assert_array_equal(first['valid'],second['valid'])
        self.assertTrue(np.any((first['valid']>0.)&(first['valid']<1.)))
        self.assertTrue(np.all(first['valid'][0]==0.))

    def test_legacy_endpoint_center_viewport_is_converted_once(self):
        mask=np.ones((9,13),bool);target=target_for_masks({'front':mask})
        c=replace(target.constraints[0],camera=replace(target.constraints[0].camera,bounds=None))
        target=replace(target,constraints=(c,));axes,(u0,u1,v0,v1)=pixel_cell_viewport(target,c)
        query=np.array([[-1.,0.,-1.],[0.,0.,0.],[1.,0.,1.]])
        expected=project_vertices(target,c,query)
        converted=np.column_stack(((query[:,axes[0]]-u0)/(u1-u0)*13-.5,(v1-query[:,axes[1]])/(v1-v0)*9-.5))
        np.testing.assert_allclose(converted,expected,atol=1e-15)

    def test_invalid_filter_mode_is_not_silently_quadrature(self):
        with self.assertRaises(ValueError):
            prepare_ray_targets(target_for_masks({'front':np.ones((8,8),bool)}),np.zeros(3),1.,8,filter_mode='unsupported')

    def test_legacy_profile_and_support_share_the_same_pixel_cell_edges(self):
        from reconstruction.profile_evidence import observed_profile_rows
        from reconstruction.oriented_support import support_evidence
        mask=np.zeros((9,13),bool);mask[2:7,4:9]=True
        target=target_for_masks({'front':mask});c=replace(target.constraints[0],camera=replace(target.constraints[0].camera,bounds=None))
        target=replace(target,constraints=(c,))
        _,(u0,u1,v0,v1)=pixel_cell_viewport(target,c)
        row=observed_profile_rows(target,c,[0.])[0]
        self.assertAlmostEqual(row['exact_radius_world'],5/13*(u1-u0)*.5)
        self.assertAlmostEqual(row['center_world'],u0+6.5/13*(u1-u0))
        support=support_evidence(target,directions_per_view=8)
        positive_x=np.flatnonzero(np.all(support.directions==[1.,0.,0.],axis=1))[0]
        self.assertAlmostEqual(support.values[positive_x],u0+9/13*(u1-u0))

    def test_global_thin_screen_is_rotation_invariant_and_not_a_local_certificate(self):
        from reconstruction.differentiable.ray_evidence import global_thin_ray_screen
        from scipy.spatial.transform import Rotation
        corners=np.array([[x,y,z] for x in (-.6,.6) for y in (-.01,.01) for z in (-.7,.7)])
        plain=global_thin_ray_screen(corners,1.,32)
        moved=global_thin_ray_screen(corners@Rotation.from_rotvec([.7,.2,-.3]).as_matrix().T+[2.,-1.,.4],1.,32)
        np.testing.assert_allclose(plain['global_pca_widths'],moved['global_pca_widths'],atol=1e-14)
        self.assertTrue(plain['obviously_subvoxel_thin'])
        self.assertTrue(plain['global_envelope_below_two_voxels'])
        self.assertIn('local thin',plain['scope'])


if __name__=='__main__':unittest.main()
