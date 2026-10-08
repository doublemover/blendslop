"""Generated field winding/normal semantics, independent of historical mesh repair."""
from unittest.mock import patch
from types import SimpleNamespace
import unittest
import numpy as np
from volume import DenseVolumeGrid,extract_mesh
from volume.contracts import Bounds3D
from reconstruction.native_geometry import GeometryArrays
from reconstruction.grouped_solids import solid_guard


class VolumeWindingContractTests(unittest.TestCase):
    def grid(self,kind='occupancy',*,shape=(4,4,4),bounds=None):
        data=np.ones(shape,dtype=bool) if kind=='occupancy' else np.ones(shape,dtype=float)
        if kind=='signed_distance':data[1:3,1:3,1:3]=-1.
        return DenseVolumeGrid(data,bounds or Bounds3D(-1.,1.,-1.,1.,-1.,1.),value_type='occupancy_bool' if kind=='occupancy' else kind)

    def test_scalar_sign_selects_an_explicit_face_convention(self):
        calls=[]
        def marching(field,**kwargs):
            calls.append(kwargs)
            return np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),np.array([[0,1,2]]),np.tile([1.,1.,1.],(3,1)),np.ones(3)
        module=SimpleNamespace(measure=SimpleNamespace(marching_cubes=marching))
        with patch.dict('sys.modules',{'skimage':module}):
            a=extract_mesh(self.grid());b=extract_mesh(self.grid('signed_distance'))
        self.assertEqual(calls[0]['gradient_direction'],'ascent')
        self.assertEqual(calls[1]['gradient_direction'],'descent')
        self.assertFalse(calls[0]['allow_degenerate'])
        self.assertEqual(a.metrics['winding_contract'],'right-handed outward material boundary')

    def test_normals_follow_inverse_anisotropic_scale_and_field_sign(self):
        def marching(field,**kwargs):
            return np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),np.array([[0,1,2]]),np.tile([1.,1.,1.],(3,1)),np.ones(3)
        bounds=Bounds3D(0.,4.,0.,8.,0.,12.)
        with patch.dict('sys.modules',{'skimage':SimpleNamespace(measure=SimpleNamespace(marching_cubes=marching))}):
            a=extract_mesh(self.grid(bounds=bounds));b=extract_mesh(self.grid('signed_distance',bounds=bounds))
        expected=np.array([1.,.5,1./3.]);expected/=np.linalg.norm(expected)
        np.testing.assert_allclose(a.normals,np.tile(expected,(3,1)))
        np.testing.assert_allclose(b.normals,-a.normals)

    def test_actual_small_generated_occupancy_and_sdf_are_outward_when_available(self):
        try:
            from skimage import measure
        except ImportError:
            self.skipTest('actual scikit-image runtime unavailable; only adapter contract fixtures ran')
        for kind in ('occupancy','signed_distance'):
            result=extract_mesh(self.grid(kind))
            self.assertEqual(result.status,'ok',result.message)
            self.assertTrue(solid_guard(GeometryArrays.capture(result.vertices,result.faces))['valid_solid'])


if __name__=='__main__':unittest.main()
