"""Residual crop selection never manufactures an unobserved reference view."""
import unittest
import numpy as np
from evaluation.adaptive_measurement import residual_crop_request


class TestAdaptiveMeasurement(unittest.TestCase):
    def test_pixel_cell_crop_moves_camera_and_preserves_orientation(self):
        ref=np.zeros((64,64));ref[8:32,8:32]=1.
        cand=ref.copy();cand[8,8]=0.
        matrix=np.eye(4);matrix[:3,3]=[10.,20.,4.]
        camera={'projection':'ORTHO','matrix_world':matrix.tolist(),'ortho_scale':4.,
                'resolution':[64,64],'pixel_aspect':[1.,1.],'clip_start':.1,'clip_end':100.}
        result=residual_crop_request(ref,cand,camera,pixel_span=16)
        self.assertEqual(result['source_pixel_cell_bounds'],[0,0,16,16])
        self.assertEqual(result['camera']['ortho_scale'],1.)
        np.testing.assert_array_equal(np.asarray(result['camera']['matrix_world'])[:3,3],[8.5,21.5,4.])
        np.testing.assert_array_equal(np.asarray(camera['matrix_world'])[:3,3],[10.,20.,4.])

    def test_no_residual_does_not_request_new_work(self):
        image=np.zeros((64,64))
        camera={'projection':'ORTHO','matrix_world':np.eye(4).tolist(),'ortho_scale':4.,'resolution':[64,64]}
        self.assertEqual(residual_crop_request(image,image,camera,pixel_span=16)['status'],'unneeded')

    def test_shifted_and_non_square_contracts_are_explicitly_refused(self):
        image=np.zeros((64,64))
        camera={'projection':'ORTHO','matrix_world':np.eye(4).tolist(),'ortho_scale':4.,'resolution':[64,64],'shift_x':.1}
        with self.assertRaises(ValueError):residual_crop_request(image,image,camera,pixel_span=16)
        with self.assertRaises(ValueError):residual_crop_request(image[:32],image[:32],camera,pixel_span=16)
