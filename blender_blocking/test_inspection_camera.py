"""Projected display framing preserves frozen declarations and exposes clip limits."""
import copy
import unittest
import numpy as np
from evaluation.inspection_camera import inspection_frame_bounds, unclipped_inspection_camera


class InspectionCameraTests(unittest.TestCase):
    def record(self):
        m=np.eye(4);m[:3,3]=[4.,-2.,10.]
        return {'projection':'ORTHO','matrix_world':m.tolist(),'ortho_scale':1.,'shift_x':.1,'shift_y':-.2,
                'clip_start':.1,'clip_end':100.,'resolution':[512,512],'pixel_aspect':[1.,1.]}

    def test_reframes_combined_projected_extent_without_touching_gate_camera(self):
        original=self.record();saved=copy.deepcopy(original)
        points=np.array([[x,y,z] for x in (-2.,2.) for y in (-1.,1.) for z in (-.5,.5)])
        self.assertFalse(inspection_frame_bounds(original,points)['projection_enclosed'])
        actual=unclipped_inspection_camera(original,points,padding_fraction=.08)
        self.assertEqual(original,saved)
        np.testing.assert_array_equal(np.asarray(actual['matrix_world'])[:3,:3],np.eye(3))
        self.assertEqual(actual['clip_start'],original['clip_start']);self.assertEqual(actual['clip_end'],original['clip_end'])
        bounds=inspection_frame_bounds(actual,points)
        self.assertTrue(bounds['projection_enclosed']);self.assertTrue(bounds['clipping_enclosed'])
        self.assertEqual(actual['ortho_scale'],4.*1.16)
        self.assertEqual(actual['shift_x'],0.);self.assertEqual(actual['shift_y'],0.)

    def test_rotated_camera_uses_camera_plane_rather_than_world_bounds(self):
        r=self.record();angle=np.pi/4;m=np.asarray(r['matrix_world'])
        m[:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
        r['matrix_world']=m.tolist();points=np.array([[-3.,0.,0.],[3.,0.,0.],[0.,.5,0.]])
        new=unclipped_inspection_camera(r,points)
        self.assertAlmostEqual(new['ortho_scale'],6./np.sqrt(2)*1.16)
        self.assertTrue(inspection_frame_bounds(new,points)['projection_enclosed'])

    def test_clipping_or_unsupported_projection_is_explicitly_blocked(self):
        for changes in [{'clip_end':1.},{'clip_start':0.},{'projection':'PERSP'},{'resolution':[512,256]},
                        {'pixel_aspect':[2.,1.]},{'ortho_scale':float('nan')}]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                unclipped_inspection_camera({**self.record(),**changes},[[-1.,0.,0.],[1.,0.,0.]])
        for fraction in [True,0.,-.1,float('nan'),1.]:
            with self.subTest(fraction=fraction),self.assertRaises(ValueError):
                unclipped_inspection_camera(self.record(),[[-1.,0.,0.],[1.,0.,0.]],padding_fraction=fraction)


if __name__=='__main__':unittest.main()
