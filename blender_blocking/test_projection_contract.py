import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'blender_blocking')]
from blender_blocking.verify_setup import configure_dependency_paths
configure_dependency_paths()
from reconstruction.projection_contract import bounds_from_calibrated_masks, validate_view_calibration, permits_bounds_seed, project_vertices
from reconstruction.types import Bounds2D, Bounds3D
from reconstruction.targets import make_reconstruction_target

class ProjectionContractTests(unittest.TestCase):
    def test_padding_changes_viewport_not_geometry(self):
        from dataclasses import replace
        mask=np.zeros((101,101), bool); mask[30:71,20:81]=True
        t=make_reconstruction_target(masks_by_view={'front':mask}, bboxes_by_view={'front':Bounds2D(20,30,81,71)}, bounds=Bounds3D(-3,3,-1,1,-2,2))
        pixels=project_vertices(t,t.constraints[0],np.array([[-3,0,-2],[3,0,2]]))
        self.assertTrue(np.allclose(pixels, [[20,70],[80,30]]))
    def test_calibrated_crop_scale_and_inconsistency(self):
        masks={v:np.zeros((100,100),bool) for v in ['front','side','top']}
        boxes={'front':Bounds2D(25,20,75,80),'side':Bounds2D(30,20,70,80),'top':Bounds2D(25,30,75,70)}
        records={v:{'world_bounds':[-2,2,-2,2], 'projection':'orthographic'} for v in masks}
        b=bounds_from_calibrated_masks(masks,boxes,records)
        self.assertTrue(np.allclose(b.to_min_max(), [[-1,-.8,-1.2],[1,.8,1.2]]))
        records['side']['world_bounds']=[-2,2,-1,3]
        with self.assertRaisesRegex(ValueError,'inconsistent'): bounds_from_calibrated_masks(masks,boxes,records)
        with self.assertRaisesRegex(ValueError,'perspective'): validate_view_calibration({'front':{'projection':'perspective','world_bounds':[-1,1,-1,1]}})
    def test_unit_and_axis_mismatch_are_rejected(self):
        base = {'projection': 'orthographic', 'world_bounds': [-1, 1, -1, 1]}
        with self.assertRaisesRegex(ValueError, 'world_units'):
            validate_view_calibration({'front': {**base, 'world_units': 'centimetres'}})
        with self.assertRaisesRegex(ValueError, 'axes'):
            validate_view_calibration({'front': {**base, 'axes': [1, 2]}})

    def test_negative_space_prevents_bounds_seed(self):
        mask=np.ones((20,20),bool); mask[8:12,8:12]=False
        t=make_reconstruction_target(masks_by_view={'top':mask})
        allowed, why=permits_bounds_seed(t)
        self.assertFalse(allowed);self.assertEqual(why['observed_hole_pixels']['top'],16)
        mask[:]=True
        self.assertTrue(permits_bounds_seed(make_reconstruction_target(masks_by_view={'top':mask}))[0])

if __name__=='__main__': unittest.main(argv=[sys.argv[0]])
