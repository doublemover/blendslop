"""Hierarchy queries and extracted world coordinates use the same cell centers."""
from dataclasses import replace
import unittest
import numpy as np
from reconstruction.adaptive_geometry import hierarchical_hull
from reconstruction.visibility import point_support
from test_quality_geometry import target_for_masks


class AdaptiveWorldGridTests(unittest.TestCase):
    def test_nonconservative_hierarchy_equals_actual_cell_center_ray_queries(self):
        masks={}
        for view in ('front','side','top'):
            mask=np.zeros((23,31),bool);mask[4:18,8:24]=True;masks[view]=mask
        target=target_for_masks(masks);grid=hierarchical_hull(target,12,conservative=False)
        indices=np.stack(np.meshgrid(*[np.arange(12)]*3,indexing='ij'),axis=-1).reshape(-1,3)
        expected,_=point_support(target,grid.transform.index_to_world(indices))
        np.testing.assert_array_equal(grid.to_dense().ravel(),expected)

    def test_partial_unknown_rows_and_legacy_camera_keep_query_equivalence(self):
        mask=np.zeros((19,27),bool);mask[3:15,5:19]=True
        target=target_for_masks({'front':mask});c=target.constraints[0]
        valid=np.ones_like(mask);valid[:,18:]=False
        target=replace(target,constraints=(replace(c,valid_mask=valid,camera=replace(c.camera,bounds=None)),))
        grid=hierarchical_hull(target,10,conservative=False)
        indices=np.stack(np.meshgrid(*[np.arange(10)]*3,indexing='ij'),axis=-1).reshape(-1,3)
        expected,_=point_support(target,grid.transform.index_to_world(indices))
        np.testing.assert_array_equal(grid.to_dense().ravel(),expected)

    def test_adaptive_dispatch_compact_save_and_selective_extraction_integrate(self):
        from reconstruction.point_cloud.hull import visual_hull_grid_from_target
        from volume.serialization import save_volume,load_volume
        from volume.meshing import extract_mesh
        from unittest.mock import patch
        import tempfile
        mask=np.zeros((24,24),bool);mask[5:19,5:19]=True
        target=target_for_masks({v:mask for v in ('front','side','top')})
        grid=visual_hull_grid_from_target(target,resolution=16,adaptive=True)
        self.assertEqual(grid.backend,'hierarchical_occupancy')
        with tempfile.TemporaryDirectory() as root,patch.object(grid,'to_dense',side_effect=AssertionError('dense expansion')):
            save_volume(grid,root)
            restored=load_volume(root)
            with patch.object(restored,'to_dense',side_effect=AssertionError('dense expansion')):
                result=extract_mesh(restored)
        self.assertTrue(result.available,result.message)
        self.assertFalse(result.metrics['global_dense_expansion'])


if __name__=='__main__':unittest.main()
