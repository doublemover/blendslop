"""Compact hierarchy serialization and exact shared-grid boundary extraction."""
from itertools import product
from pathlib import Path
from unittest.mock import patch
import tempfile,unittest
import numpy as np
from volume.contracts import Bounds3D
from volume.dense import DenseVolumeGrid
from volume.hierarchical import HierarchicalOccupancyGrid
from volume.meshing import extract_mesh,surface_points
from volume.serialization import save_volume,load_volume


def boxes_for(data):
    boxes=[]
    def recurse(lo,hi):
        subset=data[tuple(slice(a,b) for a,b in zip(lo,hi))]
        if not subset.any():return
        if subset.all():boxes.append(np.r_[lo,hi]);return
        intervals=[[(a,b)] if b-a==1 else [(a,(a+b)//2),((a+b)//2,b)] for a,b in zip(lo,hi)]
        for axes in product(*intervals):recurse(np.array([x[0] for x in axes]),np.array([x[1] for x in axes]))
    recurse(np.zeros(3,int),np.array(data.shape))
    return boxes


class HierarchicalExtractionTests(unittest.TestCase):
    def bounds(self):return Bounds3D(-1.,1.,-1.,1.,-1.,1.)
    def grid(self,data):return HierarchicalOccupancyGrid(data.shape,self.bounds(),boxes_for(data))

    def test_lookup_chunks_and_counts_equal_explicit_volume_without_stored_dense_array(self):
        data=np.zeros((17,19,21),bool);data[3:15,5:17,4:19]=True;data[7:10,8:11,9:13]=False
        grid=self.grid(data);rng=np.random.default_rng(2);indices=rng.integers(-1,22,(2000,3))
        dense=DenseVolumeGrid(data,self.bounds())
        np.testing.assert_array_equal(grid.sample_world(grid.transform.index_to_world(indices)),dense.sample_world(dense.transform.index_to_world(indices)))
        self.assertEqual(grid.active_voxel_count(),int(data.sum()))
        np.testing.assert_array_equal(grid.to_dense(),data)
        self.assertFalse(hasattr(grid,'data'))
        reconstructed=np.zeros_like(data)
        for chunk in grid.iter_active_chunks():
            slices=tuple(slice(a,a+n) for a,n in zip(chunk.origin_index,chunk.valid_shape))
            reconstructed[slices]=chunk.data[tuple(slice(0,n) for n in chunk.valid_shape)]
        np.testing.assert_array_equal(reconstructed,data)

    def test_compact_roundtrip_never_materializes_full_volume(self):
        data=np.zeros((32,32,32),bool);data[8:24,8:24,8:24]=True;grid=self.grid(data)
        with tempfile.TemporaryDirectory() as root,patch.object(grid,'to_dense',side_effect=AssertionError('dense expansion')):
            metadata=save_volume(grid,root)
            self.assertEqual(metadata.backend,'hierarchical_occupancy')
            with np.load(Path(root)/'volume.npz',allow_pickle=False) as saved:
                self.assertEqual(set(saved.files),{'accepted_boxes'})
            recovered=load_volume(root)
            np.testing.assert_array_equal(recovered.accepted_boxes,grid.accepted_boxes)
            np.testing.assert_array_equal(recovered.to_dense(),data)

    def test_selective_mesh_matches_dense_vertices_and_oriented_volume_across_tile_seams(self):
        from reconstruction.grouped_solids import signed_volume,solid_guard
        from reconstruction.native_geometry import GeometryArrays
        data=np.zeros((64,64,64),bool);data[16:32,16:32,16:32]=True;grid=self.grid(data)
        expected=extract_mesh(DenseVolumeGrid(data,self.bounds()),method='marching_cubes')
        with patch.object(grid,'to_dense',side_effect=AssertionError('dense expansion')):
            actual=extract_mesh(grid)
        self.assertTrue(actual.available,actual.message)
        self.assertEqual(set(map(tuple,actual.vertices)),set(map(tuple,expected.vertices)))
        actual_data=GeometryArrays.capture(actual.vertices,actual.faces)
        self.assertTrue(solid_guard(actual_data)['valid_solid'])
        self.assertAlmostEqual(signed_volume(actual_data),signed_volume(GeometryArrays.capture(expected.vertices,expected.faces)))
        self.assertFalse(actual.metrics['global_dense_expansion'])
        self.assertLess(actual.metrics['sampled_nodes'],data.size)

    def test_hole_and_disconnected_blocks_do_not_get_bridged(self):
        from reconstruction.grouped_solids import solid_guard
        from reconstruction.native_geometry import GeometryArrays
        data=np.zeros((32,32,32),bool);data[3:12,3:12,3:12]=True;data[17:29,17:29,17:29]=True
        data[20:26,20:26,20:26]=False
        mesh=extract_mesh(self.grid(data))
        self.assertTrue(mesh.available,mesh.message)
        self.assertTrue(solid_guard(GeometryArrays.capture(mesh.vertices,mesh.faces))['valid_solid'])
        self.assertEqual(mesh.topology['connected_components'],3)

    def test_full_grid_padding_and_empty_hierarchy_have_explicit_output(self):
        full=self.grid(np.ones((16,16,16),bool));mesh=extract_mesh(full)
        self.assertTrue(mesh.available,mesh.message)
        np.testing.assert_array_equal(mesh.vertices.min(axis=0),[-1.,-1.,-1.])
        np.testing.assert_array_equal(mesh.vertices.max(axis=0),[1.,1.,1.])
        empty=extract_mesh(self.grid(np.zeros((16,16,16),bool)))
        self.assertEqual(empty.status,'skipped')

    def test_selective_surface_points_equal_dense_six_neighbor_boundary(self):
        data=np.zeros((33,19,17),bool);data[3:30,2:17,2:15]=True;data[12:20,7:12,5:10]=False
        grid=self.grid(data);expected=surface_points(DenseVolumeGrid(data,self.bounds()))
        with patch.object(grid,'to_dense',side_effect=AssertionError('dense expansion')):
            actual=surface_points(grid)
            points=extract_mesh(grid,method='points')
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(points.vertices,expected)
        with self.assertRaises(ValueError):surface_points(grid,max_voxels=1)

    def test_ambiguous_boolean_cells_on_seams_match_actual_dense_lewiner(self):
        data=np.zeros((33,19,17),bool)
        xx,yy,zz=np.indices((5,5,5));data[14:19,7:12,6:11]=(xx+yy+zz)%2==0
        actual=extract_mesh(self.grid(data));expected=extract_mesh(DenseVolumeGrid(data,self.bounds()))
        def triangles(mesh):
            return sorted(tuple(sorted(tuple(mesh.vertices[j]) for j in face)) for face in mesh.faces)
        self.assertEqual(triangles(actual),triangles(expected))
        self.assertEqual(actual.topology,expected.topology)

    def test_method_alias_level_and_tile_allowance_are_honest(self):
        grid=self.grid(np.ones((16,16,16),bool))
        legacy=extract_mesh(grid,method='marching_cubes_lorensen')
        self.assertTrue(legacy.available,legacy.message)
        self.assertEqual(legacy.requested_method,'marching_cubes_lorensen')
        self.assertEqual(legacy.metrics['skimage_method'],'lorensen')
        with patch.object(grid,'to_dense',side_effect=AssertionError('dense expansion')):
            self.assertEqual(extract_mesh(grid,level=.3).status,'unavailable')
        self.assertEqual(grid.extract_selective_mesh(maximum_tiles=0).status,'unavailable')

    def test_nonhierarchical_or_overlapping_intervals_are_rejected(self):
        for boxes in ([[0,0,0,8,8,8],[0,0,0,8,8,8]],[[1,1,1,7,7,7]],[[0,0,0,9,8,8]],[[0.5,0,0,8,8,8]],[[0,0,0,np.nan,8,8]]):
            with self.assertRaises(ValueError):HierarchicalOccupancyGrid((8,8,8),self.bounds(),boxes)


if __name__=='__main__':unittest.main()
