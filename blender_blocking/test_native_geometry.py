"""Geometry content invalidation, ownership and native/scalar query fixtures."""
import unittest
import numpy as np
from blender_blocking.reconstruction.native_geometry import GeometryArrays,GeometryCache,NativeOwnedGeometry,evaluated_arrays
try:
    import bpy
except ImportError:
    bpy=None


def tetra():
    return GeometryArrays.capture([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],[[0,2,1],[0,1,3],[0,3,2],[1,2,3]])


class NativeGeometryTests(unittest.TestCase):
    def test_coordinates_invalidate_bvh_and_connectivity_invalidates_topology(self):
        a=tetra();b=GeometryArrays.capture(a.vertices+1,a.faces)
        self.assertNotEqual(a.content_hash,b.content_hash);self.assertEqual(a.connectivity_hash,b.connectivity_hash)
        cache=GeometryCache();cache.topology_report(a);cache.topology_report(b)
        self.assertEqual(cache.hits,1)
        c=GeometryArrays.capture(a.vertices,a.faces[:-1]);cache.topology_report(c)
        self.assertEqual(cache.misses,2)
        with self.assertRaises(ValueError):a.vertices[0]=0

    def test_nontriangle_connectivity_is_rejected_before_reshape(self):
        vertices = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
        with self.assertRaisesRegex(ValueError, "triangle"):
            GeometryArrays.capture(vertices, [[0,1,2,3]] * 3)
        with self.assertRaisesRegex(ValueError, "triangle"):
            GeometryArrays.capture(vertices, [0,1,2])

    @unittest.skipIf(bpy is None,'native Blender required')
    def test_native_mesh_survives_scene_reset_and_releases_owned_data(self):
        from blender_blocking.integration.blender_ops.scene_setup import setup_scene
        a=tetra();owner=NativeOwnedGeometry(a,'OwnedFixture')
        name=owner.obj.name;meshname=owner.obj.data.name
        setup_scene(clear_existing=True)
        self.assertIn(name,bpy.data.objects)
        owner.attach();copied=evaluated_arrays(owner.obj)
        np.testing.assert_allclose(copied.vertices,a.vertices,atol=0)
        np.testing.assert_array_equal(copied.faces,a.faces)
        owner.detach();owner.release()
        self.assertNotIn(name,bpy.data.objects);self.assertNotIn(meshname,bpy.data.meshes)

    @unittest.skipIf(bpy is None, 'native Blender required')
    def test_raycast_and_proximity_match_scalar_fixtures_with_clipping(self):
        from blender_blocking.reconstruction.native_queries import raycast_batch,proximity_batch,scalar_raycast
        a=tetra();cache=GeometryCache()
        origins=np.array([[.1,.1,2],[2,2,2],[0,0,2],[.4,.4,2]],float)
        directions=np.tile([0,0,-1],(len(origins),1))
        for near,far in ((0,4),(0,1),(1.3,4)):
            scalar=scalar_raycast(a,origins,directions,near=near,far=far,cache=cache)
            native=raycast_batch(a,origins,directions,near=near,far=far)
            np.testing.assert_array_equal(native['native_hit'],scalar['native_hit'])
            np.testing.assert_allclose(native['native_distance'],scalar['native_distance'],atol=2e-6)
        proximity=proximity_batch(a,[[0,0,2],[2,0,0],[0,0,0]])
        np.testing.assert_allclose(proximity['native_distance'],[1,1,0],atol=2e-6)

    @unittest.skipIf(bpy is None,'native Blender required')
    def test_boolean_exact_and_manifold_receipt_gate(self):
        from blender_blocking.reconstruction.native_csg import boolean_mesh
        a=tetra()
        with self.assertRaisesRegex(ValueError,'qualification'):
            boolean_mesh(a,a,solver='MANIFOLD')
        result,receipt=boolean_mesh(a,a,operation='UNION')
        self.assertFalse(receipt['same_output_optimization'])
        report=GeometryCache().topology_report(result)
        self.assertTrue(report['watertight'])
        self.assertEqual(report['connected_components'],1)


    @unittest.skipIf(bpy is None, 'native Blender required')
    def test_sdf_extracts_zero_level_with_a_nonempty_closed_self_union(self):
        from blender_blocking.reconstruction.native_csg import sdf_grid_mesh
        data = tetra()
        result, receipt = sdf_grid_mesh(data, data, operation="UNION", resolution=128)
        self.assertGreater(len(result.faces), 0)
        topology = GeometryCache().topology_report(result)
        self.assertTrue(topology["watertight"])
        self.assertEqual(topology["connected_components"], 1)
        self.assertFalse(receipt["same_output_optimization"])
        np.testing.assert_allclose(result.vertices.min(axis=0), data.vertices.min(axis=0), atol=2/128)
        np.testing.assert_allclose(result.vertices.max(axis=0), data.vertices.max(axis=0), atol=2/128)

    def test_pixel_centers_and_top_down_axes(self):
        from blender_blocking.reconstruction.native_queries import orthographic_pixel_rays
        p,d=orthographic_pixel_rays([-1,1,-1,1],2,2,horizontal_axis=0,vertical_axis=2,depth_axis=1,depth=-3,direction_sign=1)
        np.testing.assert_array_equal(p,[[-.5,-3,.5],[.5,-3,.5],[-.5,-3,-.5],[.5,-3,-.5]])
        np.testing.assert_array_equal(d,np.tile([0,1,0],(4,1)))


if __name__=='__main__':unittest.main()
