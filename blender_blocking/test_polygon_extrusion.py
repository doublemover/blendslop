"""Editable concave/perforated plate field, cap and proposal invariants."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from primitives.polygon_extrusion import PolygonExtrusionPrimitive,triangulate_polygon
from primitives.shape_program import ShapeNode,ShapeProgram,validate_compilable_program
from reconstruction.grouped_solids import solid_guard,signed_volume
from reconstruction.native_geometry import GeometryArrays
from reconstruction.polygon_proposals import planar_extrusion_programs
from reconstruction.proposal_screening import whole_program_geometry,screen_whole_programs
from reconstruction.types import Bounds3D
from test_geometry_consistency import triangles
from test_quality_geometry import target_for_masks


try:
    from shapely import constrained_delaunay_triangles
    SHAPELY_AVAILABLE=True
except ImportError:
    SHAPELY_AVAILABLE=False


@unittest.skipUnless(SHAPELY_AVAILABLE,"optional Shapely>=2.1 constrained triangulation unavailable")
class PolygonExtrusionTests(unittest.TestCase):
    outer=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    hole=np.array([[-.5,-.5],[-.5,.5],[.5,.5],[.5,-.5]])
    def test_perforated_plate_field_caps_and_material_volume(self):
        part=PolygonExtrusionPrimitive(self.outer,(self.hole,),height=.2)
        mesh=part.to_mesh_data();data=GeometryArrays.capture(mesh.vertices,triangles(mesh))
        self.assertTrue(solid_guard(data)['valid_solid'])
        self.assertAlmostEqual(signed_volume(data),.6,places=12)
        values=part.sdf_batch(np.array([[0.,0.,0.],[.75,0.,0.],[.75,0.,.1],[1.2,0.,.25]]))
        np.testing.assert_allclose(values,[.5,-.1,0.,.25],atol=1e-14)
        np.testing.assert_allclose(part.sdf_batch(mesh.vertices),0.,atol=1e-14)
        np.testing.assert_allclose(part.sdf_batch(part.sample_surface(111)),0.,atol=1e-14)

    def test_concave_cap_uses_constraints_instead_of_a_convex_fill(self):
        outline=np.array([[0.,0.],[2.,0.],[2.,1.],[1.,1.],[1.,2.],[0.,2.]])
        part=PolygonExtrusionPrimitive(outline,height=.3)
        mesh=part.to_mesh_data();data=GeometryArrays.capture(mesh.vertices,triangles(mesh))
        self.assertAlmostEqual(signed_volume(data),.9,places=12)
        self.assertGreater(part.sdf_batch(np.array([[1.5,1.5,0.]]))[0],0.)
        self.assertTrue(solid_guard(data)['valid_solid'])

    def test_pose_units_and_serialized_reload_preserve_the_geometry(self):
        frame=Rotation.from_rotvec([.2,-.3,.1]).as_matrix()
        part=PolygonExtrusionPrimitive(self.outer,(self.hole,),center=(2.,-1.,.3),rotation=frame,height=.2,scale_xy=(.8,1.2))
        restored=PolygonExtrusionPrimitive.from_dict(part.to_dict())
        np.testing.assert_allclose(restored.to_mesh_data().vertices,part.to_mesh_data().vertices,atol=1e-14)
        points=part.sample_surface(64)
        np.testing.assert_allclose(part.sdf_batch(points),0.,atol=1e-14)
        scaled=PolygonExtrusionPrimitive(self.outer*100.,(self.hole*100.,),center=part.center*100.,rotation=frame,height=20.,scale_xy=part.scale_xy)
        queries=np.array([[1.5,-1.,.3],[2.,-1.,.3],[2.3,-1.,.5]])
        np.testing.assert_allclose(scaled.sdf_batch(queries*100.)/100.,part.sdf_batch(queries),atol=1e-14)

    def test_invalid_loops_and_touching_holes_fail_explicitly(self):
        with self.assertRaises(ValueError):PolygonExtrusionPrimitive([[0,0],[1,1],[0,1],[1,0]])
        with self.assertRaises(ValueError):PolygonExtrusionPrimitive(self.outer,([[-1,-.5],[-.5,-.5],[-.5,.5],[-1,.5]],))
        with self.assertRaises(ValueError):PolygonExtrusionPrimitive([[0,0],[1,0],[1,0],[0,1]])

    def test_program_compiler_contract_advertises_the_actual_new_representation(self):
        program=ShapeProgram('shape-program-v1','plate',(ShapeNode('a','add','polygon_extrusion',{
            'outer':self.outer.tolist(),'holes':[self.hole.tolist()],'width_world':2.,'depth_world':2.,'height_world':.2}),))
        self.assertEqual(validate_compilable_program(program),())
        data=whole_program_geometry(program)
        self.assertAlmostEqual(signed_volume(data),.6,places=12)

    def test_planar_proposal_uses_pixel_cell_edges_and_preserves_known_hole(self):
        mask=np.zeros((32,32),bool);mask[4:28,4:28]=True;mask[13:19,13:19]=False
        target=target_for_masks({'top':mask})
        target=replace(target,bounds=Bounds3D.from_min_max((-1.,-1.,-.05),(1.,1.,.05)))
        seed=ShapeProgram('shape-program-v1','p',())
        proposals=planar_extrusion_programs(target,seed)
        self.assertEqual(len(proposals),1)
        self.assertEqual(proposals[0].metadata['hole_count'],1)
        ranked=screen_whole_programs(target,proposals)
        self.assertTrue(ranked)
        self.assertTrue(ranked[0][2]['known_empty_features']['passed'])
        self.assertGreater(ranked[0][2]['coarse_mean_iou'],.8)
        thick=replace(target,bounds=Bounds3D.from_min_max((-1.,)*3,(1.,)*3))
        self.assertEqual(planar_extrusion_programs(thick,seed),[])

    def test_partial_outline_is_not_invented_as_a_closed_observation(self):
        mask=np.zeros((16,16),bool);mask[2:14,2:14]=True
        target=target_for_masks({'top':mask});valid=np.ones_like(mask);valid[:,8:]=False
        target=replace(target,bounds=Bounds3D.from_min_max((-1.,-1.,-.05),(1.,1.,.05)),
            constraints=(replace(target.constraints[0],valid_mask=valid),))
        with self.assertRaisesRegex(ValueError,'fully observed'):
            planar_extrusion_programs(target,ShapeProgram('shape-program-v1','p',()))


if __name__=='__main__':unittest.main()
