"""Geometry and artist-edit contracts for the production triangular pebble."""
import unittest
import numpy as np

from primitives.rounded_triangle import RoundedTrianglePrimitive
from primitives.shape_program import ShapeNode, ShapeProgram, validate_compilable_program
from reconstruction.proposal_screening import whole_program_geometry
from reconstruction.grouped_solids import solid_guard
from synthetic.quality_contracts import rounded_triangle_mesh
from evaluation.surface_quality import circular_profile_continuity


class RoundedTriangleTests(unittest.TestCase):
    def test_production_default_preserves_frozen_reference(self):
        vertices, faces = rounded_triangle_mesh()
        data = RoundedTrianglePrimitive().to_mesh_data()
        np.testing.assert_array_equal(data.vertices, vertices)
        np.testing.assert_array_equal(data.faces, faces)
        program = ShapeProgram("1", "triangle", (ShapeNode("dot", "add", "rounded_triangle"),))
        self.assertEqual(validate_compilable_program(program), ())
        self.assertTrue(solid_guard(whole_program_geometry(program, resolution=16))["valid_solid"])

    def test_pose_serialization_and_independent_depth_edits(self):
        angle = .4
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        part = RoundedTrianglePrimitive(front_fraction=.7, thickness=.6,
                                        center=(2, -1, 3), rotation=rotation)
        mesh = part.to_mesh_data()
        self.assertAlmostEqual(mesh.vertices[:, 2].max(), 3.42)
        self.assertAlmostEqual(mesh.vertices[:, 2].min(), 2.82)
        restored = RoundedTrianglePrimitive.from_dict(part.to_dict())
        np.testing.assert_allclose(restored.to_mesh_data().vertices, mesh.vertices, atol=1e-14)
        self.assertAlmostEqual(np.ptp(mesh.vertices[:, 2]), .6)

    def test_signed_field_handles_interior_poles_and_asymmetric_depths(self):
        part = RoundedTrianglePrimitive(front_fraction=.7, thickness=.6)
        points = [[0, 0, 0], [0, 0, .42], [0, 0, -.18], [0, 0, .5], [2, 0, 0]]
        field = part.sdf_batch(points)
        self.assertLess(field[0], 0)
        np.testing.assert_allclose(field[1:3], 0, atol=1e-14)
        self.assertGreater(field[3], 0)
        self.assertGreater(field[4], 0)
        self.assertEqual(part.sdf_batch(np.empty((0, 3))).shape, (0,))

    def test_invalid_recipe_rejected_before_compilation(self):
        for values in ({"front_fraction": 0}, {"corner_segments": 8.5},
                       {"dome_segments": True}, {"corner_radius": -1},
                       {"thickness": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RoundedTrianglePrimitive(**values)
        node = ShapeNode("dot", "add", "rounded_triangle", {"front_fraction": 2})
        self.assertTrue(validate_compilable_program(ShapeProgram("1", "bad", (node,))))

    def test_program_local_dimensions_and_refinement_controls_change_actual_geometry(self):
        from reconstruction.backends.shape_program.geometry_search import parameter_variants
        base = RoundedTrianglePrimitive()
        absolute = RoundedTrianglePrimitive.from_program_parameters({"width_world":2.3,"depth_world":1.2})
        vertices = absolute.to_mesh_data().vertices
        np.testing.assert_allclose(np.ptp(vertices[:,:2],axis=0),[2.3,1.2],atol=.001)
        program = ShapeProgram("1","dot",(ShapeNode("dot","add","rounded_triangle"),))
        variants = parameter_variants(program,limit=64)
        baseline = whole_program_geometry(program,resolution=16).content_hash
        for variant in variants:
            with self.subTest(control=variant.metadata["refinement_control"]):
                self.assertNotEqual(whole_program_geometry(variant,resolution=16).content_hash,baseline)
        kinds = {variant.metadata["refinement_control"]["kind"] for variant in variants}
        self.assertTrue({"triangle_corner","triangle_dome_balance"}.issubset(kinds))
        size = [v for v in variants if v.metadata["refinement_control"]["kind"]=="size" and
                v.metadata["refinement_control"]["axis"]==0 and v.metadata["refinement_control"]["direction"]==1][0]
        edited = RoundedTrianglePrimitive.from_program_parameters(size.root_nodes[0].parameters)
        self.assertAlmostEqual(edited.scale_xy[0],np.exp(.06))
        self.assertEqual(edited.thickness,base.thickness)

    def test_reflection_preserves_asymmetric_authored_outline_and_front_back_profile(self):
        from reconstruction.program_transforms import reflect_parameters
        from scipy.spatial.transform import Rotation
        parameters={"vertices_xy":[[-.2,.9],[-.7,-.5],[1.,-.4]],"corner_radius_world":.12,
                    "height_world":.6,"front_fraction":.7,"scale_xy":[1.2,.8],
                    "x":2.,"y":-.4,"z":.3,"rotation":Rotation.from_rotvec([.3,.2,.4]).as_matrix().tolist()}
        part=RoundedTrianglePrimitive.from_program_parameters(parameters)
        directions=Rotation.from_rotvec([.6,-.3,.1]).apply(np.eye(3))
        original=part.to_mesh_data(32).vertices
        mirrored=RoundedTrianglePrimitive.from_program_parameters(reflect_parameters(parameters,axis=0,plane=.5)).to_mesh_data(32).vertices
        expected=original.copy();expected[:,0]=1.-expected[:,0]
        np.testing.assert_allclose(np.max(mirrored@directions.T,axis=0),np.max(expected@directions.T,axis=0),atol=.002)
        restored=RoundedTrianglePrimitive.from_program_parameters(reflect_parameters(reflect_parameters(parameters,axis=0,plane=.5),axis=0,plane=.5))
        np.testing.assert_allclose(restored.to_mesh_data(32).vertices,original,atol=1e-14)

    def test_continuity_detects_shoulders_despite_correct_outer_radius(self):
        z = np.linspace(0, 2.6, 65)
        radii = .6 + .2 * np.cos(2 * np.pi * z / 2.6)
        vertices = np.column_stack((radii, np.zeros(len(z)), z))
        self.assertTrue(circular_profile_continuity(vertices)["passed"])
        stepped = np.vstack((vertices, [[radii[12] + .02, 0, z[12]]]))
        result = circular_profile_continuity(stepped)
        self.assertFalse(result["passed"])
        self.assertEqual(result["shoulder_ring_count"], 1)
        self.assertAlmostEqual(result["planar_shoulder_width_max_world"], .02)


if __name__ == "__main__":
    unittest.main()
