"""Artist pose aliases and asymmetric reflection share numeric geometry rules."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from reconstruction.program_transforms import rotation_matrix,position_vector,local_pose_edit,reflect_parameters


class ProgramPoseAliasTests(unittest.TestCase):
    def test_rigid_pose_absolute_tolerance_cannot_admit_hidden_axis_scale(self):
        frame=Rotation.from_rotvec([.2,-.3,.1]).as_matrix()
        scaled=frame@np.diag([1.+2e-6,1.,1.])
        for values in ({'rotation':scaled.tolist()},{'rotation_row_major':scaled.ravel().tolist()}):
            with self.assertRaisesRegex(ValueError,'proper orthonormal'):
                rotation_matrix(values)
            with self.assertRaisesRegex(ValueError,'proper orthonormal'):
                local_pose_edit(values,translation=[.1,0.,0.])
        np.testing.assert_array_equal(rotation_matrix({'rotation':frame.tolist()}),frame)

    def parameters(self):
        frame=Rotation.from_rotvec([.2,-.3,.1]).as_matrix()
        return {'location_x':.4,'location_y':-.2,'location_z':.7,
            'rotation_row_major':frame.ravel().tolist(),'width_world':1.2,'depth_world':.8,'height_world':1.7}

    def test_artist_row_major_frame_and_location_survive_local_edits(self):
        parameters=self.parameters();original=deepcopy(parameters);frame=rotation_matrix(parameters)
        moved=local_pose_edit(parameters,translation=[.1,-.2,.3],rotation_increment=[0.,0.,.07])
        np.testing.assert_allclose(position_vector(moved),position_vector(parameters)+frame@np.array([.1,-.2,.3]))
        np.testing.assert_allclose(rotation_matrix(moved),frame@Rotation.from_rotvec([0.,0.,.07]).as_matrix())
        np.testing.assert_allclose(np.array(moved['rotation_row_major']).reshape(3,3),rotation_matrix(moved))
        self.assertEqual(parameters,original)

    def test_numeric_screen_uses_the_same_artist_alias_pose(self):
        from primitives.shape_program import ShapeNode,ShapeProgram
        from reconstruction.proposal_screening import whole_program_geometry
        parameters=self.parameters()
        program=ShapeProgram('shape-program-v1','artist',(ShapeNode('a','add','ellipsoid',parameters),))
        data=whole_program_geometry(program,resolution=16)
        local=(data.vertices-position_vector(parameters))@rotation_matrix(parameters)
        np.testing.assert_allclose(local.min(axis=0),[-.6,-.4,-.85],atol=1e-14)
        np.testing.assert_allclose(local.max(axis=0),[.6,.4,.85],atol=1e-14)

    def test_numeric_cylinder_preserves_independent_dimensions_in_artist_frame(self):
        from primitives.shape_program import ShapeNode, ShapeProgram
        from reconstruction.proposal_screening import whole_program_geometry
        parameters = {**self.parameters(), 'width_world': 2., 'depth_world': 1., 'height_world': 3.}
        original = deepcopy(parameters)
        program = ShapeProgram('shape-program-v1', 'artist', (
            ShapeNode('a', 'add', 'cylinder', parameters),))
        data = whole_program_geometry(program, resolution=16)
        frame, center = rotation_matrix(parameters), position_vector(parameters)
        local = (data.vertices - center) @ frame
        np.testing.assert_allclose(local.min(axis=0), [-1., -.5, -1.5], atol=1e-14)
        np.testing.assert_allclose(local.max(axis=0), [1., .5, 1.5], atol=1e-14)
        np.testing.assert_allclose(data.vertices[0], np.array([1., 0., -1.5]) @ frame.T + center,
                                   atol=1e-14)
        self.assertEqual(parameters, original)

    def test_numeric_frustum_preserves_roll_about_a_tilted_axis(self):
        from primitives.shape_program import ShapeNode, ShapeProgram
        from reconstruction.proposal_screening import whole_program_geometry
        frame = Rotation.from_rotvec([.4, -.2, .1]).as_matrix()
        rolled = frame @ Rotation.from_euler('z', .8).as_matrix()
        np.testing.assert_array_equal(frame[:, 2], rolled[:, 2])
        parameters = {**self.parameters(), 'width_world': 2., 'depth_world': 1.,
                      'height_world': 3., 'radius_bottom': 1., 'radius_top': .4}
        rows = []
        for pose in (frame, rolled):
            posed = {**parameters, 'rotation_row_major': pose.ravel().tolist()}
            program = ShapeProgram('shape-program-v1', 'artist', (
                ShapeNode('a', 'add', 'frustum', posed),))
            rows.append(whole_program_geometry(program, resolution=16))
        center = position_vector(parameters)
        local = (rows[0].vertices - center) @ frame
        np.testing.assert_allclose(rows[1].vertices, local @ rolled.T + center, atol=1e-14)
        self.assertGreater(np.max(np.linalg.norm(rows[1].vertices - rows[0].vertices, axis=1)), .3)
        np.testing.assert_array_equal(rows[0].faces, rows[1].faces)
        self.assertEqual(rows[0].connectivity_hash, rows[1].connectivity_hash)

    def test_numeric_cone_aliases_agree_with_editable_compiler_dimensions(self):
        from primitives.shape_program import ShapeNode, ShapeProgram
        from primitives.shape_program_compiler import _cone_or_cylinder
        from reconstruction.proposal_screening import whole_program_geometry
        cases = (
            ({'diameter': 2., 'height': 3.}, (1., 1., 3., 1., 1.)),
            ({'radius': .75, 'height': 2.}, (.75, .75, 2., 1., 1.)),
            ({'radius_bottom': .9, 'height': 2.}, (.9, .9, 2., 1., 1.)),
            ({'width_world': 2., 'diameter': 9., 'depth_world': 1., 'height_world': 3.,
              'height': 9., 'radius': .8, 'radius_bottom': .7, 'radius_top': .35},
             (.7, .35, 3., 1., .5)),
            ({'width_world': 'invalid', 'diameter': 2., 'depth_world': None,
              'height_world': None, 'height': 3., 'radius_bottom': 'invalid',
              'radius': .6, 'radius_top': None}, (.6, .6, 3., 1., 1.)),
            ({'width_world': 0., 'depth_world': 2., 'height_world': 2.},
             (1., 1., 2., 1e-6, 1.)),
        )
        for primitive in ('cylinder', 'frustum'):
            for parameters, expected in cases:
                with self.subTest(primitive=primitive, parameters=parameters):
                    calls = []
                    obj = SimpleNamespace(scale=SimpleNamespace(x=1., y=1.))
                    fake = SimpleNamespace(context=SimpleNamespace(active_object=obj),
                        ops=SimpleNamespace(mesh=SimpleNamespace(
                            primitive_cone_add=lambda **kwargs: calls.append(kwargs))))
                    with patch('primitives.shape_program_compiler.bpy', fake):
                        _cone_or_cylinder(name='a', params=parameters, vertices=16)
                    native = calls[0]
                    self.assertEqual((native['radius1'], native['radius2'], native['depth'],
                                      obj.scale.x, obj.scale.y), expected)
                    program = ShapeProgram('shape-program-v1', 'artist', (
                        ShapeNode('a', 'add', primitive, parameters),))
                    data = whole_program_geometry(program, resolution=16)
                    bottom, top, height, scale_x, scale_y = expected
                    for z, radius in ((-height * .5, bottom), (height * .5, top)):
                        ring = data.vertices[data.vertices[:, 2] == z]
                        np.testing.assert_allclose(ring[:, :2].min(axis=0),
                            [-radius * scale_x, -radius * scale_y], atol=1e-14)
                        np.testing.assert_allclose(ring[:, :2].max(axis=0),
                            [radius * scale_x, radius * scale_y], atol=1e-14)
                    np.testing.assert_array_equal(data.vertices[:, 2].min(), -height * .5)
                    np.testing.assert_array_equal(data.vertices[:, 2].max(), height * .5)

    def test_numeric_cone_default_identity_mesh_is_unchanged(self):
        from primitives.shape_program import ShapeNode, ShapeProgram
        from primitives.superfrustum import SuperFrustum
        from reconstruction.native_geometry import GeometryArrays
        from reconstruction.proposal_screening import whole_program_geometry
        mesh = SuperFrustum(position=(0., 0., 0.), orientation=(0., 0.),
                            radius_bottom=.5, radius_top=.5, height=1.).to_mesh_data(16)
        triangles = np.asarray([(face[0], face[i], face[i + 1])
            for face in mesh.faces for i in range(1, len(face) - 1)], int)
        original = GeometryArrays.capture(mesh.vertices, triangles)
        for primitive in ('cylinder', 'frustum'):
            with self.subTest(primitive=primitive):
                program = ShapeProgram('shape-program-v1', 'artist', (
                    ShapeNode('a', 'add', primitive, {}),))
                result = whole_program_geometry(program)
                np.testing.assert_array_equal(result.vertices, original.vertices)
                np.testing.assert_array_equal(result.faces, original.faces)
                self.assertEqual(result.content_hash, original.content_hash)
                self.assertEqual(result.connectivity_hash, original.connectivity_hash)

    def test_compiler_dispatch_reads_row_major_pose_before_native_euler_conversion(self):
        from primitives.shape_program import ShapeNode
        from primitives.shape_program_compiler import _compile_node
        class FakeObject(dict):pass
        obj=FakeObject();captured=[]
        def matrix(values):
            captured.append(values)
            return SimpleNamespace(to_euler=lambda:(.1,.2,.3))
        parameters=self.parameters()
        with patch('primitives.shape_program_compiler._sphere',return_value=obj), patch.dict(sys.modules,{
                'mathutils':SimpleNamespace(Matrix=matrix)}):
            result,_=_compile_node(ShapeNode('a','add','ellipsoid',parameters),lathe_segments=16,
                bevel_modifier=False,weighted_normals=False)
        self.assertIs(result,obj)
        np.testing.assert_allclose(captured[0],rotation_matrix(parameters))
        self.assertEqual(obj.rotation_euler,(.1,.2,.3))

    def test_asymmetric_sweep_reflection_changes_actual_local_centerline(self):
        from primitives.generalized_sweep import GeneralizedSweepPrimitive
        parameters={**self.parameters(),'section_knots_normalized':[
            [-.5,.1,-.1,.4,.3],[0.,.25,.15,.3,.4],[.5,-.1,.2,.45,.35]]}
        reflected=reflect_parameters(parameters,axis=1,plane=.3)
        original=GeneralizedSweepPrimitive.from_program_parameters(parameters)
        proposal=GeneralizedSweepPrimitive.from_program_parameters(reflected)
        query=np.random.default_rng(8).normal(size=(100,3));mirror=query.copy();mirror[:,1]=.6-mirror[:,1]
        np.testing.assert_allclose(original.sdf_batch(query),proposal.sdf_batch(mirror),atol=2e-15)
        self.assertGreater(np.linalg.det(rotation_matrix(reflected)),0.)

    @unittest.skipUnless(importlib.util.find_spec('shapely'),'optional Shapely')
    def test_asymmetric_polygon_reflection_preserves_holes_and_field(self):
        from primitives.polygon_extrusion import PolygonExtrusionPrimitive
        parameters={**self.parameters(),'outer':[[-.6,-.4],[.6,-.4],[.6,0.],[0.,0.],[0.,.4],[-.6,.4]],
            'holes':[[[-.45,-.25],[-.2,-.25],[-.2,-.1],[-.45,-.1]]]}
        reflected=reflect_parameters(parameters,axis=0,plane=-.1)
        def part(row):
            return PolygonExtrusionPrimitive(row['outer'],row['holes'],center=position_vector(row),
                rotation=rotation_matrix(row),height=row['height_world'])
        original,proposal=part(parameters),part(reflected)
        query=np.random.default_rng(4).normal(size=(100,3));mirror=query.copy();mirror[:,0]=-.2-mirror[:,0]
        np.testing.assert_allclose(original.sdf_batch(query),proposal.sdf_batch(mirror),atol=1e-14)

    def test_unsupported_absolute_cloud_reflection_is_not_faked_by_pose(self):
        with self.assertRaisesRegex(ValueError,'geometry-specific'):
            reflect_parameters({'points_world':[[0.,0.,0.],[1.,1.,1.]]})


if __name__=='__main__':unittest.main()
