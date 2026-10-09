"""Triangle corner response must not accept scale or depth edits as radius edits."""
from dataclasses import replace
import unittest

import numpy as np
from primitives.rounded_triangle import RoundedTrianglePrimitive
from reconstruction.native_geometry import GeometryArrays
from reconstruction.triangle_edit_observation import triangle_corner_response


class TriangleEditObservationTests(unittest.TestCase):
    def data(self, params):
        mesh = RoundedTrianglePrimitive.from_program_parameters(params).to_mesh_data()
        return GeometryArrays.capture(mesh.vertices, mesh.faces)

    def test_radius_edit_support_response_and_depth_preservation(self):
        p = {'corner_radius_world':.16,'scale_xy':[1.2,.8],'height_world':.48,
             'front_fraction':.7}
        old = self.data(p)
        for change, accepted in (({'corner_radius_world':.176},True),
                                 ({'height_world':.5},False),
                                 ({'scale_xy':[1.3,.9]},False)):
            result = triangle_corner_response(old, self.data({**p,**change}), np.eye(4), p)
            self.assertEqual(result['passed'],accepted)

    def test_pose_is_removed_only_for_edit_observation(self):
        p = {'corner_radius_world':.16,'height_world':.48}
        old,new=self.data(p),self.data({**p,'corner_radius_world':.176})
        angle=.7;matrix=np.eye(4)
        matrix[:3,:3]=[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
        matrix[:3,3]=[2.,-1.,.3]
        def transform(data):
            return GeometryArrays.capture(data.vertices @ matrix[:3,:3].T + matrix[:3,3],data.faces)
        self.assertTrue(triangle_corner_response(transform(old),transform(new),matrix,p)['passed'])
        self.assertFalse(triangle_corner_response(transform(old),transform(new),np.eye(4),p)['passed'])

    def test_unsupported_tessellation_and_invalid_pose_are_explicit(self):
        p={'corner_radius_world':.16};data=self.data(p)
        for matrix,parameters in ((np.zeros((4,4)),p),
                                  (np.eye(4),{**p,'corner_segments':16})):
            with self.assertRaises(ValueError):
                triangle_corner_response(data,data,matrix,parameters)


if __name__=='__main__':
    unittest.main()
