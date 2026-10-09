"""Specific arch opening/tube radius controls and meaningful physical checks."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_structured_family_source_edits import edited_recipe,response

class TestStructuredFamilySourceEdits(unittest.TestCase):
    def test_arch_control_only_moves_inner_walls_not_outer_or_roof(self):
        outline=[[-.9,-.9],[-.9,.9],[.9,.9],[.9,-.9],[.5,-.9],[.5,.4],[-.5,.4],[-.5,-.9]]
        wire={"root_nodes":[{"primitive_type":"polygon_extrusion","parameters":{"outer":outline,"holes":[],"height_world":.4}}]}
        snapshot=deepcopy(wire);edited,_=edited_recipe("concave_arch",wire)
        points=edited["root_nodes"][0]["parameters"]["outer"]
        self.assertEqual(points[:4],outline[:4])
        self.assertEqual([p[1] for p in points],[p[1] for p in outline])
        self.assertEqual(points[4][0],.55);self.assertEqual(points[7][0],-.55)
        self.assertEqual(wire,snapshot)

    def test_arch_outer_scaling_or_roof_motion_does_not_pass_opening_edit(self):
        before={"extents_world":[1.8,1.8,.4],"opening_width_world":1.,"inner_roof_local_y":.4}
        after={"extents_world":[1.8,1.8,.4],"opening_width_world":1.1,"inner_roof_local_y":.4}
        self.assertTrue(response("concave_arch",before,after)["passed"])
        after["extents_world"][0]=1.98
        self.assertFalse(response("concave_arch",before,after)["passed"])
        after["extents_world"][0]=1.8;after["inner_roof_local_y"]+=.01
        self.assertFalse(response("concave_arch",before,after)["passed"])

    def test_tube_edit_keeps_major_radius_and_narrows_hole(self):
        wire={"root_nodes":[{"primitive_type":"torus","parameters":{"major_radius":.7,"minor_radius":.2}}]}
        edited,_=edited_recipe("torus",wire)
        self.assertEqual(edited["root_nodes"][0]["parameters"]["major_radius"],.7)
        self.assertAlmostEqual(edited["root_nodes"][0]["parameters"]["minor_radius"],.21)
        before={"extents_world":[1.8,1.8,.4],"tube_radius_world":.2,"major_radius_world":.7,"outer_radius_world":.9,"inner_radius_world":.5}
        after={"extents_world":[1.82,1.82,.42],"tube_radius_world":.21,"major_radius_world":.7,"outer_radius_world":.91,"inner_radius_world":.49}
        self.assertTrue(response("torus",before,after)["passed"])
        after["major_radius_world"]+=.01
        self.assertFalse(response("torus",before,after)["passed"])

if __name__=="__main__":unittest.main()
