"""Specific recipe controls and independent physical response expectations."""
from copy import deepcopy
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_family_source_edit_check import edited_recipe, response_contract, _publish_receipt


class TestFamilySourceEdits(unittest.TestCase):
    def test_completed_receipt_digest_tracks_final_publication(self):
        from tempfile import TemporaryDirectory
        from utils.run_ownership import OwnedRun, plan_run_reclamation
        with TemporaryDirectory() as temporary:
            owner = OwnedRun(temporary, producer="source-edit-publication-check")
            with owner:
                _publish_receipt(owner, {"status": "running", "cases": {}})
                _publish_receipt(owner, {"status": "completed", "cases": {"one": {"passed": True}}})
            plan = plan_run_reclamation(owner.root)
            self.assertEqual(plan["status"], "dry_run_ready")
            self.assertEqual(plan["unknown_files"], [])

    def test_recipe_edit_preserves_pose_and_original_parameters(self):
        p = dict(width_world=2., depth_world=1., height_world=3., radius_bottom=.7,
                 radius_top=.3, radius_world=.4, segment_height_world=2.2,
                 corner_radius_world=.12, rotation=[[1,0,0],[0,1,0],[0,0,1]],x=.1,y=.2,z=.3)
        original = {"root_nodes": [{"parameters": p}]}
        expected = {"sphere": {"width_world", "depth_world", "height_world"},
                    "anisotropic_ellipsoid": {"depth_world"}, "thin_plate": {"depth_world"},
                    "cylinder": {"radius_bottom", "radius_top", "width_world", "depth_world"},
                    "tapered_frustum": {"radius_top"},
                    "capsule": {"segment_height_world", "height_world"},
                    "rounded_box": {"corner_radius_world"}}
        snapshot = deepcopy(original)
        for name, changed_keys in expected.items():
            edited, _ = edited_recipe(name, original)
            actual = {key for key,value in edited["root_nodes"][0]["parameters"].items() if value != p[key]}
            self.assertEqual(actual, changed_keys)
            self.assertEqual(original, snapshot)
        with self.assertRaises(ValueError):
            edited_recipe("unknown", original)

    def test_wrong_physical_axis_response_is_rejected(self):
        before = {"extents_world": [2.,1.,3.]}
        correct = {"extents_world": [2.,1.05,3.]}
        wrong = {"extents_world": [2.1,1.,3.]}
        for family in ("anisotropic_ellipsoid", "thin_plate"):
            self.assertTrue(response_contract(family, before, correct, {})["passed"])
            self.assertFalse(response_contract(family, before, wrong, {})["passed"])

    def test_capsule_straight_length_and_frustum_top_radius_are_independent(self):
        before = {"extents_world": [1.,1.,3.]}
        after = {"extents_world": [1.,1.,3.1]}
        self.assertTrue(response_contract("capsule",before,after,{"segment_height_world":2.})["passed"])
        self.assertFalse(response_contract("capsule",before,after,{"segment_height_world":1.})["passed"])
        before.update(top_radius_world=.3,bottom_radius_world=.7)
        after = {"extents_world": [1.,1.,3.],"top_radius_world":.33,"bottom_radius_world":.7}
        self.assertTrue(response_contract("tapered_frustum",before,after,{})["passed"])
        after["bottom_radius_world"] = .71
        self.assertFalse(response_contract("tapered_frustum",before,after,{})["passed"])


if __name__ == "__main__":
    unittest.main()
