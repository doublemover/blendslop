"""Style comparison is descriptive and does not manufacture quality gates."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_reference_neutral_inspection import style_comparison


class TestReferenceNeutralStyle(unittest.TestCase):
    def test_polygon_style_mismatch_is_retained_separately(self):
        source = {"evaluated_mesh": {"polygon_style": "smooth", "sharp_edges": 0, "has_custom_normals": False}, "modifiers": []}
        candidate = deepcopy(source)
        candidate["evaluated_mesh"]["polygon_style"] = "flat"
        row = style_comparison(source, candidate)
        self.assertFalse(row["polygon_style_match"])
        self.assertTrue(row["modifier_type_sequence_match"])
        self.assertNotIn("surface_passed", row)
        self.assertNotIn("qualified_limits", row)
        self.assertEqual(source["evaluated_mesh"]["polygon_style"], "smooth")

    def test_authored_weighted_normals_are_reported_without_smoothing(self):
        source = {"evaluated_mesh": {"polygon_style": "flat", "sharp_edges": 0, "has_custom_normals": True},
                  "modifiers": [{"type": "BEVEL"}, {"type": "WEIGHTED_NORMAL"}]}
        candidate = deepcopy(source)
        candidate["modifiers"].pop()
        candidate["evaluated_mesh"]["has_custom_normals"] = False
        row = style_comparison(source, candidate)
        self.assertTrue(row["polygon_style_match"])
        self.assertFalse(row["modifier_type_sequence_match"])
        self.assertFalse(row["custom_normal_state_match"])


    def test_weighted_normal_controls_mismatch_blocks_style_equivalence_description(self):
        source = {"evaluated_mesh": {"polygon_style": "flat", "sharp_edges": 0, "has_custom_normals": True},
                  "modifiers": [{"type": "WEIGHTED_NORMAL", "keep_sharp": False}]}
        candidate = deepcopy(source)
        candidate["modifiers"][0]["keep_sharp"] = True
        row = style_comparison(source, candidate)
        self.assertTrue(row["custom_normal_state_match"])
        self.assertTrue(row["modifier_type_sequence_match"])
        self.assertFalse(row["weighted_normal_controls_match"])


if __name__ == "__main__":
    unittest.main()
