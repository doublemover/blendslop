"""Cheap admission checks for external input preparation; no mesh/render campaign."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_external_mesh_inputs.py"
SPEC = importlib.util.spec_from_file_location("external_input_preparation_fixture", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


class ExternalInputPreparationTests(unittest.TestCase):
    def test_paths_cannot_escape_authorized_repository(self):
        with self.assertRaises(ValueError):
            MODULE.repository_path(MODULE.ROOT.parent / "outside.glb")
        self.assertEqual(MODULE.repository_path(SCRIPT), SCRIPT)

    def test_review_receipt_is_bound_to_source_and_both_review_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.glb"; source.write_bytes(b"fixture")
            receipt = root / "review.json"
            data = {"source_sha256": MODULE.file_hash(source), "visual_reviewed": True,
                    "canonical_axes_reviewed": True}
            receipt.write_text(json.dumps(data))
            self.assertEqual(MODULE.reviewed_source(source, receipt), data)
            source.write_bytes(b"changed")
            with self.assertRaises(ValueError): MODULE.reviewed_source(source, receipt)
            data["source_sha256"] = MODULE.file_hash(source); data["canonical_axes_reviewed"] = False
            receipt.write_text(json.dumps(data))
            with self.assertRaises(ValueError): MODULE.reviewed_source(source, receipt)

    def test_calibration_matches_existing_reference_and_novel_view_contract(self):
        calibration, center, scale = MODULE.canonical_calibration([-1,-2,-3],[3,4,5])
        self.assertEqual(center.tolist(), [1,1,1])
        self.assertEqual(scale, 9.6)
        for view, axes in (("front", [0,2]), ("side", [1,2]), ("top", [0,1])):
            self.assertEqual(calibration[view]["axes"], axes)
            self.assertEqual(calibration[view]["world_bounds"], [-3.8,5.8,-3.8,5.8])
        with self.assertRaises(ValueError): MODULE.canonical_calibration([0,0,0],[0,0,0])

    def test_render_admission_fails_before_blender_import(self):
        args = SimpleNamespace(source=MODULE.ROOT / "temp/review-required.glb",
                               output=MODULE.ROOT / "temp/external-review-missing-fixture",
                               render=True, review_receipt=None)
        with self.assertRaisesRegex(ValueError, "review-receipt"):
            MODULE.prepare(args)


if __name__ == "__main__":
    unittest.main()
