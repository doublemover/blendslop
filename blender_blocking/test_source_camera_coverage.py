"""Source-only plan and real file-binding checks, without native captures."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_source_camera_coverage as capture
import run_family_surface_contracts as report
from reconstruction.native_geometry import GeometryArrays
from synthetic.quality_contracts import quality_workload


def camera():
    return {"projection": "ORTHO", "matrix_world": np.eye(4).tolist(), "ortho_scale": 2.,
        "shift_x": 0., "shift_y": 0., "clip_start": .1, "clip_end": 100.,
        "resolution": [512, 512], "pixel_aspect": [1., 1.]}


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf8")


def fixture(root):
    """Create explicitly synthetic file bindings; they never stand in for native evidence."""
    source_root = root / "references"
    source_root.mkdir()
    cases = {row["name"]: row for row in quality_workload()["cases"] if row["name"] in capture.FAMILIES}
    references, candidates = {"cases": {}}, {"cases": {}}
    vertices = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = np.asarray([[0, 1, 2]])
    geometry = GeometryArrays.capture(vertices, faces)
    for name in capture.FAMILIES:
        source, candidate = source_root / name, root / ("candidate-" + name)
        source.mkdir()
        candidate.mkdir()
        for folder in (source, candidate):
            np.savez_compressed(folder / "evaluated-exact.npz", vertices=vertices, faces=faces)
        originals, masks, records, inventory = {}, {}, {}, {}
        for view in capture.CANONICAL_VIEWS:
            mask_path = source / (view + "-mask.png")
            Image.new("L", (512, 512), 255).save(mask_path)
            masks[view] = capture.bind(mask_path)
            original = camera()
            original.pop("clip_start")
            original.pop("clip_end")
            originals[view] = {**original, "png_sha256": masks[view]["sha256"]}
            neutral_path = candidate / (view + "-neutral.png")
            Image.new("L", (512, 512), 125).save(neutral_path)
            neutral = capture.bind(neutral_path)
            actual = camera()
            frame = capture.camera_frame_sha256(actual)
            binding = {"path": neutral_path.name, "sha256": neutral["sha256"],
                "geometry_hash": geometry.content_hash, "camera_sha256": frame}
            records[view] = {**actual, "geometry_hash": geometry.content_hash,
                "geometry_unchanged_after_passes": True, "executed_passes": ["neutral"],
                "render_passes": {"neutral": {"engine": "BLENDER_WORKBENCH"}},
                "pass_artifacts": {"neutral": binding}}
            inventory[view] = {"camera": {"sha256": frame}, "artifacts": {"neutral": {
                **binding, "producer_status": "completed", "geometry_binding": "verified_by_producer"}}}
        write_json(candidate / "camera-snapshots.json", records)
        references["cases"][name] = {"geometry_hash": geometry.content_hash, "reference_cameras": originals}
        candidates["cases"][name] = {"artifact_directory": str(candidate), "geometry_hash": geometry.content_hash,
            "npz_sha256": capture.bind(candidate / "evaluated-exact.npz")["sha256"],
            "observed_inputs": masks, "inspection_artifacts": {"views": inventory}}
    write_json(source_root / "frozen-workload.json", {"cases": list(cases.values())})
    write_json(source_root / "results.json", references)
    candidate_receipt = root / "candidate-receipt.json"
    write_json(candidate_receipt, candidates)
    return capture.prepare_plan(source_root, candidate_receipt)


class SourceCameraCoverageTests(unittest.TestCase):
    def test_plan_freezes_only_two_authored_sources_ten_frames_and_real_producer_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = fixture(Path(directory))
            self.assertEqual(plan["native_frames"], 10)
            self.assertEqual(set(plan["cases"]), set(capture.FAMILIES))
            self.assertEqual(set(plan["input_sha256"]), capture.required_inputs(plan))
            self.assertNotIn("clip_start", plan["cases"]["torus"]["original_cameras"]["front"])
            for change in ({"native_frames": 11}, {"candidate_renders": 1}, {"fits": 1},
                           {"normal_frames": 1}, {"work_seconds": 86}, {"threads": 3}):
                changed = deepcopy(plan)
                changed.update(change)
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "frozen scope"):
                    capture.validate_plan(changed)
            changed = deepcopy(plan)
            changed["input_sha256"].pop(plan["cases"]["torus"]["candidate_camera_records"]["path"])
            with self.assertRaisesRegex(ValueError, "not all hash-bound"):
                capture.validate_plan(changed)
            changed = deepcopy(plan)
            changed["cases"]["torus"]["source_case"]["parameters"]["minor_radius"] += 1e-12
            with self.assertRaisesRegex(ValueError, "authored source"):
                capture.validate_plan(changed)
            path = Path(plan["cases"]["torus"]["candidate_neutral_artifacts"]["front"]["path"])
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "policy input bytes differ"):
                capture.validate_plan(plan)

    def test_current_source_fragment_is_compatible_with_existing_report_seam(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = fixture(root)
            entries = {}
            fresh_records = {}
            for name in capture.FAMILIES:
                entry = plan["cases"][name]
                folder = root / ("fresh-" + name)
                folder.mkdir()
                shutil.copyfile(entry["source_npz"]["path"], folder / "evaluated-exact.npz")
                geometry = capture.arrays(entry["source_npz"])
                equivalence = capture.reference_equivalence(geometry, geometry)
                records = capture.read_json(entry["candidate_camera_records"])
                for view, record in records.items():
                    shutil.copyfile(entry["candidate_neutral_artifacts"][view]["path"], folder / (view + "-neutral.png"))
                    record["pass_artifacts"]["neutral"]["actual_camera"] = camera()
                write_json(folder / "camera-snapshots.json", records)
                entries[name] = capture.source_declaration(entry, folder, geometry.content_hash, records, equivalence)
                fresh_records[name] = (folder, geometry, records, equivalence)
            report_plan = {"protocol": "family-surface-source-plan-v1", "families": entries,
                "reconstruction_pixels": 1., "normal_allowance_degrees": 1.}
            # The native/analytic policy is outside this file-binding fixture; exercise the
            # report's real archive, oriented-equivalence, camera and PNG hash validation.
            with patch.object(report, "freeze_family_surface_contract", return_value={"fixture": True}) as freezer:
                frozen = report.freeze_sources(report_plan)
            self.assertEqual(set(frozen), set(capture.FAMILIES))
            self.assertEqual(freezer.call_count, 2)
            self.assertEqual(frozen["torus"]["source_provenance"]["indexed_reference_geometry_hash"],
                             fresh_records["torus"][1].content_hash)
            folder, geometry, records, equivalence = fresh_records["torus"]
            for mode in ("clip", "geometry", "actual_camera", "image_hash"):
                changed = deepcopy(records)
                if mode == "clip":
                    changed["front"]["clip_end"] += 1e-12
                elif mode == "geometry":
                    changed["front"]["geometry_hash"] = "0" * 64
                elif mode == "actual_camera":
                    changed["front"]["pass_artifacts"]["neutral"].pop("actual_camera")
                else:
                    changed["front"]["pass_artifacts"]["neutral"]["sha256"] = "0" * 64
                write_json(folder / "camera-snapshots.json", changed)
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    capture.source_declaration(plan["cases"]["torus"], folder, geometry.content_hash, changed, equivalence)

    def test_retained_candidate_record_cannot_replace_the_actual_producer_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = fixture(Path(directory))
            entry = plan["cases"]["torus"]
            record = capture.read_json(entry["candidate_camera_records"])["front"]
            artifact = entry["candidate_neutral_artifacts"]["front"]
            producer = capture.read_json(plan["candidate_receipt"])["cases"]["torus"]["inspection_artifacts"]["views"]["front"]
            capture.require_retained_producer(record, entry["candidate_geometry_hash"], artifact, producer)
            producer["artifacts"]["neutral"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "producer receipt differs"):
                capture.require_retained_producer(record, entry["candidate_geometry_hash"], artifact, producer)


if __name__ == "__main__":
    unittest.main()