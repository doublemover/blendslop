"""Fresh canonical packet guards exact source geometry and all actual passes."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_selected_canonical_inspection as packet
from reconstruction.native_geometry import GeometryArrays


def camera():
    return {"projection": "ORTHO", "matrix_world": np.eye(4).tolist(), "ortho_scale": 2.,
            "shift_x": 0., "shift_y": 0., "clip_start": .1, "clip_end": 1000.,
            "resolution": [512, 512], "pixel_aspect": [1., 1.]}


def paired_evidence():
    inventories, records = [], []
    for geometry_hash in ("a" * 64, "b" * 64):
        inv = {"geometry": {"identity_status": "verified", "geometry_hash": geometry_hash}, "views": {}}
        cams = {}
        for view in packet.CANONICAL_VIEWS:
            record = {**camera(), "geometry_hash": geometry_hash, "geometry_unchanged_after_passes": True,
                      "pass_artifacts": {}}
            frame = packet.camera_frame_sha256(record)
            rows = {}
            for name in packet.INSPECTION_PASSES:
                digest = ("1" if name == "mask" else "2" if name == "neutral" else "3") * 64
                rows[name] = {"status": "available", "producer_status": "completed",
                    "geometry_binding": "verified_by_producer", "geometry_hash": geometry_hash,
                    "sha256": digest, "camera_sha256": frame, "path": view + "-" + name + ".png",
                    "render_settings": {"pass": name}}
                record["pass_artifacts"][name] = {"sha256": digest, "camera_sha256": frame,
                    "geometry_hash": geometry_hash, "actual_camera": camera()}
            cams[view] = record
            inv["views"][view] = {"artifacts": rows}
        inventories.append(inv)
        records.append(cams)
    return [*inventories, *records]


class SelectedCanonicalInspectionTests(unittest.TestCase):
    def test_reference_equivalence_preserves_coordinates_winding_and_diagonals(self):
        vertices = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        faces = np.asarray([[0, 1, 2], [0, 2, 3]])
        original = GeometryArrays.capture(vertices, faces)
        order = np.asarray([2, 0, 3, 1])
        inverse = np.argsort(order)
        equivalent = GeometryArrays.capture(vertices[order], inverse[faces[::-1, [1, 2, 0]]])
        self.assertNotEqual(original.content_hash, equivalent.content_hash)
        self.assertTrue(packet.reference_equivalence(original, equivalent)["equivalent"])
        changed = vertices.copy()
        changed[0, 0] += 1e-12
        for v, f in ((changed, faces), (vertices, faces[:, ::-1]),
                     (vertices, np.asarray([[0, 1, 3], [1, 2, 3]]))):
            self.assertFalse(packet.reference_equivalence(original, GeometryArrays.capture(v, f))["equivalent"])

    def test_exact_archive_hash_blocks_even_tiny_coordinate_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "exact.npz"
            vertices = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
            faces = np.asarray([[0, 1, 2]])
            data = GeometryArrays.capture(vertices, faces)
            np.savez_compressed(path, vertices=vertices, faces=faces)
            self.assertEqual(packet.load_exact(path, data.content_hash).content_hash, data.content_hash)
            vertices[0, 0] += 1e-12
            np.savez_compressed(path, vertices=vertices, faces=faces)
            with self.assertRaisesRegex(ValueError, "indexed geometry"):
                packet.load_exact(path, data.content_hash)

    def test_frame_and_clipping_match_are_both_exact(self):
        original = camera()
        self.assertEqual(packet.require_same_frame(original, deepcopy(original)), packet.camera_frame_sha256(original))
        for key in ("shift_x", "clip_start", "clip_end"):
            changed = deepcopy(original)
            changed[key] += 1e-12
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "mismatch"):
                packet.require_same_frame(original, changed)
        changed = deepcopy(original)
        changed.pop("clip_start")
        with self.assertRaises(KeyError):
            packet.require_same_frame(original, changed)

    def test_optional_inventory_normals_are_required_in_complete_three_pass_packet(self):
        evidence = paired_evidence()
        self.assertEqual(packet.matched_packet(*evidence)["status"], "complete")
        for mode in ("unrun", "wrong_hash", "geometry_binding", "clip", "unchanged", "settings"):
            data = deepcopy(evidence)
            row = data[1]["views"]["front"]["artifacts"]["normals"]
            if mode == "unrun":
                row["producer_status"] = "unrun"
            elif mode == "wrong_hash":
                row["sha256"] = "e" * 64
            elif mode == "geometry_binding":
                data[3]["front"]["geometry_hash"] = "e" * 64
            elif mode == "clip":
                data[3]["front"]["pass_artifacts"]["normals"]["actual_camera"]["clip_end"] += 1e-12
            elif mode == "unchanged":
                data[3]["front"].pop("geometry_unchanged_after_passes")
            else:
                row["render_settings"]["samples"] = 32
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                packet.matched_packet(*data)

    def test_frozen_scope_refuses_extra_frames_fits_and_other_geometry(self):
        for change in ({"native_frames": 61}, {"raw_comparisons": 1}, {"fits": 1},
                       {"qualifier_children": 1}, {"threads": 3}, {"resolution": [1024, 1024]}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "frozen scope"):
                packet.validate_plan(change)

    def test_plan_requires_bound_authored_recipe_cameras_masks_and_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            workload, references, entries, files = {"cases": []}, {"cases": {}}, {}, []
            for name, selected_hash in packet.SELECTED.items():
                case = {"name": name, "parameters": {"authored": name}}
                cameras = {view: {**camera(), "png_sha256": packet.sha(__file__)} for view in packet.CANONICAL_VIEWS}
                workload["cases"].append(case)
                references["cases"][name] = {"geometry_hash": "c" * 64, "reference_cameras": cameras}
                entry = {"source_case": case, "source_geometry_hash": "c" * 64, "candidate_geometry_hash": selected_hash,
                         "original_cameras": cameras,
                         "original_masks": {view: {"path": str(Path(__file__).resolve()), "sha256": packet.sha(__file__)}
                                            for view in packet.CANONICAL_VIEWS}}
                for key in ("source_npz", "candidate_npz", "candidate_program", "candidate_receipt"):
                    path = base / (name + key)
                    path.write_bytes(b"bound input")
                    entry[key] = str(path)
                    files.append(path)
                entries[name] = entry
            for name, value in (("workload.json", workload), ("receipt.json", references)):
                path = base / name
                path.write_text(json.dumps(value))
                files.append(path)
            files += [Path(__file__).resolve(), Path(packet.__file__).resolve()]
            plan = {"protocol": "selected_canonical_recapture_v1", "cases": entries, "views": list(packet.CANONICAL_VIEWS),
                    "passes": list(packet.INSPECTION_PASSES), "resolution": [512, 512], "native_frames": 60,
                    "fits": 0, "raw_comparisons": 0, "qualifier_children": 0, "threads": 2,
                    "deadline_seconds": 120, "rss_limit_bytes": 8 * 1024 ** 3,
                    "reference_workload": str(base / "workload.json"), "reference_receipt": str(base / "receipt.json"),
                    "input_sha256": {str(p): packet.sha(p) for p in files}}
            packet.validate_plan(plan)
            changed = deepcopy(plan)
            changed["cases"]["rounded_box"]["source_case"]["parameters"]["authored"] = "candidate-tuned"
            with self.assertRaisesRegex(ValueError, "authored reference"):
                packet.validate_plan(changed)
            changed = deepcopy(plan)
            changed["input_sha256"].pop(changed["cases"]["rounded_box"]["candidate_program"])
            with self.assertRaisesRegex(ValueError, "not all hash-bound"):
                packet.validate_plan(changed)
            files[0].write_bytes(b"changed bound input")
            with self.assertRaisesRegex(ValueError, "frozen input/source changed"):
                packet.validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
