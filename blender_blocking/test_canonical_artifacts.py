"""Pure-Python integrity checks for canonical retention and raw observations."""
from pathlib import Path
import tempfile
import hashlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from blender_blocking.evaluation.canonical_artifacts import (
    CANONICAL_VIEWS, canonical_artifact_inventory, raw_surface_observation, camera_frame_sha256,
    triangle_diagnostic_control_screen,
)
from blender_blocking.reconstruction.native_geometry import GeometryArrays


class TestCanonicalArtifacts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.geometry = GeometryArrays.capture([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
        np.savez_compressed(self.root / "evaluated-exact.npz", vertices=self.geometry.vertices,
                            faces=self.geometry.faces)
        (self.root / "evaluated.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
        self.cameras = {view: {"matrix_world": np.eye(4).tolist(), "ortho_scale": 2.,
                              "geometry_hash": self.geometry.content_hash, "geometry_unchanged_after_passes": True,
                              "clip_start": .1, "clip_end": 100.}
                        for view in CANONICAL_VIEWS}

    def pass_file(self, view, name, resolution=(512, 512)):
        path = self.root / (view + "-" + name + ".png")
        Image.new("L", resolution, color=255).save(path)
        self.cameras[view].setdefault("pass_artifacts", {})[name] = {
            "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "geometry_hash": self.geometry.content_hash, "camera_sha256": camera_frame_sha256(self.cameras[view])}
        return path

    def inspect(self, **kwargs):
        options = {"geometry_hash": self.geometry.content_hash, "camera_records": self.cameras,
                   "pass_states": {"mask": "completed", "neutral": "completed", "normals": "unrun"}}
        options.update(kwargs)
        return canonical_artifact_inventory(self.root, **options)

    def complete_passes(self):
        for view in CANONICAL_VIEWS:
            for name in ("mask", "neutral"):
                self.pass_file(view, name)

    def test_five_view_inspection_retains_hidden_side_and_top(self):
        self.complete_passes()
        row = self.inspect()
        self.assertEqual(row["status"], "complete")
        self.assertEqual(len(row["retained_paths"]), 10)
        self.assertIn("side-neutral.png", row["retained_paths"])
        self.assertIn("top-neutral.png", row["retained_paths"])
        self.assertEqual(row["geometry"]["identity_status"], "verified")
        self.assertNotIn("passed", row)

    def test_mask_only_is_explicitly_incomplete(self):
        for view in CANONICAL_VIEWS:
            self.pass_file(view, "mask")
        row = self.inspect(pass_states={"mask": "completed", "neutral": "unrun", "normals": "unrun"})
        self.assertEqual(row["status"], "incomplete")
        self.assertEqual(len(row["gaps"]), 5)
        self.assertEqual(row["views"]["front"]["artifacts"]["neutral"]["status"], "unrun")

    def test_missing_camera_cannot_claim_complete_matched_inspection(self):
        self.complete_passes()
        row = self.inspect(camera_records={})
        self.assertEqual(len(row["gaps"]), 10)
        self.assertEqual(row["views"]["front"]["camera"]["status"], "unavailable")

    def test_changed_archive_invalidates_declared_geometry_identity(self):
        self.complete_passes()
        changed = self.geometry.vertices.copy()
        changed[1, 0] += .1
        np.savez_compressed(self.root / "evaluated-exact.npz", vertices=changed, faces=self.geometry.faces)
        row = self.inspect()
        self.assertEqual(row["geometry"]["identity_status"], "mismatch")
        self.assertEqual(row["status"], "incomplete")

    def test_reference_camera_match_includes_shifts_and_excludes_image_hash(self):
        self.complete_passes()
        references = {view: dict(record, png_sha256="different-reference-mask")
                      for view, record in self.cameras.items()}
        row = self.inspect(reference_camera_records=references)
        self.assertEqual(row["views"]["front"]["camera"]["reference_match"], "matched")
        references["front"]["shift_x"] = .1
        row = self.inspect(reference_camera_records=references)
        self.assertEqual(row["views"]["front"]["camera"]["reference_match"], "mismatch")
        self.assertEqual(row["status"], "incomplete")

    def test_render_frame_and_frozen_mask_hash_drift_are_gaps(self):
        self.complete_passes()
        self.pass_file("side", "neutral", (256, 256))
        self.cameras["front"]["png_sha256"] = "0" * 64
        row = self.inspect()
        self.assertEqual(row["views"]["side"]["artifacts"]["neutral"]["status"], "unavailable")
        self.assertEqual(row["views"]["front"]["artifacts"]["mask"]["status"], "unavailable")

    def test_matching_archive_and_camera_require_producer_pass_geometry_binding(self):
        self.complete_passes()
        self.cameras["front"].pop("pass_artifacts")
        row = self.inspect()
        self.assertEqual(row["views"]["front"]["artifacts"]["neutral"]["geometry_binding"], "unavailable")
        self.assertEqual(row["status"], "incomplete")
        self.pass_file("front", "neutral")
        self.cameras["front"]["pass_artifacts"]["neutral"]["geometry_hash"] = "0" * 64
        row = self.inspect()
        self.assertEqual(row["views"]["front"]["artifacts"]["neutral"]["geometry_binding"], "mismatch")
        self.assertEqual(row["status"], "incomplete")

    def test_frame_match_does_not_grant_missing_clipping_provenance(self):
        self.complete_passes()
        references = {view: dict(record) for view, record in self.cameras.items()}
        for record in references.values():
            record.pop("clip_start")
            record.pop("clip_end")
        row = self.inspect(reference_camera_records=references)
        self.assertEqual(row["views"]["front"]["camera"]["reference_match"], "matched")
        self.assertEqual(row["views"]["front"]["camera"]["reference_clipping_match"], "unavailable")
        self.assertIn("clipping qualification is independent", row["views"]["front"]["camera"]["reference_match_scope"])

    def test_known_reference_clipping_mismatch_is_a_required_inspection_gap(self):
        self.complete_passes()
        references = {view: dict(record) for view, record in self.cameras.items()}
        references["front"]["clip_end"] = 10.
        row = self.inspect(reference_camera_records=references)
        self.assertEqual(row["views"]["front"]["camera"]["reference_match"], "matched")
        self.assertEqual(row["views"]["front"]["camera"]["reference_clipping_match"], "mismatch")
        self.assertEqual(row["status"], "incomplete")
        self.assertTrue(any(gap["reference_clipping_match"] == "mismatch" for gap in row["gaps"]))

    def test_saved_camera_replay_honors_shifts_clips_and_validates_before_mutating(self):
        with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "scripts"), *sys.path]):
            from run_surface_quality_check import _replay_orthographic_camera
        def camera():
            return SimpleNamespace(data=SimpleNamespace(type="PERSP", shift_x=.9, shift_y=.8,
                clip_start=.1, clip_end=100., ortho_scale=1.), matrix_world=None)
        record = dict(self.cameras["front"], shift_x=.25, shift_y=-.1, clip_start=.2, clip_end=20.)
        with patch.dict(sys.modules, {"mathutils": SimpleNamespace(Matrix=lambda value: value)}):
            actual = camera()
            _replay_orthographic_camera(actual, record)
            self.assertEqual((actual.data.type, actual.data.shift_x, actual.data.shift_y), ("ORTHO", .25, -.1))
            self.assertEqual((actual.data.clip_start, actual.data.clip_end), (.2, 20.))
            self.assertEqual(actual.matrix_world, record["matrix_world"])
            legacy = camera()
            _replay_orthographic_camera(legacy, {"matrix_world": np.eye(4).tolist(), "ortho_scale": 2.})
            self.assertEqual((legacy.data.shift_x, legacy.data.shift_y), (0., 0.))
            self.assertEqual((legacy.data.clip_start, legacy.data.clip_end), (.1, 100.))
            for invalid in ({"shift_x": float("nan")}, {"clip_start": float("inf")},
                            {"clip_start": -1.}, {"clip_end": .01}, {"projection": "PERSP"}):
                unchanged = camera()
                with self.assertRaises(ValueError):
                    _replay_orthographic_camera(unchanged, dict(record, **invalid))
                self.assertEqual(unchanged.data.type, "PERSP")
                self.assertIsNone(unchanged.matrix_world)

    def test_all_passes_replay_original_frame_without_second_matrix_rounding(self):
        from contextlib import contextmanager
        with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "scripts"), *sys.path]):
            from run_surface_quality_check import _renders
        import integration.blender_ops.silhouette_render as render
        class Camera:
            def __init__(self):
                self.data = SimpleNamespace(type="ORTHO", shift_x=0., shift_y=0.,
                    clip_start=.1, clip_end=100., ortho_scale=2.)
                self._matrix = np.eye(4).tolist()
            @property
            def matrix_world(self):
                return self._matrix
            @matrix_world.setter
            def matrix_world(self, value):
                # Model native transform decomposition's repeatable small round.
                self._matrix = [list(row) for row in value]
                self._matrix[0][0] += 1e-7
        camera = Camera()
        session = SimpleNamespace(camera=camera, scene=SimpleNamespace(
            render=SimpleNamespace(engine="BLENDER_EEVEE", resolution_x=512, resolution_y=512,
                pixel_aspect_x=1., pixel_aspect_y=1., image_settings=SimpleNamespace(color_mode="BW")),
            display=SimpleNamespace(shading=SimpleNamespace())))
        @contextmanager
        def sessions(**kwargs):
            yield session
        def write_frame(session, path):
            Image.new("L", (512, 512), color=255).save(path)
        cameras = {}
        with patch.dict(sys.modules, {"bpy": SimpleNamespace(), "mathutils": SimpleNamespace(Matrix=lambda value: value)}), \
                patch.object(render, "silhouette_session", sessions), \
                patch.object(render, "render_silhouette_frame", write_frame), \
                patch("reconstruction.native_geometry.evaluated_arrays", return_value=self.geometry):
            _renders(SimpleNamespace(), self.root, None, ["front"], camera_records=self.cameras,
                     captured_cameras=cameras, inspection_passes=("mask", "neutral"))
        mask = cameras["front"]["pass_artifacts"]["mask"]
        neutral = cameras["front"]["pass_artifacts"]["neutral"]
        self.assertEqual(mask["camera_sha256"], neutral["camera_sha256"])
        self.assertTrue(cameras["front"]["geometry_unchanged_after_passes"])
        self.assertEqual(cameras["front"]["matrix_world"][0][0], 1.0000001)

    def test_render_camera_geometry_binding_cannot_move_between_candidates(self):
        self.complete_passes()
        self.cameras["top"].update(geometry_hash="0" * 64, geometry_unchanged_after_passes=True)
        row = self.inspect()
        self.assertEqual(row["views"]["top"]["camera"]["geometry_binding"], "mismatch")
        self.assertEqual(row["status"], "incomplete")

    def test_existing_files_do_not_turn_unrun_passes_into_evidence(self):
        self.complete_passes()
        row = self.inspect(pass_states={name: "unrun" for name in ("mask", "neutral", "normals")})
        self.assertEqual(row["retained_paths"], [])
        self.assertEqual(row["views"]["front"]["artifacts"]["mask"]["retained_file_status"], "available")

    def test_truncated_png_does_not_count_as_available_inspection(self):
        self.complete_passes()
        path = self.root / "front-neutral.png"
        path.write_bytes(path.read_bytes()[:24])
        row = self.inspect()
        self.assertEqual(row["views"]["front"]["artifacts"]["neutral"]["status"], "unavailable")

    def test_inventory_hashing_respects_explicit_byte_budget(self):
        self.complete_passes()
        row = self.inspect(per_file_byte_limit=1)
        self.assertEqual(row["status"], "incomplete")
        self.assertEqual(row["hashed_bytes"], 0)

    def test_borrowed_triangle_control_screen_does_not_qualify_family_surface(self):
        observation = {"symmetric_mean_distance_world": .001, "normal_angle_p95_degrees": 3.,
                       "surface_status": "unqualified", "qualified_limits": None}
        before = dict(observation)
        diagnostic = triangle_diagnostic_control_screen(observation)
        self.assertFalse(diagnostic["diagnostic_passed"])
        self.assertEqual(diagnostic["borrowed_thresholds"]["symmetric_mean_distance_world_max"], .003)
        self.assertEqual(diagnostic["borrowed_thresholds"]["normal_angle_p95_degrees_max"], 2.5)
        self.assertIn("no triangle family acceptance qualification", diagnostic["scope"])
        self.assertEqual(observation, before)
        self.assertIsNone(observation["qualified_limits"])
        self.assertNotIn("surface_passed", diagnostic)

    def test_raw_surface_withholds_unqualified_family_verdict(self):
        values = {"symmetric_mean_distance_world": .1, "normal_angle_p95_degrees": 8.,
                  "surface_passed": False, "frozen_limits": {"vase_limit": .003}}
        with patch("blender_blocking.evaluation.surface_quality.compare_surface_arrays", return_value=values) as compare:
            row = raw_surface_observation(self.geometry, self.geometry)
        compare.assert_called_once_with(self.geometry, self.geometry)
        self.assertEqual(row["metric_status"], "measured")
        self.assertEqual(row["symmetric_mean_distance_world"], .1)
        self.assertEqual(row["surface_status"], "unqualified")
        self.assertIsNone(row["qualified_limits"])
        self.assertNotIn("surface_passed", row)
        self.assertNotIn("frozen_limits", row)
        self.assertEqual(raw_surface_observation(None, self.geometry)["metric_status"], "unrun")


if __name__ == "__main__":
    unittest.main()
