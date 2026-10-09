"""Producer-owned render staging/publication without native Blender or cleanup."""
from contextlib import contextmanager, ExitStack, nullcontext
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from integration.blender_ops import render_utils as render
from utils import artifact_publication as publication
from utils.run_ownership import plan_run_reclamation


class ArtifactPublicationTests(unittest.TestCase):
    def test_atomic_existing_destination_keeps_both_immutable_files(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "stage", Path(folder) / "final"
            source.write_bytes(b"complete new stage")
            target.write_bytes(b"existing owner bytes")
            with self.assertRaises(FileExistsError):
                publication.publish_file_no_clobber(source, target)
            self.assertEqual(source.read_bytes(), b"complete new stage")
            self.assertEqual(target.read_bytes(), b"existing owner bytes")

    def test_competing_complete_stages_cannot_clobber_one_final(self):
        with tempfile.TemporaryDirectory() as folder:
            sources = [Path(folder) / name for name in ("first", "second")]
            for source in sources:
                source.write_bytes(source.name.encode())
            target = Path(folder) / "final"
            barrier = Barrier(2)

            def publish(source):
                barrier.wait(timeout=3)
                try:
                    self.assertEqual(publication.publish_file_no_clobber(source, target), target)
                    return "published"
                except FileExistsError:
                    return "collision"

            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(publish, sources))
            self.assertCountEqual(outcomes, ["published", "collision"])
            self.assertIn(target.read_bytes(), (b"first", b"second"))
            self.assertEqual([p.read_bytes() for p in sources], [b"first", b"second"])

    def test_nonregular_source_and_unsupported_or_locked_link_fail_without_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(publication.os, "link") as link:
                with self.assertRaises(ValueError):
                    publication.publish_file_no_clobber(root, root / "final")
                link.assert_not_called()
            source, target = root / "stage", root / "final"
            source.write_bytes(b"retained")
            for error in (OSError("unsupported filesystem"), PermissionError("locked final")):
                with patch.object(publication.os, "link", side_effect=error):
                    with self.assertRaises(type(error)) as caught:
                        publication.publish_file_no_clobber(source, target)
                self.assertIs(caught.exception, error)
                self.assertEqual(source.read_bytes(), b"retained")
                self.assertFalse(target.exists())


class RenderRunOwnershipTests(unittest.TestCase):
    @contextmanager
    def fake_blender(self, action=None, *, targets=True):
        scene = SimpleNamespace(
            render=SimpleNamespace(engine="BLENDER_WORKBENCH", filepath="before-render"),
            cycles=SimpleNamespace(samples=12),
            view_settings=SimpleNamespace(view_transform="AgX", look="Medium High Contrast",
                                          exposure=1.5, gamma=1.2),
        )
        camera = SimpleNamespace(location=(1., 2., 3.), rotation_euler=(.1, .2, .3),
                                 data=SimpleNamespace(ortho_scale=2.))
        state = SimpleNamespace(scene=scene, calls=[], session_kwargs=[])

        @contextmanager
        def session(**kwargs):
            state.session_kwargs.append(kwargs)
            yield SimpleNamespace(scene=scene, camera=camera, resolution=kwargs["resolution"])

        def configure(camera, view, *args, **kwargs):
            camera.current_view = view
            return view in {"front", "side", "top"}

        def render_frame(session, path):
            scene.render.filepath = str(path)
            state.calls.append(path)
            self.assertTrue(path.parent.name.startswith("owned-"))
            self.assertEqual(scene.view_settings.view_transform, "Standard")
            if action is not None:
                action(session, path)
            else:
                Image.new("RGBA", tuple(session.resolution), (0, 0, 0, 255)).save(path)

        with ExitStack() as stack:
            for name, value in (("BLENDER_AVAILABLE", True),
                                ("bpy", SimpleNamespace(context=SimpleNamespace(scene=scene)))):
                stack.enter_context(patch.object(render, name, value, create=True))
            stack.enter_context(patch.object(render, "collect_target_objects",
                                             return_value=[object()] if targets else []))
            stack.enter_context(patch.object(render, "compute_bounds_world",
                                             return_value=((0., 0., 0.), (1., 1., 1.))))
            stack.enter_context(patch.object(render, "silhouette_session", session))
            stack.enter_context(patch.object(render, "_configure_validation_camera", configure))
            stack.enter_context(patch.object(render, "render_silhouette_frame", render_frame))
            yield state
        self.assertEqual(scene.cycles.samples, 12)
        self.assertEqual(vars(scene.view_settings), {
            "view_transform": "AgX", "look": "Medium High Contrast", "exposure": 1.5, "gamma": 1.2})

    def owned_receipt(self, output):
        roots = list(Path(output).glob(".render-runs/owned-*"))
        self.assertEqual(len(roots), 1)
        root = roots[0]
        return root, json.loads((root / "render-result.json").read_text())

    def assert_ready(self, root, state="succeeded"):
        manifest = json.loads((root / "run-ownership.json").read_text())
        self.assertEqual(manifest["state"], state)
        self.assertEqual(json.loads((root / "run-lease.json").read_text())["status"], "released")
        plan = plan_run_reclamation(root)
        self.assertEqual(plan["status"], "dry_run_ready")
        self.assertEqual(plan["unknown_files"], [])
        self.assertEqual(plan["eligible"], [])

    def test_default_names_metadata_and_scene_filepath_are_final(self):
        with tempfile.TemporaryDirectory() as folder, self.fake_blender() as fake:
            cfg = SimpleNamespace(resolution=(8, 8), samples=4)
            result = render.render_orthogonal_views_detailed(folder, render_config=cfg)
            self.assertEqual(result.paths, {v: str(Path(folder) / (v + ".png"))
                                            for v in ("front", "side", "top")})
            self.assertEqual(fake.scene.render.filepath, result.paths["top"])
            self.assertEqual(result.applied_samples, 4)
            self.assertEqual(len(fake.calls), 3)
            root, receipt = self.owned_receipt(folder)
            self.assertEqual(result.ownership_receipt, str(root / "render-result.json"))
            self.assertEqual(result.to_dict()["ownership_receipt"], result.ownership_receipt)
            self.assertEqual(receipt["paths"], result.paths)
            for frame in receipt["frames"]:
                staged, final = root / frame["stage"], Path(frame["path"])
                self.assertEqual(staged.read_bytes(), final.read_bytes())
                self.assertEqual(hashlib.sha256(final.read_bytes()).hexdigest(), frame["sha256"])
            self.assert_ready(root)

    def test_existing_names_prefix_index_and_legacy_mapping_remain_compatible(self):
        with tempfile.TemporaryDirectory() as folder, self.fake_blender():
            existing = Path(folder) / "recipe_front_7.png"
            existing.write_bytes(b"historical final remains external")
            outputs = render.render_orthogonal_views(folder, views=["front"], resolution=(8, 8),
                                                     filename_prefix="recipe_", start_index=7)
            self.assertEqual(outputs, {"front": str(Path(folder) / "recipe_front_8.png")})
            self.assertEqual(existing.read_bytes(), b"historical final remains external")
            root, _ = self.owned_receipt(folder)
            manifest = json.loads((root / "run-ownership.json").read_text())
            self.assertNotIn("recipe_front_7.png", [entry["path"] for entry in manifest["artifacts"]])
            self.assert_ready(root)

    def test_relative_returns_and_long_name_compaction_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder, self.fake_blender():
            relative = os.path.relpath(folder, Path.cwd())
            result = render.render_orthogonal_views_detailed(relative, views=["front"], resolution=(8, 8),
                                                            filename_prefix="x" * 300)
            self.assertFalse(Path(result.paths["front"]).is_absolute())
            self.assertLessEqual(len(result.paths["front"]), 240)
            self.assertTrue(Path(result.paths["front"]).is_file())
            root, _ = self.owned_receipt(folder)
            self.assert_ready(root)

    def test_publication_race_retries_without_another_render_or_clobber(self):
        real_publish = publication.publish_file_no_clobber
        raced = []
        with tempfile.TemporaryDirectory() as folder, self.fake_blender() as fake:
            def publish(stage, final):
                if not raced:
                    final.write_bytes(b"concurrent final")
                    raced.append(final)
                return real_publish(stage, final)

            with patch.object(render, "publish_file_no_clobber", side_effect=publish):
                result = render.render_orthogonal_views_detailed(folder, views=["front"], resolution=(8, 8))
            self.assertEqual(Path(result.paths["front"]).name, "front_2.png")
            self.assertEqual(raced[0].read_bytes(), b"concurrent final")
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(fake.scene.render.filepath, result.paths["front"])
            root, receipt = self.owned_receipt(folder)
            self.assertEqual(receipt["frames"][0]["publication_collisions"], 1)
            self.assert_ready(root)

    def test_failed_second_render_retains_first_final_and_new_partial(self):
        original = RuntimeError("render fixture failed")
        def action(session, path):
            if session.camera.current_view == "side":
                path.write_bytes(b"new partial")
                raise original
            Image.new("RGBA", session.resolution).save(path)

        with tempfile.TemporaryDirectory() as folder, self.fake_blender(action) as fake:
            with self.assertRaises(RuntimeError) as caught:
                render.render_orthogonal_views_detailed(folder, views=["front", "side"], resolution=(8, 8))
            self.assertIs(caught.exception, original)
            self.assertTrue((Path(folder) / "front.png").is_file())
            self.assertFalse((Path(folder) / "side.png").exists())
            self.assertEqual(fake.scene.render.filepath, str(Path(folder) / "side.png"))
            root, receipt = self.owned_receipt(folder)
            self.assertEqual(receipt["paths"], {"front": str(Path(folder) / "front.png")})
            self.assertEqual((root / "frame-0002.png").read_bytes(), b"new partial")
            self.assert_ready(root, "failed")

    def test_progress_failure_keeps_published_image_and_final_filepath(self):
        original = RuntimeError("progress callback failed")
        with tempfile.TemporaryDirectory() as folder, self.fake_blender() as fake:
            def callback(count):
                self.assertEqual(count, 1)
                self.assertEqual(fake.scene.render.filepath, str(Path(folder) / "front.png"))
                raise original

            with self.assertRaises(RuntimeError) as caught:
                render.render_orthogonal_views_detailed(folder, views=["front"], resolution=(8, 8),
                                                       progress_callback=callback)
            self.assertIs(caught.exception, original)
            root, receipt = self.owned_receipt(folder)
            self.assertEqual(receipt["frames"][0]["status"], "published")
            self.assertTrue((Path(folder) / "front.png").is_file())
            self.assert_ready(root, "failed")

    def test_unsupported_or_locked_publication_keeps_complete_stage_and_primary_error(self):
        original = PermissionError("locked publication fixture")
        with tempfile.TemporaryDirectory() as folder, self.fake_blender():
            with patch.object(render, "publish_file_no_clobber", side_effect=original):
                with self.assertRaises(PermissionError) as caught:
                    render.render_orthogonal_views_detailed(folder, views=["front"], resolution=(8, 8))
            self.assertIs(caught.exception, original)
            self.assertFalse((Path(folder) / "front.png").exists())
            root, receipt = self.owned_receipt(folder)
            with Image.open(root / "frame-0001.png") as image:
                image.verify()
            self.assertEqual(receipt["status"], "failed")
            self.assert_ready(root, "failed")

    def test_invalid_png_is_not_published_and_remains_diagnostic(self):
        def action(session, path):
            path.write_bytes(b"incomplete image")
        with tempfile.TemporaryDirectory() as folder, self.fake_blender(action):
            with self.assertRaises(OSError):
                render.render_orthogonal_views_detailed(folder, views=["front"], resolution=(8, 8))
            self.assertFalse((Path(folder) / "front.png").exists())
            root, _ = self.owned_receipt(folder)
            self.assertEqual((root / "frame-0001.png").read_bytes(), b"incomplete image")
            self.assert_ready(root, "failed")

    def test_secondary_receipt_failure_does_not_replace_render_exception(self):
        original = RuntimeError("primary render fixture")
        real_write = render._write_render_receipt
        def action(session, path):
            path.write_bytes(b"partial")
            raise original
        def write(owner, receipt):
            if receipt["status"] != "running":
                raise OSError("secondary receipt fixture")
            return real_write(owner, receipt)

        with tempfile.TemporaryDirectory() as folder, self.fake_blender(action):
            with patch.object(render, "_write_render_receipt", side_effect=write):
                with self.assertRaises(RuntimeError) as caught:
                    render.render_orthogonal_views_detailed(folder, views=["front"], resolution=(8, 8))
            self.assertIs(caught.exception, original)
            self.assertIn("secondary receipt fixture", original.__notes__[0])
            root, _ = self.owned_receipt(folder)
            self.assert_ready(root, "failed")

    def test_cancellation_is_retained_and_releases_without_any_child(self):
        original = KeyboardInterrupt("cancel render fixture")
        def action(session, path):
            path.write_bytes(b"partial cancellation")
            raise original
        with tempfile.TemporaryDirectory() as folder, self.fake_blender(action):
            with self.assertRaises(KeyboardInterrupt) as caught:
                render.render_orthogonal_views_detailed(folder, views=["front"], resolution=(8, 8))
            self.assertIs(caught.exception, original)
            root, receipt = self.owned_receipt(folder)
            self.assertEqual(receipt["subprocesses_started"], 0)
            self.assertEqual(receipt["status"], "cancelled")
            self.assert_ready(root, "cancelled")

    def test_unsupported_view_remains_warning_without_stage(self):
        with tempfile.TemporaryDirectory() as folder, self.fake_blender() as fake:
            result = render.render_orthogonal_views_detailed(folder, views=["unsupported"], resolution=(8, 8))
            self.assertEqual(result.paths, {})
            self.assertIn("unsupported_view_skipped:unsupported", result.warnings)
            self.assertEqual(fake.calls, [])
            root, receipt = self.owned_receipt(folder)
            self.assertEqual(receipt["frames"], [])
            self.assert_ready(root)

    def test_unavailable_and_no_target_exits_do_not_create_ownership(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "unavailable"
            with patch.object(render, "BLENDER_AVAILABLE", False), patch.object(render, "OwnedRun") as owner:
                result = render.render_orthogonal_views_detailed(str(missing))
                owner.assert_not_called()
            self.assertEqual(result.warnings, ("blender_unavailable",))
            self.assertIsNone(result.ownership_receipt)
            self.assertNotIn("ownership_receipt", result.to_dict())
            self.assertFalse(missing.exists())
            output = Path(folder) / "no-target"
            with self.fake_blender(targets=False), patch.object(render, "OwnedRun") as owner:
                result = render.render_orthogonal_views_detailed(str(output))
                owner.assert_not_called()
            self.assertIn("no_renderable_mesh_targets", result.warnings)
            self.assertIsNone(result.ownership_receipt)
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])


class SaveRenderOwnershipTests(unittest.TestCase):
    owned_receipt = RenderRunOwnershipTests.owned_receipt
    assert_ready = RenderRunOwnershipTests.assert_ready

    @contextmanager
    def current_scene(self, folder, action=None, *, file_format="PNG", use_extension=True):
        scene = SimpleNamespace(
            render=SimpleNamespace(filepath="previous-current-scene-path", engine="BLENDER_EEVEE_NEXT",
                                   image_settings=SimpleNamespace(file_format=file_format, color_mode="RGB", color_depth="8"),
                                   use_file_extension=use_extension, use_multiview=False, is_movie_format=False,
                                   resolution_x=37, resolution_y=29, resolution_percentage=73,
                                   use_border=True, use_crop_to_border=True, film_transparent=False),
            frame_current=75, camera=object(), cycles=SimpleNamespace(samples=17),
            view_settings=SimpleNamespace(view_transform="AgX", look="Medium High Contrast", exposure=1.25, gamma=1.1),
        )
        fixed_settings = {key: value for key, value in vars(scene.render).items() if key != "filepath"}
        fixed_view = dict(vars(scene.view_settings))
        fixed_image = dict(vars(scene.render.image_settings))
        state = SimpleNamespace(scene=scene, calls=[], writes=[])

        def abspath(path):
            return str(Path(folder) / path[2:]) if path.startswith("//") else path

        def native_render(*, write_still):
            self.assertIs(write_still, True)
            path = Path(scene.render.filepath)
            self.assertEqual(path.parent.name, "stage")
            self.assertTrue(path.parent.parent.name.startswith("owned-"))
            self.assertEqual(scene.frame_current, 75)
            self.assertEqual(scene.render.image_settings.file_format, file_format)
            self.assertEqual(scene.render.use_file_extension, use_extension)
            state.calls.append(path)
            if action is not None:
                return action(state, path)
            extension = ".png" if file_format == "PNG" else ".jpeg"
            actual = path if not use_extension or path.name.endswith(extension) else path.with_name(path.name + extension)
            Image.new("RGB", (9, 6), (33, 71, 109)).save(actual, format=file_format)
            state.writes.append(actual)
            return {"FINISHED"}

        fake = SimpleNamespace(context=SimpleNamespace(scene=scene), path=SimpleNamespace(abspath=abspath),
                               ops=SimpleNamespace(render=SimpleNamespace(render=native_render)))
        with patch.object(render, "BLENDER_AVAILABLE", True), patch.object(render, "bpy", fake, create=True), \
                patch.object(render, "_suppress_blender_render_stdout", side_effect=nullcontext):
            try:
                yield state
            finally:
                self.assertEqual({key: value for key, value in vars(scene.render).items() if key != "filepath"}, fixed_settings)
                self.assertEqual(vars(scene.view_settings), fixed_view)
                self.assertEqual(vars(scene.render.image_settings), fixed_image)
                self.assertEqual(scene.frame_current, 75)
                self.assertEqual(scene.cycles.samples, 17)

    def test_success_keeps_current_scene_legacy_none_and_requested_filepath(self):
        with tempfile.TemporaryDirectory() as folder, self.current_scene(folder) as fake:
            requested = str(Path(folder) / "still.png")
            historical = Path(folder) / "history.json"
            historical.write_bytes(b"old result/config bytes")
            self.assertIsNone(render.save_render(requested))
            self.assertEqual(fake.scene.render.filepath, requested)
            self.assertEqual(len(fake.calls), 1)
            root, receipt = self.owned_receipt(folder)
            final = Path(receipt["final_path"])
            self.assertEqual(final, Path(requested))
            self.assertEqual(final.read_bytes(), fake.writes[0].read_bytes())
            self.assertEqual(receipt["stage"]["sha256"], hashlib.sha256(final.read_bytes()).hexdigest())
            self.assertEqual(receipt["protocol"], "save-render-owned-v1")
            self.assertTrue(receipt["published"])
            self.assertEqual(receipt["subprocesses_started"], 0)
            self.assertEqual(historical.read_bytes(), b"old result/config bytes")
            self.assert_ready(root)

    def test_native_extension_frame_leaf_and_non_png_format_are_not_rewritten(self):
        # The fixture declares native filenames; it does not certify Blender's
        # format writer. The coordinator must transport exactly what was closed.
        for fmt, append, requested, actual in (
                ("PNG", True, "still_####", "still_####.png"),
                ("JPEG", True, "portrait", "portrait.jpeg"),
                ("JPEG", False, "portrait.data", "portrait.data")):
            with self.subTest(fmt=fmt, append=append), tempfile.TemporaryDirectory() as folder, \
                    self.current_scene(folder, file_format=fmt, use_extension=append) as fake:
                self.assertIsNone(render.save_render("//" + requested))
                self.assertEqual(fake.calls[0].name, requested)
                self.assertEqual(fake.scene.render.filepath, "//" + requested)
                root, receipt = self.owned_receipt(folder)
                self.assertEqual(Path(receipt["final_path"]), Path(folder) / actual)
                self.assertEqual(receipt["file_format"], fmt)
                with Image.open(Path(folder) / actual) as image:
                    self.assertEqual(image.format, fmt)
                    self.assertEqual(image.size, (9, 6))
                self.assert_ready(root)

    def test_existing_exact_final_retains_old_bytes_complete_stage_and_failure(self):
        with tempfile.TemporaryDirectory() as folder, self.current_scene(folder) as fake:
            final = Path(folder) / "still.png"
            final.write_bytes(b"previous published bytes")
            with self.assertRaises(FileExistsError):
                render.save_render(str(final))
            self.assertEqual(final.read_bytes(), b"previous published bytes")
            self.assertEqual(fake.scene.render.filepath, "previous-current-scene-path")
            self.assertEqual(len(fake.calls), 1)
            root, receipt = self.owned_receipt(folder)
            self.assertFalse(receipt["published"])
            self.assertEqual(receipt["status"], "failed")
            self.assertTrue((root / receipt["stage"]["path"]).is_file())
            self.assertFalse((Path(folder) / "still_2.png").exists())
            self.assert_ready(root, "failed")

    def test_publication_race_and_unsupported_link_do_not_retry_or_rerender(self):
        for collide in (True, False):
            with self.subTest(collide=collide), tempfile.TemporaryDirectory() as folder, self.current_scene(folder) as fake:
                final = Path(folder) / "still.png"
                failure = PermissionError("filesystem link refused")

                def publish(source, target):
                    self.assertEqual(target, final)
                    if collide:
                        with target.open("xb") as stream:
                            stream.write(b"competing owner")
                        return publication.publish_file_no_clobber(source, target)
                    raise failure

                with patch.object(render, "publish_file_no_clobber", side_effect=publish) as link:
                    with self.assertRaises(FileExistsError if collide else PermissionError) as caught:
                        render.save_render(str(final))
                    self.assertEqual(link.call_count, 1)
                if collide:
                    self.assertEqual(final.read_bytes(), b"competing owner")
                else:
                    self.assertIs(caught.exception, failure)
                    self.assertFalse(final.exists())
                self.assertEqual(len(fake.calls), 1)
                self.assertEqual(fake.scene.render.filepath, "previous-current-scene-path")
                root, _ = self.owned_receipt(folder)
                self.assert_ready(root, "failed")


    def test_native_failure_and_keyboard_cancel_retain_partial_and_primary(self):
        for original in (RuntimeError("native primary failure"), KeyboardInterrupt("native cancellation")):
            def action(state, path):
                path.write_bytes(b"partial owned image")
                raise original

            with self.subTest(error=type(original).__name__), tempfile.TemporaryDirectory() as folder, \
                    self.current_scene(folder, action) as fake:
                with self.assertRaises(type(original)) as caught:
                    render.save_render(str(Path(folder) / "still.png"))
                self.assertIs(caught.exception, original)
                self.assertEqual(fake.scene.render.filepath, "previous-current-scene-path")
                self.assertFalse((Path(folder) / "still.png").exists())
                root, receipt = self.owned_receipt(folder)
                self.assertEqual((root / "stage/still.png").read_bytes(), b"partial owned image")
                self.assertEqual(receipt["error"], repr(original))
                state = "cancelled" if isinstance(original, KeyboardInterrupt) else "failed"
                self.assertEqual(receipt["status"], state)
                self.assert_ready(root, state)

    def test_cancelled_operator_multiple_files_and_invalid_png_never_publish(self):
        def cancelled(state, path):
            Image.new("RGB", (9, 6)).save(path, format="PNG")
            return {"CANCELLED"}

        def multiple(state, path):
            for name in ("first.png", "second.png"):
                Image.new("RGB", (9, 6)).save(path.parent / name)
            return {"FINISHED"}

        def invalid(state, path):
            path.write_bytes(b"not a PNG")
            return {"FINISHED"}

        for action, error in ((cancelled, RuntimeError), (multiple, ValueError), (invalid, OSError)):
            with self.subTest(action=action.__name__), tempfile.TemporaryDirectory() as folder, \
                    self.current_scene(folder, action) as fake:
                with self.assertRaises(error):
                    render.save_render(str(Path(folder) / "still.png"))
                self.assertEqual(fake.scene.render.filepath, "previous-current-scene-path")
                self.assertFalse((Path(folder) / "still.png").exists())
                root, receipt = self.owned_receipt(folder)
                self.assertFalse(receipt["published"])
                self.assertGreater(len(list((root / "stage").iterdir())), 0)
                self.assert_ready(root, "failed")

    def test_secondary_receipt_failure_keeps_primary_and_restores_filepath(self):
        primary = RuntimeError("native primary")
        original_writer = render._write_render_receipt

        def action(state, path):
            path.write_bytes(b"partial")
            raise primary

        def write(owner, receipt):
            if receipt["status"] != "running":
                raise OSError("secondary receipt failure")
            return original_writer(owner, receipt)

        with tempfile.TemporaryDirectory() as folder, self.current_scene(folder, action) as fake:
            with patch.object(render, "_write_render_receipt", side_effect=write):
                with self.assertRaises(RuntimeError) as caught:
                    render.save_render(str(Path(folder) / "still.png"))
            self.assertIs(caught.exception, primary)
            self.assertTrue(any("secondary receipt failure" in note for note in primary.__notes__))
            self.assertEqual(fake.scene.render.filepath, "previous-current-scene-path")
            root, _ = self.owned_receipt(folder)
            self.assert_ready(root, "failed")

    def test_fresh_attempts_preserve_previous_run_history(self):
        with tempfile.TemporaryDirectory() as folder, self.current_scene(folder):
            final = Path(folder) / "still.png"
            render.save_render(str(final))
            old_root, _ = self.owned_receipt(folder)
            old_bytes = {path.relative_to(old_root): path.read_bytes() for path in old_root.rglob("*") if path.is_file()}
            with self.assertRaises(FileExistsError):
                render.save_render(str(final))
            roots = sorted((Path(folder) / ".render-runs").glob("owned-*"))
            self.assertEqual(len(roots), 2)
            for relative, value in old_bytes.items():
                self.assertEqual((old_root / relative).read_bytes(), value)
            self.assert_ready(old_root)
            self.assert_ready(next(root for root in roots if root != old_root), "failed")

    def test_unavailable_and_unsupported_preflight_preserve_scene_without_owner(self):
        with patch.object(render, "BLENDER_AVAILABLE", False), patch.object(render, "OwnedRun") as owner:
            self.assertIsNone(render.save_render("unavailable.png"))
            owner.assert_not_called()
        with tempfile.TemporaryDirectory() as folder, self.current_scene(folder) as fake:
            for setting in ("is_movie_format", "use_multiview"):
                setattr(fake.scene.render, setting, True)
                try:
                    with patch.object(render, "OwnedRun") as owner:
                        with self.assertRaises(ValueError):
                            render.save_render(str(Path(folder) / "still.png"))
                        owner.assert_not_called()
                finally:
                    setattr(fake.scene.render, setting, False)
            for requested in ("", ".", "..", "//.", "//..", str(Path(folder) / "{scene}" / "still.png")):
                with patch.object(render, "OwnedRun") as owner:
                    with self.assertRaises(ValueError):
                        render.save_render(requested)
                    owner.assert_not_called()
            self.assertEqual(fake.calls, [])
            self.assertEqual(fake.scene.render.filepath, "previous-current-scene-path")
            self.assertFalse((Path(folder) / ".render-runs").exists())


if __name__ == "__main__":
    unittest.main()
