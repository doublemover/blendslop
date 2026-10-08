"""Single-release policy and current solver validation, without native Blender."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from utils import blender_version as runtime
from config import MeshJoinConfig


class BlenderRuntimeTests(unittest.TestCase):
    def app(self, version=(5, 2, 2), cycle="release"):
        return SimpleNamespace(app=SimpleNamespace(
            version=version, version_string=".".join(map(str, version)),
            version_cycle=cycle))

    def test_exact_current_release_is_accepted(self):
        with patch.object(runtime, "BLENDER_AVAILABLE", True), patch.object(
            runtime, "bpy", self.app(), create=True
        ):
            runtime.require_supported_blender()

    def test_other_release_is_rejected(self):
        with patch.object(runtime, "BLENDER_AVAILABLE", True), patch.object(
            runtime, "bpy", self.app(version=(5, 0, 1)), create=True
        ):
            with self.assertRaisesRegex(RuntimeError, "5.2.2"):
                runtime.require_supported_blender()

    def test_preview_is_rejected_even_with_matching_version(self):
        with patch.object(runtime, "BLENDER_AVAILABLE", True), patch.object(
            runtime, "bpy", self.app(cycle="beta"), create=True
        ):
            with self.assertRaisesRegex(RuntimeError, "beta"):
                runtime.require_supported_blender()

    def test_missing_native_runtime_is_explicit(self):
        with patch.object(runtime, "BLENDER_AVAILABLE", False):
            with self.assertRaises(RuntimeError):
                runtime.require_supported_blender()

    def test_current_solver_choices_and_default(self):
        with patch.object(runtime, "BLENDER_AVAILABLE", False):
            self.assertEqual(runtime.resolve_boolean_solver(None), "EXACT")
            self.assertEqual(runtime.resolve_boolean_solver("auto"), "EXACT")
            for solver in ("EXACT", "FLOAT", "MANIFOLD"):
                self.assertEqual(runtime.resolve_boolean_solver(solver), solver)

    def test_removed_solver_does_not_silently_fall_back(self):
        with patch.object(runtime, "BLENDER_AVAILABLE", False):
            with self.assertRaises(ValueError):
                runtime.resolve_boolean_solver("FAST")

    def test_config_rejects_removed_solver(self):
        with self.assertRaises(ValueError):
            MeshJoinConfig(boolean_solver="FAST").validate()

    def test_backend_allows_current_float_and_rejects_removed_solver(self):
        from reconstruction.backends.silhouette_intersection import SilhouetteIntersectionBackend
        backend = SilhouetteIntersectionBackend()
        config = {"extrude_distance": 2, "boolean_solver": "FLOAT"}
        self.assertFalse(backend.validate_config(config))
        self.assertTrue(backend.validate_config({**config, "boolean_solver": "FAST"}))


if __name__ == "__main__":
    unittest.main()
