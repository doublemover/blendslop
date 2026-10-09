"""Bounded cleanup keeps unjoined primary/pipe transports active."""
import json
from pathlib import Path
import subprocess
import tempfile
import time
import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
from reconstruction import native_qualification as native
from reconstruction.native_geometry import GeometryArrays
from primitives.capsule import CapsulePrimitive
from reconstruction.differentiable.dvx_adapter import helper_call
from test_owned_lifecycle_processes import fixture_python
from utils.primary_process_cleanup import finish_primary_process


class PrimaryCleanupTests(unittest.TestCase):
    def child(self):
        child = Mock(returncode=None)
        child.poll.side_effect = lambda: child.returncode
        child.communicate.side_effect = subprocess.TimeoutExpired("held output pipe", .01)
        child.wait.side_effect = subprocess.TimeoutExpired("unjoined primary", .01)
        return child

    def geometry(self):
        mesh = CapsulePrimitive().to_mesh_data(8)
        faces = np.asarray([triangle for face in mesh.faces for triangle in
                            ((face[0], face[i], face[i+1]) for i in range(1, len(face)-1))])
        return GeometryArrays.capture(mesh.vertices, faces)

    def test_cleanup_has_one_shared_finite_allowance(self):
        child = self.child()
        _, _, receipt = finish_primary_process(child, timeout_s=.05)
        self.assertFalse(receipt["primary_joined"])
        self.assertFalse(receipt["transport_closed"])
        self.assertFalse(receipt["ordinary_tree_qualified"])
        drain = child.communicate.call_args.kwargs["timeout"]
        wait = child.wait.call_args.kwargs["timeout"]
        self.assertGreaterEqual(wait, 0.)
        self.assertLessEqual(wait, drain)
        self.assertLessEqual(drain, .05)
        child.communicate.assert_called_once()
        child.wait.assert_called_once()

    def test_exited_primary_with_real_inherited_pipe_returns_within_allowance(self):
        code = ("import subprocess,sys;"
                "subprocess.Popen([sys.executable,'-c','import time;time.sleep(.6)'],"
                "stdout=sys.stdout,stderr=sys.stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))")
        child = subprocess.Popen([str(fixture_python()), "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            child.wait(timeout=2.)
            started = time.monotonic()
            _, _, receipt = finish_primary_process(child, timeout_s=.05)
            self.assertLess(time.monotonic()-started, .4)
            self.assertTrue(receipt["primary_joined"])
            self.assertFalse(receipt["pipes_drained"], receipt)
            self.assertFalse(receipt["transport_closed"])
            self.assertFalse(receipt["ordinary_tree_qualified"])
        finally:
            # The deliberately fresh descendant exits naturally after .6s.
            child.communicate(timeout=2.)

    def test_cold_incomplete_cleanup_retains_lease_and_primary_timeout(self):
        child = self.child()
        with tempfile.TemporaryDirectory() as directory, patch("subprocess.Popen", return_value=child):
            with self.assertRaisesRegex(TimeoutError, "no scored checkpoint") as caught:
                helper_call(str(fixture_python()), {"target": "unjoined"}, timeout_s=.01, ownership_root=directory)
            root = next(Path(directory).glob("owned-*"))
            self.assertEqual(json.loads((root / "run-lease.json").read_text())["status"], "active")
            self.assertTrue(any("cleanup remains unconfirmed" in note for note in caught.exception.__notes__))
            receipt = json.loads((root / "helper-result.json").read_text())
            self.assertFalse(receipt["child_joined"])
            self.assertFalse(receipt["primary_cleanup"]["transport_closed"])
            self.assertTrue(all(call.kwargs.get("timeout") is not None for call in child.communicate.call_args_list))
            self.assertEqual(child.communicate.call_count, 2)

    def test_native_incomplete_cleanup_retains_lease_and_does_not_cache(self):
        child = self.child()
        with tempfile.TemporaryDirectory() as directory, patch.object(native, "toolchain_identity", return_value="fixture-id"), \
                patch.object(native.subprocess, "Popen", return_value=child), patch.dict(native._CACHE, clear=True):
            with self.assertRaisesRegex(RuntimeError, "cleanup remains unconfirmed"):
                native.qualify_geometry(self.geometry(), python="fixture-python", timeout_s=.01, ownership_root=directory)
            root = next(Path(directory).glob("owned-*"))
            self.assertEqual(json.loads((root / "run-lease.json").read_text())["status"], "active")
            self.assertEqual(json.loads((root / "run-ownership.json").read_text())["state"], "failed")
            self.assertFalse(json.loads((root / "helper-cleanup.json").read_text())["transport_closed"])
            self.assertFalse(native._CACHE)
            self.assertTrue(all(call.kwargs.get("timeout") is not None for call in child.communicate.call_args_list))

    def test_native_cancellation_preserves_original_when_cleanup_cannot_join(self):
        child = self.child()
        original = KeyboardInterrupt("primary native cancellation")
        child.communicate.side_effect = [original, subprocess.TimeoutExpired("held output pipe", .01)]
        with tempfile.TemporaryDirectory() as directory, patch.object(native, "toolchain_identity", return_value="fixture-id"), \
                patch.object(native.subprocess, "Popen", return_value=child), patch.dict(native._CACHE, clear=True):
            with self.assertRaises(KeyboardInterrupt) as caught:
                native.qualify_geometry(self.geometry(), python="fixture-python", ownership_root=directory)
            self.assertIs(caught.exception, original)
            self.assertTrue(any("cleanup remains unconfirmed" in note for note in original.__notes__))
            root = next(Path(directory).glob("owned-*"))
            self.assertEqual(json.loads((root / "run-lease.json").read_text())["status"], "active")


if __name__ == "__main__":
    unittest.main()
