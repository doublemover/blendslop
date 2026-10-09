"""Mocked orchestration proves fresh lease gating without another child campaign."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_bounded_owned_command as wrapper
from test_owned_lifecycle_processes import fixture_python
from utils.run_ownership import plan_run_reclamation


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


class BoundedOwnedCommandTests(unittest.TestCase):
    def options(self, folder, **extra):
        return {"output_parent": Path(folder) / "runs", "timeout_s": 85., "join_timeout_s": 5.,
                "max_memory_bytes": 268435456, "max_rss_bytes": 268435456, "threads": 2,
                "cwd": folder, **extra}

    def command(self):
        return [str(fixture_python()), "-c", "print('explicit argv retained')"]

    def run_mocked(self, folder, receipt, **extra):
        def supervise(command, **kwargs):
            Path(kwargs["log_path"]).write_bytes(b"mocked supervisor stdout\n")
            return dict(receipt)
        with patch.object(wrapper, "_supports_complete_tree", return_value=True), \
                patch.object(wrapper, "run_bounded_process", side_effect=supervise) as launcher:
            result = wrapper.run_owned_command(self.command(), **self.options(folder, **extra))
        return result, launcher

    def test_confirmed_success_closes_fresh_run_and_retains_explicit_inputs_and_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            shared = Path(folder) / "source.txt"
            shared.write_bytes(b"original external input")
            result, launcher = self.run_mocked(folder,
                {"status": "succeeded", "returncode": 0, "lifecycle_complete": True}, inputs=[shared])
            root = Path(result["run_root"])
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(read(root / "run-lease.json")["status"], "released")
            self.assertEqual(read(root / "run-ownership.json")["state"], "succeeded")
            self.assertEqual(plan_run_reclamation(root)["status"], "dry_run_ready")
            self.assertEqual(shared.read_bytes(), b"original external input")
            self.assertEqual(read(root / "command.json")["requested_argv"], self.command())
            self.assertEqual(read(root / "provenance.json")["explicit_shared_inputs"][0]["path"], str(shared.resolve()))
            self.assertTrue(launcher.call_args.kwargs["require_complete_tree"])
            self.assertEqual(launcher.call_args.kwargs["timeout_s"], 85.)
            self.assertEqual(launcher.call_args.kwargs["env"]["OMP_NUM_THREADS"], "2")
            self.assertTrue(read(root / "run-result.json")["lifecycle_complete"])
            self.assertEqual(read(root / "run-result.json")["lease_status_at_receipt"], "active")

    def test_confirmed_primary_failure_stays_failed_and_preserves_actual_code(self):
        with tempfile.TemporaryDirectory() as folder:
            result, _ = self.run_mocked(folder,
                {"status": "failed", "returncode": 17, "lifecycle_complete": True})
            root = Path(result["run_root"])
            self.assertEqual((result["status"], result["exit_code"], result["primary_returncode"]), ("failed", 17, 17))
            self.assertEqual(read(root / "run-ownership.json")["state"], "failed")
            self.assertEqual(read(root / "run-lease.json")["status"], "released")
            self.assertEqual(read(root / "resource-receipt.json")["returncode"], 17)

    def test_unconfirmed_and_truthy_lifecycle_leave_failed_active_manifest_without_close(self):
        for claim in (False, "true", 1, None):
            with self.subTest(claim=claim), tempfile.TemporaryDirectory() as folder, \
                    patch.object(wrapper.OwnedRun, "close", autospec=True) as closer:
                result, _ = self.run_mocked(folder,
                    {"status": "succeeded", "returncode": 0, "lifecycle_complete": claim})
                root = Path(result["run_root"])
                closer.assert_not_called()
                self.assertEqual((result["status"], result["exit_code"]), ("failed", 1))
                self.assertEqual(read(root / "run-ownership.json")["state"], "failed")
                self.assertEqual(read(root / "run-lease.json")["status"], "active")
                self.assertEqual(plan_run_reclamation(root)["status"], "blocked")

    def test_portable_primary_only_never_releases_even_with_mocked_tree_claim(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(wrapper.OwnedRun, "close", autospec=True) as closer:
            result, launcher = self.run_mocked(folder,
                {"status": "succeeded", "returncode": 0, "primary_joined": True, "lifecycle_complete": True},
                portable_primary_only=True, max_rss_bytes=None)
            closer.assert_not_called()
            self.assertFalse(launcher.call_args.kwargs["require_complete_tree"])
            self.assertFalse(result["lifecycle_complete"])
            self.assertEqual(result["lease_status"], "active")
            self.assertEqual(read(Path(result["run_root"]) / "run-ownership.json")["state"], "failed")

    def test_supervisor_exception_retains_error_and_failed_active_lease(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(wrapper, "_supports_complete_tree", return_value=True), \
                patch.object(wrapper, "run_bounded_process", side_effect=RuntimeError("fixture supervisor unavailable")), \
                patch.object(wrapper.OwnedRun, "close", autospec=True) as closer:
            result = wrapper.run_owned_command(self.command(), **self.options(folder))
            root = Path(result["run_root"])
            closer.assert_not_called()
            self.assertEqual(result["status"], "failed")
            self.assertEqual(read(root / "run-ownership.json")["state"], "failed")
            self.assertEqual(read(root / "run-lease.json")["status"], "active")
            self.assertIn("fixture supervisor unavailable", read(root / "orchestration-error.json")["error"])
            self.assertFalse(read(root / "orchestration-error.json")["resource_receipt_available"])
            unavailable = read(root / "resource-receipt.json")
            self.assertIn("unavailable", unavailable["resource_observations"])
            self.assertIsNone(unavailable["returncode"])
            self.assertFalse(unavailable["lifecycle_complete"])
            self.assertFalse((root / "stdout.log").exists())

    def test_invalid_bounds_argv_inputs_and_platform_refuse_before_owner_or_launch(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(wrapper, "_supports_complete_tree", return_value=True), \
                patch.object(wrapper, "OwnedRun") as owner, \
                patch.object(wrapper, "run_bounded_process") as launcher:
            cases = [{"timeout_s": 90.}, {"timeout_s": True}, {"timeout_s": float("nan")},
                     {"join_timeout_s": 0.}, {"max_memory_bytes": 0},
                     {"max_memory_bytes": wrapper.MAX_BYTES + 1}, {"max_rss_bytes": None},
                     {"threads": 3}, {"threads": True}, {"inputs": [Path(folder) / "missing.npz"]}]
            for extra in cases:
                with self.subTest(extra=extra), self.assertRaises((ValueError, OSError)):
                    wrapper.run_owned_command(self.command(), **self.options(folder, **extra))
            for command in ([], [""], ["missing-command-7c2fdd"], [str(fixture_python()), "NUL\x00arg"]):
                with self.subTest(command=command), self.assertRaises((ValueError, OSError)):
                    wrapper.run_owned_command(command, **self.options(folder))
            with patch.object(wrapper, "_supports_complete_tree", return_value=False), self.assertRaises(ValueError):
                wrapper.run_owned_command(self.command(), **self.options(folder))
            owner.assert_not_called()
            launcher.assert_not_called()
            self.assertFalse((Path(folder) / "runs").exists())

    def test_blender_requires_explicit_background_and_matching_native_thread_argument(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(wrapper, "_supports_complete_tree", return_value=True), \
                patch.object(wrapper.os, "access", return_value=True), \
                patch.object(wrapper, "OwnedRun") as owner, \
                patch.object(wrapper, "run_bounded_process") as launcher:
            executable = Path(folder) / "blender.exe"
            executable.write_bytes(b"mock executable; never launched")
            for arguments in (("--background",), ("--background", "--threads", "3"),
                              ("--threads", "2"), ("--background", "--", "--threads", "2"),
                              ("--background", "--threads", "2", "-t", "2")):
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    wrapper.run_owned_command([str(executable), *arguments], **self.options(folder))
            _, declaration, _ = wrapper.prepare_command(
                [str(executable), "--background", "--threads", "2", "--", "--threads", "999"],
                **self.options(folder))
            self.assertEqual(declaration["threads"], 2)
            owner.assert_not_called()
            launcher.assert_not_called()

    def test_provenance_streaming_stops_growing_file_and_rejects_replaced_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = (Path(folder) / "changing.bin").resolve()
            for mode in ("grow", "replacement"):
                path.write_bytes(b"four")
                original_stat = Path.stat
                observed = {"changed": False, "read_bytes": 0}
                class ChangingReader:
                    def __enter__(self):
                        self.handle = open(path, "rb")
                        return self
                    def __exit__(self, *args):
                        self.handle.close()
                    def fileno(self):
                        return self.handle.fileno()
                    def read(self, amount):
                        chunk = self.handle.read(amount)
                        observed["read_bytes"] += len(chunk)
                        if not observed["changed"]:
                            if mode == "grow":
                                with open(path, "ab") as writer:
                                    writer.write(b"x" * 64)
                            observed["changed"] = True
                        return chunk
                def path_stat(target, *args, **kwargs):
                    actual = original_stat(target, *args, **kwargs)
                    if mode == "replacement" and target == path and observed["changed"]:
                        return SimpleNamespace(st_dev=actual.st_dev, st_ino=actual.st_ino + 1,
                            st_size=actual.st_size, st_mtime_ns=actual.st_mtime_ns)
                    return actual
                with self.subTest(mode=mode), patch.object(Path, "open", return_value=ChangingReader()), \
                        patch.object(Path, "stat", new=path_stat), self.assertRaises(ValueError):
                    wrapper._file_record(path, max_bytes=16)
                self.assertLessEqual(observed["read_bytes"], 17)

    def test_cli_requires_separator_and_preserves_literal_argument_array(self):
        command = ["native executable.exe", "literal spaces", "$not-a-shell"]
        arguments = ["--output-parent", "fresh", "--timeout-s", "85", "--max-memory-bytes", "1024",
                     "--max-rss-bytes", "1024", "--threads", "2"]
        with patch.object(wrapper, "run_owned_command", return_value={"exit_code": 17}) as runner, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(wrapper.main([*arguments, "--", *command]), 17)
            self.assertEqual(runner.call_args.args[0], command)
        with patch.object(wrapper, "run_owned_command") as runner, redirect_stdout(io.StringIO()), \
                patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit) as error:
            wrapper.main([*arguments, *command])
        self.assertEqual(error.exception.code, 2)
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
