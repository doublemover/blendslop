"""Real fresh stdlib children prove Windows pre-resume job containment and joins."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import utils.owned_process_supervisor as supervisor

from test_owned_lifecycle_processes import fixture_python
from utils.owned_process_supervisor import run_bounded_process


class OwnedProcessSupervisorTests(unittest.TestCase):
    def run_child(self, directory, code, *, timeout=5., name="child.log"):
        return run_bounded_process([str(fixture_python()), "-c", code], log_path=Path(directory) / name,
                                   timeout_s=timeout, max_memory_bytes=268435456,
                                   max_rss_bytes=268435456 if os.name == "nt" else None,
                                   require_complete_tree=os.name == "nt")

    def test_invalid_bounds_and_existing_log_refuse_before_child_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "prior.log"
            log.write_bytes(b"retained prior")
            for extra in ({"timeout_s": True}, {"timeout_s": float("nan")}, {"max_memory_bytes": 0}):
                arguments = {"log_path": log, "timeout_s": 5., "max_memory_bytes": 268435456,
                             "require_complete_tree": os.name == "nt", **extra}
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    run_bounded_process([str(fixture_python()), "-c", "raise SystemExit(99)"], **arguments)
            with self.assertRaisesRegex(ValueError, "fresh file"):
                self.run_child(folder, "raise SystemExit(99)", name="prior.log")
            self.assertEqual(log.read_bytes(), b"retained prior")

    def test_primary_has_real_wait_and_portable_mode_never_claims_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_child(folder, "print('actual child output')")
            self.assertEqual(result["status"], "succeeded", result)
            self.assertTrue(result["primary_joined"])
            self.assertEqual(result["returncode"], 0)
            self.assertIn("actual child output", (Path(folder) / "child.log").read_text())
            if os.name == "nt":
                self.assertTrue(result["job_assigned_before_resume"])
                self.assertTrue(result["lifecycle_complete"])
                self.assertTrue(result["job_active_zero"])
                self.assertGreater(result["peak_observed_tree_rss_bytes"], 0)
                self.assertGreater(result["rss_sample_count"], 0)
                self.assertFalse(result["rss_unavailable_samples"])
                self.assertTrue(all(row["kernel_joined"] for row in result["observed_processes"]))
            else:
                self.assertFalse(result["lifecycle_complete"])
                self.assertFalse(result["ordinary_createprocess_tree_owned"])
                with self.assertRaises(RuntimeError):
                    run_bounded_process([str(fixture_python()), "-c", "pass"],
                        log_path=Path(folder) / "not-launched.log", timeout_s=5., max_memory_bytes=268435456)

    @unittest.skipUnless(os.name == "nt", "actual Windows Job Object/handle contract")
    def test_crashed_parent_waits_descendant_and_retains_exact_kernel_exit_statuses(self):
        code = "import os,subprocess,sys;child=subprocess.Popen([sys.executable,'-c',\"import time;time.sleep(.35);print('descendant finished')\"],creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),stdout=sys.stdout,stderr=sys.stderr);print('fixture_descendant_pid='+str(child.pid),flush=True);os._exit(17)"
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_child(folder, code)
            self.assertEqual(result["returncode"], 17, result)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["lifecycle_complete"], result)
            self.assertGreaterEqual(result["job_accounting"]["total_processes"], 2)
            self.assertEqual(result["job_accounting"]["active_processes"], 0)
            output = (Path(folder) / "child.log").read_text()
            self.assertIn("descendant finished", output)
            descendant_pid = int(next(line.split("=", 1)[1] for line in output.splitlines() if line.startswith("fixture_descendant_pid=")))
            actual_descendant = next(row for row in result["observed_processes"] if row["pid"] == descendant_pid)
            self.assertTrue(actual_descendant["kernel_joined"])
            self.assertEqual(actual_descendant["exit_status"], 0)
            self.assertTrue(all(row["kernel_joined"] for row in result["observed_processes"]))
            self.assertEqual({row["exit_status"] for row in result["observed_processes"]}, {0, 17})
            primary = next(row for row in result["observed_processes"] if row["pid"] == result["pid"])
            self.assertEqual(primary["creation_filetime"], result["primary_identity"]["creation_filetime"])

    @unittest.skipUnless(os.name == "nt", "actual Windows Job Object/handle contract")
    def test_timeout_terminates_only_owned_job_and_actually_joins_it(self):
        code = "import subprocess,sys;child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10.)'],creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))"
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_child(folder, code, timeout=.5)
            self.assertEqual(result["status"], "failed", result)
            self.assertEqual(result["limit_reason"], "wall_timeout")
            self.assertTrue(result["primary_joined"])
            self.assertTrue(result["lifecycle_complete"], result)
            self.assertEqual(result["job_accounting"]["active_processes"], 0)
            self.assertTrue(all(row["kernel_joined"] for row in result["observed_processes"]))

    @unittest.skipUnless(os.name == "nt", "actual Windows working-set HANDLE contract")
    def test_rss_bound_and_unavailable_readings_stop_only_owned_job(self):
        for mode in ("tiny_bound", "unavailable"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                arguments = {"log_path": Path(folder) / "rss.log", "timeout_s": 5.,
                             "max_memory_bytes": 268435456, "max_rss_bytes": 1 if mode == "tiny_bound" else 268435456}
                command = [str(fixture_python()), "-c", "import time;time.sleep(10.)"]
                if mode == "unavailable":
                    unavailable = {"status": "unavailable", "tree_rss_bytes": None, "unavailable": [{"reason": "fixture live read unavailable"}]}
                    with patch.object(supervisor._WindowsJob, "rss_sample", return_value=unavailable):
                        result = run_bounded_process(command, **arguments)
                else:
                    result = run_bounded_process(command, **arguments)
                self.assertEqual(result["status"], "failed", result)
                self.assertEqual(result["limit_reason"], "tree_rss_limit" if mode == "tiny_bound" else "rss_monitor_unavailable")
                self.assertTrue(result["lifecycle_complete"], result)
                if mode == "unavailable":
                    self.assertIsNone(result["peak_observed_tree_rss_bytes"])
                    self.assertTrue(result["rss_unavailable_samples"])

    @unittest.skipUnless(os.name == "nt", "actual Windows cancellation Job/handle contract")
    def test_cancellation_preserves_exception_after_bounded_complete_tree_join(self):
        import time
        for interrupted in (KeyboardInterrupt("owner cancelled"), SystemExit(23)):
            with self.subTest(kind=type(interrupted).__name__), tempfile.TemporaryDirectory() as folder:
                marker = Path(folder) / "descendant-started"
                code = ("import pathlib,subprocess,sys,time;"
                        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(5.)'],"
                        "creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));"
                        f"pathlib.Path({str(marker)!r}).write_text('ready');time.sleep(5.)")
                actual_sleep = time.sleep
                marker_seen = cancelled = False

                def cancel_after_observation(seconds):
                    nonlocal marker_seen, cancelled
                    if marker_seen and not cancelled:
                        cancelled = True
                        raise interrupted
                    marker_seen = marker.is_file()
                    actual_sleep(seconds)

                with patch.object(supervisor.time, "sleep", side_effect=cancel_after_observation):
                    with self.assertRaises(type(interrupted)) as caught:
                        self.run_child(folder, code)
                self.assertIs(caught.exception, interrupted)
                result = interrupted.owned_process_receipt
                self.assertEqual(result["status"], "cancelled", result)
                self.assertEqual(result["limit_reason"], "caller_cancelled")
                self.assertTrue(result["job_assigned_before_resume"])
                self.assertTrue(result["lifecycle_complete"], result)
                self.assertTrue(result["job_active_zero"])
                self.assertGreaterEqual(result["job_accounting"]["total_processes"], 2)
                self.assertTrue(all(row["kernel_joined"] for row in result["observed_processes"]))
                self.assertLess(result["elapsed_seconds"], 10.)

    @unittest.skipUnless(os.name == "nt", "actual Windows retained HANDLE failure history")
    def test_read_exception_preserves_observed_identities_without_claiming_tree_join(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(supervisor._WindowsJob, "rss_sample", side_effect=RuntimeError("fixture query error")):
                result = self.run_child(folder, "import time;time.sleep(10.)")
            self.assertEqual(result["status"], "failed", result)
            self.assertIn("fixture query error", result["error"])
            self.assertTrue(result["primary_joined"])
            self.assertFalse(result["lifecycle_complete"])
            self.assertTrue(result["observed_processes"])
            primary = next(row for row in result["observed_processes"] if row["pid"] == result["pid"])
            self.assertEqual(primary["creation_filetime"], result["primary_identity"]["creation_filetime"])
            self.assertFalse(primary["kernel_joined"])
            self.assertIsNone(primary["exit_status"])

    @unittest.skipUnless(os.name == "nt", "actual Windows Job Object/handle contract")
    def test_concurrent_supervisors_have_distinct_jobs_and_complete_own_trees(self):
        code = "import subprocess,sys;child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.25)'],creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))"
        with tempfile.TemporaryDirectory() as folder, ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.run_child, folder, code, name=f"child-{i}.log") for i in range(2)]
            results = [future.result(timeout=10.) for future in futures]
            for result in results:
                self.assertEqual(result["status"], "succeeded", result)
                self.assertTrue(result["lifecycle_complete"])
                self.assertGreaterEqual(result["job_accounting"]["total_processes"], 2)
            identities = [{(row["pid"], row["creation_filetime"]) for row in item["observed_processes"]} for item in results]
            self.assertFalse(identities[0] & identities[1])


if __name__ == "__main__":
    unittest.main()
