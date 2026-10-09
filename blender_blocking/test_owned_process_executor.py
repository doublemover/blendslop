"""Small real-process checks for asynchronous Windows ownership and pool reuse."""
from pathlib import Path
import os
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import test_process_executor as fixture
from test_owned_lifecycle_processes import fixture_python
from blender_blocking.reconstruction.process_executor import (PersistentProcessExecutor,
    WorkerProcessBudget, candidate_payload)
from blender_blocking.utils.owned_process import OwnedProcess


class _DescendantBackend(fixture._ProcessFixtureBackend):
    def reconstruct(self, request):
        if request.config.get("action") == "spawn_crash":
            marker = Path(request.config["marker"])
            code = ("import os,pathlib,time;"
                    f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()));time.sleep(20.)")
            subprocess.Popen([request.config["python"], "-c", code],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.monotonic() + 3.
            while not marker.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("descendant did not start")
                time.sleep(.01)
            time.sleep(.08)  # Permit exact descendant HANDLE observation.
            os._exit(23)
        return super().reconstruct(request)


@unittest.skipUnless(os.name == "nt", "actual Windows Job Object ownership")
class TestAsyncOwnedProcess(unittest.TestCase):
    def make_scope(self, directory, code, **extra):
        return OwnedProcess([str(fixture_python()), "-c", code],
            **{"log_path": Path(directory) / "fresh.log", "timeout_s": 5.,
               "max_memory_bytes": 268435456, "join_timeout_s": 2., **extra})

    def test_idle_owner_still_enforces_wall_limit_and_joins(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = self.make_scope(directory, "import time;time.sleep(20.)", timeout_s=.25)
            process = scope.start()
            try:
                time.sleep(.65)
                self.assertIsNotNone(process.poll(), "automatic wall monitor must work without caller polls")
                result = scope.stop()
                self.assertEqual(result["limit_reason"], "wall_timeout")
                self.assertTrue(result["lifecycle_complete"], result)
                self.assertTrue(result["job_assigned_before_resume"])
                self.assertTrue(all(row["kernel_joined"] for row in result["observed_processes"]))
            finally:
                scope.stop()

    def test_exited_primary_descendant_is_owned_and_joined(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "child.pid"
            descendant = f"import os,pathlib,time;pathlib.Path({str(marker)!r}).write_text(str(os.getpid()));time.sleep(20.)"
            code = ("import os,subprocess,sys,time;"
                    f"subprocess.Popen([sys.executable,'-c',{descendant!r}],creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));"
                    "time.sleep(.2);os._exit(17)")
            scope = self.make_scope(directory, code)
            scope.start()
            try:
                deadline = time.monotonic() + 3.
                while scope.process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(scope.process.poll(), 17)
                result = scope.stop()
                pid = int(marker.read_text())
                child = next(row for row in result["observed_processes"] if row["pid"] == pid)
                self.assertTrue(child["kernel_joined"], result)
                self.assertEqual(child["exit_status"], 124)
                self.assertTrue(result["lifecycle_complete"])
                self.assertEqual(result["job_accounting"]["active_processes"], 0)
            finally:
                scope.stop()

    def test_incomplete_join_retains_same_owner_for_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = self.make_scope(directory, "import time;time.sleep(20.)")
            scope.start()
            job = scope.job
            try:
                with patch.object(job, "join_observed", return_value=False):
                    result = scope.stop()
                self.assertFalse(result["lifecycle_complete"])
                self.assertFalse(scope.released)
                self.assertIs(scope.job, job)
                self.assertIsNotNone(job.job)
                retried = scope.stop()
                self.assertTrue(retried["lifecycle_complete"], retried)
                self.assertTrue(scope.released)
                self.assertTrue(all(row["kernel_joined"] for row in retried["observed_processes"]))
            finally:
                scope.stop()

    def test_automatic_monitor_fault_stops_owned_tree_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = self.make_scope(directory, "import time;time.sleep(20.)")
            scope.start()
            try:
                with patch.object(scope.job, "rss_sample", side_effect=RuntimeError("fixture live memory read failed")):
                    time.sleep(.15)
                self.assertEqual(scope.snapshot()["limit_reason"], "monitor_failed")
                result = scope.stop()
                self.assertTrue(result["lifecycle_complete"], result)
                self.assertTrue(any("fixture live memory read failed" in error for error in result["errors"]))
            finally:
                scope.stop()

    def test_launch_error_preserves_exception_and_complete_empty_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = OwnedProcess([str(Path(directory) / "absent.exe")],
                log_path=Path(directory) / "fresh.log", timeout_s=1., max_memory_bytes=268435456)
            with self.assertRaises(FileNotFoundError) as caught:
                scope.start()
            self.assertIs(caught.exception.owned_process_scope, scope)
            self.assertTrue(caught.exception.owned_process_receipt["lifecycle_complete"])
            self.assertFalse(caught.exception.owned_process_receipt["job_assigned_before_resume"])


@unittest.skipUnless(os.name == "nt", "actual opt-in Windows persistent worker trees")
class TestOwnedProcessExecutor(unittest.TestCase):
    request = fixture.TestProcessExecutor.request
    submit_candidate = fixture.TestProcessExecutor.submit_candidate

    def setUp(self):
        fixture.TestProcessExecutor.setUp(self)
        self.backend = _DescendantBackend()
        self.registry_patch.stop()
        from blender_blocking.reconstruction import registry
        self.registry_patch = patch.dict(registry._BACKENDS, {self.backend.name: self.backend})
        self.registry_patch.start()
        self.addCleanup(self.registry_patch.stop)

    def make_pool(self, directory, **extra):
        return PersistentProcessExecutor(1, root=directory, threads=1, blender_binary=False,
            process_budget=WorkerProcessBudget(wall_s=25., max_memory_bytes=1073741824,
                                               join_timeout_s=2., **extra))

    def test_invalid_budget_refuses_before_launch(self):
        with tempfile.TemporaryDirectory() as directory, patch("subprocess.Popen") as launch:
            for budget in (WorkerProcessBudget(True, 100), WorkerProcessBudget(1., 0),
                           WorkerProcessBudget(1., 100, max_restarts=-1)):
                with self.subTest(budget=budget), self.assertRaises(ValueError):
                    PersistentProcessExecutor(root=directory, process_budget=budget)
            launch.assert_not_called()

    def test_reuse_and_crashed_primary_descendant_restart_share_pool_budget(self):
        python = str(fixture_python())
        with tempfile.TemporaryDirectory() as directory, patch("sys.executable", python):
            pool = self.make_pool(directory, max_restarts=1)
            try:
                first = pool.result(self.submit_candidate(pool, self.request(pool, "first"), timeout_s=8.))
                second = pool.result(self.submit_candidate(pool, self.request(pool, "second"), timeout_s=8.))
                self.assertEqual((first.status, second.status), ("success", "success"))
                self.assertEqual(first.worker_pid, second.worker_pid)
                deadline = pool._budget_deadline
                marker = pool.root / "descendant.pid"
                failed = pool.result(self.submit_candidate(pool, self.request(pool, "crash",
                    action="spawn_crash", marker=str(marker), python=python), timeout_s=5.))
                self.assertEqual(failed.status, "failed")
                pid = int(marker.read_text())
                original = pool.process_scopes[0].snapshot()
                child = next(row for row in original["observed_processes"] if row["pid"] == pid)
                self.assertTrue(child["kernel_joined"], original)
                self.assertEqual(child["exit_status"], 124)
                self.assertTrue(original["lifecycle_complete"])
                following = pool.result(self.submit_candidate(pool, self.request(pool, "replacement"), timeout_s=8.))
                self.assertEqual(following.status, "success", following.error)
                self.assertNotEqual(first.worker_pid, following.worker_pid)
                self.assertEqual(pool._budget_deadline, deadline)
                self.assertLess(pool.process_scopes[1].timeout_s, pool.process_scopes[0].timeout_s)
            finally:
                pool.close()
            receipt = pool.ownership_receipt()
            self.assertTrue(receipt["lifecycle_complete"], receipt)
            self.assertEqual(receipt["launch_count"], 2)
            self.assertEqual(len(list(pool.ownership_root.glob("*/lifecycle.json"))), 2)
            self.assertEqual(len(list(pool.ownership_root.glob("*/worker.log"))), 2)

    def test_close_retries_incomplete_tree_and_preserves_primary_exception(self):
        python = str(fixture_python())
        with tempfile.TemporaryDirectory() as directory, patch("sys.executable", python):
            pool = self.make_pool(directory)
            pool.start()
            scope = pool.process_scopes[0]
            with patch.object(scope.job, "join_observed", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "incomplete owned tree join"):
                    pool.close()
            self.assertFalse(pool.ownership_receipt()["lifecycle_complete"])
            self.assertEqual(len(pool.workers), 1)
            pool.close()
            self.assertTrue(pool.ownership_receipt()["lifecycle_complete"])
            self.assertFalse(pool.workers)
            original = KeyboardInterrupt("caller cancelled")
            with patch.object(pool, "close", side_effect=RuntimeError("fixture cleanup failure")):
                self.assertFalse(pool.__exit__(type(original), original, None))
            self.assertIn("fixture cleanup failure", original.__notes__[0])

    def test_fresh_pool_preserves_old_transport_and_caps_replacements(self):
        python = str(fixture_python())
        with tempfile.TemporaryDirectory() as directory, patch("sys.executable", python):
            old = Path(directory) / "workers" / "0"
            old.mkdir(parents=True)
            for name in ("ready.pkl", "inbox.pkl", "stop", "worker.log"):
                (old / name).write_bytes(b"unrelated retained transport")
            pool = self.make_pool(directory, max_restarts=0)
            try:
                self.assertNotEqual(pool.root, Path(directory))
                warm = pool.result(self.submit_candidate(pool, self.request(pool, "warm"), timeout_s=8.))
                self.assertEqual(warm.status, "success", warm.error)
                crashed = pool.result(self.submit_candidate(pool, self.request(pool, "crash", action="crash"), timeout_s=5.))
                self.assertEqual(crashed.status, "failed")
                following = pool.result(self.submit_candidate(pool, self.request(pool, "never"), timeout_s=5.))
                self.assertEqual(following.status, "failed")
                self.assertEqual(pool._launch_count, 1)
            finally:
                pool.close()
            self.assertTrue(pool.ownership_receipt()["lifecycle_complete"])
            for path in old.iterdir():
                self.assertEqual(path.read_bytes(), b"unrelated retained transport")

    def test_pool_expiry_stops_queue_and_does_not_reset(self):
        python = str(fixture_python())
        with tempfile.TemporaryDirectory() as directory, patch("sys.executable", python):
            pool = self.make_pool(directory)
            try:
                warm = pool.result(self.submit_candidate(pool, self.request(pool, "warm"), timeout_s=8.))
                self.assertEqual(warm.status, "success", warm.error)
                active = self.submit_candidate(pool, self.request(pool, "active", action="sleep", sleep_s=2.), timeout_s=5.)
                queued = self.submit_candidate(pool, self.request(pool, "queued"), timeout_s=5.)
                pool.poll()
                pool._budget_deadline = time.monotonic() - .01
                outcome = pool.result(active)
                self.assertEqual(outcome.status, "timeout", outcome)
                self.assertEqual(pool.result(queued).status, "timeout")
                self.assertFalse(pool.workers)
                self.assertEqual(pool._launch_count, 1)
            finally:
                pool.close()
            self.assertTrue(pool.ownership_receipt()["lifecycle_complete"])


if __name__ == "__main__":
    unittest.main()
