"""Bounded process scheduling checks; no reconstruction timing campaign."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import os
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

# Worker packets use canonical package names, including when unittest runs here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blender_blocking.reconstruction.backend import BaseBackend
from blender_blocking.reconstruction.ensemble import EnsembleRunner
from blender_blocking.reconstruction import registry
from blender_blocking.reconstruction.process_executor import (PersistentProcessExecutor,
    candidate_payload, candidate_outcome, read_packet)
from blender_blocking.reconstruction.native_geometry import GeometryArrays
from blender_blocking.reconstruction.types import (CandidateBudget, CandidateMetrics,
    CandidateRequest, CandidateResult, ReconstructionTarget)
from blender_blocking.placement.resfit.config import ResFitPipelineConfig
from blender_blocking.placement.resfit_initialization import PrimitiveInitializationConfig
from blender_blocking.placement.resfit_optimizer import CoordinateDescentConfig


class _ProcessFixtureBackend(BaseBackend):
    """Portable backend fixture transported through the real candidate IPC path."""

    def __init__(self):
        super().__init__(name="isolated_process_fixture")

    def reconstruct(self, request):
        action = request.config.get("action", "success")
        if action == "barrier":
            gate = Path(request.config["gate"])
            gate.mkdir(parents=True, exist_ok=True)
            (gate / (request.candidate_id + ".ready")).write_text(str(os.getpid()), encoding="utf-8")
            limit = time.monotonic() + 5.0
            expected = request.config["expected"]
            while not all((gate / (name + ".ready")).exists() for name in expected):
                if time.monotonic() >= limit:
                    raise RuntimeError("parallel admission barrier did not receive all jobs")
                time.sleep(0.01)
        elif action == "checkpoint":
            from blender_blocking.reconstruction.process_executor import publish_progress
            publish_progress({"scored": True, "retained": request.candidate_id}, recorded_evaluations=3)
            time.sleep(2.)
        elif action == "sleep":
            time.sleep(float(request.config.get("sleep_s", 2.0)))
        elif action == "crash":
            os._exit(23)
        elif action == "raise":
            raise RuntimeError("fixture backend rejected job")
        geometry = (GeometryArrays.capture([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
                    [[0,2,1],[0,1,3],[0,3,2],[1,2,3]]) if request.config.get("geometry") else None)
        return CandidateResult(request.candidate_id, self.name, "success", geometry=geometry,
            payload={"pid": os.getpid()}, metric_result=CandidateMetrics(per_view={
                "front": {"area_iou": 0.75, "boundary_iou": 0.6,
                          "passed": request.config.get("passed", True)}}))


class TestProcessExecutor(unittest.TestCase):
    def setUp(self):
        self.backend = _ProcessFixtureBackend()
        self.registry_patch = patch.dict(registry._BACKENDS, {self.backend.name: self.backend})
        self.registry_patch.start()
        self.addCleanup(self.registry_patch.stop)

    @contextmanager
    def pool(self, workers=2):
        with tempfile.TemporaryDirectory(prefix="blendslop-executor-test-") as directory:
            with PersistentProcessExecutor(workers, root=directory, threads=1) as pool:
                self.wait_ready(pool)
                processes = [state["process"] for state in pool.workers.values()]
                # The ready handshake precedes backend imports. Warm actual jobs so
                # short deadline checks exercise running work instead of cold imports.
                expected = [f"warm_{index}" for index in range(workers)]
                warm = [self.request(pool, name, action="barrier", gate=str(pool.root / "warm"),
                                     expected=expected) for name in expected]
                outcomes = pool.map([("candidate", candidate_payload(request, pool), 8.0)
                                     for request in warm], timeout_s=10.0)
                self.assertTrue(all(outcome.status == "success" and outcome.value.succeeded
                                    for outcome in outcomes),
                                [(outcome.error, getattr(outcome.value, "errors", ())) for outcome in outcomes])
                try:
                    yield pool
                finally:
                    processes.extend(state["process"] for state in pool.workers.values())
            self.assertTrue(all(process.poll() is not None for process in processes),
                            "owned worker processes must be joined on close")

    def wait_ready(self, pool):
        limit = time.monotonic() + 15.0
        while len(pool.workers) != pool.max_workers or not all(
                state["ready"] for state in pool.workers.values()):
            pool.poll()
            if time.monotonic() >= limit:
                logs = {worker: (pool.root / "workers" / worker / "worker.log").read_text(
                    encoding="utf-8", errors="replace") for worker in pool.workers}
                self.fail(f"workers did not become ready: {logs}")
            time.sleep(0.01)

    def worker_pids(self, pool):
        # On Windows a venv launcher is the Popen child and Python runs below it.
        return {read_packet(pool.root / "workers" / worker / "ready.pkl")["pid"]
                for worker in pool.workers}

    def request(self, pool, candidate_id="fixture", **config):
        return CandidateRequest(candidate_id, self.backend.name, ReconstructionTarget(),
            config=config, budget=CandidateBudget(timeout_s=8.0),
            artifact_root=pool.root / "candidate-artifacts",
            context=SimpleNamespace(process_executor=pool, blender_available=False))

    def submit_candidate(self, pool, request, **limits):
        return pool.submit("candidate", candidate_payload(request, pool), **limits)

    def test_worker_pid_persists_across_completed_jobs(self):
        with self.pool(workers=1) as pool:
            first = pool.result(self.submit_candidate(pool, self.request(pool, "first"), timeout_s=5.0))
            second = pool.result(self.submit_candidate(pool, self.request(pool, "second"), timeout_s=5.0))
            self.assertEqual((first.status, second.status), ("success", "success"))
            self.assertGreater(first.worker_pid, 0)
            self.assertEqual(first.worker_pid, second.worker_pid)
            self.assertNotEqual(first.worker_pid, os.getpid())

    def test_timeout_retains_scored_checkpoint_and_honest_queue_accounting(self):
        with self.pool(workers=1) as pool:
            request = self.request(pool, "partial", action="checkpoint")
            outcome = pool.result(self.submit_candidate(pool, request, timeout_s=.3))
            self.assertEqual(outcome.status, "timeout")
            self.assertTrue(outcome.partial)
            self.assertEqual(outcome.value, {"scored": True, "retained": "partial"})
            self.assertEqual(outcome.recorded_evaluations, 3)
            self.assertGreaterEqual(outcome.total_wall_s, outcome.elapsed_s)
            self.assertGreaterEqual(outcome.queue_s, 0.)
            self.assertIn("deadline", outcome.stop_reason)
            self.assertFalse(candidate_outcome(request, outcome).succeeded)

    def test_parallel_normal_ensemble_keeps_total_deadline(self):
        with self.pool() as pool:
            gate = pool.root / "parallel-normal"
            requests = [self.request(pool, name, action="barrier", gate=str(gate), expected=["a", "b"])
                        for name in ("a", "b")]
            result = EnsembleRunner().run_requests(requests, total_timeout_s=8.0,
                                                  max_parallel_candidates=2)
            self.assertTrue(all(candidate.succeeded for candidate in result.candidates),
                            [candidate.errors for candidate in result.candidates])
            self.assertEqual(len({candidate.payload["pid"] for candidate in result.candidates}), 2)
            self.assertIsNotNone(result.selected)
            self.assertEqual([candidate.candidate_id for candidate in result.candidates], ["a", "b"])

    def test_total_deadline_preserves_completed_winner(self):
        with self.pool() as pool:
            requests = [self.request(pool, "slow", action="sleep", sleep_s=2.0),
                        self.request(pool, "winner")]
            result = EnsembleRunner().run_requests(requests, total_timeout_s=0.4,
                                                  max_parallel_candidates=2)
            candidates = {candidate.candidate_id: candidate for candidate in result.candidates}
            self.assertTrue(candidates["winner"].succeeded, candidates["winner"].errors)
            self.assertEqual(candidates["slow"].status, "failed")
            self.assertTrue(any("deadline" in error for error in candidates["slow"].errors))
            self.assertEqual(result.selected.candidate_id, "winner")

    def test_failed_worker_does_not_discard_success_and_is_replaced(self):
        with self.pool() as pool:
            initial_pids = self.worker_pids(pool)
            good_request = self.request(pool, "retained")
            bad_request = self.request(pool, "crash", action="crash")
            good_id = self.submit_candidate(pool, good_request, timeout_s=6.0)
            bad_id = self.submit_candidate(pool, bad_request, timeout_s=6.0)
            good = pool.result(good_id)
            bad = pool.result(bad_id)
            self.assertEqual(good.status, "success")
            self.assertEqual(bad.status, "failed")
            self.assertEqual(candidate_outcome(good_request, good).status, "success")
            self.assertEqual(candidate_outcome(bad_request, bad).status, "failed")
            self.wait_ready(pool)
            replacement_pids = self.worker_pids(pool)
            self.assertEqual(len(initial_pids & replacement_pids), 1)
            gate = pool.root / "replacement"
            requests = [self.request(pool, name, action="barrier", gate=str(gate),
                                     expected=["reused", "replaced"])
                        for name in ("reused", "replaced")]
            outcomes = pool.map([("candidate", candidate_payload(request, pool), 6.0)
                                 for request in requests], timeout_s=8.0)
            self.assertTrue(all(outcome.status == "success" for outcome in outcomes))
            self.assertEqual({outcome.worker_pid for outcome in outcomes}, replacement_pids)
            self.assertEqual(pool.result(good_id).value.candidate_id, "retained")

    def test_backend_exception_preserves_result_and_worker(self):
        with self.pool(workers=1) as pool:
            failed_request = self.request(pool, "failed", action="raise")
            failed = pool.result(self.submit_candidate(pool, failed_request, timeout_s=5.0))
            self.assertEqual(failed.status, "success", "backend failure is a transported CandidateResult")
            candidate = candidate_outcome(failed_request, failed)
            self.assertEqual(candidate.status, "failed")
            self.assertIn("fixture backend rejected job", candidate.errors[0])
            following = pool.result(self.submit_candidate(pool, self.request(pool, "following"), timeout_s=5.0))
            self.assertEqual(following.status, "success")
            self.assertEqual(failed.worker_pid, following.worker_pid)

    def test_routing_uses_parallel_pool_and_rejects_unmeasured_metrics(self):
        with self.pool() as pool:
            gate = pool.root / "parallel-routing"
            requests = [self.request(pool, name, action="barrier", gate=str(gate),
                                     expected=["routed_a", "routed_b"], passed=False)
                        for name in ("routed_a", "routed_b")]
            result = EnsembleRunner(evidence_routing=True, max_render_candidates=2).run_requests(
                requests, total_timeout_s=8.0, max_parallel_candidates=2)
            self.assertEqual(len({candidate.payload["pid"] for candidate in result.candidates}), 2)
            self.assertIsNone(result.selected)
            ledger = result.pareto_report["routing_ledger"]
            self.assertEqual(ledger["submitted_candidates"], 2)
            self.assertEqual(ledger["rendered_candidates"], 0)
            self.assertEqual(ledger["max_workers"], 2)

    def test_transport_preserves_geometry_immutability_and_hashes(self):
        with self.pool(workers=1) as pool:
            request = self.request(pool, "geometry", geometry=True)
            outcome = pool.result(self.submit_candidate(pool, request, timeout_s=5.0))
            self.assertEqual(outcome.status, "success", outcome.error)
            geometry = outcome.value.geometry
            self.assertIsInstance(geometry, GeometryArrays)
            self.assertFalse(geometry.vertices.flags.writeable)
            self.assertFalse(geometry.faces.flags.writeable)
            recaptured = GeometryArrays.capture(geometry.vertices, geometry.faces)
            self.assertEqual(geometry.content_hash, recaptured.content_hash)
            self.assertEqual(geometry.connectivity_hash, recaptured.connectivity_hash)
            with self.assertRaises(ValueError):
                geometry.vertices[0] = 0.0

    def test_unknown_backend_does_not_abort_valid_candidates(self):
        with self.pool() as pool:
            unknown = replace(self.request(pool, "unknown"), backend_name="unregistered_process_backend")
            valid = self.request(pool, "valid")
            result = EnsembleRunner().run_requests([unknown, valid], total_timeout_s=6.0,
                                                  max_parallel_candidates=2)
            candidates = {candidate.candidate_id: candidate for candidate in result.candidates}
            self.assertEqual(candidates["unknown"].status, "failed")
            self.assertTrue(candidates["unknown"].errors)
            self.assertTrue(candidates["valid"].succeeded)
            self.assertEqual(result.selected.candidate_id, "valid")

    def test_unsafe_job_payload_does_not_abort_valid_candidates(self):
        with self.pool() as pool:
            unsafe = self.request(pool, "unsafe", callback=lambda: None)
            valid = self.request(pool, "valid")
            result = EnsembleRunner().run_requests([unsafe, valid], total_timeout_s=6.0,
                                                  max_parallel_candidates=2)
            candidates = {candidate.candidate_id: candidate for candidate in result.candidates}
            self.assertEqual(candidates["unsafe"].status, "failed")
            self.assertTrue(any("process-safe" in error for error in candidates["unsafe"].errors))
            self.assertTrue(candidates["valid"].succeeded)
            self.assertEqual(result.selected.candidate_id, "valid")

    def test_simultaneous_nested_multistart_jobs_share_flat_workers(self):
        with self.pool() as pool:
            initial_pids = self.worker_pids(pool)
            points = np.array([[-0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[0.5,-0.5,-0.5],[0.5,0.5,-0.5],
                               [-0.5,-0.5,0.5],[-0.5,0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,0.5]])
            config = ResFitPipelineConfig(primitive_family="superfrustum",
                initialization=PrimitiveInitializationConfig(primitive_count=1, target_point_count=8),
                optimizer=CoordinateDescentConfig(iterations=1, max_objective_evaluations=4))
            jobs = [("fit_multistart", dict(target_points=points + offset, config=config, max_attempts=2), 10.0)
                    for offset in (0.0, 0.2)]
            outcomes = pool.map(jobs, timeout_s=12.0)
            self.assertTrue(all(outcome.status == "success" for outcome in outcomes),
                            [outcome.error for outcome in outcomes])
            self.assertEqual({outcome.worker_pid for outcome in outcomes}, initial_pids)
            fit_jobs = [job for job in pool.jobs.values() if job["kind"] == "fit_start"]
            self.assertEqual(len(fit_jobs), 4)
            self.assertTrue(all(job["outcome"].status == "success" for job in fit_jobs))
            self.assertLessEqual({job["outcome"].worker_pid for job in fit_jobs}, initial_pids)
            self.assertEqual(len(pool.workers), 2)
            self.assertFalse(pool.groups)
            self.assertTrue(all(not state["stack"] for state in pool.workers.values()))
            for outcome in outcomes:
                self.assertEqual(len(outcome.value.attempts), 2)
                self.assertEqual(outcome.value.objective_evaluations, 4)
                self.assertTrue(np.isfinite(outcome.value.final_loss.total))


if __name__ == "__main__":
    unittest.main()
