"""Asynchronous fresh Windows process-tree ownership for reusable workers.

Polling does not perform kernel waits. Stop joins within one caller-visible allowance and retains
handles after an incomplete join so the same owner can retry. It does not adopt
old PIDs or release artifact leases, and broker-created processes are excluded.
"""
from __future__ import annotations

from copy import deepcopy
from functools import wraps
from pathlib import Path
import subprocess
import time
import threading

from .owned_process_supervisor import _WindowsJob, validate_process_bounds


def _locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class OwnedProcess:
    def __init__(self, command, *, log_path, timeout_s, max_memory_bytes,
                 join_timeout_s=5., max_rss_bytes=None, cwd=None, env=None):
        validate_process_bounds(command, timeout_s=timeout_s, max_memory_bytes=max_memory_bytes,
                                join_timeout_s=join_timeout_s, max_rss_bytes=max_rss_bytes)
        self._lock = threading.RLock()
        self._monitor_stop = threading.Event()
        self._monitor = None
        self.command = [str(arg) for arg in command]
        self.log_path = Path(log_path)
        if self.log_path.exists() or not self.log_path.parent.is_dir():
            raise ValueError("owned process log must be fresh in an existing directory")
        self.timeout_s = float(timeout_s)
        self.join_timeout_s = float(join_timeout_s)
        self.max_memory_bytes, self.max_rss_bytes = max_memory_bytes, max_rss_bytes
        self.cwd, self.env = cwd, env
        self.process = self.job = self.stream = None
        self.started = self.last_sample = None
        self.released = False
        self.receipt = {
            "protocol": "fresh_async_owned_process_v1", "command": self.command,
            "status": "not_started", "lifecycle_complete": False, "primary_joined": False,
            "job_active_zero": False, "observed_handles_joined": False,
            "job_assigned_before_resume": False, "ordinary_createprocess_tree_owned": False,
            "broker_process_ownership": "unsupported", "artifact_lease_release": "unsupported",
            "timeout_s": self.timeout_s, "join_timeout_s": self.join_timeout_s,
            "max_job_committed_bytes": max_memory_bytes, "max_rss_bytes": max_rss_bytes,
            "log_byte_limit": 4194304, "log_path": str(self.log_path),
            "limit_reason": None, "stop_reason": None, "returncode": None,
            "rss_sample_count": 0, "peak_observed_tree_rss_bytes": None,
            "rss_unavailable_samples": [], "errors": [], "observed_processes": [],
            "rss_sampling_scope": "automatic monitor plus caller polling, at most 50Hz; inter-poll peaks unavailable",
        }

    @_locked
    def snapshot(self):
        if self.started is not None:
            self.receipt["elapsed_seconds"] = time.monotonic() - self.started
        if self.job is not None:
            self.receipt["observed_processes"] = deepcopy(self.job.records)
        return deepcopy(self.receipt)

    @_locked
    def start(self):
        if self.started is not None or self.released:
            raise RuntimeError("an owned process scope can launch only once")
        self.started = time.monotonic()
        try:
            self.job = _WindowsJob(self.max_memory_bytes)
            self.stream = self.log_path.open("xb")
            self.process = subprocess.Popen(
                self.command, stdout=self.stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, cwd=self.cwd, env=self.env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | 4)
            self.receipt["pid"] = self.process.pid
            self.receipt["primary_identity"] = self.job.attach_and_resume(self.process)
            self.receipt.update(status="running", job_assigned_before_resume=True,
                                ordinary_createprocess_tree_owned=True)
            self._monitor = threading.Thread(target=self._monitor_bounds,
                name="blendslop-owned-process-" + str(self.process.pid), daemon=True)
            self._monitor.start()
        except BaseException as error:
            self.receipt["errors"].append(repr(error))
            result = self.stop(reason="launch_failed")
            error.owned_process_receipt = result
            error.owned_process_scope = self
            raise
        return self.process

    def _monitor_bounds(self):
        while not self._monitor_stop.wait(.02):
            try:
                observation = self.poll()
                if observation["limit_reason"]:
                    return
            except Exception as error:
                with self._lock:
                    if self.released:
                        return
                    self.receipt["errors"].append("automatic monitor: " + repr(error))
                    self.receipt["limit_reason"] = "monitor_failed"
                    try:
                        self.job.terminate()
                    except Exception as termination_error:
                        self.receipt["errors"].append("monitor termination: " + repr(termination_error))
                    return

    @_locked
    def poll(self):
        """Observe bounds and primary exit without consuming a join allowance."""
        if self.process is None or self.released:
            return self.snapshot()
        now = time.monotonic()
        if self.last_sample is None or now - self.last_sample >= .02:
            self.job.observe()
            self.receipt["job_accounting"] = self.job.accounting()
            sample = self.job.rss_sample()
            self.receipt["rss_sample_count"] += 1
            self.last_sample = now
            if sample["status"] == "available":
                self.receipt["peak_observed_tree_rss_bytes"] = max(
                    self.receipt["peak_observed_tree_rss_bytes"] or 0, sample["tree_rss_bytes"])
            elif len(self.receipt["rss_unavailable_samples"]) < 128:
                self.receipt["rss_unavailable_samples"].append(sample)
            if self.max_rss_bytes is not None:
                if sample["status"] != "available":
                    self.receipt["limit_reason"] = "rss_monitor_unavailable"
                elif sample["tree_rss_bytes"] > self.max_rss_bytes:
                    self.receipt["limit_reason"] = "tree_rss_limit"
        if now - self.started >= self.timeout_s:
            self.receipt["limit_reason"] = self.receipt["limit_reason"] or "wall_timeout"
        elif self.log_path.stat().st_size > self.receipt["log_byte_limit"]:
            self.receipt["limit_reason"] = self.receipt["limit_reason"] or "log_byte_limit"
        if self.receipt["limit_reason"]:
            # Wall/log/RSS limits still apply while the caller is idle.
            self.job.terminate()
        code = self.process.poll()
        self.receipt["returncode"] = code
        self.receipt["status"] = "limit_reached" if self.receipt["limit_reason"] else (
            "primary_exited" if code is not None else "running")
        return self.snapshot()

    @_locked
    def stop(self, *, reason="owner_closed", join_timeout_s=None):
        """Stop only this fresh tree; a failed join keeps the owner retryable."""
        allowance = self.join_timeout_s if join_timeout_s is None else join_timeout_s
        if isinstance(allowance, bool) or not isinstance(allowance, (int, float)) or not 0 < allowance <= self.join_timeout_s:
            raise ValueError("stop allowance must be positive and within the configured join bound")
        if self.released:
            return self.snapshot()
        self._monitor_stop.set()
        self.receipt["stop_reason"] = reason
        deadline = time.monotonic() + allowance
        errors = self.receipt["errors"]
        if self.job is not None:
            try:
                self.job.observe()
            except Exception as error:
                errors.append("stop observation: " + repr(error))
            try:
                # A primary that already exited can still have live descendants.
                self.job.terminate()
            except Exception as error:
                errors.append("job termination: " + repr(error))
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    self.process.kill()
                self.receipt["returncode"] = self.process.wait(timeout=max(0., deadline-time.monotonic()))
                self.receipt["primary_joined"] = True
            except Exception as error:
                errors.append("primary join: " + repr(error))
        else:
            # No child launched; there is no primary or descendant to join.
            self.receipt["primary_joined"] = True
        if self.job is not None:
            try:
                while True:
                    self.job.observe()
                    self.receipt["job_accounting"] = self.job.accounting()
                    if self.receipt["job_accounting"]["active_processes"] == 0:
                        self.receipt["job_active_zero"] = True
                        break
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(min(.02, max(0., deadline-time.monotonic())))
                self.receipt["observed_handles_joined"] = self.job.join_observed(deadline)
            except Exception as error:
                errors.append("tree join: " + repr(error))
        else:
            self.receipt.update(job_active_zero=self.process is None,
                                observed_handles_joined=self.process is None)
        self.receipt["lifecycle_complete"] = all(self.receipt[key] for key in (
            "primary_joined", "job_active_zero", "observed_handles_joined"))
        self.receipt["status"] = "stopped" if self.receipt["lifecycle_complete"] else "join_incomplete"
        self.snapshot()
        if self.receipt["lifecycle_complete"]:
            if self.job is not None:
                self.job.close()
                self.job = None
            if self.stream is not None:
                self.stream.close()
                self.stream = None
            self.released = True
        return self.snapshot()
