"""Fresh bounded process ownership; Windows jobs precede child execution.

No historical PID adoption, lease release, deletion or universal broker ownership.
Callers release artifact leases only when the returned lifecycle is complete.
"""
from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import subprocess
import time


class _WindowsJob:
    """An unnamed no-breakaway job and retained exact process handles."""
    def __init__(self, memory_limit):
        from ctypes import wintypes as w
        self.w = w
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.handles = {}
        self.records = []
        self.job = None
        size_t = ctypes.c_size_t

        class BasicLimit(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", w.DWORD), ("min_ws", size_t), ("max_ws", size_t),
                        ("active_limit", w.DWORD), ("affinity", size_t),
                        ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
        class Extended(ctypes.Structure):
            _fields_ = [("basic", BasicLimit), ("io", IO), ("process_limit", size_t),
                        ("job_limit", size_t), ("peak_process", size_t), ("peak_job", size_t)]
        class Accounting(ctypes.Structure):
            _fields_ = [(name, ctypes.c_int64) for name in ("user_time", "kernel_time", "period_user", "period_kernel")]
            _fields_ += [(name, w.DWORD) for name in ("page_faults", "total", "active", "terminated")]
        class PIDList(ctypes.Structure):
            _fields_ = [("assigned", w.DWORD), ("count", w.DWORD), ("pids", size_t * 128)]
        class ThreadEntry(ctypes.Structure):
            _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("tid", w.DWORD),
                        ("pid", w.DWORD), ("priority", w.LONG), ("delta", w.LONG), ("flags", w.DWORD)]
        class MemoryCounters(ctypes.Structure):
            _fields_ = [("size", w.DWORD), ("faults", w.DWORD)]
            _fields_ += [(name, size_t) for name in ("peak_ws", "working_set", "peak_paged", "paged",
                                                  "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
        self.MemoryCounters = MemoryCounters
        self.Extended, self.Accounting, self.PIDList, self.ThreadEntry = Extended, Accounting, PIDList, ThreadEntry
        declarations = {
            "CreateJobObjectW": (w.HANDLE, [ctypes.c_void_p, w.LPCWSTR]),
            "K32GetProcessMemoryInfo": (w.BOOL, [w.HANDLE, ctypes.POINTER(MemoryCounters), w.DWORD]),
            "SetInformationJobObject": (w.BOOL, [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]),
            "QueryInformationJobObject": (w.BOOL, [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p]),
            "AssignProcessToJobObject": (w.BOOL, [w.HANDLE, w.HANDLE]),
            "IsProcessInJob": (w.BOOL, [w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)]),
            "TerminateJobObject": (w.BOOL, [w.HANDLE, w.UINT]),
            "OpenProcess": (w.HANDLE, [w.DWORD, w.BOOL, w.DWORD]),
            "GetProcessId": (w.DWORD, [w.HANDLE]),
            "GetProcessTimes": (w.BOOL, [w.HANDLE, ctypes.POINTER(w.FILETIME), ctypes.POINTER(w.FILETIME), ctypes.POINTER(w.FILETIME), ctypes.POINTER(w.FILETIME)]),
            "GetExitCodeProcess": (w.BOOL, [w.HANDLE, ctypes.POINTER(w.DWORD)]),
            "WaitForSingleObject": (w.DWORD, [w.HANDLE, w.DWORD]),
            "CloseHandle": (w.BOOL, [w.HANDLE]),
            "CreateToolhelp32Snapshot": (w.HANDLE, [w.DWORD, w.DWORD]),
            "Thread32First": (w.BOOL, [w.HANDLE, ctypes.POINTER(ThreadEntry)]),
            "Thread32Next": (w.BOOL, [w.HANDLE, ctypes.POINTER(ThreadEntry)]),
            "OpenThread": (w.HANDLE, [w.DWORD, w.BOOL, w.DWORD]),
            "ResumeThread": (w.DWORD, [w.HANDLE]),
            "GetProcessIdOfThread": (w.DWORD, [w.HANDLE]),
            "GetThreadId": (w.DWORD, [w.HANDLE]),
        }
        for name, (result, args) in declarations.items():
            function = getattr(self.kernel, name)
            function.restype, function.argtypes = result, args
        self.job = self.kernel.CreateJobObjectW(None, None)
        if not self.job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        # KILL_ON_JOB_CLOSE + JOB_MEMORY + ACTIVE_PROCESS; no breakaway flags.
        limits.basic.flags = 0x2000 | 0x200 | 0x8
        limits.basic.active_limit = 128
        limits.job_limit = memory_limit
        try:
            self.check(self.kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    def identity(self, handle):
        times = [self.w.FILETIME() for _ in range(4)]
        self.check(self.kernel.GetProcessTimes(handle, *(ctypes.byref(v) for v in times)))
        return {"pid": int(self.kernel.GetProcessId(handle)),
                "creation_filetime": (int(times[0].dwHighDateTime) << 32) | int(times[0].dwLowDateTime)}

    def attach_and_resume(self, child):
        # Popen owns the actual CreateProcess HANDLE, not a later PID lookup.
        handle = int(child._handle)
        identity = self.identity(handle)
        if identity["pid"] != child.pid:
            raise ValueError("fresh child HANDLE/PID mismatch")
        self.check(self.kernel.AssignProcessToJobObject(self.job, handle))
        self.observe()
        snapshot = self.kernel.CreateToolhelp32Snapshot(4, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        tids = []
        try:
            entry = self.ThreadEntry()
            entry.size = ctypes.sizeof(entry)
            more = self.kernel.Thread32First(snapshot, ctypes.byref(entry))
            while more:
                if entry.pid == child.pid:
                    tids.append(int(entry.tid))
                more = self.kernel.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            self.kernel.CloseHandle(snapshot)
        if len(tids) != 1:
            raise ValueError("fresh suspended child must expose its sole primary thread")
        thread = self.check(self.kernel.OpenThread(2 | 0x800, False, tids[0]))
        try:
            if (self.kernel.GetProcessIdOfThread(thread) != child.pid or
                    self.kernel.GetThreadId(thread) != tids[0]):
                raise ValueError("fresh suspended thread HANDLE identity differs")
            current = self.check(self.kernel.OpenProcess(0x100000 | 0x1000, False, child.pid))
            try:
                if self.identity(current) != identity or self.kernel.WaitForSingleObject(handle, 0) != 258:
                    raise ValueError("fresh suspended process identity/active state differs")
            finally:
                self.kernel.CloseHandle(current)
            # No child user code can run before job assignment and this resume.
            previous = self.kernel.ResumeThread(thread)
            if previous != 1:
                raise OSError("fresh primary thread suspension count differs")
        finally:
            self.kernel.CloseHandle(thread)
        return identity

    def query(self, cls, value):
        self.check(self.kernel.QueryInformationJobObject(self.job, cls, ctypes.byref(value), ctypes.sizeof(value), None))
        return value

    def accounting(self):
        value = self.query(1, self.Accounting())
        limits = self.query(9, self.Extended())
        return {"active_processes": int(value.active), "total_processes": int(value.total),
                "terminated_processes": int(value.terminated), "peak_job_committed_bytes": int(limits.peak_job)}

    def observe(self):
        value = self.query(3, self.PIDList())
        for pid in value.pids[:value.count]:
            handle = self.kernel.OpenProcess(0x100000 | 0x1000, False, int(pid))
            if not handle:
                # A short-lived process can end before an individual HANDLE is
                # captured. Job accounting still owns its completion; no exit
                # status or individual kernel join is invented for it.
                if ctypes.get_last_error() == 87:
                    continue
                raise ctypes.WinError(ctypes.get_last_error())
            keep = False
            try:
                included = self.w.BOOL()
                self.check(self.kernel.IsProcessInJob(handle, self.job, ctypes.byref(included)))
                identity = self.identity(handle)
                if not included.value or identity["pid"] != int(pid):
                    raise ValueError("observed HANDLE is not in this fresh owned job")
                key = identity["pid"], identity["creation_filetime"]
                if key not in self.handles:
                    if len(self.handles) >= 128:
                        raise ValueError("retained process HANDLE bound exceeded")
                    self.handles[key] = handle
                    self.records.append({**identity, "kernel_joined": False, "exit_status": None})
                    keep = True
            finally:
                if not keep:
                    self.kernel.CloseHandle(handle)

    def rss_sample(self):
        total, reads, unavailable = 0, 0, []
        for identity, handle in self.handles.items():
            state = self.kernel.WaitForSingleObject(handle, 0)
            if state == 0:
                continue
            try:
                if state != 258 or self.identity(handle) != {"pid": identity[0], "creation_filetime": identity[1]}:
                    raise ValueError("RSS process HANDLE identity/state differs")
                counters = self.MemoryCounters()
                counters.size = ctypes.sizeof(counters)
                if not self.kernel.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.size):
                    # A member may finish between the live-handle check and
                    # memory query. An actual signaled HANDLE confirms that
                    # race; other unavailable live reads remain explicit.
                    if self.kernel.WaitForSingleObject(handle, 0) == 0:
                        continue
                    raise ctypes.WinError(ctypes.get_last_error())
                total += int(counters.working_set)
                reads += 1
            except Exception as exc:
                unavailable.append({"pid": identity[0], "creation_filetime": identity[1], "reason": repr(exc)})
        return {"status": "unavailable" if unavailable else "available",
                "tree_rss_bytes": None if unavailable else total,
                "readable_working_set_bytes": total, "live_handle_reads": reads,
                "unavailable": unavailable}

    def join_observed(self, deadline):
        for record in self.records:
            handle = self.handles[record["pid"], record["creation_filetime"]]
            remaining_ms = max(0, min(5000, int((deadline - time.monotonic()) * 1000)))
            outcome = self.kernel.WaitForSingleObject(handle, remaining_ms)
            if outcome == 258:
                return False
            if outcome != 0:
                raise ctypes.WinError(ctypes.get_last_error())
            identity = self.identity(handle)
            if identity != {key: record[key] for key in ("pid", "creation_filetime")}:
                raise ValueError("retained process HANDLE identity changed")
            code = self.w.DWORD()
            self.check(self.kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
            record.update(kernel_joined=True, exit_status=int(code.value))
        return True

    def terminate(self):
        self.check(self.kernel.TerminateJobObject(self.job, 124))

    def close(self):
        for handle in self.handles.values():
            self.kernel.CloseHandle(handle)
        self.handles.clear()
        if self.job:
            self.kernel.CloseHandle(self.job)
            self.job = None


def validate_process_bounds(command, *, timeout_s, max_memory_bytes, join_timeout_s=5.,
                            max_rss_bytes=None, require_complete_tree=True):
    """Reject unsupported ownership and invalid resource bounds before launch."""
    if (not isinstance(command, (list, tuple)) or not command or
            any(not isinstance(arg, (str, os.PathLike)) for arg in command) or
            not isinstance(require_complete_tree, bool) or
            isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or
            not math.isfinite(timeout_s) or not 0 < timeout_s <= 600 or
            isinstance(join_timeout_s, bool) or not isinstance(join_timeout_s, (int, float)) or
            not math.isfinite(join_timeout_s) or not 0 < join_timeout_s <= 30 or
            isinstance(max_memory_bytes, bool) or not isinstance(max_memory_bytes, int) or
            not 0 < max_memory_bytes <= 8 * 1024 ** 3 or
            (max_rss_bytes is not None and (isinstance(max_rss_bytes, bool) or not isinstance(max_rss_bytes, int) or
                                           not 0 < max_rss_bytes <= 8 * 1024 ** 3))):
        raise ValueError("bounded fresh command/time/memory/join parameters required")
    if os.name != "nt" and (require_complete_tree or max_rss_bytes is not None):
        raise RuntimeError("complete ordinary process-tree ownership requires Windows Job Objects")


def run_bounded_process(command, *, log_path, timeout_s, max_memory_bytes,
                        join_timeout_s=5., max_rss_bytes=None, require_complete_tree=True, cwd=None, env=None):
    """Launch fresh work and return actual join/job evidence, retaining its log.

    Windows ordinary CreateProcess descendants inherit an unnamed job assigned
    before the suspended child's user code runs. Broker/WMI-created unrelated
    processes are outside this API. Portable opt-in proves Popen wait only.
    This function never releases an artifact lease; callers require lifecycle
    completion before doing so, independently of the primary program's status.
    KeyboardInterrupt/SystemExit are re-raised with an owned_process_receipt
    attribute after bounded cancellation joins; incomplete joins remain explicit.
    """
    validate_process_bounds(command, timeout_s=timeout_s, max_memory_bytes=max_memory_bytes,
                            join_timeout_s=join_timeout_s, max_rss_bytes=max_rss_bytes,
                            require_complete_tree=require_complete_tree)
    log_path = Path(log_path)
    if log_path.exists() or not log_path.parent.is_dir():
        raise ValueError("supervisor log must be a fresh file in an existing caller-owned directory")
    receipt = {"protocol": "fresh_owned_process_supervision_v1", "command": [str(arg) for arg in command],
               "status": "failed", "primary_joined": False, "lifecycle_complete": False,
               "job_assigned_before_resume": False, "job_active_zero": False,
               "ordinary_createprocess_tree_owned": False, "broker_process_ownership": "unsupported",
               "observed_processes": [], "returncode": None, "limit_reason": None,
               "timeout_s": timeout_s, "join_timeout_s": join_timeout_s,
               "max_job_committed_bytes": max_memory_bytes, "job_memory_limit_enforced": os.name == "nt",
               "memory_scope": "job-accounted committed memory; not an RSS quota", "log_path": str(log_path),
               "ownership_scope": "Windows no-breakaway job" if os.name == "nt" else "portable primary Popen only",
               "max_rss_bytes": max_rss_bytes, "peak_observed_tree_rss_bytes": None,
               "rss_sample_count": 0, "rss_unavailable_samples": [],
               "rss_sampling_scope": "20ms polling; sum current WorkingSetSize from exact retained in-job HANDLEs; unsampled/inter-poll peaks unavailable",
               "total_declared_supervision_budget_s": timeout_s + join_timeout_s}
    started = time.monotonic()
    job, child = None, None
    cancellation = None
    try:
        if os.name == "nt":
            job = _WindowsJob(max_memory_bytes)
        with log_path.open("xb") as stream:
            flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0) | 4) if job else 0
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, cwd=cwd, env=env, creationflags=flags)
            receipt["pid"] = child.pid
            if job:
                receipt["primary_identity"] = job.attach_and_resume(child)
                receipt.update(job_assigned_before_resume=True, ordinary_createprocess_tree_owned=True)
            while True:
                if job:
                    job.observe()
                    receipt["job_accounting"] = job.accounting()
                    sample = job.rss_sample()
                    receipt["rss_sample_count"] += 1
                    if sample["status"] == "available":
                        receipt["peak_observed_tree_rss_bytes"] = max(receipt["peak_observed_tree_rss_bytes"] or 0, sample["tree_rss_bytes"])
                    elif len(receipt["rss_unavailable_samples"]) < 128:
                        receipt["rss_unavailable_samples"].append(sample)
                    if max_rss_bytes is not None and (sample["status"] != "available" or sample["tree_rss_bytes"] > max_rss_bytes):
                        receipt["limit_reason"] = "rss_monitor_unavailable" if sample["status"] != "available" else "tree_rss_limit"
                        job.terminate()
                        break
                primary_done = child.poll() is not None
                tree_done = receipt.get("job_accounting", {}).get("active_processes") == 0 if job else primary_done
                if primary_done and tree_done:
                    break
                if time.monotonic() - started >= timeout_s or log_path.stat().st_size > 4194304:
                    receipt["limit_reason"] = "wall_timeout" if time.monotonic() - started >= timeout_s else "log_byte_limit"
                    if job:
                        job.terminate()
                    else:
                        child.kill()
                    break
                time.sleep(.02)
            join_deadline = min(time.monotonic() + join_timeout_s, started + timeout_s + join_timeout_s)
            try:
                receipt["returncode"] = child.wait(timeout=max(0., join_deadline - time.monotonic()))
                receipt["primary_joined"] = True
            except subprocess.TimeoutExpired:
                receipt["limit_reason"] = receipt["limit_reason"] or "primary_join_timeout"
            if job:
                while time.monotonic() < join_deadline:
                    job.observe()
                    receipt["job_accounting"] = job.accounting()
                    if receipt["job_accounting"]["active_processes"] == 0:
                        receipt["job_active_zero"] = True
                        break
                    time.sleep(.02)
                receipt["observed_processes"] = job.records
                receipt["observed_handles_joined"] = job.join_observed(join_deadline)
                receipt["lifecycle_complete"] = (receipt["primary_joined"] and receipt["job_active_zero"] and
                                                 receipt["observed_handles_joined"])
            else:
                receipt["observed_handles_joined"] = receipt["primary_joined"]
                receipt["portable_tree_qualification"] = "unavailable; only primary Popen actually waited"
            receipt["status"] = "succeeded" if (receipt["returncode"] == 0 and not receipt["limit_reason"] and
                                                (receipt["lifecycle_complete"] if job else receipt["primary_joined"])) else "failed"
    except BaseException as exc:
        # Cancellation keeps its original exception while the fresh Job is
        # terminated and actually joined within the existing join allowance.
        # Ordinary observation errors retain their conservative incomplete
        # history; a later stop cannot repair a failed evidence read.
        cancellation = exc if not isinstance(exc, Exception) else None
        receipt["error"] = repr(exc)
        if cancellation is not None:
            receipt.update(status="cancelled", limit_reason="caller_cancelled")
        if child is not None:
            try:
                join_deadline = min(time.monotonic() + join_timeout_s,
                                    started + timeout_s + join_timeout_s)
                if job:
                    # Capture current members before termination; completion
                    # for uncaptured short-lived members remains Job-accounted.
                    if cancellation is not None:
                        job.observe()
                    job.terminate()
                if child.poll() is None:
                    child.kill()
                receipt["returncode"] = child.wait(timeout=max(0., join_deadline - time.monotonic()))
                receipt["primary_joined"] = True
                if cancellation is not None and job:
                    while time.monotonic() < join_deadline:
                        job.observe()
                        receipt["job_accounting"] = job.accounting()
                        if receipt["job_accounting"]["active_processes"] == 0:
                            receipt["job_active_zero"] = True
                            break
                        time.sleep(.02)
                    receipt["observed_processes"] = job.records
                    receipt["observed_handles_joined"] = job.join_observed(join_deadline)
                    receipt["lifecycle_complete"] = (receipt["primary_joined"] and
                        receipt["job_active_zero"] and receipt["observed_handles_joined"])
            except BaseException as join_error:
                receipt["join_error"] = repr(join_error)
        if cancellation is not None:
            raise
    finally:
        if job is not None:
            # Preserve observed identity history even when a read/launch/join
            # failed. An incomplete normal join path stays unqualified.
            receipt["observed_processes"] = job.records
            try:
                receipt["job_accounting"] = job.accounting()
            except Exception as accounting_error:
                receipt["final_job_accounting_error"] = repr(accounting_error)
            # Last-handle kill is restricted to this job. It is a containment
            # backstop, never retroactively claimed as a verified kernel join.
            job.close()
        receipt["elapsed_seconds"] = time.monotonic() - started
        if cancellation is not None:
            # Producer callers may release only from this actual receipt;
            # they must re-raise cancellation and preserve an incomplete lease.
            try:
                cancellation.owned_process_receipt = receipt
            except Exception as attachment_error:
                if hasattr(cancellation, "add_note"):
                    cancellation.add_note("Process cancellation receipt attachment failed: " + repr(attachment_error))
    return receipt
