# Fresh Windows process-tree supervision

`blender_blocking/utils/owned_process_supervisor.py` supplies `run_bounded_process` for fresh bounded launches. It creates an unnamed Job Object, launches the primary process suspended and hidden, assigns that exact process handle to the job, verifies the sole primary thread and process creation identity, then resumes it. A failed assignment or unexpected thread layout blocks user-code execution. No existing PID is adopted, and the utility does not alter artifact leases or delete files.

The public call is:

```python
receipt = run_bounded_process(
    command,
    log_path=owner.root / "child.log",
    timeout_s=120,
    join_timeout_s=5,
    max_memory_bytes=8 * 1024 ** 3,
    max_rss_bytes=8 * 1024 ** 3,
    cwd=workspace,
)
```

`log_path` must be a new file in an existing caller-owned directory. The log is opened exclusively and retained. The caller keeps the returned primary-program status separate from lifecycle completion: an intentional exit 17 remains `status="failed"` even when every required lifecycle check passes. To release an `OwnedRun` lease, a caller must require `lifecycle_complete=True`, then register the log and receipt and close the owned run with its actual success/failure state. An unconditional `OwnedRun` context-manager exit is inappropriate when join proof is unavailable.

The Windows receipt requires the actual primary `Popen.wait`, Job Object active-process count zero, and kernel waits plus exact creation-FILETIME/exit-status records for each observed process handle. Handles captured while members are alive remain valid after exit. A short-lived member that ends before its individual handle is captured can be covered by zero-active job accounting, but receives no invented individual exit status. The receipt retains observed identities and unavailable join state even if a later query or join fails. Closing the job is a containment backstop for its members; it does not manufacture a completed join. The utility never releases a previous run's lease.

The job sets no breakaway flags. Ordinary `CreateProcess` descendants inherit this ownership; broker/WMI launches outside that inheritance are unsupported. Host job restrictions or an unusual initial thread layout can make assignment/resume fail and remain a failed receipt. The implementation uses the actual CPython Windows `Popen` process handle and was checked under the existing bundled Python 3.13.13, without a Blender launch. On other platforms, the default complete-tree request refuses before launch; explicit `require_complete_tree=False` permits a real primary-only `Popen.wait`, while `lifecycle_complete` remains false and tree ownership unavailable.

## Separate resource observations

`max_memory_bytes` is a Windows Job Object accounted committed-memory quota, with `peak_job_committed_bytes` reported separately. This kernel quota can reject new allocations; it is not an RSS observation or a guaranteed whole-job termination notification.

`max_rss_bytes`, when declared, is independently enforced from sampled sums of current `WorkingSetSize` obtained with `K32GetProcessMemoryInfo` on retained in-job handles after exact creation-identity checks. Polling aims for 20 ms between samples; query and launch overhead can increase that interval. An observed RSS breach, an unavailable live-process read under a declared RSS ceiling, the wall bound, or the 4 MiB log bound terminates only this fresh job and follows the bounded join path. Successful receipts record sample count, sampled peak RSS and unavailable readings. The observed peak excludes unobserved short-lived/inter-poll peaks; it is not a continuous maximum. Summing process working sets may count shared resident pages more than once.

The work timeout and final join timeout are separately declared. Final joins and rescue waits share a deadline at most `timeout_s + join_timeout_s` after supervision starts; synchronous operating-system call overhead is not an independently interruptible operation. The primary command, 128-process/retained-handle bounds and fresh log remain explicit. No security descriptor, ACL or shared system setting is changed.

## Retained checks

Seven focused regressions cover real primary waits, a crashed parent with a live descendant, bounded job termination, disjoint concurrent jobs, declared RSS breach, unavailable RSS reads, invalid/existing-log refusal, and query-exception history that stays lifecycle-unqualified. The final test count/resource details and byte hashes are in [evidence.json](evidence.json). Root owns test-runner registration; no broad suite or Blender test was run for this chunk.

The retained tiny crash fixture is `temp/tasks/quality-continuation-20261009/process-supervisor-pure-01/owned-n9h8lfoy/supervision.json`. It intentionally exits 17, retains four actual job members with kernel-waited exit statuses 17/0/0/0, and proves active-process count zero. It completed in 3.880121 s with sampled peak RSS 28,622,848 B and separate peak job committed memory 17,092,608 B. Its fresh artifact owner is correctly failed/released, and its read-only ownership audit is ready. The receipt's original source hashes are retained separately from the subsequent exception-history-only correction and final focused checks.

The earlier selected-canonical supervisor remains failed. Its exact-PID follow-up found absence and supplied no kernel wait or exit status. Neither that receipt nor any historical lease is rewritten or qualified by this fresh capability. No native Blender run has been launched using the new supervisor; future producer adoption remains a separately reviewed launch.

## API references

The implementation follows Microsoft documentation for [Job Object inheritance and limits](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects), [job assignment](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject), [suspended process creation](https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags), [thread resume](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-resumethread), and [process memory counters](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters).
