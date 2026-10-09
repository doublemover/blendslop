# Persistent worker process ownership

Persistent workers now have an explicit optional process budget. A worker is created suspended, assigned to its own fresh no-breakaway Windows Job, and only then resumed. The same worker handles successive candidate and nested fitting jobs. A crashed primary no longer leaves its ordinary descendants outside the stop path.

```python
from blender_blocking.reconstruction.process_executor import (
    PersistentProcessExecutor, WorkerProcessBudget,
)

budget = WorkerProcessBudget(
    wall_s=85., max_memory_bytes=8 * 1024 ** 3,
    join_timeout_s=5., max_restarts=8,
)
with PersistentProcessExecutor(2, root=scratch_parent,
                               process_budget=budget) as pool:
    # Supply this shared pool through the existing process_executor context.
    # pool.root is a fresh generated directory inside scratch_parent.
    outcomes = pool.map(jobs, timeout_s=60.)
receipt = pool.ownership_receipt()
```

The wall allowance belongs to the whole pool and never resets on replacement. Committed-memory and optional sampled RSS allowances are divided across concurrent workers. The automatic monitor enforces wall/log/memory-monitor limits even while the caller is idle. Each worker log is fresh and capped by the existing 4 MiB monitoring limit; generation-specific lifecycle receipts are retained. Restarts are limited explicitly.

Stop waits on the primary and the exact retained member HANDLEs, and requires Job active-process count zero. An incomplete join keeps its owner and handles for a bounded `close()` retry. Caller and launch exceptions remain primary when receipt publication or close also fails. `ownership_receipt()` reports process completion separately from job success. The old unbudgeted API remains available and makes no new process-tree ownership claim.

This is an opt-in Windows process ownership seam, not artifact reclamation. It does not adopt existing transport, PIDs or leases, delete retained output, claim broker/WMI ownership, or release artifact leases. The supplied root is a scratch parent in this mode; existing files remain external. Reclamation and adoption by other production entry points remain separate work.

Validation used existing Blender-bundled Python 3.13.13 and small stdlib processes, with no new Blender reconstruction/render campaign. Fifteen distinct focused methods passed across the changed ownership path, invalid-bound/synchronous compatibility checks, and existing reuse, partial-checkpoint and flat nested-job scheduling contracts. The actual crash fixture created a live descendant, exited its primary, and retained the descendant's exact joined HANDLE and termination status. The incomplete-join fixture retained the same Job/handles and completed a later retry. Existing full Blender and surface qualification results were not rerun or treated as missing.
