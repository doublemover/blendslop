Ensemble and measured routing callers can now opt into the existing persistent complete-tree pool budget directly:

```python
from blender_blocking.reconstruction.process_executor import WorkerProcessBudget
run = runner.run_requests(requests, max_parallel_candidates=2,
    process_budget=WorkerProcessBudget(wall_s=85., max_memory_bytes=8*1024**3))
```

Existing context callers may instead supply `context.worker_process_budget`. One allowance covers the entire pool and all candidate contexts; conflicting declarations are rejected before pool allocation. An existing shared pool must already own the requested budget. Reuse never re-enters/closes that pool or resets its deadline, resource caps or restart accounting. With no opt-in, the old constructor/default path remains unchanged.

Candidate transport serializes only the actual owning pool declaration. The worker binds it once to the same WorkerClient and request context; nested fit/program/native-union/DVX users keep the existing coordinated map protocol. Recursive candidate ensembles require submit/result and are refused clearly; no child-pool workaround is introduced.

Nine new focused caller/transport guards and three existing real stdlib worker reuse/nested scheduling/immutability methods passed. No Blender/native job or repeated process campaign ran. [Evidence](evidence.json) records exact affected methods and source bytes. Other standalone backend fallback pools remain the next concrete adoption seam; artifact reclamation and process budget ownership stay independent.
