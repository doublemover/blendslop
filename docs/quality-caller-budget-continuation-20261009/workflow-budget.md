The normal workflow can now opt into worker ownership through explicit configuration:

```python
cfg.ensemble.worker_process_budget = {
    "wall_s": 85., "max_memory_bytes": 8*1024**3,
    "max_rss_bytes": 8*1024**3, "join_timeout_s": 5., "max_restarts": 8,
}
```

None preserves historical defaults. Required wall/memory limits must be declared; unknown/incomplete/invalid controls are refused without worker allocation. The resource fragment serializes as ordinary JSON with no inferred caps. Main reconstruction forwards it into the real request context, supporting both ensemble and direct backend callers with context-aware scopes. Each created pool retains one lifetime allowance; this does not imply a whole-workflow cap across independent pools.

All coordinator dispatch envelopes now carry the actual owning pool declaration, closing the noncandidate fitting/native-union client gap. Binding occurs before any job kind executes and cannot replace the client's owner budget. Legacy envelopes without the new field remain compatible. No restart, deadline, cap or ownership reset is introduced.

Two envelope guards and four configuration/main-workflow forwarding guards passed; one affected real stdlib worker reuse method passed after the envelope change. [Evidence](workflow-budget-evidence.json) records exact source scope. No native job or broad test campaign ran. Reclamation, broker ownership and unsupported recursive candidate ensembles remain separate.
