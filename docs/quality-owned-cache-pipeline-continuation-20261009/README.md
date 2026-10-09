# Owned chunk cache option through the visual-hull pipeline

The existing visual-hull option seam now exposes `cache_owned_writes: bool = False`. `BlockingConfig.visual_hull.to_dict()` carries the strict boolean through `visual_hull_run_config`, `VisualHullBackend`, and the existing chunked/sparse-hash/OpenVDB-labeled point-cloud builders into the single `VolumeChunkCache` constructor. Invalid integers, strings and `None` fail validation instead of becoming true by coercion. Existing positional dataclass parameters are preserved; public builder parameters remain keyword-only.

Example:

```python
config.visual_hull.backend = "chunked"
config.visual_hull.cache_directory = "temp/volume-chunk-cache"
config.visual_hull.cache_owned_writes = True
```

This flag does not enable caching by itself. Existing cache-directory/enable-cache routing, namespace/key contents, read/write flags, volume values, reconstruction outputs and legacy overwrite behavior remain unchanged. Dense and adaptive builders still bypass this chunk-cache producer. With the default false value, existing cache diagnostics retain their previous shape.

For an enabled sparse/chunked cache with the opt-in true, the existing `chunk_cache` metrics and volume metadata additionally report `owned_writes`, `ownership_status`, `last_ownership_receipt`, and the scope of that receipt. `retained` means a receipt for the last fresh store attempt during this grid construction exists; it does not mean that write succeeded. A failed deterministic-key collision retains the earlier NPZ, the new failed stage/receipt, and existing `write_errors` semantics. `unrun` and a null path describe a construction with no fresh store attempt, including a cache-hit-only build. The constructor-local receipt is forwarded directly; no previous ownership directory or receipt is discovered/adopted. Earlier receipts remain on disk but are not presented as belonging to the new build.

Six pure checks passed in 1.426 seconds with Blender's bundled Python 3.13.13, without a Blender workload or new child-launch campaign. The two focused pipeline checks cover strict/default config semantics and actual one-chunk backend serialization: unchanged source mask and dense grid values, NPZ boolean/metadata compatibility, fresh receipt retention, a hit-only null/unrun receipt, and a forced read-disabled key collision preserving earlier archive bytes and success receipt while retaining a distinct failed receipt. Three existing configuration default checks and one existing backend parser/helper contract also passed. Python AST parsing and `git -c core.whitespace=cr-at-eol diff --check` passed; existing runtime CRLF bytes are preserved. The previously validated owned cache producer files were not changed by this adoption.

`test_owned_cache_pipeline.py` is registered in `test_runner.py`; this batch is committed locally on the separate follow-up branch. The exact file identities and checks are recorded in [evidence.json](evidence.json). No native requalification, cache deletion, historical adoption, default-policy change or publication is part of this change.
