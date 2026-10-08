# Development

### Tests and benchmarks

The project uses its custom runner rather than a blanket `pytest` invocation. From the repository root:

```bash
# Pure-Python phase; requires the core dependencies in this Python environment.
python blender_blocking/test_runner.py --phase pure

# Blender examples below assume its executable is on PATH.
blender --background --python-exit-code 1 --python blender_blocking/test_runner.py -- --quick
blender --background --python-exit-code 1 --python blender_blocking/test_runner.py

# Inspect microbenchmark options before choosing a workload.
python blender_blocking/benchmarks/benchmark_perf.py --help
```

The quick phase skips slow end-to-end work; it is not the full suite. Read [quality/performance gates](QUALITY_PERF_GATES.md) for synthetic matrices, budgets, and metric-specific acceptance. Microbenchmark timings are not end-to-end reconstruction speedups.

The [existing CI workflow](../.github/workflows/blender-tests.yml) runs on filtered pushes to `main`/`develop` and PRs targeting `main`; it does not automatically validate every feature-branch or documentation push. Check the workflow and results for the exact commit before calling a branch validated. Follow [AGENTS.md](../AGENTS.md) for repository-specific development and execution constraints.

### Where to work

| Path | Responsibility |
|---|---|
| [`main_integration.py`](../blender_blocking/main_integration.py) | Image-to-blockout orchestration and API |
| [`config_models/`](../blender_blocking/config_models/) | Typed configuration groups and validation |
| [`reconstruction/`](../blender_blocking/reconstruction/) | Targets, backend registry, candidate selection, geometry/evidence contracts |
| [`integration/`](../blender_blocking/integration/) | Image processing and native Blender scene/mesh/render operations |
| [`geometry/`](../blender_blocking/geometry/), [`primitives/`](../blender_blocking/primitives/), [`placement/`](../blender_blocking/placement/) | Geometry building blocks and fitting |
| [`e2e/`](../blender_blocking/e2e/), [`evaluation/`](../blender_blocking/evaluation/), [`validation/`](../blender_blocking/validation/) | CLI, rendered/geometry evidence, and acceptance checks |
| [`refinement_lab/`](../blender_blocking/refinement_lab/), [`synthetic/`](../blender_blocking/synthetic/) | Experiment plans, reports, and deterministic fixtures |
| [`scripts/`](../scripts/), [`configs/`](../configs/) | Focused utilities, presets, and budgets |

To add a reconstruction method, implement the [backend protocol](../blender_blocking/reconstruction/backend.py), return a structured `CandidateResult`, and register it in the [registry](../blender_blocking/reconstruction/registry.py). Registry registration alone does not expose a top-level workflow/CLI mode: configuration, dispatch, and validation also need wiring. Keep Blender imports and optional numerical dependencies at the appropriate boundary, declare skips/failures, and add focused contract tests before native integration checks.

Commit source, documentation, and small deterministic fixtures. Keep generated meshes, renders, volumes, and run reports under ignored output roots such as `temp/`. Preserve failed cases and provenance when sharing comparison evidence.

## Public evaluation gallery

[Build, evidence sanitization and Pages deployment](EVALUATION_SITE.md) use a separate static artifact. Only the curated synthetic inspection PNGs under `site/` are committed for public display; large or unrelated generated evidence remains under ignored run roots. Native reconstruction work and website publishing have separate validation scopes.
