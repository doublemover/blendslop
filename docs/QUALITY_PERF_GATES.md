# Quality And Performance Gates

This repo keeps quality gates at the harness layer. The runner, e2e validator, and benchmark tool emit JSON artifacts, then `scripts/quality_budget.py` compares those artifacts against declarative budget files under `configs/`.

`docs/IMPLEMENTATION_SPEC.md` lines 556-563 define the `quality_budget` config surface: `budget_json`, `compare_baseline`, `fail_on_regression`, and `environment_compatibility`. The files added here make that surface executable without adding backend dependencies.

## Budget Files

- `configs/quality_perf_budget.schema.json` documents the budget format.
- `configs/quality_perf_budget-smoke.json` gates short quality-smoke benchmark rows and synthetic e2e matrix rows.
- `configs/quality_perf_budget-nightly.json` provides broader nightly thresholds and baseline comparison rules.
- `configs/quality-gated-synthetic-smoke.json` is a workflow config example that points at the smoke budget.

Budget selectors match normalized artifact records:

```json
{
  "artifact": "benchmark",
  "case": "quality-smoke",
  "name": "canonicalize_mask",
  "metric": "metrics.per_iter_ms",
  "mode": "max",
  "threshold": 100.0
}
```

For e2e matrix artifacts, use `artifact: "e2e"` plus optional `case`, `mode`, or `shape_id` selectors. Supported comparison modes are `max_percent_increase`, `max_percent_decrease`, `max_absolute_increase`, and `max_absolute_decrease`.

## Runner Phases

From `blender_blocking/`:

```bash
# Pure tests only, no Blender API required
python test_runner.py --phase pure

# Existing quick Blender pass
blender --background --python test_runner.py -- --phase quick

# Full Blender phase
blender --background --python test_runner.py -- --phase blender

# Benchmark case with JSON output and budget comparison
python test_runner.py --phase bench --bench-case quality-smoke --budget-json ../configs/quality_perf_budget-smoke.json

# Synthetic smoke matrix plus budgeted benchmark smoke
blender --background --python test_runner.py -- --phase quality-smoke --budget-json ../configs/quality_perf_budget-smoke.json
```

`quality-smoke` writes under `blender_blocking/test_output/runner/quality-smoke/` by default:

- `e2e-matrix.json`
- `e2e-budget-report.json`
- `benchmarks.json`
- `bench-budget-report.json`
- synthetic references and per-case e2e results under `synthetic/`

## E2E Synthetic Matrix

The e2e validator can run a suite across multiple reconstruction modes:

```bash
blender --background --python blender_blocking/test_e2e_validation.py -- \
  --synthetic-matrix \
  --synthetic-suite smoke \
  --synthetic-modes legacy,loft_profile,silhouette_intersection,visual_hull_voxel \
  --result-json blender_blocking/test_output/e2e_synthetic_matrix.json \
  --quality-budget-json configs/quality_perf_budget-smoke.json \
  --quality-report-json blender_blocking/test_output/e2e_budget_report.json \
  --no-progress
```

The matrix renders Blender-backed synthetic fixtures through the public synthetic builder, then calls the existing e2e custom-image path. Pure 2D synthetic definitions are marked as allowed skips unless `--synthetic-strict-skips` is set.

## Artifact Policy

Do not commit routine outputs from `blender_blocking/test_output/`, `benchmarks/results/`, or ad hoc runner artifact roots. Keep generated JSON, rendered PNGs, synthetic references, and imported/exported meshes as CI artifacts or local diagnostics.

Commit only small, intentional fixtures and config examples:

- budget schemas and budget JSON under `configs/`
- documentation under `docs/`
- hand-curated baselines only when the team explicitly chooses a new baseline

When replacing a baseline, store the old and new benchmark/e2e JSON in the review artifact bundle so the `comparison_checks` section can be inspected.
