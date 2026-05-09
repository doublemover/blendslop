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

Evaluation bundles can be gated directly with `artifact: "evaluation"`.
The budget reader expands bundle metric names into dotted paths, so
`export.qa_score` is selected as `metrics.export.qa_score` and
`editability.editable_reconstruction_index` is selected as
`metrics.editability.editable_reconstruction_index`.

The SOTA-oriented evaluation contract is intentionally multi-axis:

- Silhouette gates use `silhouette.min_view_iou`, `silhouette.average_iou`,
  `silhouette.mean_boundary_iou`, and
  `silhouette.mean_signed_distance_loss`. Boundary IoU catches contour and
  thin-feature errors that area IoU can hide, while signed-distance loss keeps
  near-miss boundaries visible.
- Synthetic geometry gates use `geometry.true.*` for exact ground-truth
  distance and `geometry.recoverable.*` for the best silhouette-recoverable
  envelope. `geometry.ambiguity_gap_*` records how much of the error is caused
  by silhouette-only underdetermination rather than backend failure.
- Editable-output gates use `editability.editable_reconstruction_index` plus
  topology/export metrics because a dense visual result is not enough when the
  deliverable is an editable Blender asset.

This lines up with `docs/IMPLEMENTATION_SPEC.md` lines 314-324: optional paths
must be explicit, visual-hull and research paths must emit structured metrics,
and synthetic artifacts must be deterministic enough to support measured
regression gates.

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

The quality-smoke and nightly benchmark registries include dedicated rows for:

- `geometry_metrics`: Chamfer/F-score/volumetric-IoU metric throughput.
- `shape_program_build`: editable shape-program construction from profile bands.

Evaluation bundles now expose a first-class `export_qa` metric group for
editable Blender delivery checks. The group records export target count,
round-trip status, object/vertex/face/material counts, per-target QA scores,
and the aggregate `export.qa_score` used by reports and quality budgets.

Evaluation bundles also expose first-class `geometry` and `recoverability`
groups. For analytic synthetic fixtures, `synthetic.ground_truth` can build a
bundle-ready extras payload containing Chamfer L1/L2, F-score at tolerance,
volumetric IoU, and recoverability-gap metrics from the deterministic SDF and
occupancy grids. Candidate scoring and reports can then compare true geometry,
recoverable geometry, and editable output without conflating them into one
opaque score.

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

For research/editability passes, add `shape_program` to `--synthetic-modes` and
use `--validation-mode backend-status`. `research_only` is treated as an
acceptable backend contract status because it emits a structured editable
artifact without claiming render-IoU success.

For Blender-side editable asset delivery checks, enable shape-program export
round-trip QA:

```bash
blender --background --python blender_blocking/test_e2e_validation.py -- \
  --synthetic-matrix \
  --synthetic-suite smoke \
  --synthetic-modes shape_program \
  --validation-mode backend-status \
  --shape-run-export-qa \
  --shape-export-qa-targets obj,glb \
  --result-json blender_blocking/test_output/shape_program_export_qa.json \
  --no-progress
```

When Blender compilation runs with an artifact root, this produces `export_qa`
evaluation metrics such as `export.qa_score`, `export.status_ok`, and
`export.reimport_ok`.

The matrix renders Blender-backed synthetic fixtures through the public synthetic builder, then calls the existing e2e custom-image path. Pure 2D synthetic definitions are marked as allowed skips unless `--synthetic-strict-skips` is set.

## Artifact Policy

Do not commit routine outputs from `blender_blocking/test_output/`, `benchmarks/results/`, or ad hoc runner artifact roots. Keep generated JSON, rendered PNGs, synthetic references, and imported/exported meshes as CI artifacts or local diagnostics.

Commit only small, intentional fixtures and config examples:

- budget schemas and budget JSON under `configs/`
- documentation under `docs/`
- hand-curated baselines only when the team explicitly chooses a new baseline

When replacing a baseline, store the old and new benchmark/e2e JSON in the review artifact bundle so the `comparison_checks` section can be inspected.
