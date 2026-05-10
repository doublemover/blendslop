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
Mode-specific quality floors can set `allow_missing: true`: absent modes do not fail a smaller matrix, but any matching row must satisfy the threshold. This is how the smoke budget keeps hard floors for optional research modes without forcing every command to run every backend.

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

`quality-smoke` writes under repo-root `temp/runner/quality-smoke/` by default:

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

Selection now uses those bundle metrics directly. Ensemble policies such as
`quality_first`, `editability_first`, `fast_preview`, and `research_explore`
score required-view failures, missing required metrics, Boundary IoU,
signed-distance loss, geometry/recoverability metrics, topology, editability,
export QA, warnings, failures, and wall time. A candidate can no longer win
only because its backend returned `success`; hard silhouette/view failures and
missing required fields carry explicit score penalties.

Evaluation bundles also expose first-class `geometry` and `recoverability`
groups. For analytic synthetic fixtures, `synthetic.ground_truth` can build a
bundle-ready extras payload containing Chamfer L1/L2, F-score at tolerance,
volumetric IoU, and recoverability-gap metrics from the deterministic SDF and
occupancy grids. Candidate scoring and reports can then compare true geometry,
recoverable geometry, and editable output without conflating them into one
opaque score.

Synthetic e2e matrix rows compute a recoverable visual-hull envelope from the
same rendered reference silhouettes used by the reconstruction run. Matrix JSON
therefore includes `synthetic_geometry_recoverable_*` and
`synthetic_recoverability_ambiguity_gap_*` fields when a candidate mesh is
available, making silhouette-only ambiguity measurable instead of subjective.

Autopsy packs turn high ambiguity, calibration, and geometry failures into an
`active_view_plan`. The plan lists concrete next captures such as diagonal or
top-oblique silhouettes, along with expected information gain and capture notes,
so refinement can ask for better input instead of only sweeping backend knobs.
When visual-hull diagnostics flag axis, transform, bounds, or framing suspicion,
the same autopsy pack now carries a `calibration_plan`. That plan points at the
`visual-hull-transform` refinement track and enumerates safe bounds-padding,
global-scale, per-view offset, and axis-role permutation probes with explicit
acceptance deltas.
Boundary-specific failures also carry `boundary_refinement_plan`, which turns
Boundary IoU and signed-distance failures into concrete mask threshold,
morphology, boundary-band, content-adaptive patch, and differentiable loss-weight
sweeps. Boundary IoU remains the metric that catches contour errors normal area
IoU can hide.

Visual-hull rows include `diagnostics.visual_hull.*` metrics when projection
diagnostics are available. These flag suspected axis/transform mismatches,
catastrophic per-view collapse, failed view counts, and top-like failures.
The smoke and nightly budgets gate catastrophic view collapse when the metric is
present. Visual-hull carving also honors boundary refinement by expanding the
silhouette boundary band before voxel rejection, which reduces quantization
loss around thin structures and hard view edges.

Render-IoU validation treats front/side/top as required views by default.
`average_iou` is still reported, but it cannot hide a required-view failure:
the E2E payload includes `min_view_iou`, `required_views_passed`,
`failed_required_view_count`, `missing_required_metric_count`, and a
`silhouette_summary`. The same comparison path emits per-view Boundary IoU,
signed-distance loss, precision, recall, area ratio, centroid delta, and
optional gates from `--boundary-iou-threshold` and
`--signed-distance-loss-threshold`.

Visual-hull postprocess mode `topology_repair` is a pure-Python conservative
repair pass. It drops invalid/degenerate faces, duplicate faces, loose vertices,
and optional non-largest connected components without moving vertices or filling
holes. The result includes before/after topology reports and an operation log;
hole filling remains a higher-risk remesh/Poisson step and is called out in the
autopsy `topology_repair_plan` instead of being silently synthesized.
Postprocess mode `smooth_guarded` is also pure-Python: it runs bounded Laplacian
smoothing, freezes boundary vertices by default, clamps per-iteration
displacement, and rejects the result if topology score would regress.

Primitive and differentiable research backends now record optimization evidence
instead of only final proxy metrics. Primitive fit runs deterministic multistart
attempts and writes selected-attempt, accepted-move, rejected-move, and
per-attempt objective data. CPU differentiable refinement performs a bounded
coordinate-search loop over primitive parameters and reports initial, final,
and zero-baseline losses separately.

Gaussian/ellipsoid proxy runs also distill their fitted splats into an editable
shape-program artifact. Each Gaussian becomes a parameterized ellipsoid node
with center, radii, rotation metadata, density/opacity, and confidence, so the
research proxy can be inspected or rebuilt as Blender primitives instead of
remaining only a dense visual mesh/proxy. The evaluation extras expose
`editable_proxy` with node counts, validation errors, and the embedded
shape-program payload; artifact runs write
`shape-program/gaussian-ellipsoid-editable-proxy.json`.

Ensemble runtime budgets are part of the contract, not just CLI decoration.
`--ensemble-timeout` is passed to each candidate as `CandidateBudget.timeout_s`,
and `--ensemble-total-timeout` bounds the remaining candidates in serial
ensemble runs. CPU differentiable refinement honors the smaller of
`--diff-max-runtime` and the request budget as its optimizer elapsed-time
limit. For bounded e2e acceptance runs that still exercise the research stack:

```bash
blender --background --python blender_blocking/test_e2e_validation.py -- \
  --reconstruction-mode ensemble \
  --validation-mode backend-status \
  --ensemble-candidates visual_hull_voxel,primitive_fit_refine,gaussian_ellipsoid_proxy,differentiable_refine \
  --ensemble-policy quality_first \
  --ensemble-timeout 20 \
  --ensemble-total-timeout 120 \
  --diff-primitive-count 6 \
  --diff-target-points 512 \
  --diff-visual-hull-resolution 24 \
  --diff-optimization-steps 3 \
  --diff-max-objective-evaluations 96 \
  --diff-max-runtime 15 \
  --result-json temp/validation/ensemble-quality-push.json \
  --no-progress
```

## E2E Synthetic Matrix

The e2e validator can run a suite across multiple reconstruction modes:

```bash
blender --background --python blender_blocking/test_e2e_validation.py -- \
  --synthetic-matrix \
  --synthetic-suite smoke \
  --synthetic-modes legacy,loft_profile,silhouette_intersection,visual_hull_voxel \
  --result-json temp/e2e_synthetic_matrix.json \
  --quality-budget-json configs/quality_perf_budget-smoke.json \
  --quality-report-json temp/e2e_budget_report.json \
  --no-progress
```

For research/editability passes, add `shape_program` to `--synthetic-modes` and
use `--validation-mode backend-status`. `research_only` is treated as an
acceptable backend contract status because it emits a structured editable
artifact without claiming render-IoU success.

With `--shape-residual-policy suggest_patches`, profile rows that contain
multiple foreground intervals or interior holes now become first-class editable
residual nodes in the shape program. Multi-interval details are emitted as
attachable residual patch objects; hole evidence is emitted as subtractive patch
objects. The residual patch annotations still preserve source view, confidence,
row position, and the suggested node id so Blender-side review can keep,
resize, convert to booleans, or discard each detail patch without losing the
audit trail.

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
  --result-json temp/shape_program_export_qa.json \
  --no-progress
```

When Blender compilation runs with an artifact root, this produces `export_qa`
evaluation metrics such as `export.qa_score`, `export.status_ok`, and
`export.reimport_ok`.

The matrix renders Blender-backed synthetic fixtures through the public synthetic builder, then calls the existing e2e custom-image path. Pure 2D synthetic definitions are marked as allowed skips unless `--synthetic-strict-skips` is set.
Pure 2D adversarial and capture-noise definitions now run through a backend-status mask path. The runner writes deterministic reference masks under the selected `temp/` output root, builds a `ReconstructionTarget` directly from those masks, and records backend metrics for each selected mode instead of skipping the row solely because no Blender mesh builder exists.

Synthetic matrix rows also carry a `ground_truth` object. For analytic fixtures
with readable mesh artifacts, the row metrics include flattened true-geometry
measurements such as `synthetic_geometry_true_chamfer_l1`; non-analytic rows
still report quality targets and the ground-truth level so failures can be
separated from known silhouette-only ambiguity.

## Artifact Policy

Do not commit routine outputs from repo-root `temp/`, `benchmarks/results/`, or ad hoc runner artifact roots. Keep generated JSON, rendered PNGs, synthetic references, and imported/exported meshes as CI artifacts or local diagnostics.

Commit only small, intentional fixtures and config examples:

- budget schemas and budget JSON under `configs/`
- documentation under `docs/`
- hand-curated baselines only when the team explicitly chooses a new baseline

When replacing a baseline, store the old and new benchmark/e2e JSON in the review artifact bundle so the `comparison_checks` section can be inspected.
