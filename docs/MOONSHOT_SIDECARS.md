# Moonshot Sidecars

Moonshot sidecars are deterministic, Blender-free evidence runners. They do not
change reconstruction pass/fail contracts. When enabled, refinement rows keep
their normal backend status, render metrics, promotion decision, and score; the
sidecars only add artifacts, metrics, and diagnostics under `moonshots/`.

## Enabling

Refinement runs keep sidecars off by default:

```bash
python -m blender_blocking.refinement_lab.cli loop --moonshot-sidecars
```

Limit execution to specific runners with:

```bash
--moonshot-experiments shape_grammar_search,active_view_planning
```

The quality smoke orchestrator also has a cheap profile:

```bash
python scripts/run_quality_refinement_smoke.py --profile moonshot-smoke --dry-run
```

## Artifact Shape

Each runner writes a `moonshot_run_bundle_v1` JSON file containing:

- `status`: `ran`, `skipped`, `unsupported`, or `error`;
- `metrics`: numeric sidecar measurements;
- `evidence`: runner-specific payloads for next-step decisions;
- `artifacts`: JSON artifacts written by the sidecar;
- `warnings`, `errors`, and `next_steps`.

Refinement writes a consolidated `moonshots/summary.json` beside each candidate.
Leaderboards expose `moonshot_summary`, and quality smoke summaries aggregate
status counts, top deltas, active-view suggestions, and portfolio actions.

## Runners

`shape_grammar_search` builds a seed editable shape program from target signals
or existing shape-program evidence, runs bounded deterministic grammar search,
and scores candidates by silhouette proxy, editability, topology, grammar gain,
and complexity. Evidence includes family hints, beam layers, validation
summaries, a selected program fingerprint, and a Blender compile plan with
guards.

`active_view_planning` ranks next camera requests from per-view weakness,
candidate disagreement, uncertainty signals, and capture cost. It emits expected
metric deltas without marking the candidate successful. Evidence includes a
sequence plan, view-pair pressure, ambiguity index, and a stop rule so repeated
captures can rerank after every new view.

`implicit_sdf_proxy` creates a sparse SDF and mesh-proxy manifest from
visual-hull/candidate metrics and target signals. It estimates SDF smoothing,
volumetric IoU, chamfer, and normal-consistency deltas. It writes separate
`sdf-proxy.json`, `mesh-proxy.json`, `sdf-levels.json`, and
`extraction-plan.json` artifacts when an artifact root is available.

`editable_retopology` reads mesh/topology metrics, classifies editability
defects, and writes a repair plan with silhouette-regression guards. Evidence
now includes a phased execution plan, acceptance gates, and a human-review flag
for risky topology edits.

`human_constraint_learning` mines failed rows, target signals, and optional
human labels into deterministic learned-constraint records for future
refinement plans. It also emits training examples and ranking priors that the
portfolio optimizer can apply without hiding the original constraint source.

`differentiable_primitives` emits a CPU finite-difference probe manifest when
primitive evidence exists. Without primitive evidence it reports `unsupported`
with a dependency/status explanation. The manifest includes parameter groups,
objective terms, finite-difference stages, probe counts, and trust-region
guards.

`moonshot_portfolio_optimizer` runs after the other sidecars in the smoke
profile. It consumes their evidence, ranks cross-sidecar actions under a small
budget, and emits selected actions plus explicit guards. Typical actions include
capturing a next-best view, compiling a selected shape program, extracting a
guarded SDF proxy, applying a retopology repair plan, injecting learned
constraints, or running a finite-difference primitive probe. These actions stay
advisory until the normal render/topology/editability gates pass. Portfolio
evidence includes ranking-prior bonuses, dependency edges, a staged execution
plan, and a risk register so the summary can distinguish "do this next" from
"this is blocked by evidence, review, or render QA."
