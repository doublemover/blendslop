# Reconstruction Refinement Lab

The refinement lab is the repeatable experiment harness for improving reconstruction quality. It builds a deterministic run plan, executes mode/parameter variants, writes structured result indexes, creates failure autopsies, generates bounds diagnostics, and renders a static HTML report. Generated outputs stay under ignored roots such as `temp/refinement-runs/` and must not be committed.

## What It Produces

Each run root contains:

- `plan.json` and `manifest.json` for reproducibility.
- `index.jsonl`, `leaderboard.json`, and `leaderboard.md` for ranking.
- `report.html` with per-variant metrics, warnings, artifacts, overlays, and failure summaries.
- `cases/<case>/variants/<variant>/command.txt`, `config.json`, `result.json`, and optional `autopsy.json` / `bounds-debug.json`.
- `human-labels.jsonl` when reviewers label candidates.

## Built-In Suites

List suites:

```bash
python -m blender_blocking.refinement_lab.cli list-suites
```

Current suite categories include:

- `default-vase`: fast real-image regression using the committed sample references.
- `synthetic-smoke`: small deterministic synthetic sanity cases.
- `synthetic-profile-band`: profile/lathe cases for loft tuning.
- `synthetic-visual-hull`: visual-hull volume and meshing cases.
- `synthetic-primitive-fit`: primitive, Gaussian, and differentiable fitting cases.
- `synthetic-adversarial`: silhouette edge cases and capture degradations.
- `synthetic-nightly`: broad heavy regression suite.

## Built-In Tracks

List tracks:

```bash
python -m blender_blocking.refinement_lab.cli list-tracks
```

Useful tracks:

- `mask-refinement`: silhouette thresholding, morphology, component filtering, and canonicalization.
- `profile-loft-refinement`: profile sampling, smoothing, radial segments, caps, and mesh editability.
- `visual-hull-transform`: bounds/projection/axis/framing diagnosis for `visual_hull_voxel`.
- `visual-hull-quality`: sparse/chunked volume, mesh method, uncertainty aggregation, and postprocess tuning.
- `primitive-fit`: primitive families, counts, objective weights, optimizer steps, and regression handling.
- `gaussian-proxy`: Gaussian/ellipsoid counts, initialization, radii, opacity, and proxy export.
- `differentiable-refine`: CPU/GPU differentiable-style loss sweeps and optional dependency policy.
- `ensemble-selection`: candidate sets and selection policies.

`visual-hull-transform` keeps its render-IoU sweep to renderable mesh outputs.
Use `--validation-mode backend-status --vh-mesh-method points` directly when
you need the point-cloud diagnostic path.

## Plan Without Running

```bash
python -m blender_blocking.refinement_lab.cli plan `
  --suite default-vase `
  --track visual-hull-transform `
  --max-runs 4 `
  --out temp\refinement-runs\plan-smoke.json
```

Use this when changing presets or search logic. The command writes only the plan file.

## Run From Blender

Inside Blender's Python, call the e2e CLI through Blender:

```powershell
$blender = "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"

& $blender --background --python blender_blocking\test_e2e_validation.py -- `
  --refinement-suite default-vase `
  --refinement-track profile-loft-refinement `
  --refinement-max-runs 1 `
  --refinement-result-root temp\refinement-runs\smoke-profile `
  --refinement-html-report `
  --refinement-autopsy `
  --refinement-bounds-debug `
  --no-progress
```

Run the visual-hull transform diagnostic:

```powershell
& $blender --background --python blender_blocking\test_e2e_validation.py -- `
  --refinement-suite default-vase `
  --refinement-track visual-hull-transform `
  --refinement-search coordinate `
  --refinement-objective visual_hull_alignment `
  --refinement-max-runs 48 `
  --refinement-result-root temp\refinement-runs\default-vase-vh-transform `
  --refinement-html-report `
  --refinement-bounds-debug `
  --refinement-autopsy `
  --no-progress
```

## Run From System Python With Blender Subprocesses

Use this form when you want the planner/reporting process outside Blender but each variant executed by Blender:

```powershell
python blender_blocking\test_e2e_validation.py `
  --refinement-subprocess `
  --refinement-blender-exe "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" `
  --refinement-suite default-vase `
  --refinement-track visual-hull-transform `
  --refinement-max-runs 4 `
  --refinement-result-root temp\refinement-runs\vh-subprocess-smoke `
  --no-progress
```

The pure CLI also supports subprocess execution:

```powershell
python -m blender_blocking.refinement_lab.cli run `
  --suite default-vase `
  --track profile-loft-refinement `
  --max-runs 1 `
  --result-root temp\refinement-runs\cli-smoke `
  --blender-exe "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
```

## Rank, Report, Autopsy, Label, Promote

Regenerate rankings:

```bash
python -m blender_blocking.refinement_lab.cli rank --run-root temp/refinement-runs/default-vase-vh-transform
```

Regenerate the HTML report:

```bash
python -m blender_blocking.refinement_lab.cli report --run-root temp/refinement-runs/default-vase-vh-transform
```

Write or refresh autopsies:

```bash
python -m blender_blocking.refinement_lab.cli autopsy --run-root temp/refinement-runs/default-vase-vh-transform
```

Turn a result/autopsy payload into the next adaptive batch:

```bash
python -m blender_blocking.refinement_lab.cli adapt `
  --result-json temp\refinement-runs\default-vase-vh-transform\cases\case\variants\variant\result.json `
  --out temp\refinement-runs\adaptive-proposals.json `
  --variants-out temp\refinement-runs\adaptive-variants.json `
  --max-proposals 8
```

The adaptive planner consumes metric bundles plus autopsy plans such as
`boundary_refinement_plan`, `calibration_plan`, `topology_repair_plan`, and
`active_view_plan`, then emits runnable variants for the next plan.

Run a closed adaptive loop when you want each generation to execute, rank the
best parents, and feed their proposals into the next generation automatically:

```bash
python -m blender_blocking.refinement_lab.cli loop `
  --suite synthetic-smoke `
  --track refinement-maximal `
  --generations 3 `
  --parent-top-k 3 `
  --children-per-parent 4 `
  --result-root temp\refinement-runs\closed-loop `
  --blender-exe "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
```

The loop writes one `generation-XX/` run root per iteration plus
`adaptive-loop-summary.json` at the loop root. Each generation records the
selected parent variants, emitted proposals, and child variants. Child variants
carry `parent_variant_id`, `adaptive-loop` tags, and generation metadata so
reports and lineage tooling can reconstruct the search path.

Append a human label:

```bash
python -m blender_blocking.refinement_lab.cli label `
  --run-root temp\refinement-runs\default-vase-vh-transform `
  --variant visual_hull_voxel-vh_resolution-00-abcd1234 `
  --label accept `
  --score 5 `
  --tag sculptable `
  --notes "Best silhouette agreement and cleanest bounds."
```

Promote a result into a small preset JSON:

```bash
python -m blender_blocking.refinement_lab.cli promote `
  --run-root temp\refinement-runs\default-vase-vh-transform `
  --variant visual_hull_voxel-vh_resolution-00-abcd1234 `
  --out temp\refinement-runs\promoted\vh-best.json
```

Promotion is blocked by default for candidates whose backend status is
`degraded`, `research_only`, `skipped`, `failed`, `error`, `unreported`, or whose
metrics are proxy-only without required per-view render evidence. Use
`--allow-review-required` only when intentionally preserving one of those risky
artifacts; the exported preset records the promotion tier and blockers.

## Quality Rules

- A high average score is not enough if a required view fails; check per-view IoU in `result.json` and the report table.
- Leaderboards include a promotion tier. `promotable` means a passing backend result with required-view evidence; `degraded`, `research_only`, `metric_only`, `unverified`, and `blocked` are intentionally ranked below full validated results.
- Optional dependencies must be explicit: skipped/fail behavior is recorded in result metadata instead of silently degrading.
- Bounds diagnostics are the first stop for visual-hull failures: inspect `bounds-debug.json` and projection overlays before changing reconstruction math.
- Commit only source, tiny deterministic fixtures/specs, and docs. Do not commit generated `temp/`, `test_output/`, meshes, renders, volumes, reports, or suite result JSON.

## Pure Validation

The pure refinement tests run as part of the pure phase:

```bash
python blender_blocking/test_runner.py --phase pure --no-progress
```

They cover contracts, presets, plan construction, search/ranking, result indexes, autopsies, bounds diagnostics, reports, labels, and CLI surfaces. Blender smoke runs should be executed after source changes when Blender is available.
