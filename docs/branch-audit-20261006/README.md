# Blendslop branch audit and bounded refinement pass — 2026-10-06

Current quality follow-up: [surface, coverage and rounded-triangle backlog](quality-backlog-20261007.md). These remain independently qualified work.

Latest saved-solid implementation and validation: [solid follow-through](solid-followthrough.md). Previous corrective implementation: [corrective batch](corrective-batch.md). Measurements of the earlier frozen source: [frozen final campaign results](quality-final-results.md). Frozen campaign preparation snapshot: [quality connections and handoff](quality-connections.md). Previous implementation checkpoint: [quality implementation handoff](quality-implementation.md). Earlier checkpoint: [scoped implementation follow-up](implementation-remainder.md). Measurements: [frozen native performance and metric phase](native-phase.md). This document retains its earlier snapshot.


The figures and open items below describe the earlier audit/compatibility snapshot. The linked native phase records its frozen implementation, held-out measurements, matched timings and validation limits.

Baseline: `endblay_opslay`, `fbde785b4f112e3f49151e924c17279ef5319c73`; initial
checkout was clean. Source repair commit: `44f493b8f3dd4fbd60c91500f041ed03b097c5e8`. This is a scoped repair/refinement pass, with original
reports retained. No installs, resets, deletions, pushes or publication occurred.

## What the task records establish

The saved documentation contains **2,686 checkbox/TODO records**, **1,363 recorded
open records**, and **1,092 specification headings**. Of the open records,
1,344 come from the old granular refinement checklist. Its unchecked boxes are
stale: much of that code exists and is exercised by the pure runner. Checked
boxes elsewhere are historical claims, not fresh acceptance evidence.

- [task-records.csv](task-records.csv) preserves every task's source, line,
  heading, recorded state and referenced-file existence.
- [recorded-open.csv](recorded-open.csv) is the complete retained unchecked/TODO
  list. Nothing was silently checked off or discarded.
- [spec-sections.csv](spec-sections.csv) covers the master spec, ambitious
  15-part spec, evaluation 29-part spec, refinement and modularization packets.
- [readiness.csv](readiness.csv) provides **49 current readiness rows**, with
  verified code locations and explicit remaining work. It separates implemented
  foundations, incomplete validation, studies without data, and blockers.

This is exhaustive inventory of the retained task records, plus subsystem
reconciliation. It is not a claim that every one of the old 1,344 individual
acceptance subtasks was manually proved complete. Unverified claims stay open.
The scope anchor is `docs/IMPLEMENTATION_SPEC.md`, lines 3–18; its canonical
defaults and legacy/profile contracts are preserved.

## Repairs completed

1. Gaussian/ellipsoid `area_iou` now measures polygon-projected exported geometry
   against target masks. Coverage/confidence retain their own fields; confidence
   is no longer presented as IoU. These projections have their own namespace;
   they do not replace external Blender renders.
2. E2E retains the exact evaluated world-space Z-up mesh used for rendering.
   It has priority over nested proxy/branch mesh paths. Hybrid used to expose
   hull geometry while rendering a profile mesh; that discrepancy is now explicit
   and exact rendered geometry is used for 3D evidence.
3. Synthetic geometry compares evaluated reference and candidate meshes in the
   same declared uniform bbox frame. Area-weighted deterministic surface samples
   replace tessellation-biased raw vertices. Aspect and rotation errors remain.
4. Geometry bundles are populated after measurement; selected silhouette bundles,
   failures and autopsies are refreshed from external-render evidence. Other
   ensemble candidates keep their original evidence, because they were not rendered.
5. `_normalized` Chamfer aliases are emitted only for the explicit normalization
   source. Old unaligned measurements are not relabeled normalized.
6. Compiled shape-program Empty roots now render/export all mesh descendants,
   preserving the editable scene. Compilation no longer produces a missing-mesh
   verdict merely because the root is an Empty.
7. Farthest-point sampling keeps a running nearest-distance array: O(N*K) work and
   O(N) auxiliary storage rather than repeated O(N*K²) work. The saved 2,048-point,
   256-selection seed 1234 test produced identical byte hashes, with indicative
   sampler times 2.2166 s→0.01249 s (~177×). This is not an end-to-end speed claim.
8. E2E resolves running Blender render-engine capabilities, guards removed EEVEE
   sample properties, and records applied settings.

## Equal comparison contract and the single trial

The experiment uses box, vase and torus at seed 1234, with exactly shared 512 px
front/side/top PNGs and evaluated reference meshes. The synthetic box includes a
small rotation; this is disclosed in its saved spec, not fitted away. Eight modes
run serially in fresh Blender processes. Child timeout 100 s, four configured
threads, primitive fitting 8 s and differentiable refinement 20 s. Inputs, configs,
source and mesh hashes, raw per-view precision/recall/IoU, timing scopes and
failure rows are retained.

One joint trial changes hull resolution 64→96, profile samples 100→160, radial
segments 24→48, Gaussian primitives 24→32 and initialization farthest-point→kmeans,
alongside the code repairs. Other defaults stay fixed. This is a single
multi-factor experiment, not a factorial ablation or exhaustive search. Trial
settings remain in the harness; no production tuning defaults were promoted.

Silhouette gates: per-view area IoU≥.7, boundary IoU≥.1 and signed-distance
loss≤.1. A gate pass is not full geometry/topology/editability acceptance.
Geometry: independent bbox centering and uniform longest-extent scaling,
8,192 area-weighted samples, seed 1234, Chamfer as SUM of directional means,
F-score tolerance .02. Coarse 24³ volume parity is reported only for one closed
component on each side. Multiple overlapping parts are not treated as a valid
union by parity. Self-intersections are not certified. Sampling resolution and
normalization materially affect these measurements.

The six baseline profile/hybrid instrumentation reruns must match the original
mean rendered IoU exactly before their exported meshes can backfill the missing
3D evidence. They do not search parameters. Original run outputs are preserved;
the finalizer writes separate refreshed evidence. Single-run wall times include
workflow/render/validation, exclude startup and separate geometry evaluation,
and are noisy on a shared host. The pure suite overlapped early trial cases.

Results and reproduction:

- [comparison.md](../../temp/bounded-pass-20261006/comparison.md): compact technique table.
- [comparison.csv](../../temp/bounded-pass-20261006/comparison.csv): all 48 rows,
  including per-shape 3D precision, topology, timing and unavailable metrics.
- [paired-deltas.csv](../../temp/bounded-pass-20261006/paired-deltas.csv): all 24
  paired decisions, exposing weaker-view regressions.
- [measured.json](../../temp/bounded-pass-20261006/measured.json): provenance and
  fresh evidence paths; original baseline.json/candidate.json remain unchanged.
- [scripts/run_comparable_pass.py](../../scripts/run_comparable_pass.py): `--blender`,
  `--output`, `--arm baseline|candidate`, `--cases`, `--modes`, `--seed`, `--timeout`.
- [scripts/finalize_comparable_pass.py](../../scripts/finalize_comparable_pass.py)
  backfills exact-mesh evidence; [summarize_comparable_pass.py](../../scripts/summarize_comparable_pass.py)
  creates small tables. Run outputs remain ignored; no meshes/renders are committed.

## Measured outcome

Mean rendered IoU across the same three shapes; failures stay in the per-case table.

| Method | Baseline | Single trial | Trial render gate passes |
|---|---:|---:|---:|
| profile_loft | 0.9344 | 0.9359 | 3/3 |
| visual_hull_voxel | 0.9698 | 0.9728 | 3/3 |
| gaussian_ellipsoid_proxy | 0.9028 | 0.8597 | 2/3 |
| primitive_fit_refine | 0.8069 | 0.7699 | 2/3 |
| differentiable_refine | 0.8353 | 0.7768 | 2/3 |
| hybrid_loft_hull | 0.9344 | 0.9359 | 3/3 |
| shape_program | unavailable | 0.7248 | 1/3 |
| ensemble | 0.9698 | 0.9728 | 3/3 |

Profile loft improved mean/weakest-view rendered IoU, normalized Chamfer and F-score on each of the three cases. Mean F-score moved .5911→.5937; vase alone reached .9640. The tested opt-in [profile configuration](../../configs/profile-loft-quality-20261006.json) is saved as an experimental preset, with full configuration provenance. Production defaults are unchanged. Its single-run mean workflow/render/validation time increased 5.95 s→7.43 s on the shared host; this is not a throughput claim.

The hull trial improved aggregate IoU and F-score, but its box weakest-view IoU regressed .87864→.87757. Gaussian, primitive and differentiable trials each regress materially on at least one shape. Their tested settings are rejected for general promotion. Shape programs now produce rendered measurements but pass only 1/3 render gates. Ensemble selects the hull, so it must not receive credit for independent geometry improvement.

The source fixes and exact sampler optimization are retained. The tested quality settings are opt-in. Broader seeds, frozen camera calibration and repeated isolated timings are required before promoting new defaults.

## Validation and runtime certification

Baseline pure runner: 73 groups passed, 0 failed. Repaired pure runner: 74 groups
passed, 0 failed. The final 25 focused metric/engine/selection tests and 22 backend contract tests pass. The installed
Blender identity/reference integration checks have their logs under the run root.
The identity/reference integration check passes with Chamfer 0, F-score 1 and volume IoU 1 for the same mesh. The installed Blender quick suite initially found a Windows MAX_PATH failure in moonshot reports; generated experiment directories now compact within the given artifact root. The fixed quick run passes 82 groups, with 0 failures and 2 documented skips (procedural workflow and full E2E, outside quick scope).

The first comparison used Blender 5.0.1 (`a3db93c5b259`, Python 3.11.13).
The installed **Blender 5.2.2 LTS** (`d13f752e3b9c`, Python 3.13.13) follow-up
passes 82 quick groups, the identity/reference contract and 24/24 fresh-process
reconstruction checks; 19/24 pass the render quality gate. Actual blend/OBJ/GLB/STL
round trips are recorded with transformed children, evaluated bevels, materials
and units. See [versioned verification](blender52-verification.md) for scope,
metrics, export results and remaining optional paths. The earlier missing-installation
blocker is withdrawn: the user identified an existing executable. No download or
installation was attempted. No general quality setting is promoted by this repeat.

## Remaining work, ordered by impact

P0: joint camera/scale/framing contract and transform-aware geometry; preselection
external-render checks for every shortlisted candidate; remaining aggregate-only
backend evidence; held-out geometry guardrails; volume union and self-intersection
validity; metric threshold calibration. High 2D IoU
with low 3D F-score in this experiment makes camera/proportion recovery urgent.

P1: per-technique weakest-view improvements across many shapes/seeds, primitive
fit stability under hard budgets, cavity/thin-feature preservation, guarded hull
resolution/simplification, shape-program residual fitting, fair resource accounting,
normalized recoverable-envelope metrics, broader export asset/texture/animation round trips.

P2: permitted external dataset/weights acquisition, matched SOTA reproduction,
held-out RGB/LPIPS evaluation, human editability study, real UV/PBR benchmarks,
optional GPU/OpenVDB/Poisson/FlexiCubes certification and executed moonshot
improvements. See all 49 readiness rows for topic-specific code evidence and gaps.

Brickworks principles that apply here are small isolated stages, typed boundaries,
exact provenance, explicit degradation and acceptance guards. No Brickworks code
or framework migration is part of this pass.

## External comparisons and retained historical evidence

The upstream PR's reported Gaussian average/min IoU`.963/.942` and differentiable
`.910/.874` are **PR-reported measurements**, not goals and not these runs. They
cannot be pooled with this experiment without identical inputs and protocols.
The retained May findings report 318 scanned results, 81 proxy/render disagreements
and 48 topology-blocked rows. Its g00 weakest-view averages were ensemble`.9571`,
hull/hybrid`.9224`, Gaussian`.8041`, differentiable`.7036`, primitive`.4435`.
Some later “Gaussian/primitive” rows actually selected ensemble; credit follows
the selected backend. The original large refinement directories were absent at
initial inspection; summary artifacts remain. The pure tests may create new
tiny fixture runs, which do not restore those historical directories.

No reproduced external SOTA baseline exists in the retained evidence.
[PartGS](https://arxiv.org/abs/2408.10789) uses calibrated multiview RGB and masks;
[Light-SQ](https://arxiv.org/abs/2509.24986) concerns mesh abstraction;
[SparseSurf](https://arxiv.org/abs/2511.14633) uses sparse RGB surface reconstruction;
[PrimitiveAnything](https://arxiv.org/abs/2505.04622) learns primitive assembly.
Their inputs, training and outputs differ from three silhouette blocking, so
paper scores do not establish a ranking for this task. Targets such as volume
IoU`.82` in old specs are acceptance proposals, not measured achievements.

Final source hashes and the trial/backfill provenance notes are in `temp/bounded-pass-20261006/final-source.json`. Twelve native/imported OBJ vertex-array checks confirm coordinate agreement within 1e-6; six baseline instrumentation runs match every per-view area IoU.
