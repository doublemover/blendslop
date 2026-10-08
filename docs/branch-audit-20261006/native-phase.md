# Accepted native performance and metric phase — 2026-10-06

Historical checkpoint. Current implementation and validation status: [frozen final campaign results](quality-final-results.md). Measurements below retain their original source revisions and do not qualify the new candidate.

Subsequent source changes: [implementation follow-up](implementation-remainder.md). Measurements below remain tied to the frozen revisions named here; no new matrix or timing campaign accompanied that follow-up.

The ensemble is **35.1% faster** on the mean of six per-case warm medians: **8.988 → 5.832 seconds**. All 96 matched timing runs keep exactly the same selected backend, raw 512-pixel alpha masks, silhouette IoU and legacy F-score. Strong profile/hull and default Gaussian results remain stable. Weak refinement means improve, with box/chair 3D regressions explicitly retained.

The complete 48-cell matrix ran. Silhouette gate passes rise from 23/48 archived controls to 31/48. The full native suite passed 88 groups at the frozen measurement revision; 107 focused tests passed on the accepted source. This is evidence on six synthetic cases, including three reserved cases.

## Revisions and acceptance boundaries

- Quality control: `b4186fbbf447105ed1106e59e086a296d828170b`; 48 archived mask-control rows.
- Matched unprofiled speed baseline: `34de2d358ee8dd23a2ce5d8981a7ee4be802bea1`.
- Frozen measurements: `e707b3fa2b79307ae041099ce5121edcba234175`.
- Invalid-input/cost/failure guards: `716b46eeb9f7d88b41ee3eb9a5894d4892035d86`.
- Accepted source: `55d10f1a6a464c80e45e2e3d3c16be4e9bef69e0`; zero-level SDF extraction repair.

The original campaign manifest remains incomplete because its final opt-in prototype job failed. The first repair exposed a second SDF extraction-default issue. Both logs remain. Scoped repairs resolved the multi-input RNA socket name and set the SDF isosurface to zero. The final four-case native ladder passed. Default-output replay and all 144 common metric/sample bundles exactly match the original frozen receipts. Timing rows retain `e707b3f`; they are never relabeled as measurements of later commits. Source hashes for the accepted executable code remain unchanged.

## What changed

Allocation tracing is explicitly optional and disabled in reported performance runs. Nested spans record exclusive costs; the observed validation workflow total includes the backend once. Missing timing remains null and cannot pass a requested cost gate. Unstructured legacy receipts do not invent exclusive subspans or complete workflow totals.

Candidates retain copied, read-only numeric geometry. Index-only topology caches use connectivity plus an evaluator version; scalar BVH caches include coordinates and faces. Blender evaluated meshes are copied in bulk and released immediately. A resident original mesh survives scene resets and serves selection/final renders; discarded owners are released. Exported OBJ files and artist sources remain.

The measured route skips polygon projections that fresh Blender renders replace. Fresh calibrated alpha gates still determine admission. Exhaustive diagnostic paths retain the polygon option; consumed voxel objectives are still evaluated. `native_resident`, `projection_diagnostics` and `diagnostic_allocations` are configuration flags.

Native CPU GN Raycast/Proximity and Exact/qualified-Manifold/SDF operations are opt-in APIs. They use owned Blender data, world-space buffers, explicit clipping and pixel centers. No bpy worker threads or default reconstruction replacement was introduced. SDF changes geometry and remains experimental: the shifted 5mm wall vanishes at 128 and returns at 256. Two Exact fixture outputs show a measured intersection pair. Volume or watertight edges alone do not establish a valid solid.

Versioned metric adapters preserve the legacy evaluator and add a shared target-derived frame, exact directed-distance formulas, raw sample/NN bundles, source/camera/mask hashes, seeds, units, dependency versions and per-object failure-preserving aggregates. Native paper scores stay unavailable when their released/prepared data is missing.

## Measurement contract

Training: box/vase/torus, seed 1234. Reserved: cylinder/bottle/chair, seed 77. Same 512-square front/side/top orthographic PNGs, exact calibration and ground-truth meshes. Legacy geometry uses 8192 area samples per surface, F-score tolerance .02 and independent uniform bbox extent 1. No rotation or anisotropic fitting. Native source coordinates differ by at most 5.9e-8 from the OBJ transport; triangle connectivity and masks match exactly.

Performance uses one cold and three warm runs per case/variant, serial children and four configured Blender/BLAS threads. Warm means retained interpreter/imports with fresh scenes and candidate caches. Process startup, geometry evaluation and novel-view evaluation are outside the reported reconstruction/validation timer. Child bounds:160s for four performance samples;100s for matrix children;2400s coordinated campaign. Repair acceptance and solid screens have separate explicit caps. Three warm repetitions support medians/ranges, not p95 or a population-wide speed claim.

`instrumentation_only` is a historical variant name: it also includes current safe bulk-export/cache improvements. Its difference from baseline is not isolated instrumentation overhead. Chair has visible timing noise and is slower in the native variant than projection bypass alone in this sample, while still improving over the matched baseline.

Historical control rows lack the later per-row provenance fields. Matching checks use their original preserved reference files/manifests: image/mesh hashes, seed, calibration, sample counts, threshold, sampler and normalization contract. Whole-harness hashes differ because the harness changed; both hashes are preserved. Historical traced runtime is excluded from the new speed comparison. Time-capped refinement can perform more search when tracing is disabled; those quality changes are a separate result.

## Validation and limits

- 48 new matrix cells;48 matched archived controls; all process jobs completed. Failed silhouette gates remain in the metric means.
- 88/0/0 native groups at the frozen measurement revision;107 focused checks on accepted code; full suite was not needlessly repeated after scoped repairs.
- 144 metric bundles; repaired points, directed distances and scores exactly equal; three repaired default outputs exactly equal.
- Eight candidate OBJ/GLB round trips;18 saved primitive parameter replays with nonzero edit responses; maximum replay vertex error 4.90e-9.
- Metre/centimetre transformed-hierarchy/modifier/material fixtures passed OBJ/GLB/STL/BLEND checks. Unitless formats retain declared Blender-unit coordinates; texture/UV/animation fidelity is unverified.
- 57 solid records: 26 fresh native children,25 exact prior mesh/evaluator receipt reuses, 30 records using fresh receipts including duplicate hashes, and two bounded SDF 256 skips. 27/48 candidate meshes qualify as single solids; all six selected ensembles qualify.
- Two native query shapes, four 128/256 SDF self-unions, four analytic thin-wall diagnostics and six Exact/Manifold operation fixtures. Query timings are single samples and support no parallel speed claim.
- Three prior actual Open3D 0.19 no-trim/no-crop Poisson rows preserved: all nonwatertight, vase self-intersecting. No Poisson promotion or new installation.

Unfinished: broad real-world/generalization studies, native paper datasets/upstream preparation and DTU official sampler execution, Blender 4.2/5.0 reruns, semantic artist judgment, texture/UV/animation fidelity, universal weak-method 3D success, classification of the two Exact intersection pairs, and SDF 256 self-intersection qualification. The complete scoped checklist follows. Historical 49 readiness rows and 1,363 recorded-open task records remain inventories, not silently completed acceptance criteria.

No installs, paid compute, pushes, publication or unrelated filesystem actions occurred. Owned validation children were joined; prior work and receipts remain intact.

## Deliverables and reproduction

[Numeric results](native-results.md), [machine-readable summary](native-summary.json), [all48 cells](native-cells.csv), [technique means](native-techniques.csv), [timing rows](native-performance.csv), [common metrics](native-common-metrics.csv), [checklist CSV](native-checklist.csv).

Evidence root: `temp/native-phase-20261006/frozen-e707b3f`. Hashes of the principal manifests/receipts are in `native-summary.json`. Earlier phase: `temp/improvement-phase-20261006/final-candidate`. Both remain local. The campaign driver is `scripts/run_native_performance_phase.py`; metric driver is `scripts/evaluate_protocol_campaign.py`; opt-in driver is `scripts/validate_native_prototypes.py`. Repair/solid/feature helpers and their logs remain under `temp/native-phase-20261006`.

## Complete scoped done/not-done list

| ID | Requirement | State | Evidence / remaining work |
|---|---|---|---|
| 1.01 | Freeze source/config/masks/reference/camera/seed | DONE | e707b3f manifest; 48 matched cells and 96 timing rows;  |
| 1.02 | Retain earlier dirty work and measurements | DONE | 974821c/34de2d3 and earlier phase receipts preserved;  |
| 1.03 | Keep every failure and per-object macro denominator | DONE | 48-cell table; unavailable bundle fixture; macro returns null on failure;  |
| 1.04 | Independently validate historical comparison inputs | DONE | Original reference-file hashes, seeds, calibration and surface contracts match; Whole harness hashes differ and are retained |
| 1.05 | Publish full failure/regression map | DONE | native-cells.csv and native-results.md; Box/chair weak-method 3D regressions remain |
| 2.01 | Primitive normalized fit/seeding/scale refinements | DONE | Inherited approved implementation; training and held-out reruns; No universal geometry improvement |
| 2.02 | Differentiable objective/refinement improvements | DONE | Inherited implementation; training and held-out reruns; Box/chair geometry regressions remain |
| 2.03 | Preserve Gaussian defaults; isolate experimental k-means | DONE | Default config/backends tests; six default Gaussian rows stable; Experimental Gaussian gains did not justify a default change |
| 2.04 | Preserve strong profile and hull outputs | DONE | Matched original controls; stable F-score/IoU; Tiny export rounding is disclosed |
| 2.05 | Remove all weak-case geometry regressions | NOT_DONE | Five decreases over 1e-4 reported; Four weak refinement cases; one small shape-program sampling change |
| 2.06 | Learned/semantic shape-program quality improvements | NOT_DONE | Current structured program remains measured; Requires a separate implementation/data study |
| 3.01 | Cost-aware measured route on reserved examples | DONE | All six ensemble selections retained; three held-out cases; All six choose hull; broader routing usefulness unproved |
| 3.02 | Three warm repetitions plus cold boundary | DONE | 96 serial measurements; rotated variant order; Six synthetic cases; no p95 estimate |
| 3.03 | Match output and sampling across speed variants | DONE | All selection/alpha/IoU/F-score invariants; identical triangle connectivity; Coordinates differ at most 5.9e-8 |
| 3.04 | Separate tracing diagnostics from reported performance | DONE | Both historical import namespaces overridden; tracemalloc disabled; Historical traced times are not speed baselines |
| 3.05 | Exclusive cost spans and nonduplicated workflow total | DONE | Named spans sum to observed total; E2E backend is nested; Missing receipts remain null |
| 3.06 | Broad real-world selection/generalization certification | NOT_DONE | Reserved synthetic split reported separately; No real dataset or artist evaluation |
| 4.01 | Fixed camera/crop/unit/axis matching | DONE | 512-square orthographic views; exact raw-alpha match; full suite;  |
| 4.02 | World-space numeric buffers and bulk mesh transfer | DONE | Native buffer counters and export round trips;  |
| 4.03 | Metres/centimetres and hierarchy/modifier fixtures | DONE | OBJ/GLB/STL/BLEND at scene scale 1/.01; Unitless formats retain declared Blender-unit convention |
| 4.04 | Cross-DCC units/texture/animation certification | NOT_DONE | Installed Blender round trips only; External viewers and UV/texture/animation fidelity unverified |
| 5.01 | Immutable candidate buffers and content/versioned caches | DONE | Coordinate/BVH and connectivity/topology invalidation fixtures; Index topology cache does not certify geometric degeneracy |
| 5.02 | Owned resident mesh across selection/final rendering | DONE | Native ownership/reset/release fixture; 96 measurements;  |
| 5.03 | Bypass discarded polygon projection safely | DONE | Fresh Blender alpha gates retained; opt-in diagnostic flag; Voxel objective/projection stages remain where consumed |
| 5.04 | Retain artist source/parts/parameters | DONE | 18 saved parameter replays and nonzero edit responses; Semantic usefulness requires human judgment |
| 5.05 | Preserve exports and evaluated modifiers | DONE | Eight candidate OBJ/GLB pairs and two transformed unit fixtures; OBJ/STL flatten hierarchy; STL lacks materials |
| 5.06 | Structural diagnostics for every final cell | DONE | 48 candidate records plus saved scene fixtures; Concatenation is not solid union |
| 5.07 | Bounded actual self-intersection qualification | DONE | 57 records; 26 fresh children and exact prior receipt reuse; Two SDF 256 outputs exceed 200000-face cap |
| 5.08 | All six selected ensemble meshes qualify | DONE | Six zero-pair self-intersection screens; one closed component each; Bounded mesh screens, not arbitrary future inputs |
| 5.09 | Certify every proxy/refinement output as one solid | NOT_DONE | 27/48 final candidates eligible; Assemblies, intersections and disconnected pieces are explicitly retained |
| 5.10 | Artist usability study | NOT_DONE | Native parameter response and stack preservation verified; Pleasantness and semantic editing not measured |
| 6.01 | Comparable common-case metric contracts | DONE | 144 raw point/NN bundles; explicit frames/formulas/seeds/hashes; Formula diagnostics on our cases only |
| 6.02 | Existing Poisson no-trim/no-crop qualification | DONE | Three prior actual Open3D 0.19 rows retained; All three nonwatertight; vase self-intersects; no promotion |
| 6.03 | Run learned/native paper baselines | NOT_DONE | No native scores fabricated; Released/prepared datasets and complete upstream pipelines unavailable |
| 6.04 | Install optional missing dependencies | NOT_DONE | No installations requested or performed; Outside authorized scope |
| 7.01 | GN batch Raycast and Proximity CPU prototype | DONE | Torus/chair masks match scalar BVH; distance error <=1.2e-7; 1024 rays per shape; one timing each |
| 7.02 | Explicit coordinates/pixel centers/near/far/no-hit | DONE | Tetrahedron clipping/pixel-center fixtures; read-only buffers; No bpy worker threads |
| 7.03 | Establish parallel query performance gain | NOT_DONE | Indicative timings retained without a speed claim; Needs repeated profiles and larger query sets |
| 7.04 | Exact and qualified Manifold opt-in operations | DONE | Six volume/topology fixtures; actual input hash qualifications; Two Exact outputs show one measured intersection pair |
| 7.05 | SDF 128/256 native ladder and output qualification | DONE | Torus/chair four self-unions; holes/volume/surface metrics; Dense mesh; SDF 256 intersection budget exceeded |
| 7.06 | Quantify thin-wall discretization failure | DONE | Centered/shifted 5mm walls at both resolutions; Shifted wall disappears at 128;256 recovers with 2.28% volume error |
| 7.07 | Promote new algorithms into defaults | NOT_DONE | Prototypes remain separate callable APIs; Changed geometry and qualification limits preclude promotion |
| 7.08 | General multi-object CSG robustness | NOT_DONE | Two qualified analytic fixtures and SDF self-unions; Broader geometry/operation cases and Exact-contact classification remain |
| 8.01 | Legacy independent bbox extent 1 contract unchanged | DONE | 8192 deterministic area samples; F-score tolerance .02; Legacy numbers are not common-frame native scores |
| 8.02 | Shared target-derived common transform | DONE | Translation/scaling fixtures; 144 deterministic bundles;  |
| 8.03 | SuperFlex native strict sample/frame adapter | DONE | 4096 FPS/reference and area prediction guards; identity native frame; Released datasets/occupancy queries unavailable |
| 8.04 | SuperFit native preparation/occupancy contract | DONE | Pinned preparation receipt gate; union/SDF-sign fixtures; Actual upstream cleanup/FlexiCubes/SDF pipeline not reproduced |
| 8.05 | DTU modes/mm/20mm/asymmetry contract | DONE | DP-GS, SparseSurf, PartGS point and block adapters; numeric fixtures; Official assets/prepared sampling unavailable |
| 8.06 | Reproduce official DTU sampling/culling pipeline | NOT_DONE | Adapter requires verified official prepared arrays/assets; No official sampler execution on real assets |
| 9.01 | One coordinated frozen campaign and scoped repairs | DONE | e707 campaign; 716 guard repair;55d opt-in SDF repair; Two original prototype failures retained |
| 9.02 | Full applicable native suite and focused acceptance | DONE | 88 full groups at e707;107 focused checks at 55d; Full suite not repeated after isolated guards/prototype repairs |
| 9.03 | Recheck repaired samples and default output | DONE | 144 arrays/metrics exactly equal; three default replays exactly equal;  |
| 9.04 | Fresh Blender 4.2/5.0 qualification | NOT_DONE | Installed5.2.2 used; compatibility checks retained; No alternate installed Blender or CI run |
| 9.05 | Coherent local commits and complete final docs | DONE | Source commits plus documentation commit; clean final checkout; No push or publication |
| 9.06 | Owned child cleanup and bounded execution | DONE | All runners joined their owned children; timeout tree cleanup coded; No unrelated process manipulation |
