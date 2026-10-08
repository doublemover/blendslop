# Native phase measurements — 2026-10-06

Historical checkpoint. Current implementation and validation status: [frozen final campaign results](quality-final-results.md). Measurements below retain their original source revisions and do not qualify the new candidate.

Accepted source: `55d10f1`; measured timing/matrix source: `e707b3f`. The [phase report](native-phase.md) states the precise repair and comparison boundaries. Values are equal-weight case means unless identified as medians. Every failed silhouette gate remains in the quality means.

## Matched ensemble speed

Unprofiled baseline `34de2d3` versus native resident path `e707b3f`. One cold sample plus three warm samples per variant per case; 96 measurements total. Seconds include reconstruction and validation, excluding process startup, novel views and geometry evaluation. Warm retains interpreter/imports, with fresh scenes and caches.

| Case | Baseline warm median [min,max] | Projection bypass median | Native warm median [min,max] | Reduction vs baseline | Cold baseline → native |
|---|---:|---:|---:|---:|---:|
| bottle | 9.007 [8.970,10.351] | 6.066 | 5.232 [5.188,5.297] | 41.9% | 10.422 → 7.404 |
| box | 8.970 [8.524,9.086] | 6.235 | 5.799 [5.493,7.298] | 35.4% | 11.309 → 7.527 |
| chair | 7.958 [7.862,7.982] | 5.825 | 6.437 [5.846,9.370] | 19.1% | 10.141 → 8.205 |
| cylinder | 8.020 [8.008,8.640] | 6.203 | 5.268 [5.185,5.418] | 34.3% | 10.429 → 9.472 |
| torus | 10.448 [10.339,12.132] | 7.347 | 6.394 [6.358,6.458] | 38.8% | 12.186 → 8.371 |
| vase | 9.524 [9.436,9.546] | 6.843 | 5.860 [5.839,5.886] | 38.5% | 11.597 → 7.858 |

Mean of case warm medians: **8.988 → 5.832s, 35.1% reduction**. Projection bypass mean is 6.420s. The `instrumentation_only` mean is 8.503s, but that variant includes shared bulk/cache improvements and does not isolate instrumentation overhead. Chair is noisier and native is slower than bypass alone in this sample. All six still improve over the matched baseline. No p95 or generalization claim. [All variant/cold/range/memory rows](native-performance.csv).

All selection decisions, raw alpha masks, IoU/F-score values and inputs match across 96 runs. Triangle connectivity matches; maximum coordinate difference 5.9e-8; maximum legacy Chamfer-L1 difference 2.25e-7. Removal of the OBJ round trip changes rounding, so byte-identical meshes are not claimed.

## Exclusive costs and memory

Current recorder spans sum to their observed totals. E2E combined cost equals its validation total, which contains the backend once. Missing measurements remain null; optional allocation tracing is disabled. Median stage costs are computed within each case and then averaged across cases; they are not additive workflow speed estimates from independently chosen medians.

| Native backend stage | Mean of case median exclusive ms |
|---|---:|
| build_target | 1247.90 |
| extraction_topology | 1058.23 |
| topology | 686.65 |
| serialization | 468.97 |
| ensemble_reconstruct | 350.31 |
| render | 147.58 |
| projection | 108.96 |
| candidate_backend | 93.61 |
| native_evaluation | 50.59 |
| extraction | 41.89 |
| construction | 36.49 |
| editable_proxy | 14.40 |
| unaccounted | 0.43 |

Process memory is own-process Windows working set and lifetime peak, including Python, Blender and native libraries. Peaks span the cold/warm interpreter lifetime and are not isolated native allocation measurements.

| Case | Baseline maximum lifetime peak MiB | Native maximum lifetime peak MiB |
|---|---:|---:|
| bottle | 746.4 | 767.9 |
| box | 761.6 | 756.2 |
| chair | 748.1 | 747.0 |
| cylinder | 748.1 | 755.0 |
| torus | 852.0 | 828.6 |
| vase | 758.4 | 778.4 |

Mean case-median counters: native builds 1.67; topology cache hits 1; numeric transfer to native 2,678,184bytes; native-to-Python 19,544bytes; retained OBJ boundary writes 4,191,773bytes. These distinguish numeric transfers, cache use and retained export work; they do not measure all memory traffic.

## Per-technique before/after

Before: 48 retained mask controls at `b4186fb`. After: 48 cells at `e707b3f`. Images, ground-truth files, calibration, seeds and declared surface protocol match. Whole-harness hashes differ and are retained. Legacy metrics: independent bbox extent 1, 8192 area samples, tolerance .02. Historical traced runtime is excluded. Refinement remains capped at 8/20 seconds; disabled tracing allows additional search, so its quality improvement is separate from the same-output ensemble result.

### Training: box/vase/torus, seed 1234

| Technique | IoU before → after | F.02 before → after | CD-L1 before → after | Novel IoU before → after | Gates before → after /3 |
|---|---:|---:|---:|---:|---:|
| profile_loft | 0.8923 → 0.8923 | 0.6415 → 0.6415 | 0.0496 → 0.0496 | 0.8934 → 0.8934 | 2 → 2 |
| visual_hull_voxel | 0.9809 → 0.9809 | 0.5114 → 0.5114 | 0.0514 → 0.0514 | 0.9076 → 0.9076 | 3 → 3 |
| gaussian_ellipsoid_proxy | 0.8653 → 0.8653 | 0.3590 → 0.3590 | 0.0685 → 0.0685 | 0.8280 → 0.8280 | 2 → 2 |
| primitive_fit_refine | 0.7073 → 0.8701 | 0.1713 → 0.3073 | 0.1180 → 0.0916 | 0.7070 → 0.8694 | 0 → 1 |
| differentiable_refine | 0.5005 → 0.8838 | 0.2424 → 0.3339 | 0.1044 → 0.0877 | 0.5790 → 0.8649 | 0 → 3 |
| hybrid_loft_hull | 0.8923 → 0.8923 | 0.6415 → 0.6415 | 0.0496 → 0.0496 | 0.8934 → 0.8934 | 2 → 2 |
| shape_program | 0.7798 → 0.7798 | 0.1332 → 0.1332 | 0.0931 → 0.0931 | 0.7373 → 0.7373 | 1 → 1 |
| ensemble | 0.9809 → 0.9809 | 0.5114 → 0.5114 | 0.0514 → 0.0514 | 0.9076 → 0.9076 | 3 → 3 |

### Held out: cylinder/bottle/chair, seed 77

| Technique | IoU before → after | F.02 before → after | CD-L1 before → after | Novel IoU before → after | Gates before → after /3 |
|---|---:|---:|---:|---:|---:|
| profile_loft | 0.8152 → 0.8152 | 0.4882 → 0.4882 | 0.1261 → 0.1261 | 0.7572 → 0.7572 | 1 → 1 |
| visual_hull_voxel | 0.9722 → 0.9722 | 0.4583 → 0.4583 | 0.0931 → 0.0931 | 0.8597 → 0.8597 | 3 → 3 |
| gaussian_ellipsoid_proxy | 0.7644 → 0.7644 | 0.2482 → 0.2482 | 0.1294 → 0.1294 | 0.6883 → 0.6883 | 1 → 1 |
| primitive_fit_refine | 0.6265 → 0.8489 | 0.1724 → 0.3115 | 0.1231 → 0.1073 | 0.7091 → 0.8499 | 0 → 2 |
| differentiable_refine | 0.4339 → 0.8458 | 0.2229 → 0.2604 | 0.1260 → 0.1072 | 0.4856 → 0.8051 | 0 → 2 |
| hybrid_loft_hull | 0.8152 → 0.8152 | 0.4882 → 0.4882 | 0.1261 → 0.1261 | 0.7572 → 0.7572 | 1 → 1 |
| shape_program | 0.6953 → 0.6953 | 0.0447 → 0.0447 | 0.1810 → 0.1810 | 0.6085 → 0.6085 | 1 → 1 |
| ensemble | 0.9722 → 0.9722 | 0.4583 → 0.4583 | 0.0931 → 0.0931 | 0.8597 → 0.8597 | 3 → 3 |

All 48 process cells completed. Gate passes 23 →31;27/48 meshes qualify as single solids. Every ensemble selects hull; this proves the bounded route preserves these selections, not that it identifies the best 3D technique generally. The complete [cell table](native-cells.csv) and [means including previous-phase values](native-techniques.csv) retain the full comparison.

### Remaining geometry decreases

Cases with an F-score decrease or CD-L1 increase greater than 1e-4 are listed below. No ground-truth-specific fallback was added to conceal them. Shape-program surface sampling is sensitive to the tiny evaluated-export rounding change.

| Split/case | Technique | ΔF.02 | ΔCD-L1 | Δsilhouette IoU |
|---|---|---:|---:|---:|
| paired/box | primitive_fit_refine | -0.05826 | +0.02221 | +0.09916 |
| paired/box | differentiable_refine | -0.00391 | -0.00981 | +0.26947 |
| paired/torus | shape_program | -0.00015 | +0.00000 | +0.00000 |
| heldout/chair | primitive_fit_refine | -0.03644 | +0.04198 | +0.18016 |
| heldout/chair | differentiable_refine | -0.08057 | +0.01385 | +0.28436 |

## Metric-aligned common-case diagnostics

These are formulas on our synthetic objects, with a single transform derived from the full target mesh and applied to both surfaces. They are not native dataset scores or paper-level comparisons. Samples, raw directed NN distances, hashes, units, seeds, revisions and per-object macros are saved in 144 bundles. Failed silhouette cases remain included. [All three profiles](native-common-metrics.csv).

SuperFlex formula profile: extent 1, 4096 area samples per surface on our cases; Euclidean NN; inclusive thresholds .01/.015/.02; F=2pr/(p+r+1e-6); Chamfers are half-sums, displayed×100. Native mode instead requires released 4096 FPS reference points in the original ShapeNet frame, deterministic area prediction samples and identity transformation (`normalize=False`). Native data is missing.

### paired — SuperFlex formula on common cases

| Technique | F.01 | F.015 | F.02 | CD-L1×100 | CD-L2×100 |
|---|---:|---:|---:|---:|---:|
| profile_loft | 0.2483 | 0.4486 | 0.5927 | 2.7392 | 0.1645 |
| visual_hull_voxel | 0.1042 | 0.2688 | 0.4322 | 2.7612 | 0.1262 |
| gaussian_ellipsoid_proxy | 0.0670 | 0.1717 | 0.2941 | 3.6078 | 0.1967 |
| primitive_fit_refine | 0.0660 | 0.1711 | 0.3035 | 4.2602 | 0.3810 |
| differentiable_refine | 0.0727 | 0.1840 | 0.3051 | 4.2653 | 0.3239 |
| hybrid_loft_hull | 0.2483 | 0.4486 | 0.5927 | 2.7392 | 0.1645 |
| shape_program | 0.0920 | 0.2048 | 0.3006 | 3.8952 | 0.2256 |
| ensemble | 0.1042 | 0.2688 | 0.4322 | 2.7612 | 0.1262 |

### heldout — SuperFlex formula on common cases

| Technique | F.01 | F.015 | F.02 | CD-L1×100 | CD-L2×100 |
|---|---:|---:|---:|---:|---:|
| profile_loft | 0.1912 | 0.3423 | 0.4543 | 5.8618 | 0.9652 |
| visual_hull_voxel | 0.1730 | 0.4223 | 0.6199 | 2.6860 | 0.2050 |
| gaussian_ellipsoid_proxy | 0.0542 | 0.1346 | 0.2311 | 5.9850 | 0.8265 |
| primitive_fit_refine | 0.0716 | 0.1891 | 0.3217 | 4.2275 | 0.3956 |
| differentiable_refine | 0.0650 | 0.1716 | 0.3031 | 4.8787 | 0.5605 |
| hybrid_loft_hull | 0.1912 | 0.3423 | 0.4543 | 5.8618 | 0.9652 |
| shape_program | 0.0940 | 0.2313 | 0.3471 | 6.1982 | 1.1189 |
| ensemble | 0.1730 | 0.4223 | 0.6199 | 2.6860 | 0.2050 |

SuperFit formula profile: shared target-centered extent 1.8, 2048 area samples each; CD=100×half-sum of directional mean squared NN distances. Upstream cleanup/FlexiCubes and 128³ native SDF-sign occupancy are not reproduced. The implemented common-grid sign/active-union fixtures are separately labeled diagnostics.

| Technique | Training reported CD | Held-out reported CD |
|---|---:|---:|
| profile_loft | 0.5853 | 3.1378 |
| visual_hull_voxel | 0.4709 | 0.7354 |
| gaussian_ellipsoid_proxy | 0.6936 | 2.7321 |
| primitive_fit_refine | 1.3122 | 1.3887 |
| differentiable_refine | 1.1304 | 1.8385 |
| hybrid_loft_hull | 0.5853 | 3.1378 |
| shape_program | 0.7969 | 3.5342 |
| ensemble | 0.4709 | 0.7354 |

Pinned revisions: SuperFlex `15b63b23375c6a80c3ce0dec4be0c9b11055c722`; SuperFit `42b263343683221f7c1081baa069b029e9aafeee`. Native adapters are unavailable for both. DTU DP-GS/no-cull, SparseSurf/point-cull, PartGS/point-cull and PartGS/block/no-cull modes also remain unavailable. The adapter requires native mm calibration, official assets and verified prepared asymmetric samples; directed distances ≥20 mm are excluded and the overall score is their half-sum without a multiplier. Synthetic fixtures validate the boundary/formula, not the upstream official sampler.

## Native prototypes

Two 1024-ray shapes exactly match scalar BVH hit masks. Torus maximum hit-distance error 0; chair 1.19e-7; sampled on-surface proximity distance 0. Single native/scalar timings are retained in the summary; they support no parallel performance claim. Clipping/no-hit/pixel-center cases are covered by focused fixtures.

SDF rows are self-unions of source hulls. F.02 below compares against the source hull in a common frame, not synthetic ground truth; silhouette gates compare fresh 512 renders against observed masks. Signed triangle volumes are separate from the coarse 24³ ray-parity diagnostic, whose aliasing is material.

| Case | Grid ladder | Triangles | Min observed IoU | Common F.02 vs hull | Signed-volume error | Self-intersection qualification |
|---|---:|---:|---:|---:|---:|---|
| torus | 128 | 93416 | 0.9698 | 0.8939 | 0.279% | zero measured pairs |
| torus | 256 | 372116 | 0.9697 | 0.8926 | 0.093% | not checked:200000-face cap |
| chair | 128 | 115508 | 0.8600 | 0.7786 | 0.082% | zero measured pairs |
| chair | 256 | 462672 | 0.8604 | 0.7770 | 0.024% | not checked:200000-face cap |

All four are structurally watertight with one component. The observed top-view torus hole remains 1 in reference/hull/SDF. Medial-ridge radius 3/8/12 pixel proxies had no qualifying pixels, so no preservation score is invented. Separate analytic 5 mm walls test lattice sensitivity:

| Wall center x(m) | Resolution | Result | Signed-volume error |
|---|---:|---|---:|
| 0.0 | 128 | available | 1.940% |
| 0.0 | 256 | available | 0.970% |
| 0.004 | 128 | disappears / empty mesh | unavailable |
| 0.004 | 256 | available | 2.281% |

The wall phase changes the result; SDF stays opt-in. Thin-wall outputs have structural checks only, with no self-intersection certification. Dense meshes retain native coordinate editing but no primitive editability claim.

Exact/Manifold use actual hash-qualified overlapping-cube and disconnected-closed-cube input fixtures. Both solvers reproduce analytic union/intersection/difference volumes 2.5/1/.5; union correctly has two components. Native output checks:

| Operation | Exact intersection pairs | Manifold intersection pairs | Expected components |
|---|---:|---:|---:|
| UNION | 1 | 0 | 2 |
| INTERSECT | 1 | 0 | 1 |
| DIFFERENCE | 0 | 0 | 1 |

Two Exact rows require contact/intersection classification; they are not silently accepted as valid solids. Union is a valid disconnected result and is not labeled one solid. No broader CSG robustness or solver speed claim. The first validation-helper lookup failure and the all-zero intersection assertion failure remain in the local receipts.

## Validation receipt locations

Root: `temp/native-phase-20261006/frozen-e707b3f`

- `validation.json` and `native-full.log`: original freeze/jobs and 88-group full result.
- `repair-acceptance/`: guard acceptance, 144 equivalent sample bundles and three exact default replays; first repaired prototype failure.
- `sdf-zero-acceptance/`: accepted source hashes and successful four-row ladder; feature diagnostics.
- `solid-qualification/result.json`: 57 records, exact receipt reuse, bounded skips and all native outcomes.
- `manifold-qualification-repaired/` and `thin-sdf/`: additional bounded qualification/analytic diagnostics.
- `candidate-exports/`, `exports-metres/`, `exports-centimetres/`, `solids/`: saved scenes, parameter replays and format/unit checks.

Installed Blender 5.2.2 LTS/build `d13f752e3b9c`, bundled Python 3.13.13; existing isolated Open3D 0.19/Python 3.12 helper. No alternate Blender, learned baseline, paid compute, new install, push or publication. Principal receipt hashes are in [native-summary.json](native-summary.json).
