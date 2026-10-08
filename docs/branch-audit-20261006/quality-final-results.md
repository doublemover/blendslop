# Frozen final quality campaign - 2026-10-06

One campaign on implementation `06a7f0219c98612a03b6e7e127d698fd9cc6c541`: **247/252 completed cells, 5 process failures, 0 unreached**; 171 completed cells pass the render gates.

The measurements support several method improvements relative to the retained mask-control workflow, plus specific quality/cost tradeoffs. DVX 32^3/12-step and fitting budgets are practical initial limits, not empirically chosen convergence plateaus. DVX stops for steps/time/numerical validity; CPU fitters additionally shrink unsuccessful moves to min_step. Process watchdogs are separate. Longer matched refinement/grid/step sweeps have not been measured; history lengths and final loss trends are retained in JSON. They do **not** support universal weak-case improvement, blanket promotion of the quality preset, or an ensemble guarantee of best 3D fidelity. Failures, regressions, skipped candidates and partial repetitions remain in the evidence.

## Frozen boundary

Source tree: `ac61394e40b1c117c8712d068aa3390385d0ed3bd24935be75dd3443adff8984`. Manifest: `1337ef0d27bbd37de232d5dcd62faedd42550c19d89441f8c192a354453060d1`. The implementation source/configuration and 108 prepared input/provenance files were checked before each job. Source unchanged at campaign completion: `True`.

Started 2026-10-06 at approximately 12:34:57 UTC, immediately after the scoped activity snapshot. Installed Blender 5.2.2 LTS d13f752e3b9c; four configured threads, serial top-level children, 100s ordinary child / 600s ensemble repetition child / 20000s global limits. No new installs, pushes, downloads, or new candidate variants.

Default means the ordinary adaptive preset under the shared comparison fidelity/budgets. Legacy geometry uses 8192 area-weighted surface samples, F.02 and independent uniform bbox normalization; novel views are evaluation only. Common profiles use their declared shared target frame and sample counts. These protocols are reported separately.

## Completion by split

| Split | Planned | Completed | Process failed | Unreached | Render gates passed |
|---|---:|---:|---:|---:|---:|
| external | 144 | 142 | 2 | 0 | 92 |
| heldout | 54 | 54 | 0 | 0 | 41 |
| paired | 54 | 51 | 3 | 0 | 38 |

Ensembles: 42/42 saved cold rows, 119/126 saved warm rows; 39 case/variant groups completed all three warm repetitions. Successful cold outputs from later-failed repetition groups are preserved separately and excluded from primary driver-cell macros, candidate export validation and common metrics.

Reconstruction-level counts (including cold outputs from failed repetition groups): `{'planned': 378, 'completed': 369, 'process_failed': 5, 'unreached': 4}`.

## Historical before/after by technique

Retained b4186fb mask-control workflow, including its recorded harness overlay, versus the frozen current ordinary preset. Compatibility checks verify image/GT hashes, cameras, seed/sampling, normalization/reference frame, F tolerance, fidelity, observation policy, gate thresholds, Blender build and evaluator code (line endings normalized). The changed harness/export path is declared. This compares complete workflows, including their declared implementation settings; it is not an isolated source-only effect or a historical timing claim.

| Split | Technique | Compatible n/planned | Surface F.02 before -> after | Image IoU before -> after | Gates before -> after |
|---|---|---:|---:|---:|---:|
| paired | differentiable_refine | 3/3 | 0.242 -> 0.303 | 0.500 -> 0.881 | 0 -> 3 |
| paired | ensemble | 3/3 | 0.511 -> 0.511 | 0.981 -> 0.981 | 3 -> 3 |
| paired | gaussian_ellipsoid_proxy | 3/3 | 0.359 -> 0.359 | 0.865 -> 0.865 | 2 -> 2 |
| paired | hybrid_loft_hull | 3/3 | 0.642 -> 0.511 | 0.892 -> 0.942 | 2 -> 3 |
| paired | primitive_fit_refine | 3/3 | 0.171 -> 0.252 | 0.707 -> 0.805 | 0 -> 1 |
| paired | profile_loft | 3/3 | 0.642 -> 0.720 | 0.892 -> 0.914 | 2 -> 2 |
| paired | shape_program | 3/3 | 0.133 -> 0.576 | 0.780 -> 0.912 | 1 -> 2 |
| paired | visual_hull_voxel | 3/3 | 0.511 -> 0.511 | 0.981 -> 0.981 | 3 -> 3 |
| heldout | differentiable_refine | 3/3 | 0.223 -> 0.258 | 0.434 -> 0.824 | 0 -> 2 |
| heldout | ensemble | 3/3 | 0.458 -> 0.458 | 0.972 -> 0.972 | 3 -> 3 |
| heldout | gaussian_ellipsoid_proxy | 3/3 | 0.248 -> 0.248 | 0.764 -> 0.764 | 1 -> 1 |
| heldout | hybrid_loft_hull | 3/3 | 0.488 -> 0.458 | 0.815 -> 0.936 | 1 -> 2 |
| heldout | primitive_fit_refine | 3/3 | 0.172 -> 0.357 | 0.627 -> 0.817 | 0 -> 2 |
| heldout | profile_loft | 3/3 | 0.488 -> 0.638 | 0.815 -> 0.903 | 1 -> 2 |
| heldout | shape_program | 3/3 | 0.045 -> 0.655 | 0.695 -> 0.938 | 1 -> 3 |
| heldout | visual_hull_voxel | 3/3 | 0.458 -> 0.458 | 0.972 -> 0.972 | 3 -> 3 |

Every compatible current cell is included; failures are unavailable, never zero-imputed. Historical quality-preset comparisons are retained in the JSON. Default hull and Gaussian legacy surface scores are unchanged on all six synthetic controls. Hybrid and individual weak-case regressions remain visible.

## Current ordinary -> quality / CPU-DVX

| Split | After preset | Technique | Matched n/planned | Surface F.02 | Image IoU | Novel IoU | Gates before -> after |
|---|---|---|---:|---:|---:|---:|---:|
| external | quality | profile_loft | 8/8 | 0.533 -> 0.621 | 0.852 -> 0.906 | 0.805 -> 0.832 | 4 -> 6 |
| external | quality | visual_hull_voxel | 8/8 | 0.636 -> 0.659 | 0.963 -> 0.974 | 0.846 -> 0.856 | 8 -> 8 |
| external | quality | gaussian_ellipsoid_proxy | 8/8 | 0.388 -> 0.388 | 0.742 -> 0.742 | 0.678 -> 0.678 | 3 -> 3 |
| external | quality | primitive_fit_refine | 7/8 | 0.341 -> 0.338 | 0.695 -> 0.690 | 0.700 -> 0.674 | 0 -> 0 |
| external | quality | differentiable_refine | 8/8 | 0.407 -> 0.407 | 0.787 -> 0.787 | 0.760 -> 0.760 | 3 -> 3 |
| external | quality | hybrid_loft_hull | 8/8 | 0.636 -> 0.684 | 0.920 -> 0.946 | 0.846 -> 0.880 | 8 -> 8 |
| external | quality | shape_program | 8/8 | 0.574 -> 0.604 | 0.902 -> 0.902 | 0.852 -> 0.848 | 6 -> 6 |
| external | quality | ensemble | 8/8 | 0.640 -> 0.685 | 0.964 -> 0.973 | 0.848 -> 0.876 | 8 -> 8 |
| external | cpu_dvx | differentiable_refine | 8/8 | 0.407 -> 0.416 | 0.787 -> 0.779 | 0.760 -> 0.761 | 3 -> 2 |
| external | cpu_dvx | ensemble | 7/8 | 0.666 -> 0.717 | 0.963 -> 0.972 | 0.836 -> 0.868 | 7 -> 7 |
| heldout | quality | profile_loft | 3/3 | 0.638 -> 0.673 | 0.903 -> 0.938 | 0.905 -> 0.915 | 2 -> 3 |
| heldout | quality | visual_hull_voxel | 3/3 | 0.458 -> 0.475 | 0.972 -> 0.978 | 0.860 -> 0.864 | 3 -> 3 |
| heldout | quality | gaussian_ellipsoid_proxy | 3/3 | 0.248 -> 0.248 | 0.764 -> 0.764 | 0.688 -> 0.688 | 1 -> 1 |
| heldout | quality | primitive_fit_refine | 3/3 | 0.357 -> 0.322 | 0.817 -> 0.754 | 0.801 -> 0.736 | 2 -> 2 |
| heldout | quality | differentiable_refine | 3/3 | 0.258 -> 0.258 | 0.824 -> 0.824 | 0.785 -> 0.785 | 2 -> 2 |
| heldout | quality | hybrid_loft_hull | 3/3 | 0.458 -> 0.556 | 0.936 -> 0.946 | 0.860 -> 0.922 | 2 -> 3 |
| heldout | quality | shape_program | 3/3 | 0.655 -> 0.661 | 0.938 -> 0.938 | 0.924 -> 0.924 | 3 -> 3 |
| heldout | quality | ensemble | 3/3 | 0.458 -> 0.556 | 0.972 -> 0.979 | 0.860 -> 0.922 | 3 -> 3 |
| heldout | cpu_dvx | differentiable_refine | 3/3 | 0.258 -> 0.248 | 0.824 -> 0.789 | 0.785 -> 0.735 | 2 -> 0 |
| heldout | cpu_dvx | ensemble | 3/3 | 0.458 -> 0.556 | 0.972 -> 0.979 | 0.860 -> 0.922 | 3 -> 3 |
| paired | quality | profile_loft | 3/3 | 0.720 -> 0.686 | 0.914 -> 0.933 | 0.896 -> 0.921 | 2 -> 2 |
| paired | quality | visual_hull_voxel | 3/3 | 0.511 -> 0.521 | 0.981 -> 0.986 | 0.908 -> 0.911 | 3 -> 3 |
| paired | quality | gaussian_ellipsoid_proxy | 3/3 | 0.359 -> 0.359 | 0.865 -> 0.865 | 0.828 -> 0.828 | 2 -> 2 |
| paired | quality | primitive_fit_refine | 2/3 | 0.237 -> 0.246 | 0.794 -> 0.777 | 0.819 -> 0.788 | 0 -> 0 |
| paired | quality | differentiable_refine | 3/3 | 0.303 -> 0.303 | 0.881 -> 0.881 | 0.889 -> 0.889 | 3 -> 3 |
| paired | quality | hybrid_loft_hull | 3/3 | 0.511 -> 0.521 | 0.942 -> 0.946 | 0.908 -> 0.911 | 3 -> 3 |
| paired | quality | shape_program | 3/3 | 0.576 -> 0.582 | 0.912 -> 0.907 | 0.960 -> 0.960 | 2 -> 2 |
| paired | quality | ensemble | 2/3 | 0.541 -> 0.546 | 0.983 -> 0.987 | 0.956 -> 0.961 | 2 -> 2 |
| paired | cpu_dvx | differentiable_refine | 3/3 | 0.303 -> 0.283 | 0.881 -> 0.848 | 0.889 -> 0.840 | 3 -> 0 |
| paired | cpu_dvx | ensemble | 2/3 | 0.541 -> 0.546 | 0.983 -> 0.987 | 0.956 -> 0.961 | 2 -> 2 |

Paired macros use only the same completed cases on both sides, with their denominators shown. Ordinary/quality differentiable refinement may be identical when the adapter consumes no effective quality change. CPU-DVX is an actual separate CPU helper at 32^3, 12 steps; reading generic switches does not prove proposal admission.

## Common shared-frame geometry

These values use the declared common target frame, distinct from the independent-bbox scores above. Native paper reproduction remains data-blocked. Every matched available pair is included.

| Split | Common profile | After preset | Technique | Matched n/planned | Shared-frame F.02 |
|---|---|---|---|---:|---:|
| paired | common_surface_v1 | quality | profile_loft | 3/3 | 0.648 -> 0.629 |
| paired | common_surface_v1 | quality | visual_hull_voxel | 3/3 | 0.432 -> 0.440 |
| paired | common_surface_v1 | quality | gaussian_ellipsoid_proxy | 3/3 | 0.294 -> 0.294 |
| paired | common_surface_v1 | quality | primitive_fit_refine | 2/3 | 0.210 -> 0.207 |
| paired | common_surface_v1 | quality | differentiable_refine | 3/3 | 0.282 -> 0.282 |
| paired | common_surface_v1 | quality | hybrid_loft_hull | 3/3 | 0.432 -> 0.440 |
| paired | common_surface_v1 | quality | shape_program | 3/3 | 0.560 -> 0.546 |
| paired | common_surface_v1 | quality | ensemble | 2/3 | 0.448 -> 0.457 |
| paired | common_surface_v1 | cpu_dvx | differentiable_refine | 3/3 | 0.282 -> 0.321 |
| paired | common_surface_v1 | cpu_dvx | ensemble | 2/3 | 0.448 -> 0.457 |
| paired | superflex_formula_common_v1 | quality | profile_loft | 3/3 | 0.648 -> 0.629 |
| paired | superflex_formula_common_v1 | quality | visual_hull_voxel | 3/3 | 0.432 -> 0.440 |
| paired | superflex_formula_common_v1 | quality | gaussian_ellipsoid_proxy | 3/3 | 0.294 -> 0.294 |
| paired | superflex_formula_common_v1 | quality | primitive_fit_refine | 2/3 | 0.210 -> 0.207 |
| paired | superflex_formula_common_v1 | quality | differentiable_refine | 3/3 | 0.282 -> 0.282 |
| paired | superflex_formula_common_v1 | quality | hybrid_loft_hull | 3/3 | 0.432 -> 0.440 |
| paired | superflex_formula_common_v1 | quality | shape_program | 3/3 | 0.560 -> 0.546 |
| paired | superflex_formula_common_v1 | quality | ensemble | 2/3 | 0.448 -> 0.457 |
| paired | superflex_formula_common_v1 | cpu_dvx | differentiable_refine | 3/3 | 0.282 -> 0.321 |
| paired | superflex_formula_common_v1 | cpu_dvx | ensemble | 2/3 | 0.448 -> 0.457 |
| heldout | common_surface_v1 | quality | profile_loft | 3/3 | 0.666 -> 0.758 |
| heldout | common_surface_v1 | quality | visual_hull_voxel | 3/3 | 0.620 -> 0.628 |
| heldout | common_surface_v1 | quality | gaussian_ellipsoid_proxy | 3/3 | 0.231 -> 0.231 |
| heldout | common_surface_v1 | quality | primitive_fit_refine | 3/3 | 0.356 -> 0.291 |
| heldout | common_surface_v1 | quality | differentiable_refine | 3/3 | 0.302 -> 0.302 |
| heldout | common_surface_v1 | quality | hybrid_loft_hull | 3/3 | 0.620 -> 0.712 |
| heldout | common_surface_v1 | quality | shape_program | 3/3 | 0.785 -> 0.797 |
| heldout | common_surface_v1 | quality | ensemble | 3/3 | 0.620 -> 0.712 |
| heldout | common_surface_v1 | cpu_dvx | differentiable_refine | 3/3 | 0.302 -> 0.370 |
| heldout | common_surface_v1 | cpu_dvx | ensemble | 3/3 | 0.620 -> 0.712 |
| heldout | superflex_formula_common_v1 | quality | profile_loft | 3/3 | 0.666 -> 0.758 |
| heldout | superflex_formula_common_v1 | quality | visual_hull_voxel | 3/3 | 0.620 -> 0.628 |
| heldout | superflex_formula_common_v1 | quality | gaussian_ellipsoid_proxy | 3/3 | 0.231 -> 0.231 |
| heldout | superflex_formula_common_v1 | quality | primitive_fit_refine | 3/3 | 0.356 -> 0.291 |
| heldout | superflex_formula_common_v1 | quality | differentiable_refine | 3/3 | 0.302 -> 0.302 |
| heldout | superflex_formula_common_v1 | quality | hybrid_loft_hull | 3/3 | 0.620 -> 0.712 |
| heldout | superflex_formula_common_v1 | quality | shape_program | 3/3 | 0.785 -> 0.797 |
| heldout | superflex_formula_common_v1 | quality | ensemble | 3/3 | 0.620 -> 0.712 |
| heldout | superflex_formula_common_v1 | cpu_dvx | differentiable_refine | 3/3 | 0.302 -> 0.370 |
| heldout | superflex_formula_common_v1 | cpu_dvx | ensemble | 3/3 | 0.620 -> 0.712 |
| external | common_surface_v1 | quality | profile_loft | 8/8 | 0.525 -> 0.559 |
| external | common_surface_v1 | quality | visual_hull_voxel | 8/8 | 0.561 -> 0.585 |
| external | common_surface_v1 | quality | gaussian_ellipsoid_proxy | 8/8 | 0.348 -> 0.348 |
| external | common_surface_v1 | quality | primitive_fit_refine | 7/8 | 0.318 -> 0.350 |
| external | common_surface_v1 | quality | differentiable_refine | 8/8 | 0.356 -> 0.356 |
| external | common_surface_v1 | quality | hybrid_loft_hull | 8/8 | 0.561 -> 0.608 |
| external | common_surface_v1 | quality | shape_program | 8/8 | 0.545 -> 0.572 |
| external | common_surface_v1 | quality | ensemble | 8/8 | 0.565 -> 0.610 |
| external | common_surface_v1 | cpu_dvx | differentiable_refine | 8/8 | 0.356 -> 0.466 |
| external | common_surface_v1 | cpu_dvx | ensemble | 7/8 | 0.598 -> 0.649 |
| external | superflex_formula_common_v1 | quality | profile_loft | 8/8 | 0.525 -> 0.559 |
| external | superflex_formula_common_v1 | quality | visual_hull_voxel | 8/8 | 0.561 -> 0.585 |
| external | superflex_formula_common_v1 | quality | gaussian_ellipsoid_proxy | 8/8 | 0.348 -> 0.348 |
| external | superflex_formula_common_v1 | quality | primitive_fit_refine | 7/8 | 0.318 -> 0.350 |
| external | superflex_formula_common_v1 | quality | differentiable_refine | 8/8 | 0.356 -> 0.356 |
| external | superflex_formula_common_v1 | quality | hybrid_loft_hull | 8/8 | 0.561 -> 0.608 |
| external | superflex_formula_common_v1 | quality | shape_program | 8/8 | 0.545 -> 0.572 |
| external | superflex_formula_common_v1 | quality | ensemble | 8/8 | 0.565 -> 0.610 |
| external | superflex_formula_common_v1 | cpu_dvx | differentiable_refine | 8/8 | 0.356 -> 0.466 |
| external | superflex_formula_common_v1 | cpu_dvx | ensemble | 7/8 | 0.598 -> 0.649 |

All common-profile cells: 247 available and five unavailable per profile. All six paper-native tracks retain unavailable cells and their data requirements in JSON.

## Ensemble cold/warm costs

Warm means one persistent interpreter/import state with a fresh scene and fresh geometry caches. Startup, novel rendering and extra geometry evaluation are excluded from the reconstruction/validation timing below and retained separately in receipts. Three repetitions justify per-case median/range, not population latency or p95. Backend/output variation is explicit.

| Case | Preset | Cold s | Warm n | Warm median [min,max] s | Complete | Warm selected backends | Identical output signatures |
|---|---|---:|---:|---|---|---|---|
| box | default | 14.15 | 3 | 12.66 [12.63, 12.72] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| box | quality | 40.71 | 3 | 37.38 [36.93, 37.65] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| box | cpu_dvx | 39.24 | 3 | 38.89 [37.20, 39.61] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| vase | default | 15.13 | 3 | 13.87 [13.61, 15.86] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| vase | quality | 49.01 | 2 | 46.42 [44.08, 48.77] | False | visual_hull_voxel, visual_hull_voxel | True |
| vase | cpu_dvx | 47.28 | 0 | unavailable [unavailable, unavailable] | False |  | True |
| torus | default | 14.91 | 3 | 13.46 [12.85, 15.56] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| torus | quality | 47.56 | 3 | 44.80 [44.60, 44.96] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| torus | cpu_dvx | 46.47 | 3 | 47.58 [45.72, 48.08] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| cylinder | default | 14.23 | 3 | 12.57 [12.47, 12.88] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| cylinder | quality | 36.57 | 3 | 35.17 [34.78, 35.67] | True | profile_loft, profile_loft, profile_loft | True |
| cylinder | cpu_dvx | 36.12 | 3 | 34.50 [34.36, 34.63] | True | profile_loft, profile_loft, profile_loft | True |
| bottle | default | 14.07 | 3 | 12.68 [12.47, 13.05] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| bottle | quality | 37.86 | 3 | 38.66 [36.91, 40.22] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| bottle | cpu_dvx | 41.47 | 3 | 38.35 [37.43, 41.11] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| chair | default | 13.17 | 3 | 11.83 [11.81, 17.99] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| chair | quality | 34.64 | 3 | 32.62 [32.59, 36.51] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| chair | cpu_dvx | 33.45 | 3 | 33.02 [32.35, 35.71] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext01_3b4702e | default | 13.58 | 3 | 11.76 [11.75, 11.81] | True | silhouette_intersection, silhouette_intersection, silhouette_intersection | False |
| ext01_3b4702e | quality | 35.43 | 3 | 33.87 [33.12, 33.87] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext01_3b4702e | cpu_dvx | 35.93 | 3 | 33.45 [32.78, 33.68] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext02_963c5f8 | default | 14.46 | 3 | 12.83 [12.66, 12.95] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext02_963c5f8 | quality | 42.61 | 3 | 37.88 [36.94, 40.86] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext02_963c5f8 | cpu_dvx | 44.79 | 3 | 41.30 [38.06, 43.62] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext03_8a5c0fd | default | 19.89 | 3 | 15.57 [15.52, 15.91] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext03_8a5c0fd | quality | 42.05 | 3 | 36.98 [36.22, 39.45] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext03_8a5c0fd | cpu_dvx | 42.37 | 3 | 35.46 [35.42, 39.01] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext04_94f6eb4 | default | 14.92 | 3 | 12.36 [12.27, 12.49] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext04_94f6eb4 | quality | 42.61 | 3 | 40.57 [34.67, 40.90] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext04_94f6eb4 | cpu_dvx | 39.22 | 3 | 38.41 [38.19, 38.57] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext05_1b41e3e | default | 15.69 | 3 | 14.06 [13.97, 14.67] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext05_1b41e3e | quality | 51.09 | 3 | 48.17 [47.97, 48.29] | True | silhouette_intersection, silhouette_intersection, silhouette_intersection | False |
| ext05_1b41e3e | cpu_dvx | 49.41 | 3 | 47.87 [47.80, 47.93] | True | silhouette_intersection, silhouette_intersection, silhouette_intersection | False |
| ext06_0c3ca2b | default | 14.64 | 3 | 13.41 [13.15, 15.52] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext06_0c3ca2b | quality | 36.70 | 3 | 27.65 [27.30, 29.55] | True | profile_loft, profile_loft, profile_loft | True |
| ext06_0c3ca2b | cpu_dvx | 34.44 | 3 | 34.42 [33.48, 34.48] | True | profile_loft, profile_loft, profile_loft | True |
| ext07_01fe14f | default | 12.46 | 3 | 11.41 [11.21, 13.66] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext07_01fe14f | quality | 44.35 | 3 | 43.92 [41.24, 46.90] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext07_01fe14f | cpu_dvx | 46.27 | 0 | unavailable [unavailable, unavailable] | False |  | True |
| ext08_186ecaa | default | 17.33 | 3 | 16.01 [15.54, 26.15] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext08_186ecaa | quality | 47.72 | 3 | 42.64 [42.23, 48.71] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |
| ext08_186ecaa | cpu_dvx | 43.91 | 3 | 39.86 [39.51, 40.60] | True | visual_hull_voxel, visual_hull_voxel, visual_hull_voxel | True |

Complete three-repetition timing ratios and all candidate admission reasons are in the JSON. Many ensembles return a hull/loft and skip refinement/program candidates for budget; those times cannot describe DVX throughput or evaluation of the full nominal candidate pool. The oracle standalone result may exceed the ensemble selection, but GT never drives selection.

## Failures and coordinated validation

| Job | Exit/status | Child wall s |
|---|---|---:|
| vase-quality-ensemble | 2 | 212.03 |
| vase-cpu_dvx-ensemble | 2 | 106.44 |
| torus-quality-primitive_fit_refine | 2 | 13.61 |
| ext03_8a5c0fd-quality-primitive_fit_refine | 2 | 16.26 |
| ext07_01fe14f-cpu_dvx-ensemble | 2 | 102.78 |
| native-full | 1 | 205.59 |
| saved-parts | 0 | 6.38 |
| exports-metres | 0 | 4.18 |
| exports-centimetres | 0 | 4.36 |
| candidate-exports | 2 | 886.29 |
| metrics | 0 | 192.80 |

Full native runner suite summary: 94 passed, 3 failed, 0 skipped.

Candidate output checks: 247 mesh-export receipts, 82 multipart parameter/edit-response receipts, 28 program artist receipts; 14 validation failures. Unavailable candidate entries: 5.

All five failures retain raw diagnostics: two quality primitive fits (torus and ext03) exhaust worker/queued-start bounds; three ensemble warm runs (vase quality, vase CPU-DVX and ext07 CPU-DVX) exhaust hull workers and remaining candidate routing budgets. The ext07 intersection also fails for insufficient unique contour points. Each saved no-mesh payload then triggers the harness KeyError(rendered_paths). Cold outputs and successful earlier repetitions remain available. No retries or source repairs changed this freeze.

Solid topology screens and actual boundary qualification are separate. A watertight/component screen does not establish no self-intersection. Open3D qualification has a 60000-triangle bound and configured timeout. OBJ/GLB checks measure world geometry; they do not certify UV/PBR appearance or cross-DCC behavior. Editable multipart sources do not imply one connected solid. DVX seed parameters do not reproduce final deformation. Qualified baked programs require recompile after artist edits.

Native DTU, released SuperFlex FPS/native frame and prepared SuperFit target/receipts remain data-blocked only for their own tracks. Common/formula metrics are not paper-native method reproduction. The eight external cases are a pinned capability subset, not a representative generalization sample. Blender 4.2/5.0, GPU performance, broad semantic artist certification and universal weak-case improvement are not done.

Bounded Open3D qualification: 46 outputs validated, 49 reported self-intersections, and 152 unavailable under input guards/60000-triangle bounds. Unavailable is not a pass. All 28 program artist receipts were measured; this is a scoped edit-response check, not broad semantic certification.

## Exhaustive scoped done/not-done list

| ID | Requirement | Final state | Evidence / remaining limit |
|---|---|---|---|
| 1.1 | Earlier baseline, input hashes, camera/evaluator and failure map | PRESERVED_COMPATIBILITY_CHECKED | All available historical comparisons verify exact inputs/cameras/GT/seed/fidelity/gates and normalized evaluator code; recorded harness/export changes declared. |
| 1.2 | New source and configuration preset freeze | FROZEN_VERIFIED | 533 source files and 108 inputs/provenance; original manifest retained; final hash audit {'source_sha256_map': True, 'source_tree': True, 'manifest_bytes': True, 'inputs': True} |
| 2.1 | Default Gaussian isolation and strong-method safeguards | SAMPLED_DEFAULT_STABILITY_VERIFIED | Ordinary hull and Gaussian legacy F.02 unchanged on all six synthetic controls; quality remains explicit. Global/default byte identity is not claimed. |
| 2.2 | Family adapters, exterior objective, changed-part reuse and analytic fitting | IMPLEMENTED_CHECKED | Earlier cbcf143 changes retained; final derivative/process/objective checks pass. |
| 2.3 | Camera-basis profile interval union and confidence | IMPLEMENTED_CHECKED | Actual part sections, pixel centers, offsets and known intervals. Confidence is evidence weight. Frozen matrix now measured with mixed gains/regressions; see per-cell ledger and admission receipts. |
| 2.4 | Non-ellipse footprints and finest finishing | IMPLEMENTED_CHECKED | Actual generated shapes, ellipse analytic gradients and other-shape numeric fallback; shared coarse/fine allowance. Frozen matrix now measured with mixed gains/regressions; see per-cell ledger and admission receipts. |
| 2.5 | Compiled/rendered program alternatives and parameter dedup | IMPLEMENTED_CHECKED | Geometry-scheduled program families are interleaved under the cap; actual compiled/rendered admission and parameter identity retained. Frozen matrix now measured with mixed gains/regressions; see per-cell ledger and admission receipts. |
| 2.6 | Add/split/mirror/repeat/subtract and best-program refinement | IMPLEMENTED | Actual positioned instances and known-empty-supported cuts; true geometry admission. No semantic learning claim. Frozen matrix now measured with mixed gains/regressions; see per-cell ledger and admission receipts. |
| 2.7 | Adaptive loft/contour sections and hierarchical thin hull | IMPLEMENTED_CHECKED | Neck/thin-strip contracts pass; boundary proposals use geometry/solid guards. Matrix pending. Frozen matrix now measured with mixed gains/regressions; see per-cell ledger and admission receipts. |
| 2.8 | Connected partitions/regrowth/fill, mixed families, merge/prune | IMPLEMENTED_CHECKED | Separate missing-region seeds retained; complete shared-budget objective admits alternatives. Frozen matrix now measured with mixed gains/regressions; see per-cell ledger and admission receipts. |
| 2.9 | PCA cuboid hierarchy and adjacent priority queue merges | IMPLEMENTED_CHECKED | Input correspondence and complexity levels; every program is compiled/scored. Frozen matrix now measured with mixed gains/regressions; see per-cell ledger and admission receipts. |
| 2.10 | Spatial convex hull proxies and support/boundary merging | IMPLEMENTED_CHECKED | Native convex fixture passes; artist sources/correspondence retained. Broad quality pending. Frozen matrix now measured with mixed gains/regressions; see per-cell ledger and admission receipts. |
| 2.11 | Exact CPU DVX adapter/helper and compatibility | IMPLEMENTED_CHECKED_MEASURED | Existing pinned CPU-DVX helper runs real 32^3/12-step matrix; direct results, option reads and admitted output hashes retained. All 14 direct outputs lack editable initialization parameter artifacts (R3). Ensemble hull/loft timings are not DVX throughput. |
| 2.12 | Universal weak-case 3D improvement | NOT_DONE | Universal improvement rejected: box/source-method and quality primitive regressions plus failed bounded fitting attempts remain. CPU-DVX is mixed; no blanket promotion. |
| 3.1 | Shared persistent queue for ensemble and fitting starts | IMPLEMENTED_CHECKED_MEASURED | Actual frozen workers consume typed settings; queue deadlines and budget-skipped candidates retained. Increased nominal pool is not proof all candidates were evaluated. |
| 3.2 | Actual hybrid seed reuse and output metrics | IMPLEMENTED_CHECKED | Real hybrid reuses completed seeds and one resident proximity state across residual rounds; scoped updates and cleanup verified. |
| 3.3 | Current held-out cost, selection and performance | MEASURED_COST_QUALITY_TRADEOFF | Same-input heldout and eight external cases, complete three-warm-repetition pairs plus explicit partial groups. Quality ensemble costs more and can miss better standalone geometry. |
| 4.1 | Foreground/background/unknown, camera/crop/unit consistency | IMPLEMENTED_CHECKED | Image/validity/camera crop moves together; holes require known evidence. Actual 48x64 cropped render and unknown top exclusion in diagnostics and serialized evidence pass. |
| 4.2 | Current camera/unit/export replay campaign | CHECKED_WITH_FAILURES_OR_UNREACHED | Actual metre/centimetre fixtures and matrix output replay; job status and geometry errors retained. Crop/unknown-view native component fixture remains evidence. |
| 5.1 | Grouped balanced native union, qualified Manifold and thin-safe SDF | IMPLEMENTED_NATIVE_CHECKED | Actual per-operand Open3D Manifold qualification includes input/toolchain hashes and cost; queued production Manifold volume1.5, stale rejection, guarded128 SDF and thin/unknown rejection pass. |
| 5.2 | Balanced live CSG and retained artist sources | IMPLEMENTED_CHECKED | Selected program can replay positive union as a live Exact artist tree. Qualified numeric CSG is the baked render/export output and requires recompile after edits. |
| 5.3 | Resident GN query graph/mesh and bulk updates | IMPLEMENTED_NATIVE_CHECKED | Actual repeated hybrid proximity caller retains graph/query mesh and target object, bulk updates coordinates, rebuilds topology, releases failure resources; native fixture confirms identity and results. |
| 5.4 | New-path primitive cap orientation and polygon transport | IMPLEMENTED_CHECKED | Fixed at DVX/hybrid boundaries; coordinates and legacy default meshes preserved. |
| 5.5 | Output contract qualification, exports and semantic editability | CHECKED_WITH_FAILURES_OR_UNREACHED | 247 matrix export receipts; 82 multipart edit checks; 28 program artist receipts; 14 validation failures. 46 outputs validated by Open3D, 49 report self-intersections, 152 unavailable under guards/size bounds. Topology, bounded native qualification and semantic usability remain separate. |
| 5.6 | Blender4.2/5.0 compatibility | NOT_DONE | Only installed Blender5.2.2 LTS exercised; no additional Blender install. |
| 6.1 | External/native metric adapters and bounded prior Poisson | COMMON_MEASURED_NATIVE_DATA_BLOCKED | Selected common profiles consume same saved outputs; native DTU/SuperFlex/SuperFit data absent only for own tracks. Prior three actual Poisson meshes nonwatertight, unpromoted. |
| 6.2 | Eight semantic external GLB inputs | PREPARED_AND_MEASURED | All eight original GLBs/material/axis/UV reviews and calibrated inputs retained; all attempted cells and failures reported. Capability subset only. |
| 6.3 | Native DTU inputs and camera/culling/preparation receipts | DATA_BLOCKED_NATIVE_ONLY | Native DTU arrays/frame/camera/mask and verified culling receipts absent. Released SuperFlex FPS/native frame and SuperFit prepared-target receipts also absent; common campaign proceeds independently. |
| 6.4 | Broad real-world/generalization/paper/artist certification | NOT_DONE | A capability subset and six synthetic cases cannot certify representative population results. |
| 7.1 | Local commits, preserved failures and owned process cleanup | DONE_WITH_FAILURES_RETAINED | Local implementation and report commits; original evidence preserved; owned campaign children joined and scoped final activity captured; no pushes/installs/downloads/unrelated process termination. |
| 7.2 | One coordinated campaign after implementation | ONE_CAMPAIGN_COMPLETE | {'planned': 252, 'completed': 247, 'process_failed': 5, 'unreached': 0, 'render_gate_passes': 171}; reconstruction counts {'planned': 378, 'completed': 369, 'process_failed': 5, 'unreached': 4}; six validation jobs retained separately. No rerun after freeze. |

## Source work remaining after frozen validation

- R1 Failure-safe campaign harness (NOT_DONE): Five bounded reconstruction/repetition failures then KeyError(rendered_paths); preserve structured failure rows without assuming renders.
- R2 Quality preset admission/iteration budgets (NOT_DONE): Quality fitting regresses and some starts/hulls exhaust worker/queue allowances. No empirically justified plateau/grid/steps sweep; increasing bounds was not tested.
- R3 DVX editable initialization artifacts (NOT_DONE): Direct DVX geometry exports but selected multipart parameter sources are missing in candidate validation.
- R4 Three full native suite regressions (NOT_DONE): Native receipt-gate error-text expectation; grammar sidecar returns error; differentiable empty-camera max() and fake dependency_state error handler. No source repair/rerun after freeze.
- R5 Solid geometry qualification failures (NOT_DONE): Open3D reports self-intersections on 49 tested outputs; 152 are unavailable under guards/size bounds. Watertight topology and exports cannot certify these as solids.

The three failed runner suites contain four assertion failures and one error. Individual failing/error cases:
- test_boolean_exact_and_manifold_receipt_gate (test_native_geometry.NativeGeometryTests.test_boolean_exact_and_manifold_receipt_gate)
- test_shape_grammar_search_is_deterministic (test_moonshot_sidecars.MoonshotSidecarTests.test_shape_grammar_search_is_deterministic)
- test_nvdiffrast_available_backend_routes_through_renderer (test_differentiable_render.TestDifferentiableRender.test_nvdiffrast_available_backend_routes_through_renderer)
- test_cpu_soft_silhouette_backend_emits_artifacts_and_metrics (test_differentiable_render.TestDifferentiableRender.test_cpu_soft_silhouette_backend_emits_artifacts_and_metrics)
- test_objective_regression_can_fail_strict_candidate (test_differentiable_render.TestDifferentiableRender.test_objective_regression_can_fail_strict_candidate)

## Retained evidence

Committed summaries: [machine-readable results](quality-final-results.json), [every primary cell](quality-final-cells.csv), [all 378 planned reconstructions](quality-final-reconstructions.csv). Frozen source/input handoff: [quality-connections.md](quality-connections.md), [manifest](quality-campaign-manifest.json). Earlier checkpoints keep their original revisions.

Full local evidence: temp/quality-final-campaign-20261006 (validation.json, campaign-rows.json, performance/*, per-cell comparison/config/dispatch/result receipts, candidate-exports, metric-bundles and raw logs). The prepared original GLBs/material/axis reviews and calibrated references remain in temp/quality-campaign-inputs-20261006. Receipt-only analysis helpers do not reconstruct or resample geometry.

Owned campaign children exited/joined; scoped final activity receipt and source/input verification are retained. Windows is released for the hardware worker. No unrelated process was stopped.
