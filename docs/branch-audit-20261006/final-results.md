# Frozen improvement candidate: matched results

Historical checkpoint. Current implementation and validation status: [frozen final campaign results](quality-final-results.md). Measurements below retain their original source revisions and do not qualify the new candidate.

Current accepted continuation: [native performance and metric phase](native-phase.md). This document retains its earlier snapshot and measurements.


Control rows reuse the verified saved camera-control source. All final cells were rerun against one frozen source tree.

| Set | Technique | Silhouette IoU before → after | F-score before → after | Workflow seconds before → after | Gates |
|---|---|---:|---:|---:|---:|
| paired | profile_loft | 89.2% → 89.2% | 64.2% → 64.2% | 6.3 → 6.3 | 2/3 → 2/3 |
| paired | visual_hull_voxel | 98.1% → 98.1% | 51.1% → 51.1% | 22.0 → 23.3 | 3/3 → 3/3 |
| paired | gaussian_ellipsoid_proxy | 86.5% → 86.5% | 35.9% → 35.9% | 7.7 → 8.3 | 2/3 → 2/3 |
| paired | primitive_fit_refine | 70.7% → 85.2% | 17.1% → 30.6% | 27.0 → 23.3 | 0/3 → 1/3 |
| paired | differentiable_refine | 50.0% → 84.0% | 24.2% → 28.7% | 29.2 → 29.0 | 0/3 → 2/3 |
| paired | hybrid_loft_hull | 89.2% → 89.2% | 64.2% → 64.2% | 23.2 → 22.1 | 2/3 → 2/3 |
| paired | shape_program | 78.0% → 78.0% | 13.3% → 13.3% | 7.1 → 7.0 | 1/3 → 1/3 |
| paired | ensemble | 98.1% → 98.1% | 51.1% → 51.1% | 71.1 → 51.5 | 3/3 → 3/3 |
| heldout | profile_loft | 81.5% → 81.5% | 48.8% → 48.8% | 5.8 → 5.6 | 1/3 → 1/3 |
| heldout | visual_hull_voxel | 97.2% → 97.2% | 45.8% → 45.8% | 21.4 → 18.0 | 3/3 → 3/3 |
| heldout | gaussian_ellipsoid_proxy | 76.4% → 76.4% | 24.8% → 24.8% | 9.6 → 7.1 | 1/3 → 1/3 |
| heldout | primitive_fit_refine | 62.7% → 83.6% | 17.2% → 31.1% | 27.7 → 22.9 | 0/3 → 1/3 |
| heldout | differentiable_refine | 43.4% → 82.3% | 22.3% → 22.1% | 28.9 → 29.7 | 0/3 → 2/3 |
| heldout | hybrid_loft_hull | 81.5% → 81.5% | 48.8% → 48.8% | 19.4 → 17.9 | 1/3 → 1/3 |
| heldout | shape_program | 69.5% → 69.5% | 4.5% → 4.5% | 6.7 → 6.4 | 1/3 → 1/3 |
| heldout | ensemble | 97.2% → 97.2% | 45.8% → 45.8% | 65.0 → 39.5 | 3/3 → 3/3 |

Native intersection checks: 27/48 final meshes meet the stated single-solid structural screen.
Assemblies can remain useful editable parts while failing solid acceptance. Per-case failures and all regressions are in paired-cells.csv.

Geometry uses 8192 area-weighted samples, uniform bbox center/longest-extent normalization, no rotation or anisotropic fitting, and F-score tolerance .02. Cameras are fixed recorded 512×512 orthographic views.
Workflow times include reconstruction and observed-view validation, exclude startup, novel-view and 3D evaluation. Single-run timings are noisy; no statistical speed claim.

Orbit-45 views and reference geometry are evaluation only. Selection uses observed masks and recorded cameras. The ensemble admission budget is soft; each worker has a separate 100-second hard cap.
