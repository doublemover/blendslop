# Spec 06: Validation, Benchmarks, Quality Gates, And Artifacts

## Scope

Owns F04, F05, F06, F11, F17, F20, F21, and F22. This spec makes performance and reconstruction quality hard to regress. It is the enforcement layer for all other specs.

## Current Code To Modify

- `blender_blocking/test_runner.py:35-54`: pure test registry does not include benchmarks.
- `blender_blocking/test_runner.py:131-223`: Blender test sections omit existing loft and voxel-remesh tests.
- `blender_blocking/test_runner.py:372-390`: summary pass/fail counts only tests, not perf/quality budgets.
- `blender_blocking/benchmarks/benchmark_perf.py:194-323`: visual hull and surface voxel benchmarks.
- `blender_blocking/benchmarks/benchmark_perf.py:1016-1251`: CLI benchmark registry and dispatch.
- `blender_blocking/test_suite_quick.py:61-65`: writes `quick_test_results.json`.
- `blender_blocking/test_suite_quick.py:79-93`: reports averages and improvements.
- `blender_blocking/test_suite_quick.py:103-107`: succeeds if objects complete, not if quality passes.
- `blender_blocking/test_suite_multiview.py:367-402`: voxelizes ground truth with per-voxel ray casts.
- `blender_blocking/test_suite_multiview.py:557-604`: computes 3-view/12-view IoU and improvement.
- `blender_blocking/test_ground_truth_iou.py:87-148`: raycast voxelization.
- `blender_blocking/test_ground_truth_iou.py:251-254`: voxel IoU.
- `blender_blocking/test_ground_truth_iou.py:365-372`: prints success/warning rather than integrating with a general gate.
- `docs/IMPLEMENTATION_SPEC.md:252-260`: manifest requirements.
- `docs/IMPLEMENTATION_SPEC.md:262-278`: deterministic tests and E2E validation.
- `docs/IMPLEMENTATION_SPEC.md:291-293`: acceptance criteria.

## Artifact Layout

All validation and benchmark runs should produce a run-scoped directory:

```text
test_output/runs/<run_id>/
  manifest.json
  config.json
  environment.json
  stages.json
  metrics/
    silhouette.json
    topology.json
    performance.json
    quality-budget.json
  images/
    raw/
    masks/
    canonical/
    diffs/
  meshes/
    baseline/
    candidate/
  logs/
```

The manifest from `docs/IMPLEMENTATION_SPEC.md:252-260` should expand to include:

- git commit and dirty status if available,
- Blender version,
- Python version,
- dependency versions,
- input artifact hashes,
- config hash,
- per-stage timings,
- peak memory where measurable,
- selected reconstruction mode and fallback/degradation status,
- links to metrics and visual artifacts.

## Quality Gates

Area IoU is insufficient. Add gates:

- per-view canonical area IoU,
- per-view Boundary IoU,
- average IoU as summary only,
- required-view all-pass gate,
- empty-mask diagnostics,
- topology gates: loose vertices, non-manifold edges, disconnected components, watertightness,
- mesh complexity gates: vertices/faces within configured budgets,
- synthetic 3D IoU where ground truth is reliable,
- optional Chamfer/Hausdorff/F-score for point/mesh comparisons.

Quality gates should be expressed in JSON:

```json
{
  "fixture": "asymmetric_vase_front_side_top",
  "required_views": ["front", "side", "top"],
  "thresholds": {
    "area_iou_min": 0.82,
    "boundary_iou_min": 0.70,
    "non_manifold_edges_max": 0,
    "loose_vertices_max": 0
  }
}
```

## Performance Gates

Use `benchmark_perf.py` as a starting point, but make budgets enforceable:

- Add `--budget-json`.
- Add `--compare-baseline`.
- Add `--fail-on-regression`.
- Store machine/environment metadata to avoid comparing incompatible runs blindly.
- Support absolute thresholds and percent regression thresholds.

Benchmarks to add:

- canonicalization cache warm/cold by size,
- silhouette extraction by image size and cleanup policy,
- luma candidate scoring matrix,
- vectorized surface voxel fallback vs SciPy erosion,
- dense/chunked/adaptive visual hull memory and throughput,
- marching cubes extraction,
- BMesh vs `mesh.from_pydata` loft construction in Blender,
- join strategies by slice count,
- render engine/sample timing,
- ResFit optimization quality per second.

## Ground Truth Refactor

The current voxel ground truth in `test_suite_multiview.py:367-402` and `test_ground_truth_iou.py:87-148` is slow and fragile for open/thin geometry.

Implement a tiered ground-truth strategy:

1. Analytic fixtures:
   - sphere,
   - box,
   - cylinder,
   - ellipsoid,
   - torus-like explicit SDF if needed.
   - Occupancy comes from analytic equations, not ray casts.

2. Blender evaluated mesh fixtures:
   - use a robust BVH/winding or multi-direction ray approach,
   - record ambiguity near surfaces,
   - cache voxel grids by mesh hash, bounds, and resolution.

3. Surface point fixtures:
   - sample surfaces and compare Chamfer/Hausdorff/F-score.

Ground truth artifacts must be stored under run-scoped directories with hashes, not overwritten fixed files.

## Test Runner Integration

Update `test_runner.py`:

- Add existing Blender tests from Spec 03.
- Add optional benchmark phase:
  - off by default for quick pure tests,
  - enabled with `--bench`,
  - budget-enforced in CI if configured.
- Add quality-gate phase for fixtures:
  - pure fixtures can run outside Blender,
  - Blender fixtures skipped outside Blender with explicit reason.
- Exit nonzero on quality budget failures, not just exceptions.

## CI Strategy

CI should have lanes:

- pure tests,
- pure benchmarks with small budgets,
- Blender 4.2 full tests,
- Blender 5.0 full tests,
- optional nightly heavy quality suite.

Do not make heavyweight visual hull or primitive optimization mandatory for every commit. Instead:

- smoke fixtures run per PR,
- heavy fixtures run nightly or manually,
- budget files are versioned when source implementation begins.

## Acceptance

- Running quick tests can fail on quality regression, not just exceptions.
- Per-view IoU failure with average pass exits nonzero.
- Benchmark budget regression exits nonzero when enabled.
- Two E2E runs produce distinct artifact dirs and manifests.
- Existing orphaned Blender tests appear in runner output.
- Ground-truth cache is keyed and reproducible.

## Commands

Pure Python:

```powershell
python blender_blocking/test_runner.py --quick --no-progress
python blender_blocking/benchmarks/benchmark_perf.py --all --no-progress --iterations 200 --repeat 1 --resolution 32 --json temp/perf-quality-audit-20260509/benchmarks/pure-all-r32.json
```

Blender local:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python blender_blocking/test_runner.py
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python blender_blocking/test_runner.py -- --quick
& "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" --background --python blender_blocking/test_runner.py
```

## Papers And References

- Tanks and Temples benchmark discipline: `temp/perf-quality-audit-20260509/lane-e-validation-quality/paper-pdfs/tanks-and-temples-2017.pdf`.
- Boundary IoU: `temp/perf-quality-audit-20260509/lane-e-validation-quality/paper-pdfs/boundary-iou-cvpr2021.pdf`.
- 3D-R2N2 voxel IoU context: `temp/perf-quality-audit-20260509/lane-e-validation-quality/paper-pdfs/3d-r2n2-1604.00449.pdf`.
- Point-set distances: `temp/perf-quality-audit-20260509/lane-e-validation-quality/paper-pdfs/point-set-distances-iccv2021.pdf`.
- Open3D metrics: https://www.open3d.org/docs/release/python_api/open3d.t.geometry.Metric.html.
- Python `perf_counter`: https://docs.python.org/3/library/time.html#time.perf_counter.
