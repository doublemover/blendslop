# Ambitious Implementation Spec Pack

Source audit: `temp/perf-quality-audit-20260509/final/exhaustive-findings.md`.

Purpose: convert every performance and reconstruction-quality finding into implementation-grade specs. These specs deliberately do not settle for a narrow v1 patch set. They define foundations that can support robust silhouette processing, loft reconstruction, exact-ish visual hulls, topology-aware meshing, primitive fitting, and objective quality gates.

## Operating Principles

- Treat correctness and measurement as foundation work, not cleanup.
- Prefer typed data contracts over tuples and ad hoc dictionaries.
- Preserve every diagnostic needed to reproduce a reconstruction: input masks, extracted constraints, config, stage timings, mesh topology, rendered masks, metric details, and fallback decisions.
- Make risky techniques opt-in but real. Avoid wrapper-only modes that dispatch into the same old behavior.
- Do not silently degrade from an ambitious mode into a weak fallback. Return structured failure metadata and make manifests say what happened.
- Use `docs/IMPLEMENTATION_SPEC.md:3-18` as historical context, but update it when implementation starts because it currently excludes full visual hull and ML-style reconstruction at `docs/IMPLEMENTATION_SPEC.md:15-16` while this spec pack intentionally creates foundations for those directions.

## Spec Files

| File | Primary findings | Core code areas |
| --- | --- | --- |
| `01-silhouette-metrics-foundation.md` | F02, F03, F04, F05, F06, F20, F22, F23, F24 | `geometry/silhouette.py`, `validation/silhouette_iou.py`, `test_e2e_validation.py`, `shape_matcher.py`, `image_processor.py`, `contour_analyzer.py`, manifests |
| `02-reconstruction-architecture.md` | F01, F07, F08, F09, F10, F13, F15 | `main_integration.py`, `mesh_generator.py`, `dual_profile.py`, `slicing.py`, `profile_extractor.py`, `primitive_placement.py`, `config.py` |
| `03-blender-mesh-render-ops.md` | F10, F11, F12, F13, F14, F15 | `profile_loft_mesh.py`, `render_utils.py`, `primitive_placement.py`, `test_runner.py`, Blender-only tests |
| `04-visual-hull-volume-meshing.md` | F08, F16, F17, F26, F27 | `multi_view/visual_hull.py`, future voxel mesh extraction, marching cubes, Poisson postprocess |
| `05-primitive-fitting-silhouette-refinement.md` | F07, F22, F25, F28 | `primitives/superfrustum.py`, `placement/resfitting.py`, future objective/refinement modules |
| `06-validation-benchmarks-artifacts.md` | F04, F05, F06, F11, F20, F21, F22 | `test_runner.py`, `benchmarks/benchmark_perf.py`, `test_suite_quick.py`, `test_suite_multiview.py`, `test_ground_truth_iou.py` |
| `07-synthetic-shape-factory.md` | Cross-cutting fixture foundation | `create_test_images.py`, `test_suite_multiview.py`, `test_ground_truth_iou.py`, future `synthetic/` package |
| `08-candidate-ensemble-reconstruction.md` | Cross-cutting mode selection | `main_integration.py`, backend dispatch, manifests, scoring, metrics |
| `09-uncertainty-aware-reconstruction.md` | F02, F03, F07, F08, F16, F22, F24, F28 | masks, profile bands, visual hull occupancy, scoring |
| `10-differentiable-rendering-backend.md` | F22, F25, F28 | optional render/loss backend interface, finite-difference baseline, GPU plugin path |
| `11-gaussian-ellipsoid-primitive-research.md` | F07, F16, F22, F25, F26, F28 | Gaussian/ellipsoid primitive cloud research backend |
| `12-sparse-volume-interchange.md` | F16, F17, F21, F26, F27 | future `volume/` package, sparse hash grids, optional OpenVDB adapter |
| `13-human-correction-constraints.md` | F02, F03, F07, F08, F22, F24, F28 | future `constraints/` package, scribbles, dimensions, symmetry, axes |
| `14-topology-aware-objectives.md` | F04, F05, F08, F11, F14, F15, F21, F22, F26 | mesh QA, topology scoring, optimization penalties |
| `15-backend-plugin-contract.md` | F08, F09, F16, F20, F25, F26, F28 | future `reconstruction/backend.py`, registry, capabilities, plugin-safe dispatch |

## Finding Coverage

| Finding | Spec ownership |
| --- | --- |
| F01 bbox tuple mismatch | `02-reconstruction-architecture.md` |
| F02 noisy IoU crop | `01-silhouette-metrics-foundation.md` |
| F03 luma polarity inversion | `01-silhouette-metrics-foundation.md` |
| F04 per-view IoU hidden by average | `01-silhouette-metrics-foundation.md`, `06-validation-benchmarks-artifacts.md` |
| F05 debug artifacts not scoped/linked | `01-silhouette-metrics-foundation.md`, `06-validation-benchmarks-artifacts.md` |
| F06 quick-suite quality regressions do not fail | `06-validation-benchmarks-artifacts.md` |
| F07 one-width-per-height profiles | `02-reconstruction-architecture.md`, `05-primitive-fitting-silhouette-refinement.md` |
| F08 brittle front/side silhouette intersection | `02-reconstruction-architecture.md`, `04-visual-hull-volume-meshing.md` |
| F09 mode schema drift | `02-reconstruction-architecture.md` |
| F10 legacy slice booleans scale poorly | `02-reconstruction-architecture.md`, `03-blender-mesh-render-ops.md` |
| F11 orphaned Blender tests | `03-blender-mesh-render-ops.md`, `06-validation-benchmarks-artifacts.md` |
| F12 render config not applied | `03-blender-mesh-render-ops.md` |
| F13 radial segment spec drift | `02-reconstruction-architecture.md`, `03-blender-mesh-render-ops.md` |
| F14 silent voxel/simple fallback | `03-blender-mesh-render-ops.md` |
| F15 degenerate loft topology | `02-reconstruction-architecture.md`, `03-blender-mesh-render-ops.md` |
| F16 dense visual hull cubic cost | `04-visual-hull-volume-meshing.md` |
| F17 surface voxel triple-loop fallback | `04-visual-hull-volume-meshing.md`, `06-validation-benchmarks-artifacts.md` |
| F18 canonicalization copy/cache overhead | `01-silhouette-metrics-foundation.md`, `06-validation-benchmarks-artifacts.md` |
| F19 tuple-list profile overhead | `02-reconstruction-architecture.md`, `06-validation-benchmarks-artifacts.md` |
| F20 benchmarks not gated | `06-validation-benchmarks-artifacts.md` |
| F21 slow/fragile voxel ground truth | `06-validation-benchmarks-artifacts.md` |
| F22 IoU-only metric gaps | `01-silhouette-metrics-foundation.md`, `06-validation-benchmarks-artifacts.md` |
| F23 canonicalize config ignored | `01-silhouette-metrics-foundation.md` |
| F24 legacy Canny contour path | `01-silhouette-metrics-foundation.md` |
| F25 SuperFrustum/ResFit disconnected | `05-primitive-fitting-silhouette-refinement.md` |
| F26 topology-safe marching cubes | `04-visual-hull-volume-meshing.md` |
| F27 Poisson postprocess | `04-visual-hull-volume-meshing.md` |
| F28 differentiable-rendering-inspired loss | `05-primitive-fitting-silhouette-refinement.md` |

## Recommended Execution Order

1. Implement the silhouette and metric foundation first. This makes every later experiment measurable and debuggable.
2. Build the synthetic shape factory early. It supplies the fixtures, ground truth, adversarial masks, and benchmark suites needed to prove every ambitious path.
3. Refactor reconstruction contracts next: typed bounds, silhouette sets, profile bands, explicit modes, and hard failure metadata.
4. Add the backend plugin contract before adding more modes, so new work does not pile into `main_integration.py`.
5. Stabilize Blender mesh/render ops and wire all existing Blender tests into the runner.
6. Add candidate ensemble reconstruction so loft, silhouette intersection, visual hull, primitive fit, and research backends can compete through one scoring system.
7. Add uncertainty and human constraints so hard masks are not the only source of truth.
8. Build visual hull, sparse volume interchange, and topology-safe meshing as real ambitious modes, not hidden helpers.
9. Add primitive fitting, Gaussian/ellipsoid proxies, and differentiable-rendering-backed refinement on top of measured targets.
10. Gate everything through benchmark budgets, manifest artifacts, synthetic fixtures, and local Blender 5.0/4.2 validation.

## Ambitious Additions

The expanded specs add these foundations beyond the original audit findings:

- synthetic shape factory with deterministic mesh, mask, volume, SDF, and degradation generation,
- candidate ensemble reconstruction with explicit scoring and all-candidate artifacts,
- uncertainty-aware masks/profiles/voxels instead of early binary-only decisions,
- optional differentiable rendering backend interface with CPU/Blender baseline and GPU plugin path,
- Gaussian/ellipsoid primitive proxy research track inspired by 3D Gaussian Splatting but focused on editable geometry,
- sparse volume interchange with pure-Python sparse hash grids and optional OpenVDB-oriented adapters,
- human correction constraints for scribbles, dimensions, symmetry, axes, centerlines, and known view roles,
- topology-aware objectives so silhouette IoU cannot reward broken geometry,
- backend plugin contract so future reconstruction modes are isolated, measurable, and replaceable.
