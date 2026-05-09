# Spec 07: Synthetic Shape Factory

## Scope

This is the primary expansion spec. It creates a deterministic, parameterized, adversarial shape generator for proving reconstruction quality, performance, and robustness. The existing repo already has hand-written image and mesh fixtures, but they are not a general factory and they do not preserve enough ground-truth metadata for ambitious reconstruction work.

Related findings: F02, F03, F04, F05, F06, F07, F08, F10, F11, F13, F14, F15, F16, F20, F21, F22, F24, F25, F26, F28.

## Current Code To Modify

- `blender_blocking/create_test_images.py:20-140`: creates simple bottle, vase, and cube silhouettes as static helper images.
- `blender_blocking/test_suite_multiview.py:4-15`: describes a diverse mesh validation suite.
- `blender_blocking/test_suite_multiview.py:51-123`: hand-written `TEST_OBJECTS` list.
- `blender_blocking/test_suite_multiview.py:135-180`: `create_mesh` dispatches fixture types.
- `blender_blocking/test_suite_multiview.py:195-365`: hand-built vase, bottle, bowl, cup, table, chair, lamp, car, and dog fixtures.
- `blender_blocking/test_suite_multiview.py:367-402`: slow raycast voxelization for ground truth.
- `blender_blocking/test_suite_multiview.py:405-447`: turntable rendering.
- `blender_blocking/test_suite_multiview.py:450-524`: silhouette loading and visual-hull reconstruction from rendered views.
- `blender_blocking/test_suite_multiview.py:554-604`: end-to-end object test computes 3-view and 12-view IoU.
- `blender_blocking/test_ground_truth_iou.py:42-57`: creates one vase mesh.
- `blender_blocking/test_ground_truth_iou.py:60-148`: another voxelization implementation.
- `blender_blocking/integration/blender_ops/silhouette_render.py:270-295`: render settings application.
- `blender_blocking/integration/blender_ops/silhouette_render.py:333-368`: camera orbit/top setup and render frame.
- `blender_blocking/utils/generation_context.py:39-94`: run id, seed, timing metadata.
- `blender_blocking/utils/manifest.py:43-60`: manifest construction.
- `docs/IMPLEMENTATION_SPEC.md:262-278`: deterministic tests, E2E validation, and artifact expectations.

## Goals

- Generate families of known 3D objects with exact parameters and reproducible seeds.
- Render canonical orthographic references and noisy/ambiguous variants.
- Export mesh, volume, analytic SDF, silhouette masks, camera metadata, expected failure modes, and ground-truth quality thresholds.
- Cover easy, medium, hard, and adversarial cases.
- Make fixture generation available both inside Blender and in pure Python where possible.
- Produce artifacts that can drive candidate ensemble scoring, visual hull, primitive fitting, differentiable refinement, and regression gates.

## New Package Layout

Create:

```text
blender_blocking/synthetic/
  __init__.py
  registry.py
  specs.py
  random_params.py
  analytic_sdf.py
  blender_builders.py
  silhouette_render_jobs.py
  degradations.py
  ground_truth.py
  artifact_writer.py
  quality_targets.py
  cli.py
```

Test files:

```text
blender_blocking/test_synthetic_shape_specs.py
blender_blocking/test_synthetic_degradations.py
blender_blocking/test_synthetic_ground_truth.py
blender_blocking/test_synthetic_factory_blender.py
```

Generated outputs:

```text
test_output/synthetic/<run_id>/<shape_id>/
  spec.json
  mesh/
    source.blend
    ground_truth.obj
    ground_truth.glb
  volume/
    occupancy-r64.npz
    occupancy-r128.npz
    sdf-samples.npz
  views/
    front.png
    side.png
    top.png
    orbit_000.png
    orbit_030.png
  masks/
    clean/
    noisy/
    antialias/
    partial/
  metrics/
    expected.json
  manifest.json
```

## Core Data Model

```python
@dataclass(frozen=True)
class SyntheticShapeSpec:
    shape_id: str
    family: str
    seed: int
    parameters: Mapping[str, object]
    transforms: Mapping[str, object]
    materials: Mapping[str, object]
    intended_challenges: tuple[str, ...]
    symmetry: tuple[str, ...]
    known_dimensions: Mapping[str, float]
    expected_failure_modes: tuple[str, ...]

@dataclass(frozen=True)
class SyntheticViewSpec:
    view_name: str
    camera_type: Literal["orthographic", "perspective"]
    azimuth_deg: float
    elevation_deg: float
    roll_deg: float
    resolution: tuple[int, int]
    padding: float
    occluders: tuple[Mapping[str, object], ...]
    expected_visible_features: tuple[str, ...]

@dataclass(frozen=True)
class SyntheticArtifactSet:
    shape_spec_path: Path
    mesh_paths: Mapping[str, Path]
    volume_paths: Mapping[str, Path]
    image_paths: Mapping[str, Path]
    mask_paths: Mapping[str, Path]
    metric_paths: Mapping[str, Path]
    manifest_path: Path
```

Every artifact must be hashable and reproducible from `(shape_id, seed, generator_version)`.

## Shape Families

### 1. Analytic Primitives

Purpose: exact SDF/occupancy, simple sanity baselines.

- box/cube,
- sphere,
- ellipsoid,
- cylinder,
- cone/frustum,
- capsule,
- torus,
- rounded box,
- superquadric/superellipse lathe,
- tapered superquadric.

Required parameters:

- dimensions,
- bevel/rounding,
- taper,
- rotation,
- position,
- scale,
- topology expectation: watertight, genus, component count.

Use analytic occupancy wherever possible instead of raycast voxelization from `test_suite_multiview.py:367-402`.

### 2. Profile And Lathe Objects

Purpose: stress `loft_profile`, offsets, and profile bands.

- vase with neck/body/base,
- bottle with shoulders,
- bowl/cup with hollow rim,
- chess-pawn style stacked radii,
- asymmetric vase with shifted centerline,
- multi-lobe profile with row intervals that cannot be represented by one radius.

The factory should export the exact generating profile as JSON so reconstruction can compare extracted profile bands to ground truth.

### 3. Furniture And Thin Support Objects

Purpose: stress topology, visual hull, holes, thin structures, and fallback joins.

- table with four legs,
- chair with back and legs,
- stool,
- lamp with shade and stem,
- shelf/bookcase,
- arch/gate,
- ladder,
- sawhorse.

Expected failure modes should include silhouette ambiguity, thin-leg loss, internal overlap, non-manifold risk, and visual-hull concavity fill.

### 4. Vehicle And Mechanical Blockouts

Purpose: stress multi-part silhouettes and editability.

- car body with cabin and wheel gaps,
- truck,
- robot torso/limbs,
- wrench-like silhouette,
- pipe elbow,
- gear-like object.

These fixtures should include holes/gaps where `silhouette_intersection` and visual hull should beat simple loft.

### 5. Adversarial Silhouette Families

Purpose: directly test findings F02, F03, F07, F08, F15, F22, F24.

- off-center dark object on light background,
- object touching one border,
- single outlier pixel far from object,
- dust clusters,
- anti-aliased thin diagonal struts,
- holed silhouettes,
- disconnected foreground components,
- repeated zero-radius rings,
- tiny object in large canvas,
- full-canvas near-threshold background,
- ambiguous light-on-dark and dark-on-light pairs,
- front/side profiles that disagree deliberately.

These can be generated in pure Python without Blender.

### 6. Capture-Style Noise Families

Purpose: make segmentation and uncertainty handling real.

- JPEG artifacts,
- blur,
- paper sketch lines,
- partial occlusion,
- uneven lighting,
- low contrast,
- alpha premultiplication artifacts,
- transparent background with faint RGB noise,
- missing top view,
- rotated/tilted input for future non-axis-aligned handling.

Each degradation must store its parameters and expected effect on masks.

## Builder Backends

### Pure Python Builder

Use for:

- 2D adversarial masks,
- analytic occupancy grids,
- SDF samples,
- simple profile JSON,
- tests that can run here.

### Blender Builder

Use for:

- real mesh creation,
- bevels and booleans,
- turntable renders,
- mesh QA,
- local Blender 4.2/5.0 validation.

Do not duplicate shape logic between backends if it can be represented as parameters and evaluated in both.

## Ground Truth

Replace or augment raycast voxelization from `test_suite_multiview.py:367-402` and `test_ground_truth_iou.py:60-148`.

Ground truth levels:

- `analytic_exact`: SDF equation gives inside/outside.
- `mesh_evaluated`: Blender-evaluated mesh plus BVH/winding/raycast with ambiguity tracking.
- `rendered_silhouette`: clean orthographic mask from known view.
- `expected_ambiguous`: fixture deliberately lacks enough views for exact reconstruction.

Store:

- occupancy grids at multiple resolutions,
- SDF samples,
- surface point samples,
- face/vertex counts,
- topology stats,
- known limits, such as visual hull cannot recover a hidden concavity.

## Quality Targets

Each fixture defines thresholds by mode:

```json
{
  "shape_id": "table_thin_legs_seed_0004",
  "targets": {
    "profile_loft": {
      "front_area_iou_min": 0.80,
      "side_area_iou_min": 0.75,
      "known_limit": "topology will not preserve leg gaps"
    },
    "silhouette_intersection": {
      "front_area_iou_min": 0.90,
      "side_area_iou_min": 0.88,
      "non_manifold_edges_max": 0
    },
    "visual_hull_voxel": {
      "volume_iou_min": 0.72,
      "surface_chamfer_max": 0.08
    }
  }
}
```

Targets must distinguish:

- expected exactness,
- expected ambiguity,
- expected failure,
- research candidate.

## CLI

Add:

```powershell
python -m blender_blocking.synthetic.cli list
python -m blender_blocking.synthetic.cli generate --suite smoke --out test_output/synthetic
python -m blender_blocking.synthetic.cli generate --suite adversarial-silhouettes --count 200 --seed 1234
python -m blender_blocking.synthetic.cli validate-manifest test_output/synthetic/<run_id>
```

Blender:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python -m blender_blocking.synthetic.cli -- generate --suite blender-smoke
```

If Blender cannot execute `-m` cleanly in this repo, provide a thin `synthetic_runner.py` script.

## Suite Definitions

- `smoke`: 5 shapes, pure Python where possible.
- `quick-blender`: cube, vase, table, asymmetric lathe, holed object.
- `adversarial-silhouettes`: pure 2D masks only.
- `profile-band`: holes, multi-lobes, offsets, top constraints.
- `visual-hull`: multi-view calibrated fixtures.
- `primitive-fit`: superquadrics, SuperFrusta, compound primitives.
- `nightly-heavy`: randomized sweep across families.
- `paper-regression`: fixtures inspired by visual hull, marching cubes, superquadric, and differentiable rendering papers.

## Manifest Requirements

Extend `utils/manifest.py:43-60` for factory artifacts:

- generator version,
- seed,
- shape family,
- full parameter dictionary,
- artifact hashes,
- camera matrices,
- expected metrics,
- known ambiguity notes,
- degradation parameters,
- source code version,
- Blender version when generated in Blender.

## Acceptance

- Pure Python smoke suite generates deterministic masks and manifests.
- Blender smoke suite generates meshes, renders, masks, topology stats, and manifests.
- Re-running with the same seed gives identical shape specs and artifact hashes where deterministic.
- At least one fixture fails each known bad behavior from F02/F03/F04/F07/F15/F21/F22.
- Every ambitious backend has at least one fixture where it is expected to beat `profile_loft`, and one where it is expected to lose or be ambiguous.

## References

- Existing repo fixture generation: `blender_blocking/test_suite_multiview.py:51-365`.
- Tanks and Temples benchmark practice: `temp/perf-quality-audit-20260509/lane-e-validation-quality/paper-pdfs/tanks-and-temples-2017.pdf`.
- 3D-R2N2 voxel IoU: `temp/perf-quality-audit-20260509/lane-e-validation-quality/paper-pdfs/3d-r2n2-1604.00449.pdf`.
- Point-set distances: `temp/perf-quality-audit-20260509/lane-e-validation-quality/paper-pdfs/point-set-distances-iccv2021.pdf`.
- Visual hull concept: https://iris.polito.it/handle/11583/1401917.
