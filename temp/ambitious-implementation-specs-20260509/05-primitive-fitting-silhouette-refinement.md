# Spec 05: Primitive Fitting And Silhouette-Loss Refinement

## Scope

Owns F07, F22, F25, and F28. This spec turns existing SuperFrustum and ResFit code into an ambitious refinement layer. The goal is not to add an ML dependency. The goal is to use objective silhouette and surface losses to fit editable primitives and improve reconstruction quality after profile or hull initialization.

## Current Code To Modify

- `blender_blocking/primitives/superfrustum.py:19-46`: `SuperFrustum` primitive and parameters.
- `blender_blocking/primitives/superfrustum.py:400-421`: serialization helpers.
- `blender_blocking/placement/resfitting.py:29-64`: `ResidualFitter` initialization and state.
- `blender_blocking/placement/resfitting.py:69-188`: primitive initialization.
- `blender_blocking/placement/resfitting.py:190-529`: primitive optimization and residual computation.
- `blender_blocking/placement/resfitting.py:529-553`: `compute_residual_error`.
- `blender_blocking/placement/resfitting.py:565-611`: residual primitive addition.
- `blender_blocking/placement/resfitting.py:611-738`: iterative `fit`.
- `blender_blocking/benchmarks/benchmark_perf.py:693-933`: ResFit benchmark functions.
- `docs/IMPLEMENTATION_SPEC.md:233-249`: current objective is canonical IoU only.
- `docs/IMPLEMENTATION_SPEC.md:15-16`: old scope excludes ML reconstruction; this spec keeps refinement deterministic but uses differentiable-rendering papers as loss-design references.

## Target Role

Primitive fitting is a second-stage optimizer. It should not be fed raw images directly. It consumes:

- `ReconstructionTarget` from Spec 02,
- profile-band loft mesh,
- visual-hull occupancy or mesh from Spec 04,
- metrics and canonical render artifacts from Spec 01.

It produces:

- editable primitive set,
- optional refined mesh,
- loss history,
- residual heatmaps,
- per-view silhouette metrics before/after,
- manifest entries for every iteration or checkpoint.

## Objective Function

Create `reconstruction/objectives.py`:

```python
@dataclass(frozen=True)
class ReconstructionLossWeights:
    area_iou: float
    boundary_iou: float
    silhouette_sdf: float
    surface_chamfer: float
    visual_hull_occupancy: float
    primitive_count: float
    overlap_penalty: float
    topology_penalty: float

@dataclass(frozen=True)
class ReconstructionLossResult:
    total: float
    terms: Mapping[str, float]
    per_view: Mapping[str, SilhouetteMetricResult]
    warnings: tuple[str, ...]
```

Loss terms:

- silhouette area IoU loss from `validation/silhouette_iou.py:212-228`,
- Boundary IoU from Spec 01,
- signed-distance silhouette loss using canonical masks,
- surface Chamfer/Hausdorff/F-score when ground truth or visual-hull surface points exist,
- occupancy consistency against visual hull,
- primitive parsimony,
- primitive overlap penalty,
- topology/mesh QA penalty from Spec 03.

This makes Kato/SoftRas ideas useful without importing a neural stack.

## Primitive Families

Extend beyond current SuperFrustum:

- SuperFrustum: keep current SDF path.
- Superquadric: add deterministic implementation from Solina/Bajcsy and Liu et al.
- Ellipsoid/capsule/cylinder/cone: simple baselines for sanity and editability.
- Mesh proxy primitive: optional for segments that cannot fit parametric shapes.

Each primitive must implement:

```python
class FittablePrimitive(Protocol):
    def sdf_batch(self, points: np.ndarray) -> np.ndarray: ...
    def sample_surface(self, n: int) -> np.ndarray: ...
    def to_mesh(self, resolution: int) -> MeshBuildResult: ...
    def to_dict(self) -> dict: ...
```

## Refactor ResFit

`placement/resfitting.py` currently mixes initialization, optimization, residual computation, and primitive addition. Split it:

- `placement/resfit_initialization.py`: seed primitives from profile bands, hull components, or mesh segmentation.
- `placement/resfit_objective.py`: objective evaluation and loss decomposition.
- `placement/resfit_optimizer.py`: deterministic optimizers.
- `placement/resfit_pipeline.py`: orchestration, checkpointing, manifest output.

Requirements:

- Keep vectorized SDF paths as the default.
- Bound memory for target point sets with sampling strategies.
- Record optimization history, per-term losses, primitive parameters, and timings.
- Add constraints for stable parameters: scale bounds, orientation bounds, exponent bounds, overlap bounds.
- Do not use finite-difference gradients everywhere if analytic or batched approximations are available.

## Silhouette-Loss Refinement

Implement a non-ML silhouette optimizer first:

1. Render current primitive set from required views.
2. Canonicalize rendered masks with Spec 01.
3. Compute per-view area IoU, Boundary IoU, and silhouette-distance losses.
4. Update primitive/profile parameters with derivative-free optimizer or finite-difference over a bounded parameter subset.
5. Stop on plateau, regression, or budget exhaustion.

Candidate parameter subsets:

- profile offsets,
- slice radii,
- superellipse exponents,
- primitive centers/scales,
- primitive taper parameters,
- limited mesh vertex radial offsets for loft rings.

Do not adopt a full neural differentiable renderer until the deterministic objective proves useful.

## Integration Modes

Add mode from Spec 02:

- `primitive_fit_refine`: initialize from profile loft or visual hull, fit primitives, optionally render a primitive mesh.
- `hybrid_loft_hull`: use visual hull as a constraint and loft as an initialization.

Both modes must expose:

- max iterations,
- target point count,
- primitive count bounds,
- loss weights,
- checkpoint cadence,
- fail-on-regression behavior,
- output primitive JSON path.

## Validation

Pure Python:

- SDF correctness for known primitive points.
- superquadric sample/mesh sanity.
- residual error decreases on known synthetic target.
- primitive addition targets high-error residuals.
- optimizer obeys parameter bounds.

Blender:

- render primitive fit before/after and compare per-view metrics.
- check mesh QA for primitive mesh exports.
- inspect residual heatmap artifacts.

Benchmarks:

- target point count scaling,
- primitive count scaling,
- optimizer step scaling,
- silhouette render/compare loop timing,
- quality delta per second.

## Papers And References

- SuperFrusta/ResFit: `temp/perf-quality-audit-20260509/lane-b-reconstruction/paper-pdfs/ganeshan-2026-superfrusta-resfit.pdf`.
- Superquadrics Revisited: `temp/perf-quality-audit-20260509/lane-b-reconstruction/paper-pdfs/paschalidou-2019-superquadrics-revisited.pdf`.
- Robust superquadric recovery: `temp/perf-quality-audit-20260509/lane-f-literature/paper-pdfs/liu-superquadric-recovery.pdf`.
- Neural 3D Mesh Renderer: `temp/perf-quality-audit-20260509/lane-f-literature/paper-pdfs/kato-neural-3d-mesh-renderer.pdf`.
- Soft Rasterizer: `temp/perf-quality-audit-20260509/lane-f-literature/paper-pdfs/liu-soft-rasterizer.pdf`.
- Boundary IoU: `temp/perf-quality-audit-20260509/lane-e-validation-quality/paper-pdfs/boundary-iou-cvpr2021.pdf`.
- Open3D metrics: https://www.open3d.org/docs/release/python_api/open3d.t.geometry.Metric.html.
