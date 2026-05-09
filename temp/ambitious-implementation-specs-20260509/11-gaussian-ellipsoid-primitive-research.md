# Spec 11: Gaussian, Ellipsoid, And Splat Primitive Research Track

## Scope

This expands idea 5. Add a research track for anisotropic Gaussian/ellipsoid primitive clouds as an intermediate representation. The target is not photorealistic radiance fields first. The target is a compact, optimizable, editable shape proxy that can bridge visual hulls, surface point sets, SuperFrusta, and mesh extraction.

Related findings: F07, F16, F22, F25, F26, F28.

## Current Code To Modify

- `blender_blocking/primitives/superfrustum.py:19-421`: existing parametric primitive pattern.
- `blender_blocking/placement/resfitting.py:127-188`: initializes primitives from voxels.
- `blender_blocking/placement/resfitting.py:190-738`: optimizes primitive sets against target points.
- `blender_blocking/integration/multi_view/visual_hull.py:477-504`: extracts point cloud from voxel grid.
- `blender_blocking/benchmarks/benchmark_perf.py:693-933`: ResFit benchmarks can be extended for Gaussian/ellipsoid fitting.

## Representation

```python
@dataclass(frozen=True)
class AnisotropicGaussianPrimitive:
    center: np.ndarray
    covariance: np.ndarray
    opacity: float
    color: tuple[float, float, float] | None
    semantic_role: str | None

@dataclass(frozen=True)
class EllipsoidPrimitive:
    center: np.ndarray
    radii: np.ndarray
    rotation: np.ndarray
    density: float
    confidence: float
```

For shape blocking, opacity/density represents occupancy confidence, not final material.

## Use Cases

- approximate visual-hull surface with a compact primitive cloud,
- initialize SuperFrustum/superquadric fitting,
- represent uncertain boundary bands from Spec 09,
- provide an editable intermediate before meshing,
- enable fast differentiable-style silhouette rendering through projected ellipses.

## Fitting Pipeline

1. Input source:
   - surface points from visual hull,
   - profile-band samples,
   - mesh surface samples from synthetic factory.
2. Initialize:
   - k-means or farthest-point centers,
   - local PCA covariance,
   - density from local occupancy/confidence,
   - merge tiny or redundant primitives.
3. Optimize:
   - silhouette reprojection loss,
   - surface coverage loss,
   - overlap/compactness penalty,
   - primitive-count penalty.
4. Convert:
   - ellipsoid mesh,
   - voxel occupancy,
   - point cloud,
   - SuperFrustum seeds.

## Rendering

Implement a CPU projected-ellipse renderer first:

- project ellipsoid center/covariance into each orthographic view,
- rasterize soft elliptical footprint,
- composite occupancy probability,
- compare to uncertain masks from Spec 09.

Later optional:

- GPU splat renderer,
- differentiable backend from Spec 10.

## Relation To 3D Gaussian Splatting

3D Gaussian Splatting uses anisotropic 3D Gaussians for efficient real-time radiance-field rendering. This repo should borrow:

- anisotropic covariance optimization,
- density control,
- empty-space avoidance,
- fast splat rendering.

This repo should not initially borrow:

- photorealistic color training,
- SfM dependency,
- large GPU-only training loop,
- viewer-specific format as primary output.

## Export

Artifact formats:

- `gaussians.json`: exact primitive parameters,
- `gaussians.npz`: efficient numeric arrays,
- `ellipsoid_mesh.obj`: optional mesh proxy,
- `splat_preview.png`: rendered preview per view,
- `conversion_report.json`: primitive count, coverage, losses.

## Tests

- fit ellipsoids to synthetic sphere/ellipsoid fixtures,
- projected ellipse renderer matches analytic silhouette,
- primitive cloud approximates visual hull points within threshold,
- conversion to mesh passes topology QA where expected,
- candidate ensemble can score Gaussian proxy against loft/hull.

## References

- 3D Gaussian Splatting: https://arxiv.org/abs/2308.04079.
- Instant Neural Graphics Primitives/hash encoding: https://arxiv.org/abs/2201.05989.
- Robust superquadric recovery: `temp/perf-quality-audit-20260509/lane-f-literature/paper-pdfs/liu-superquadric-recovery.pdf`.
- SuperFrusta/ResFit: `temp/perf-quality-audit-20260509/lane-b-reconstruction/paper-pdfs/ganeshan-2026-superfrusta-resfit.pdf`.
