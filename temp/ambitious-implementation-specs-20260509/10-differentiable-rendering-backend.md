# Spec 10: Optional Differentiable Rendering Backend Interface

## Scope

This expands idea 4. The project should define a concrete differentiable-rendering interface now, while keeping implementation optional. The baseline path can use Blender renders and finite-difference or derivative-free optimization; GPU differentiable backends can plug in later.

Related findings: F22, F25, F28.

## Current Code To Modify

- `blender_blocking/integration/blender_ops/render_utils.py:29-129`: current orthographic rendering path.
- `blender_blocking/integration/blender_ops/silhouette_render.py:270-368`: render settings and frame rendering.
- `blender_blocking/validation/silhouette_iou.py:212-228`: hard IoU metric.
- `blender_blocking/placement/resfitting.py:190-529`: optimization currently works on SDF residuals and finite differences.
- `blender_blocking/primitives/superfrustum.py:152-210`: finite-difference gradient batch support.
- `docs/IMPLEMENTATION_SPEC.md:15-16`: old scope excludes ML reconstruction; this spec keeps differentiable rendering optional and isolated.

## Backend Interface

Create `reconstruction/differentiable_render.py`:

```python
class DifferentiableRenderBackend(Protocol):
    name: str
    supports_gradients: bool
    supports_silhouette: bool
    supports_depth: bool

    def render(self, scene: RenderableScene, cameras: Sequence[CameraSpec]) -> RenderBatch: ...
    def loss(self, render_batch: RenderBatch, target: ReconstructionTarget, weights: LossWeights) -> LossResult: ...
    def backward(self, loss: LossResult) -> GradientBatch: ...
```

Backends:

- `BlenderFiniteDifferenceBackend`: always available inside Blender, slow, no direct gradients.
- `CpuSoftSilhouetteBackend`: pure Python approximate soft rasterizer for simple primitives/profiles.
- `NvdiffrastBackend`: optional CUDA/GPU research plugin.
- `Torch3DBackend`: optional if dependency exists later.

The interface must be optional. Missing GPU dependencies should skip tests, not break core functionality.

## Renderable Scene Contract

```python
@dataclass(frozen=True)
class RenderablePrimitive:
    primitive_type: str
    parameters: Mapping[str, np.ndarray]
    mesh_proxy: MeshData | None

@dataclass(frozen=True)
class RenderableScene:
    primitives: tuple[RenderablePrimitive, ...]
    mesh: MeshData | None
    transform: np.ndarray
```

This decouples optimization from Blender object state.

## Losses

Use losses from Spec 05:

- hard area IoU for gates,
- soft silhouette loss for gradients,
- Boundary IoU for diagnostics,
- signed-distance transform loss,
- depth silhouette loss when depth exists,
- topology penalties outside differentiable loop.

## Implementation Phases

1. Define protocol and data classes.
2. Implement Blender finite-difference backend for profile parameters.
3. Implement CPU soft silhouette for simple 2D projected profiles.
4. Add optional GPU backend behind dependency checks.
5. Add candidate ensemble entries that compare refined vs unrefined outputs.

## Nvdiffrast Track

NVIDIA nvdiffrast provides modular differentiable rendering primitives: rasterization, interpolation, texturing, and antialiasing. Use it as the design reference for a backend that focuses on rasterization and leaves loss/shading to this repo.

Rules:

- Never make nvdiffrast a required dependency.
- All GPU code must live behind plugin checks.
- Store GPU device, driver/CUDA info, backend version, and determinism notes.
- Provide CPU/Blender fallback metrics for every GPU result.

## Tests

- protocol conformance tests with a fake backend,
- finite-difference optimizer improves a simple ellipse silhouette,
- missing optional backend is reported as skipped,
- gradient backend result is compared to finite-difference baseline on tiny fixture,
- no GPU backend can bypass manifest or quality gates.

## References

- Nvdiffrast docs: https://nvlabs.github.io/nvdiffrast/.
- Nvdiffrast paper: https://arxiv.org/abs/2011.03277.
- Neural 3D Mesh Renderer: `temp/perf-quality-audit-20260509/lane-f-literature/paper-pdfs/kato-neural-3d-mesh-renderer.pdf`.
- Soft Rasterizer: `temp/perf-quality-audit-20260509/lane-f-literature/paper-pdfs/liu-soft-rasterizer.pdf`.
