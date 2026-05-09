# Spec 03: Blender Mesh, Render, And Join Operations

## Scope

Owns F10, F11, F12, F13, F14, and F15. This spec makes Blender operations measurable, version-aware, and explicit. Ambitious reconstruction modes will fail without reliable render settings, mesh topology checks, and join/fallback observability.

## Current Code To Modify

- `blender_blocking/config.py:123-169`: `RenderConfig` defines engine, resolution, samples, color mode, view names, and camera padding.
- `blender_blocking/config.py:92-118`: `LoftMeshConfig` accepts `radial_segments >= 3`.
- `blender_blocking/integration/blender_ops/render_utils.py:29-47`: `render_orthogonal_views` does not accept or apply engine/samples.
- `blender_blocking/integration/blender_ops/profile_loft_mesh.py:31-51`: degenerate rings can become one vertex and adjacent point rings produce no faces.
- `blender_blocking/integration/blender_ops/profile_loft_mesh.py:82-114`: cap creation skips rings with fewer than three vertices.
- `blender_blocking/integration/blender_ops/profile_loft_mesh.py:125-139`: loft mesh builder accepts `radial_segments >= 3`.
- `blender_blocking/integration/blender_ops/profile_loft_mesh.py:144-168`: BMesh creation, remove doubles, normals.
- `blender_blocking/placement/primitive_placement.py:289-334`: `MeshJoiner.join_objects` chooses boolean or voxel and silently falls through.
- `blender_blocking/placement/primitive_placement.py:365-379`: sequential boolean union.
- `blender_blocking/placement/primitive_placement.py:384-427`: voxel remesh path.
- `blender_blocking/placement/primitive_placement.py:427-460`: simple join path uses `bpy.ops.object.join()`.
- `blender_blocking/test_runner.py:35-50`: pure suites are listed, but loft and voxel-remesh tests are absent.
- `blender_blocking/test_runner.py:131-223`: Blender suites are manually invoked and should include existing missing tests.
- `blender_blocking/test_profile_loft_mesh.py:33-105`: existing Blender loft tests.
- `blender_blocking/test_mesh_joiner_voxel_remesh.py:12-47`: existing Blender voxel remesh test.
- `docs/IMPLEMENTATION_SPEC.md:106`: render/canonicalization settings must be easy to override and passed through.
- `docs/IMPLEMENTATION_SPEC.md:180-189`: loft mesh generation and degenerate ring handling.
- `docs/IMPLEMENTATION_SPEC.md:208-214`: join-mode fallback order.
- `docs/IMPLEMENTATION_SPEC.md:268-270`: Blender tests must validate loft bounds, manifoldness, join modes, and camera framing.
- `docs/IMPLEMENTATION_SPEC.md:394-402`: render silhouette schema.

## Render Pipeline

Replace `render_orthogonal_views` with a config-driven signature:

```python
def render_orthogonal_views(
    obj,
    output_dir,
    *,
    render_config: RenderConfig,
    camera_config: CameraConfig | None = None,
    context: GenerationContext | None = None,
) -> RenderResult:
    ...
```

Requirements:

- Apply `RenderConfig.engine` from `config.py:127` with Blender-version-aware enum resolution.
- Apply `RenderConfig.samples` from `config.py:129` to the selected engine where supported.
- Apply resolution from `config.py:124-126`.
- Apply background/color mode deterministically.
- Record engine requested, engine applied, sample count, resolution, camera transform, render path, and timing per view.
- Never inherit uncontrolled scene render settings.
- If an engine is unavailable in Blender 4.2 or 5.0, record the substitution in the manifest.

Tests:

- Blender test sets WORKBENCH and EEVEE/available enum and asserts the render scene settings.
- Render output masks should be stable enough for canonical comparison across 4.2 and 5.0.

## Mesh QA Contract

Introduce `MeshQualityReport` under `integration/blender_ops/mesh_quality.py`:

```python
@dataclass(frozen=True)
class MeshQualityReport:
    object_name: str
    vertices: int
    edges: int
    faces: int
    loose_vertices: int
    non_manifold_edges: int
    boundary_edges: int
    self_intersection_warnings: tuple[str, ...]
    bbox: Bounds3D
```

Every mesh-generating path must run this report before returning a success result:

- profile loft,
- silhouette intersection,
- visual hull mesh extraction,
- primitive fitting mesh export,
- legacy joined primitives.

Failure thresholds should be mode-aware: research modes may return warnings, but stable modes must fail if they produce loose vertices or empty meshes.

## Loft Mesh Generation

Update `profile_loft_mesh.py`:

- Align validation with `docs/IMPLEMENTATION_SPEC.md:371-381`. For ambitious modes, prefer `radial_segments >= 16` unless explicitly configured lower for tests.
- Add adaptive radial segments based on curvature and target screen-space edge error.
- Replace unstructured degenerate behavior at `profile_loft_mesh.py:31-51` with explicit pole/ring topology:
  - consecutive point rings should collapse into one pole region or be skipped before face creation,
  - point-to-ring transitions must create fan triangles,
  - ring-to-point transitions must create fan triangles,
  - interior zero-radius slices must be rejected or converted into two separate lobes, not silently create loose topology.
- Benchmark `bmesh` construction against `mesh.from_pydata` for large ring counts. Keep BMesh if topology operations matter, but do not assume it is faster.
- Store mesh-generation metadata in the returned result: radial segments, ring count, degenerate count, face count, cap mode, weld threshold, and QA report.

## Join Strategy

Replace `MeshJoiner.join_objects` at `primitive_placement.py:289-334` with a result-returning API:

```python
@dataclass(frozen=True)
class JoinAttempt:
    mode: str
    success: bool
    elapsed_s: float
    warnings: tuple[str, ...]
    quality: MeshQualityReport | None

@dataclass(frozen=True)
class JoinResult:
    object: Any | None
    attempts: tuple[JoinAttempt, ...]
    selected_mode: str | None
    degraded: bool
    fatal_error: str | None
```

Rules:

- Do not silently return simple join after voxel/boolean failure as in `primitive_placement.py:314-326`.
- If simple join is used, mark `degraded=True` and record that the output may contain internal overlaps or disconnected shells.
- Replace sequential boolean union at `primitive_placement.py:365-379` with measured strategies:
  - balanced boolean tree,
  - direct voxel remesh,
  - loft path when input is slice-derived,
  - simple join only as an explicit degraded debug artifact.
- Add per-object and aggregate timings.
- Record solver requested and resolved through `resolve_boolean_solver`.

## Test Runner Wiring

Add existing tests to `test_runner.py`:

- `test_profile_loft_mesh.py` from `test_profile_loft_mesh.py:33-105`.
- `test_mesh_joiner_voxel_remesh.py` from `test_mesh_joiner_voxel_remesh.py:12-47`.

Runner behavior:

- In non-Blender Python, mark these skipped with reason.
- In Blender, include them in the normal full run and in quick mode if they are fast enough. If not quick, label them clearly as full-only.
- CI should report section names so failures are obvious.

## Acceptance

- Render settings test proves engine and samples are applied.
- Loft degenerate ring fixtures have zero loose vertices and expected face counts.
- Voxel failure and boolean failure are forced in tests; manifests include degraded fallback metadata.
- Existing Blender tests are visible in runner output.
- Legacy high-slice path reports join timings and selected strategy.

## Papers And References

- Blender BooleanModifier API: https://docs.blender.org/api/current/bpy.types.BooleanModifier.html.
- Blender RemeshModifier API: https://docs.blender.org/api/current/bpy.types.RemeshModifier.html.
- Blender BMesh API: https://docs.blender.org/api/master/bmesh.html.
- Curless and Levoy volumetric fusion: `temp/perf-quality-audit-20260509/lane-c-blender-mesh/paper-pdfs/CurlessLevoy-1996-volumetric-range-images.pdf`.
- Kazhdan Poisson reconstruction: `temp/perf-quality-audit-20260509/lane-c-blender-mesh/paper-pdfs/Kazhdan-2006-poisson-surface-reconstruction.pdf`.
- Screened Poisson: `temp/perf-quality-audit-20260509/lane-c-blender-mesh/paper-pdfs/KazhdanHoppe-2013-screened-poisson.pdf`.
