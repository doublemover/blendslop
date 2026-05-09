# Implementation Master Spec (Single Source of Truth)

This document replaces all prior roadmaps/specs and is the authoritative baseline for the Blender Automated Blocking Tool. The legacy pipeline remains default, while the 2026 ambitious reconstruction extensions add typed backend contracts, candidate ensembles, visual hull volumes, primitive fitting, uncertainty, constraints, synthetic fixtures, and topology-aware validation as opt-in foundations.

## 1) Scope, Non-Goals, and Defaults

### Scope (In)
- Deterministic, testable reconstruction from orthogonal silhouettes.
- Legacy slice-primitive-boolean path (maintained).
- Optional dual-profile loft path (opt-in flag).
- Canonical silhouette extraction, framing, and IoU validation.
- Metadata tagging and run manifest for outputs.
- Typed reconstruction backend registry and candidate ensemble execution.
- Optional visual-hull volume reconstruction and mesh extraction.
- Optional primitive/ellipsoid/Gaussian proxy fitting with differentiable-rendering-inspired losses.
- Synthetic shape factory fixtures, quality budgets, uncertainty maps, and human correction constraints.

### Non-Goals (Out)
- Required external ML dependencies or mandatory GPU reconstruction.
- Blender UI add-ons or operator UX beyond scripts.
- Non-axis-aligned inputs (tilted silhouettes) in v1.

### Default Behavior (Must Match Current)
- `unit_scale = 0.01` (pixel to Blender units).
- Legacy reconstruction path remains the default.
- Boolean union remains default join mode unless explicitly changed.
- Render resolution defaults to 512x512; canonical output size defaults to 256 (both configurable).

## 2) Naming Map (Resolve Conflicts)
This spec defines canonical names. Any legacy names must be treated as aliases.
- `unit_scale` (canonical) == `pixel_size` (legacy) when scale policy is pixel-based.
- `world_height` (canonical) == `height_u` (legacy).
- `BBox2D` uses `(x0, y0, x1, y1)` exclusive bounds (Python slicing). Legacy `(x, y, w, h)` must be converted.
- `cap_mode` uses `"fan" | "none"` (canonical). `"ngon"` is allowed as a legacy optional value for compatibility but is not the default.

## 3) Module Layout (Canonical)
Pure-Python logic lives in feature packages (`geometry/`, `validation/`, `metrics/`, `reconstruction/`, `constraints/`, `synthetic/`, and `volume/`) and must not import `bpy`. Blender adapters stay under `integration/blender_ops/` and `placement/`.

```
blender_blocking/
  geometry/
    __init__.py
    profile_models.py
    silhouette.py
    dual_profile.py
    slicing.py
  metrics/
    silhouette.py
    surface.py
    topology.py
    budgets.py
  validation/
    __init__.py
    silhouette_iou.py
  reconstruction/
    backend.py
    registry.py
    ensemble.py
    target_builder.py
    point_cloud.py
    backends/
  constraints/
  synthetic/
  volume/
  integration/
    blender_ops/
      camera_framing.py
      profile_loft_mesh.py
      render_utils.py  (modified)
  utils/
    generation_context.py
    manifest.py
  config.py
```

Compatibility re-export (no new definitions):
- `blender_blocking/integration/shape_matching/contracts.py` should re-export types from `geometry/profile_models.py` if older docs or modules expect them.

## 4) Data Contracts (Canonical)
Define all data contracts in `blender_blocking/geometry/profile_models.py`.

### 4.1 BBox2D
- `x0, y0` inclusive
- `x1, y1` exclusive
- Properties: `w = x1 - x0`, `h = y1 - y0`
- Invariants: `0 <= x0 < x1 <= W`, `0 <= y0 < y1 <= H`

### 4.2 PixelScale
```python
@dataclass(frozen=True)
class PixelScale:
    unit_per_px: float

    @staticmethod
    def from_target_height(target_height_units: float, silhouette_height_px: int) -> "PixelScale":
        return PixelScale(unit_per_px=target_height_units / float(silhouette_height_px))
```
- Invariant: `unit_per_px > 0`

### 4.3 VerticalWidthProfilePx
Arrays length N; all arrays are float32/float64; `valid` is bool.
- `heights_t`: monotonic in [0, 1], bottom to top.
- `left_x`, `right_x`, `width_px`, `center_x` (NaN where invalid).
- `bbox`: BBox2D used for sampling.
- `source_view`: `"front"` or `"side"`.

### 4.4 EllipticalProfileU
- `heights_t`: length N, [0,1].
- `rx`, `ry`: >= 0 in Blender units.
- `world_height`: > 0 in Blender units.
- `z0`: base z (default 0.0).
- Optional `cx`, `cy` arrays (default disabled).

### 4.5 EllipticalSlice
Per-slice loft data:
- `z`, `rx`, `ry`, optional `cx`, `cy`.

## 5) Configuration (Canonical)
All configuration fields and valid values are defined in the embedded schema in **Section 19**. Treat the schema as the single source of truth for config structure and defaults.

Implementation rules:
- `BlockingConfig` composes these groups: `ReconstructionConfig`, `SilhouetteExtractConfig`, `ProfileSamplingConfig`, `LoftMeshOptions`, `RenderConfig`, `CanonicalizeConfig`, `VisualHullConfig`, `VolumeConfig`, `EnsembleConfig`, `PrimitiveFitConfig`, `GaussianEllipsoidConfig`, `DifferentiableRenderConfig`, `ConstraintConfig`, `SyntheticFactoryConfig`, and `QualityBudgetConfig`.
- Defaults must preserve current behavior unless a phase explicitly changes them (see Section 1 defaults).
- Render and canonicalization sizes must be easy to override via config and passed through to `render_orthogonal_views` and `canonicalize_mask`.

## 6) Silhouette Extraction (Canonical Algorithm)
Implemented in `blender_blocking/geometry/silhouette.py`.

### 6.1 Input Validation
- Accept `image.ndim` of 2 (gray) or 3 (RGB/RGBA).
- Raise `ValueError` for any other shape.

### 6.2 Alpha-First Logic
- If RGBA and alpha varies, use alpha as silhouette driver.
- `mask = alpha > alpha_threshold`.

### 6.3 Luma Path with Auto Polarity
- Convert to grayscale (uint8) if not already.
- Compute `bg_mean` from border pixels; `center_mean` from central ROI.
- If `center_mean < bg_mean`: object is darker -> invert threshold.
- Use Otsu threshold if `gray_threshold` is None.

### 6.4 Morphology and Cleanup
- Apply close then open with odd kernel sizes (>=3) if configured.
- Keep largest component if `largest_component_only=True`.
- Fill holes by drawing external contour filled.

### 6.5 Outputs
- Return `bool` mask.
- Provide `bbox_from_mask(mask)` that returns `BBox2D` or raises if empty.

### Example (Synthetic)
- Input: 100x100 grayscale with black rectangle 20x60 centered.
- Output: mask with bbox `x0=40,x1=60,y0=20,y1=80` and nonzero area.

## 7) Vertical Profile Extraction (Legacy Wrapper)
Maintain legacy signature in `integration/shape_matching/profile_extractor.py` while delegating to canonical functions.

### 7.1 Wrapper Rules
- `extract_silhouette_from_image`:
  - Calls `geometry.silhouette.extract_binary_silhouette` and returns uint8 0/255 for legacy callers.
- `extract_vertical_profile(image, num_samples, *, bbox=None, already_silhouette=False, smoothing_window=3)`:
  - Converts to silhouette, crops to bbox, samples widths, interpolates, smooths, normalizes.
  - Returns `List[(height_norm, width_norm)]`.

## 8) Dual-Profile Elliptical Reconstruction (Pure Python)
Implemented in `blender_blocking/geometry/dual_profile.py`.

### 8.1 Sampling Width Profiles
`extract_vertical_width_profile_px(mask, bbox=None, num_samples, sample_policy, fill_strategy, smoothing_window)`:
- Sample N rows from bottom to top.
- For each row, find leftmost and rightmost True pixel.
- width_px = right - left + 1
- center_x = (left + right) / 2
- Missing rows -> NaN then interpolated.
- Smooth widths with median filter (window size).

### 8.2 Build EllipticalProfileU
`build_elliptical_profile_from_views(front_mask, side_mask, scale, num_samples, z0, height_strategy, fallback_policy, min_radius_u)`:
- Compute `rx` from front, `ry` from side.
- `height_strategy`: `front` (default), `side`, `max`, `mean`.
- If only one view present:
  - `fallback_policy="circular"` => missing axis equals available axis.
  - `fallback_policy="error"` => raise.
- Clamp radii to `min_radius_u`.
- If offsets enabled, compute `cx/cy` from `center_x` minus bbox center.

## 9) Slice Sampling
Implemented in `blender_blocking/geometry/slicing.py`.

`sample_elliptical_slices(profile, num_slices, sampling)`:
- `cell_centers`: `t = (i+0.5)/num_slices`.
- `endpoints`: `t = i/(num_slices-1)`.
- `z = z0 + t * world_height`.
- Interpolate `rx/ry` linearly at each `t`.

## 10) Mesh-From-Profile Loft (Blender bmesh)
Implemented in `integration/blender_ops/profile_loft_mesh.py`.

### 10.1 Algorithm
- For each slice:
  - If `rx` or `ry` near 0 and `weld_degenerate_rings=True`, use a single point vertex.
  - Else create ring of `radial_segments` vertices.
- Bridge adjacent rings:
  - Ring->Ring: quads (triangulate if desired).
  - Ring->Point: triangle fan.
  - Point->Ring: reverse triangle fan.
- Caps:
  - `cap_mode="fan"`: create center vertex and triangles.
  - `cap_mode="ngon"` (legacy optional): create a single face from the ring; triangulation may vary by Blender version.
  - `cap_mode="none"`: do nothing (mesh is open).
- Post:
  - Optional remove doubles (merge threshold).
  - Recalc normals.
  - Shade smooth.

### 10.2 Blender Adapter Rules
- Must not use `bpy.ops.mesh.primitive_*`.
- Must not use boolean modifiers.
- Prefer bmesh or `mesh.from_pydata`.

## 11) Legacy Slice + Primitive Path (Keep Working)
- `SliceAnalyzer` and `PrimitivePlacer` remain unchanged except for scale/heuristic improvements.
- Join uses MeshJoiner (now supports multiple modes).

## 12) Mesh Join Modes
Implemented in `placement/primitive_placement.py`:
- `boolean`: sequential boolean union.
- `simple`: join objects without booleans.
- `voxel`: join then voxel remesh.
- `auto`: boolean for small N, voxel for large N.
- Fallback order if remesh fails: voxel -> boolean -> simple.

## 13) Render Framing (Orthographic)
Implemented in `integration/blender_ops/camera_framing.py`.

### 13.1 Bounds
Compute world bounds by transforming each object’s `bound_box` with `matrix_world`.

### 13.2 Ortho Scale
For each view:
- compute view-plane extents.
- `ortho_scale = max(width_extent * (1+2*margin), height_extent * aspect * (1+2*margin))`.

### 13.3 Camera Placement
- front: look along -Y
- side: look along -X
- top: look along -Z
- distance = `max_dim * distance_factor` (default 2.0).

## 14) Canonical Silhouette IoU
Implemented in `validation/silhouette_iou.py`.

### 14.1 mask_from_image_array
- RGBA: use alpha channel.
- RGB/gray: luma and threshold.
- Return bool mask.

### 14.2 canonicalize_mask
- Tight crop -> add padding -> resize nearest -> paste into canvas.
- `bottom_center` alignment for front/side, `center` for top.
- Optional small morphological close for rendered silhouettes.

### 14.3 compute_mask_iou
- Intersection/union on canonical masks.
- If union == 0, IoU = 0 and note why.
- Return IoUResult with diagnostics.

## 15) Generation Context + Manifest
Implemented in `utils/generation_context.py` and `utils/manifest.py`.

### 15.1 Tags
Add custom properties:
- `blocktool_schema`, `blocktool_run_id`, `blocktool_seed`, `blocktool_role`, `blocktool_index` (if primitive), plus optional `blocktool_params`.

### 15.2 Manifest
Store a JSON-like dict in `scene["blocktool_manifest"]` with:
- `manifest_version`, `run_id`, `created_utc`, `context`, `stages`, `outputs`, `warnings`, `errors`.

## 16) Testing Strategy

### 16.1 Pure Python Tests
- Use synthetic masks; no Blender.
- Validate: silhouette extraction, bbox, width profiles, slice sampling, canonical IoU invariance.

### 16.2 Blender Integration Tests
- Minimal geometry; clean scene before/after.
- Validate loft mesh bounds, manifoldness, join modes, camera framing.

### 16.3 E2E Tests
- Render, canonicalize, compare IoU.
- Save debug artifacts on failure: raw, canonical, diff.

### 16.4 Test Runner
- Pure-Python tests should run even without Blender.
- Blender-only tests behind a flag or in Blender environments.

## 17) Integration Touchpoints (Where to Modify)
- `main_integration.py`: add `config` + `context`, add `create_3d_blockout_loft`, branch by `reconstruction_mode`.
- `image_processor.py`: add `_to_gray_uint8`, support RGBA.
- `profile_extractor.py`: wrap canonical silhouette, update vertical profile extraction.
- `render_utils.py`: use camera framing and `RenderConfig`.
- `shape_matcher.py`: use canonical IoU pipeline.
- `primitive_placement.py`: add join modes and heuristics.
- `test_e2e_validation.py`: use canonical IoU and framed renders.

## 18) Acceptance Criteria
- Legacy pipeline behaves identically when `reconstruction_mode="legacy"`.
- Optional loft pipeline generates a watertight mesh without booleans.
- IoU validation is robust to resolution/framing differences.
- Metadata and manifest present for every run.
- Every non-legacy reconstruction mode returns a structured `CandidateResult` with status, warnings/errors, artifacts, and metrics.
- Ensemble mode writes all-candidate metadata and selects by explicit score terms.
- Visual hull, primitive fitting, Gaussian/ellipsoid, and differentiable-refine paths are opt-in and must report optional dependency skips explicitly.
- Tests are deterministic and partitioned by environment.

### 18.1 Ambitious Backend Completion Requirements
- Non-legacy reconstruction modes are routed through `reconstruction/registry.py` and backend-owned implementations under `reconstruction/backends/`; `main_integration.py` may keep direct legacy behavior only.
- Visual hull backends must support dense, chunked, sparse-hash, and OpenVDB-labeled modes. Chunked/sparse/OpenVDB modes must build directly into chunk storage instead of requiring a dense-first conversion.
- OpenVDB is optional. When `pyopenvdb` or `openvdb` bindings are installed, `volume/openvdb_adapter.py` must export and import direct `.vdb` grids. When bindings are absent or lack required APIs, the backend must return structured `OpenVDBStatus` data and keep NPZ sparse interchange artifacts.
- Visual hull mesh postprocess modes `poisson` and `screened_poisson` must run Open3D Poisson reconstruction when `open3d` is installed. Missing or unusable Open3D must be an explicit skip or failure according to `postprocess_required`, `require_postprocess`, or `fail_on_postprocess_skip`.
- Differentiable rendering must provide a deterministic CPU soft-silhouette backend and an optional `nvdiffrast.torch` backend. Missing `nvdiffrast`/`torch`, missing raster context, or unusable GPU runtime must obey `optional_dependency_policy="skip" | "fail"` and must not silently fall back to CPU when the user requested `nvdiffrast`.
- Primitive, Gaussian/ellipsoid, visual hull, and differentiable backends must emit `CandidateResult` metrics with per-view data where views exist, topology reports for mesh-producing paths, artifacts, warnings/errors, and objective or loss histories where an objective is evaluated.
- Synthetic fixtures must remain deterministic and generated outputs must stay under ignored artifact roots. Commit only small fixture specs, budget JSON, schemas, and docs.
- Quality budgets must be runnable against current artifacts and optional baseline artifacts; `fail_on_regression` must produce a nonzero runner/benchmark exit when required checks fail.

## 19) Canonical Schemas (Embedded)
These replace standalone JSON schema files.

### 19.1 EllipticalProfile Schema
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://blendtec.local/schema/elliptical_profile_v1.json",
  "title": "EllipticalProfile (v1)",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "heights": {
      "type": "array",
      "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
      "minItems": 2
    },
    "rx": { "type": "array", "items": { "type": "number", "minimum": 0.0 }, "minItems": 2 },
    "ry": { "type": "array", "items": { "type": "number", "minimum": 0.0 }, "minItems": 2 },
    "cx": { "type": ["array", "null"], "items": { "type": "number" }, "minItems": 2 },
    "cy": { "type": ["array", "null"], "items": { "type": "number" }, "minItems": 2 },
    "world_height": { "type": "number", "exclusiveMinimum": 0.0 },
    "z0": { "type": "number" },
    "meta": { "type": ["object", "null"] }
  },
  "required": ["heights", "rx", "ry", "world_height", "z0"]
}
```

### 19.2 Reconstruction Config Schema
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://blendtec.local/schema/reconstruction_config_v1.json",
  "title": "ReconstructionConfig (v1)",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "reconstruction_mode": {
      "type": "string",
      "enum": [
        "legacy",
        "loft_profile",
        "profile_loft",
        "silhouette_intersection",
        "visual_hull_voxel",
        "hybrid_loft_hull",
        "primitive_fit_refine",
        "gaussian_ellipsoid_proxy",
        "differentiable_refine",
        "ensemble"
      ]
    },
    "unit_scale": { "type": "number", "minimum": 0.0 },
    "num_slices": { "type": "integer", "minimum": 1 },
    "mesh_join_mode": { "type": "string", "enum": ["auto", "boolean", "voxel", "simple"] },
    "silhouette_extract_ref": { "$ref": "#/definitions/silhouette_extract" },
    "silhouette_extract_render": { "$ref": "#/definitions/silhouette_extract" },
    "profile_sampling": { "$ref": "#/definitions/profile_sampling" },
    "mesh_from_profile": { "$ref": "#/definitions/mesh_from_profile" },
    "canonicalize": { "$ref": "#/definitions/canonicalize" },
    "render_silhouette": { "$ref": "#/definitions/render_silhouette" },
    "visual_hull": { "$ref": "#/definitions/visual_hull" },
    "volume": { "$ref": "#/definitions/volume" },
    "ensemble": { "$ref": "#/definitions/ensemble" },
    "primitive_fit": { "$ref": "#/definitions/primitive_fit" },
    "gaussian_ellipsoid": { "$ref": "#/definitions/gaussian_ellipsoid" },
    "differentiable_render": { "$ref": "#/definitions/differentiable_render" },
    "constraints": { "$ref": "#/definitions/constraints" },
    "synthetic_factory": { "$ref": "#/definitions/synthetic_factory" },
    "quality_budget": { "$ref": "#/definitions/quality_budget" }
  },
  "required": ["reconstruction_mode", "unit_scale", "num_slices"],
  "definitions": {
    "silhouette_extract": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "prefer_alpha": { "type": "boolean" },
        "alpha_threshold": { "type": "integer", "minimum": 0, "maximum": 255 },
        "gray_threshold": { "type": ["integer", "null"], "minimum": 0, "maximum": 255 },
        "invert_policy": { "type": "string", "enum": ["auto", "invert", "no_invert"] },
        "morph_close_px": { "type": "integer", "minimum": 0 },
        "morph_open_px": { "type": "integer", "minimum": 0 },
        "fill_holes": { "type": "boolean" },
        "largest_component_only": { "type": "boolean" },
        "polarity": { "type": "string", "enum": ["auto", "dark_foreground", "light_foreground", "alpha_foreground"] },
        "min_area_frac": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "max_area_frac": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "min_component_area_px": { "type": "integer", "minimum": 0 },
        "emit_uncertainty": { "type": "boolean" }
      }
    },
    "profile_sampling": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "num_samples": { "type": "integer", "minimum": 2 },
        "sample_policy": { "type": "string", "enum": ["endpoints", "cell_centers"] },
        "fill_strategy": { "type": "string", "enum": ["interp_linear", "interp_nearest", "constant"] },
        "smoothing_window": { "type": "integer", "minimum": 1 }
      }
    },
    "mesh_from_profile": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "radial_segments": { "type": "integer", "minimum": 12 },
        "cap_mode": { "type": "string", "enum": ["fan", "none", "ngon"] },
        "min_radius_u": { "type": "number", "minimum": 0.0 },
        "merge_threshold_u": { "type": "number", "minimum": 0.0 },
        "recalc_normals": { "type": "boolean" },
        "shade_smooth": { "type": "boolean" },
        "weld_degenerate_rings": { "type": "boolean" },
        "adaptive_radial_segments": { "type": "boolean" },
        "max_adaptive_radial_segments": { "type": "integer", "minimum": 3 },
        "topology_strict": { "type": "boolean" }
      }
    },
    "canonicalize": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "output_size": { "type": "integer", "minimum": 32 },
        "padding_frac": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "anchor": { "type": "string", "enum": ["center", "bottom_center"] },
        "interp": { "type": "string", "enum": ["nearest"] }
      }
    },
    "render_silhouette": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "resolution": { "type": "array", "items": { "type": "integer", "minimum": 32 }, "minItems": 2, "maxItems": 2 },
        "engine": { "type": "string", "enum": ["BLENDER_EEVEE", "WORKBENCH"] },
        "transparent_bg": { "type": "boolean" },
        "samples": { "type": "integer", "minimum": 1 },
        "margin_frac": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
      }
    },
    "visual_hull": {
      "type": "object",
      "properties": {
        "backend": { "type": "string", "enum": ["dense", "chunked", "sparse_hash", "openvdb"] },
        "resolution": { "type": "integer", "minimum": 1 },
        "max_resolution": { "type": "integer", "minimum": 1 },
        "chunk_size": { "type": ["integer", "null"], "minimum": 1 },
        "adaptive_max_depth": { "type": "integer", "minimum": 0 },
        "boundary_refine": { "type": "boolean" },
        "mesh_method": { "type": "string", "enum": ["marching_cubes", "lewiner", "dual_contouring", "points"] },
        "postprocess": { "type": "string", "enum": ["none", "poisson", "screened_poisson"] },
        "postprocess_required": { "type": "boolean" },
        "require_postprocess": { "type": "boolean" },
        "fail_on_postprocess_skip": { "type": "boolean" },
        "poisson_depth": { "type": "integer", "minimum": 1 },
        "poisson_scale": { "type": "number", "exclusiveMinimum": 0.0 },
        "poisson_linear_fit": { "type": "boolean" },
        "poisson_crop_to_input_bounds": { "type": "boolean" },
        "poisson_crop_scale": { "type": "number", "exclusiveMinimum": 0.0 },
        "poisson_density_quantile": { "type": ["number", "null"], "minimum": 0.0, "maximum": 1.0 },
        "memory_budget_mb": { "type": ["integer", "null"], "minimum": 1 },
        "occupancy_threshold": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "uncertainty_aggregation": { "type": "string", "enum": ["min", "product", "logit_sum"] }
      }
    },
    "volume": {
      "type": "object",
      "properties": {
        "backend": { "type": "string", "enum": ["dense", "chunked", "sparse_hash", "openvdb"] },
        "sparse_chunk_size": { "type": "integer", "minimum": 1 },
        "serialization": { "type": "string", "enum": ["npz"] },
        "export_openvdb": { "type": "boolean" }
      }
    },
    "ensemble": {
      "type": "object",
      "properties": {
        "selection_policy": { "type": "string", "enum": ["best_score", "quality_first", "editability_first", "fast_preview", "pareto"] },
        "max_parallel_candidates": { "type": "integer", "minimum": 1 },
        "keep_all_artifacts": { "type": "boolean" }
      }
    },
    "primitive_fit": {
      "type": "object",
      "properties": {
        "primitive_families": { "type": "array", "items": { "type": "string", "enum": ["superfrustum", "ellipsoid", "superquadric"] } },
        "target_point_count": { "type": "integer", "minimum": 1 },
        "min_primitives": { "type": "integer", "minimum": 0 },
        "max_primitives": { "type": "integer", "minimum": 1 },
        "optimization_steps": { "type": "integer", "minimum": 0 },
        "checkpoint_cadence": { "type": "integer", "minimum": 1 },
        "fail_on_regression": { "type": "boolean" },
        "loss_weights": { "type": "object" }
      }
    },
    "gaussian_ellipsoid": {
      "type": "object",
      "properties": {
        "primitive_count": { "type": "integer", "minimum": 1 },
        "initialization": { "type": "string", "enum": ["farthest_point", "kmeans", "grid"] },
        "min_radius": { "type": "number", "exclusiveMinimum": 0.0 },
        "max_radius": { "type": ["number", "null"] },
        "opacity_min": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "opacity_max": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "renderer": { "type": "string", "enum": ["cpu_projected_ellipse", "gpu_splat"] },
        "export_mesh_proxy": { "type": "boolean" }
      }
    },
    "differentiable_render": {
      "type": "object",
      "properties": {
        "backend": { "type": "string", "enum": ["cpu_soft_silhouette", "blender_finite_difference", "nvdiffrast"] },
        "optional_dependency_policy": { "type": "string", "enum": ["skip", "fail"] },
        "gradient_mode": { "type": "string", "enum": ["finite_difference", "backend"] },
        "finite_difference_epsilon": { "type": "number", "exclusiveMinimum": 0.0 },
        "softness": { "type": "number", "exclusiveMinimum": 0.0 },
        "min_variance": { "type": "number", "exclusiveMinimum": 0.0 },
        "visual_hull_resolution": { "type": "integer", "minimum": 1 },
        "primitive_count": { "type": "integer", "minimum": 1 },
        "target_point_count": { "type": "integer", "minimum": 1 },
        "min_radius": { "type": "number", "exclusiveMinimum": 0.0 },
        "covariance_floor": { "type": "number", "minimum": 0.0 },
        "kmeans_iterations": { "type": "integer", "minimum": 1 },
        "chunk_size": { "type": ["integer", "null"], "minimum": 1 },
        "loss_weights": { "type": "object" }
      }
    },
    "constraints": { "type": "object" },
    "synthetic_factory": { "type": "object" },
    "quality_budget": {
      "type": "object",
      "properties": {
        "budget_json": { "type": ["string", "null"] },
        "compare_baseline": { "type": ["string", "null"] },
        "fail_on_regression": { "type": "boolean" },
        "environment_compatibility": { "type": "string", "enum": ["warn", "strict", "ignore"] }
      }
    }
  }
}
```
