# Spec 02: Reconstruction Architecture And Shape Constraints

## Scope

Owns F01, F07, F08, F09, F10, F13, F15, and F19. This spec replaces the current split between legacy slices, loft profiles, and brittle silhouette intersection with a typed reconstruction architecture that can support profiles, contour extrusion, visual hulls, and primitive refinement.

## Current Code To Modify

- `blender_blocking/config.py:9`: `_VALID_RECON_MODES` currently includes `legacy`, `loft_profile`, and `silhouette_intersection`.
- `blender_blocking/config.py:92-118`: `LoftMeshConfig` validates `radial_segments >= 3`, conflicting with `docs/IMPLEMENTATION_SPEC.md:371-381`.
- `blender_blocking/main_integration.py:226-235`: silhouette bbox helper returns `(x0, y0, x1, y1)`.
- `blender_blocking/integration/blender_ops/mesh_generator.py:22`: `normalize_bounds` is an untyped tuple.
- `blender_blocking/integration/blender_ops/mesh_generator.py:79-86`: tuple is consumed as `(min_x, max_x, min_y, max_y)`.
- `blender_blocking/main_integration.py:733-741`: `create_3d_blockout_silhouette_intersection` implements front/side extrusion intersection.
- `blender_blocking/main_integration.py:747-749`: silhouette intersection falls back without both views.
- `blender_blocking/main_integration.py:871-891`: contour filtering can keep only largest outer contour.
- `blender_blocking/main_integration.py:955-1027`: boolean intersection, solver override, and empty-result FIXME.
- `blender_blocking/main_integration.py:1082-1092`: reconstruction-mode dispatch.
- `blender_blocking/geometry/dual_profile.py:73-129`: vertical width profile stores one interval/width/center per row.
- `blender_blocking/geometry/dual_profile.py:132-250`: profile build combines front/side widths into one elliptical profile; offsets are optional.
- `blender_blocking/main_integration.py:656-668`: loft path builds profiles with `enable_offsets=False`.
- `blender_blocking/geometry/slicing.py:13-39`: samples elliptical slices from the profile.
- `blender_blocking/integration/shape_matching/profile_extractor.py:83-147`: legacy vertical profile computes one normalized radius per row and converts arrays back to tuple lists.
- `blender_blocking/placement/primitive_placement.py:56`: legacy overlap ratio defaults to `2.5`.
- `blender_blocking/placement/primitive_placement.py:106-145`: legacy slices use one circular radius per height.
- `blender_blocking/placement/primitive_placement.py:164-179`: legacy profile interpolation converts tuple lists to arrays.
- `blender_blocking/placement/primitive_placement.py:289-498`: join strategy uses boolean/voxel/simple paths.
- `docs/IMPLEMENTATION_SPEC.md:149-189`: profile/loft requirements.
- `docs/IMPLEMENTATION_SPEC.md:208-214`: join fallback order.
- `docs/IMPLEMENTATION_SPEC.md:281-292`: integration and acceptance criteria.
- `docs/IMPLEMENTATION_SPEC.md:334-345`: schema currently lists only `legacy` and `loft_profile`.
- `docs/IMPLEMENTATION_SPEC.md:371-381`: `radial_segments` minimum 12.

## New Core Contracts

Create explicit data contracts under `blender_blocking/geometry/`:

```python
@dataclass(frozen=True)
class Bounds2D:
    x0: float
    y0: float
    x1: float
    y1: float

@dataclass(frozen=True)
class Bounds3D:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

@dataclass(frozen=True)
class ProfileIntervalPx:
    x0: float
    x1: float
    confidence: float
    source: str

@dataclass(frozen=True)
class ProfileBand:
    t: float
    intervals: tuple[ProfileIntervalPx, ...]
    center_x: float | None
    width_px: float
    holes: tuple[ProfileIntervalPx, ...]
    moments: Mapping[str, float]

@dataclass(frozen=True)
class ViewConstraint:
    view: Literal["front", "side", "top", "custom"]
    mask: SilhouetteMask
    camera: OrthographicCameraSpec
    bbox: Bounds2D | None

@dataclass(frozen=True)
class ReconstructionTarget:
    constraints: tuple[ViewConstraint, ...]
    profile_bands: Mapping[str, tuple[ProfileBand, ...]]
    bounds: Bounds3D
    config_hash: str
```

Every reconstruction mode consumes `ReconstructionTarget`. This avoids the current mode-specific re-reading and re-normalization in `main_integration.py`.

## Fix The BBox Contract First

F01 must be the first implementation change in this workstream:

- Replace tuple `normalize_bounds` in `mesh_generator.py:22` and `mesh_generator.py:79-86` with `Bounds2D`.
- Keep compatibility only at module boundaries: if a tuple arrives, parse it through `Bounds2D.from_xyxy`.
- Rename any `(min_x, max_x, min_y, max_y)` expectation to explicit property access.
- Update `main_integration.py:226-235`, `main_integration.py:972`, and `main_integration.py:983` to pass typed bounds.
- Add a non-square, non-origin bbox unit test that would fail with the current tuple order.

This is not optional because every ambitious mode that extrudes or normalizes contours depends on correct bounds.

## Replace One-Width Profiles With Profile Bands

The current profile representation loses recoverable detail:

- `dual_profile.py:73-129` stores one row width.
- `profile_extractor.py:83-147` returns one radius per height.
- `main_integration.py:656-668` disables offsets.

Implement `extract_profile_bands(mask, *, preserve_holes=True, preserve_components=True)`:

- Return all foreground intervals per sampled row, not just min/max.
- Preserve holes via contour hierarchy and per-row interval subtraction.
- Store centerline, component id, interval confidence, and edge softness.
- Allow resampling without discarding intervals.
- Keep `extract_vertical_profile` as a compatibility wrapper that reduces bands to the old `(height, radius)` list only for legacy callers.

Then upgrade `build_elliptical_profile_from_views`:

- Enable offsets by default for new modes.
- Support cross-section families:
  - ellipse,
  - superellipse,
  - polygonal ring from intervals,
  - multi-lobe cross-section when a row has multiple intervals.
- Store approximation error per slice so later refinement can target high-error areas.

## Reconstruction Modes

Update `config.py:9` and the schema in `docs/IMPLEMENTATION_SPEC.md:334-345` when source implementation begins. Proposed modes:

- `legacy`: compatibility only, not the quality target.
- `profile_loft`: typed profile-band loft with offsets and quality diagnostics.
- `silhouette_intersection`: explicit orthographic contour extrusion/intersection with holes and typed bounds.
- `visual_hull_voxel`: sparse/adaptive visual hull mode from Spec 04.
- `hybrid_loft_hull`: loft initialization constrained/refined by visual hull occupancy.
- `primitive_fit_refine`: profile or hull target followed by primitive fitting from Spec 05.

Do not alias ambitious modes to legacy behavior.

## Silhouette Intersection Upgrade

Current code at `main_integration.py:733-1027` should be split into:

- `geometry/contour_extrusion.py`: converts view contours and holes into extruded prisms.
- `reconstruction/silhouette_intersection.py`: coordinates view constraints, boolean operations, failure handling, and metrics.
- `integration/blender_ops/boolean_ops.py`: isolated Blender boolean application with structured result.

Requirements:

- Use contours from the canonical mask pipeline, not Canny.
- Preserve contour hierarchy from Suzuki/Abe-style retrieval.
- Keep multiple components unless config explicitly says otherwise.
- Produce debug meshes for each view prism and hole prism.
- Treat empty boolean result at `main_integration.py:1007-1027` as a structured failure with artifact paths, not a warning string alone.
- Support front/side/top orthographic constraints, not only front/side.

## Legacy Slice Path

The old slice-cylinder path in `primitive_placement.py:56-179` is not a quality foundation. Keep it as compatibility, but do not use it as the default target for ambitious work.

If retained:

- Make `z_overlap_ratio` configurable and recorded in manifests.
- Replace sequential booleans at `primitive_placement.py:365-379` with join strategy results from Spec 03.
- Track slice count, primitive count, boolean count, and join time.
- Warn when users increase slices to compensate for profile detail already discarded.

## Config And Documentation

During implementation, update the authoritative implementation spec in the following line-numbered areas:

- Revise scope at `docs/IMPLEMENTATION_SPEC.md:15-16` to include ambitious visual hull and reconstruction experiments as supported opt-in modes.
- Revise integration work at `docs/IMPLEMENTATION_SPEC.md:281-292` to include profile bands, visual hull, and primitive refinement.
- Revise schema at `docs/IMPLEMENTATION_SPEC.md:334-345` to include all real modes.
- Either keep `radial_segments` minimum 12 from `docs/IMPLEMENTATION_SPEC.md:371-381` or deliberately raise it for ambitious modes. Do not leave code at `config.py:102-103` and spec in conflict.

## Tests And Acceptance

Pure Python:

- `Bounds2D` tuple order regression.
- profile bands for holes, two lobes in one row, offset centerline, tiny components, anti-aliased edges.
- old vertical profile wrapper matches reduced band output.
- config modes validate and serialize round-trip.

Blender:

- silhouette intersection front/side/top asymmetric fixture.
- hole-preserving contour extrusion fixture.
- empty boolean failure produces manifest artifacts.
- profile-band loft with offsets improves or preserves per-view IoU against no-offset loft.

## Papers And References

- Laurentini visual hull concept: https://iris.polito.it/handle/11583/1401917.
- Matusik polyhedral visual hulls: `temp/perf-quality-audit-20260509/lane-b-reconstruction/paper-pdfs/matusik-2001-polyhedral-visual-hulls.pdf`.
- Suzuki/Abe contours: `temp/perf-quality-audit-20260509/lane-a-image-silhouette/paper-pdfs/suzuki-abe-1985-border-following.pdf`.
- Superquadrics Revisited: `temp/perf-quality-audit-20260509/lane-b-reconstruction/paper-pdfs/paschalidou-2019-superquadrics-revisited.pdf`.
- SuperFrusta/ResFit: `temp/perf-quality-audit-20260509/lane-b-reconstruction/paper-pdfs/ganeshan-2026-superfrusta-resfit.pdf`.
