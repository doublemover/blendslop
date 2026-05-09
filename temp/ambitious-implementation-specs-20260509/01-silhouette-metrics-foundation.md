# Spec 01: Canonical Silhouette And Metrics Foundation

## Scope

Owns findings F02, F03, F04, F05, F06, F18, F20, F22, F23, and F24. This is the foundation for every reconstruction path. The goal is not merely to patch a few defaults. The goal is to make silhouette extraction, canonicalization, comparison, and artifact reporting deterministic, configurable, explainable, and robust enough for ambitious visual hull and optimization workflows.

## Current Code To Modify

- `blender_blocking/config.py:32-54`: `SilhouetteExtractConfig` already carries `fill_holes` and `largest_component_only`; expand it into a full extraction policy.
- `blender_blocking/config.py:181-207`: `CanonicalizeConfig` exists but is not used consistently.
- `blender_blocking/geometry/silhouette.py:37-53`: auto luma polarity uses border mean vs central ROI mean.
- `blender_blocking/geometry/silhouette.py:74-75`: mask extraction defaults to cleanup enabled.
- `blender_blocking/geometry/silhouette.py:99-113`: thresholding applies the selected polarity.
- `blender_blocking/geometry/silhouette.py:127-138`: largest-component and fill-holes cleanup live here.
- `blender_blocking/validation/silhouette_iou.py:41-57`: validation wrapper calls mask extraction with `fill_holes=False` and `largest_component_only=False`.
- `blender_blocking/validation/silhouette_iou.py:61-111`: canonicalization crops, pads, resizes, and anchors.
- `blender_blocking/validation/silhouette_iou.py:126-149`: cache key hashes/copies full masks before lookup.
- `blender_blocking/validation/silhouette_iou.py:212-228`: IoU computes area intersection/union only.
- `blender_blocking/integration/shape_matching/shape_matcher.py:37-68`: compare helper exposes manual canonical options and ignores full config.
- `blender_blocking/test_e2e_validation.py:232-249`: E2E validation extracts and canonicalizes masks with hardcoded canonical size/padding.
- `blender_blocking/test_e2e_validation.py:251-298`: per-view threshold failures are printed but pass/fail is average-only.
- `blender_blocking/test_e2e_validation.py:253-269`: debug artifact names are fixed, not run-scoped.
- `blender_blocking/test_e2e_validation.py:733-741`: config overrides parse canonical settings but the compare path does not consume them in the core code above.
- `blender_blocking/integration/image_processing/image_processor.py:70-87`: legacy path extracts Canny edges.
- `blender_blocking/integration/shape_matching/contour_analyzer.py:46-48`: legacy path passes Canny output into `cv2.findContours`.
- `docs/IMPLEMENTATION_SPEC.md:106`: requires render/canonical sizes to be overridable and passed through.
- `docs/IMPLEMENTATION_SPEC.md:119-123`: currently describes the center-vs-border polarity heuristic.
- `docs/IMPLEMENTATION_SPEC.md:233-249`: defines canonical silhouette IoU.
- `docs/IMPLEMENTATION_SPEC.md:252-260`: requires generation context and manifest metadata.
- `docs/IMPLEMENTATION_SPEC.md:266-274`: requires tests and failure artifacts.
- `docs/IMPLEMENTATION_SPEC.md:281-287`: names files that must use canonical IoU and framed renders.
- `docs/IMPLEMENTATION_SPEC.md:291-293`: acceptance requires robust IoU and manifests.

## Target Architecture

Introduce a typed silhouette pipeline. Recommended module names:

- `blender_blocking/geometry/silhouette_types.py`
- `blender_blocking/geometry/silhouette_pipeline.py`
- `blender_blocking/validation/silhouette_metrics.py`

Core dataclasses:

```python
@dataclass(frozen=True)
class MaskBBoxPx:
    x0: int
    y0: int
    x1: int
    y1: int

@dataclass(frozen=True)
class SilhouetteMask:
    mask: np.ndarray
    bbox: MaskBBoxPx | None
    source: Literal["alpha", "luma", "manual"]
    polarity: Literal["dark_foreground", "light_foreground", "alpha_foreground"]
    threshold: float | None
    score: float
    diagnostics: Mapping[str, object]

@dataclass(frozen=True)
class CanonicalSilhouette:
    mask: np.ndarray
    source_bbox: MaskBBoxPx | None
    output_size: int
    padding_frac: float
    anchor: str
    transform: Mapping[str, float]
```

The old functions can remain as public entry points only if they delegate into this pipeline and return the same legacy type. Do not leave independent extraction logic in multiple modules.

## Extraction Policy

Expand `SilhouetteExtractConfig` in `config.py:32-54` with:

- `polarity`: `"auto" | "dark_foreground" | "light_foreground" | "alpha_foreground"`.
- `prefer_alpha`: bool, default true.
- `alpha_min_coverage`: small positive fraction so an all-opaque RGB image does not become a full mask.
- `min_area_frac` and `max_area_frac`: reject absurd candidates such as one-pixel dust and full-canvas masks.
- `max_border_contact_frac`: penalize masks that occupy the border unless configured.
- `morphology`: structured options for open, close, fill holes, largest component, min component area, kernel size, iterations.
- `anti_alias`: `"hard" | "soft_distance"` for future boundary metrics.

Replace `geometry/silhouette.py:37-53` with candidate scoring:

1. Generate alpha candidate when alpha exists and is not trivially full.
2. Generate dark-foreground Otsu candidate.
3. Generate light-foreground Otsu candidate.
4. Optionally generate adaptive threshold candidates for sketches or uneven backgrounds.
5. Apply the configured cleanup policy to each candidate.
6. Score candidates using area fraction, largest-component dominance, hole count, border contact, bbox compactness, contour count, and optional expected role hints.
7. Select the highest-scoring candidate or fail with diagnostics if all candidates violate hard constraints.

The center-vs-border heuristic from `docs/IMPLEMENTATION_SPEC.md:119-123` should be removed from the authoritative spec after implementation. It can remain as one weak scoring feature, not as a polarity decision.

## Canonicalization And Cache

Refactor `validation/silhouette_iou.py:61-111` so canonicalization consumes `CanonicalizeConfig` from `config.py:181-207` everywhere:

- `test_e2e_validation.py:241-245` must use config output size, padding, anchor, interpolation, and cleanup role.
- `shape_matcher.py:37-68` should accept `CanonicalizeConfig` or a new `SilhouetteCompareConfig`, not scattered scalar arguments.
- `mask_from_image_array` at `validation/silhouette_iou.py:41-57` must not have different cleanup defaults from `geometry/silhouette.py:74-75` unless the caller explicitly asks for raw masks.

Improve cache design at `validation/silhouette_iou.py:126-149`:

- Avoid unconditional full contiguous copies before checking cheap identity or caller-provided digest.
- Add optional `MaskIdentity` with shape, dtype, strides, version/digest, and extraction config hash.
- Benchmark warm and cold cache behavior for 128, 256, 512, and 1024 masks.
- Keep exact output equivalence tests for all cache changes.

## Metrics

Do not stop at area IoU. Add a metrics suite:

- Area IoU: current `compute_mask_iou` at `validation/silhouette_iou.py:212-228`.
- Boundary IoU: use Cheng et al., "Boundary IoU", saved at `temp/perf-quality-audit-20260509/lane-a-image-silhouette/paper-pdfs/cheng-2021-boundary-iou.pdf`.
- Empty-mask diagnostics: distinguish both empty, reference empty, render empty, and degenerate canonical crop.
- Optional signed-distance boundary loss for silhouette-loss optimization.
- Per-view threshold and aggregate summary. Aggregate is never the sole pass gate.

Create result types:

```python
@dataclass(frozen=True)
class SilhouetteMetricResult:
    view: str
    area_iou: float
    boundary_iou: float | None
    intersection: int
    union: int
    ref_area: int
    render_area: int
    pass_required: bool
    warnings: tuple[str, ...]
```

## E2E Validation Behavior

Change `test_e2e_validation.py:251-298`:

- `passed = all(view_result.iou >= view_threshold for required views)`.
- Average IoU remains a summary metric.
- Required view list should come from config or fixture metadata.
- Save debug artifacts for every failed view, and optionally for all views in audit mode.
- Debug artifacts must be under a unique run directory, not fixed filenames from `test_e2e_validation.py:253-269`.

Manifests must include:

- raw reference/render image paths,
- raw mask paths,
- canonical mask paths,
- diff image path,
- metric JSON path,
- extraction diagnostics,
- canonicalization transform,
- pass/fail reason per view.

## Legacy Canny Path Cutover

Replace the legacy path in `image_processor.py:70-87` and `contour_analyzer.py:46-48`:

- The canonical mask pipeline becomes the source for contours.
- Canny edge images become optional debug artifacts only.
- Contour extraction should use `cv2.findContours` on clean binary masks with `RETR_CCOMP` or `RETR_TREE` when holes matter.
- Preserve contour hierarchy for `silhouette_intersection` and profile-band extraction.

## Tests

Pure Python tests to add or extend:

- one-pixel outlier far from object,
- small dust clusters,
- holes,
- anti-aliased diagonal edges,
- disconnected components with largest-component on/off,
- off-center dark object on light background,
- corner-touching object,
- light object on dark background,
- all-opaque RGB image with alpha present,
- empty images and full images,
- canonical config override consumed by `shape_matcher` and E2E helpers,
- per-view fail with average pass should fail.

Blender-local tests:

- run E2E with per-view thresholds and inspect manifest links,
- verify run-scoped artifacts across two consecutive runs,
- verify render/reference extraction uses separate `SilhouetteExtractConfig` instances.

## Paper And Reference Links

- Otsu thresholding: `temp/perf-quality-audit-20260509/lane-a-image-silhouette/paper-pdfs/otsu-1979-threshold-selection.pdf`.
- Suzuki/Abe contour hierarchy: `temp/perf-quality-audit-20260509/lane-a-image-silhouette/paper-pdfs/suzuki-abe-1985-border-following.pdf`.
- Boundary IoU: `temp/perf-quality-audit-20260509/lane-a-image-silhouette/paper-pdfs/cheng-2021-boundary-iou.pdf`.
- OpenCV thresholding: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html.
- OpenCV morphology: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html.
- OpenCV contours: https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html.
