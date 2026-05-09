# Spec 09: Uncertainty-Aware Masks, Profiles, Voxels, And Decisions

## Scope

This expands idea 3. Binary masks are brittle. The pipeline should carry uncertainty from image extraction through profile bands, visual hull carving, candidate scoring, and human correction. This allows ambitious algorithms to reason about ambiguous edges instead of committing too early.

Related findings: F02, F03, F07, F08, F16, F18, F22, F23, F24, F28.

## Current Code To Modify

- `blender_blocking/geometry/silhouette.py:37-53`: polarity decision is hard.
- `blender_blocking/geometry/silhouette.py:99-113`: thresholded output becomes binary immediately.
- `blender_blocking/geometry/silhouette.py:127-138`: cleanup removes uncertainty.
- `blender_blocking/validation/silhouette_iou.py:61-111`: canonicalization returns binary masks only.
- `blender_blocking/geometry/dual_profile.py:73-129`: profile extraction stores one hard interval.
- `blender_blocking/integration/multi_view/visual_hull.py:76-90`: point-in-silhouette is a bool.
- `blender_blocking/integration/multi_view/visual_hull.py:238-300`: voxel grid is bool.
- `blender_blocking/validation/silhouette_iou.py:212-228`: metric is binary IoU.

## Uncertainty Data Types

```python
@dataclass(frozen=True)
class UncertainMask:
    foreground_prob: np.ndarray
    hard_mask: np.ndarray
    confidence: np.ndarray
    source: str
    threshold: float | None
    diagnostics: Mapping[str, object]

@dataclass(frozen=True)
class UncertainProfileBand:
    t: float
    intervals: tuple[ProfileIntervalPx, ...]
    center_mean: float
    center_std: float
    width_mean: float
    width_std: float
    confidence: float

@dataclass(frozen=True)
class UncertainVoxelGrid:
    occupancy_prob: VolumeData
    hard_occupancy: VolumeData
    confidence: VolumeData
    source_view_agreement: VolumeData
```

Keep hard masks for compatibility, but use probability/confidence for scoring and refinement.

## Mask Confidence Generation

Sources:

- alpha values,
- luma distance from threshold,
- Otsu between-class separation,
- candidate polarity score margin,
- distance to boundary,
- component stability under morphology,
- anti-alias edge gradient,
- human scribble certainty.

Implementation:

- `geometry/silhouette_pipeline.py` should produce both hard and uncertain masks.
- `degradations.py` from Spec 07 should generate known uncertainty cases.
- Confidence values should be normalized to `[0, 1]`.

## Soft Metrics

Add:

- soft IoU,
- expected area IoU under mask probabilities,
- boundary uncertainty penalty,
- disagreement heatmap,
- confidence-weighted Boundary IoU.

Use hard metrics for gates and soft metrics for ranking/refinement.

## Uncertain Visual Hull

Replace bool-only predicate from `visual_hull.py:76-90` with:

- `inside_probability(point, view)`,
- `confidence(point, view)`,
- `hard_inside(point, view, threshold)`.

Voxel occupancy:

```text
P(inside all views) = product(P_inside_view) or min(P_inside_view)
confidence = aggregate(view confidence)
```

Keep configurable aggregation:

- `min`: conservative visual hull,
- `product`: probabilistic,
- `logit_sum`: smoother optimization target.

Store:

- `occupancy_prob`,
- `hard_occupancy`,
- `boundary_band`,
- `view_disagreement`.

## Uncertain Profiles

Profile band extraction should sample multiple thresholds:

- low threshold,
- chosen threshold,
- high threshold,
- morphology variants.

For each row:

- measure interval stability,
- width variance,
- center variance,
- hole stability,
- component stability.

This directly addresses one-width-per-height fragility from F07.

## Candidate Scoring Integration

Spec 08 should consume:

- minimum confidence in matched regions,
- penalty for high-quality score achieved only in low-confidence pixels,
- reward for satisfying high-confidence constraints,
- caution flag when selected candidate depends on uncertain boundary regions.

## Artifact Outputs

For each mask/view:

- probability map PNG/NPZ,
- confidence map PNG/NPZ,
- threshold candidate comparison,
- boundary uncertainty heatmap,
- chosen hard mask,
- diagnostics JSON.

For volumes:

- occupancy probability slices,
- confidence slices,
- disagreement slices,
- histograms.

## Human Guidance Hook

Human scribbles from Spec 13 are represented as certainty overrides:

- foreground scribble: `foreground_prob=1`, `confidence=1`.
- background scribble: `foreground_prob=0`, `confidence=1`.
- uncertain brush: decrease confidence, not force foreground/background.

## Tests

- identical hard mask with different confidence should affect ranking but not hard IoU.
- outlier pixel with low confidence should not dominate canonical crop.
- polarity candidates with close scores should mark uncertainty.
- visual hull with uncertain boundary should produce boundary band.
- human foreground/background scribbles override low-confidence extraction.

## References

- Boundary IoU: `temp/perf-quality-audit-20260509/lane-a-image-silhouette/paper-pdfs/cheng-2021-boundary-iou.pdf`.
- Soft Rasterizer: `temp/perf-quality-audit-20260509/lane-f-literature/paper-pdfs/liu-soft-rasterizer.pdf`.
- Generalized reprojection error: https://pmc.ncbi.nlm.nih.gov/articles/PMC4281271/.
