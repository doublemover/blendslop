# Extraction connectivity reuse on matching actual geometry

This implements the first E5 diagnostic reuse seam. Dense and hierarchical
extraction now produce the same immutable typed index-connectivity receipt as
the geometry cache. The receipt includes the evaluator version and a hash of
vertex count plus the actual triangle indices. A matching extracted report is
passed into the visual-hull consumer and its owned geometry cache, avoiding a
second connectivity traversal. Changes to indices, vertex count or evaluator
invalidate reuse; an ordinary unbound dictionary is insufficient.

Coordinate-only edits may reuse index diagnostics because this evaluator is
explicitly index-only. They do not reuse geometric validity, orientation,
intersection, native qualification or final silhouette acceptance. A collapsed
coordinate fixture deliberately remains index-watertight while the independent
solid guard rejects it. Accepted adaptive coordinate edits now discard stale
extraction normals and field values, while retaining matching connectivity.
Topology provenance and its limited scope are visible in extraction metadata.

## Bounded evidence

Six new contracts and 59 affected CPU tests passed. Four existing native Blender
tests were skipped. All 585 Python sources parsed. Actual scikit-image 0.26 tiny
fixtures demonstrate receipt matching and invalidation. Two bounded 8-cubed
visual-hull backend fixtures, with and without an owned geometry cache, each
perform exactly one connectivity evaluator call and retain their required mesh
and volume artifacts. The accepted-coordinate-refinement fixture is mocked and
only verifies the changed consumer contract.

No matched timing or reconstruction campaign was run. These source construction
counts do not establish a speed or quality gain. This receipt is not a native
solid certificate. Other E5 work, including target-derived-field reuse and
delaying rejected coarse artifacts, remains open. Historical saved meshes and
scores are unchanged; native Blender render/export acceptance remains pending.
