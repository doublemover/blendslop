# Seeded and conditioned DVX source checkpoint

No pinned Torch/DVX execution occurred. The cloud currently has NumPy/SciPy,
while the existing approved DVX contract requires its exact CPython 3.13 CPU
runtime. That guard remains intact. This checkpoint contains independently
validated CPU mathematics and source integration, not an end-to-end solver pass,
reconstruction improvement or speed claim.

## Implemented source and independent contracts

- D1: compact retained seeds are admitted only with the same mask/validity/camera
  evidence identity, passing observed-view evidence and a closed orientation/volume
  screen. Overdense meshes exceed the declared seed allowance rather than silently
  entering the solver. Four fresh ellipsoids remain the last fallback. Retained
  evaluated surface coordinates/connectivity and available original artist JSON
  are preserved separately; unchanged artist parameters never claim to reproduce
  the deformation. Actual seed boundary certification remains separate.
- D2: observed-ray targets use original fixed camera viewports, never the DVX cube
  as replacement camera bounds. Unknown subcells are omitted. Canonical depth axes
  agree with z,y,x winding grids and the original image's vertical flip. Background
  rays penalize predicted coverage and foreground rays require intersection.
  Multipart fallback uses per-part coverage union; this surrogate does not remove
  internal surfaces or qualify an actual union. The old summed-winding occupancy
  path remains an explicit legacy mode.
- D3: deterministic subcell coverage targets carry a separate observed fraction.
  The representation says that matching the actual pinned voxelizer's filter is
  still pending; no exact footprint agreement is invented from target smoothing.
- D4: a 4x4x4 trilinear cage has at most eight weights per vertex and an independent
  scatter-add pullback. Uniform-Laplacian differential coordinates have a reusable
  CPU sparse factorization and a transposed-solve pullback. Independent finite
  derivative checks and exact affine/seed reproduction fixtures pass. Optional
  Torch pullbacks and the DVX-conditioned solver are source-integrated but unrun.
- D5: the conditioned path separates displacement, relative edge distortion,
  relative triangle-area barrier and normal-change terms. Poorly observed vertices
  receive stronger seed anchoring. Coordinate updates backtrack if the fixed cube
  would be violated. The ae377fe exact within-part boundary guard still controls
  final deformation admission and retains rejected proposal geometry.
- D6: increasing grid levels share one physical transform, seed connectivity and
  coordinate optimizer state. Every level rebuilds coverage/ray targets. Best
  states are compared within their current level, then carried forward; raw losses
  are explicitly not comparable across levels. Scarce requested updates use final
  levels instead of claiming unevaluated fine work. Progress includes stage/grid,
  counts and elapsed time. Stage setup/last evaluated level remain distinguishable.
- Primitive-fit and DVX optional typed settings now reach workflow overlays, with
  legacy alternatives available explicitly. Quality chooses ray/cage staging only
  when no alternate DVX mode was specified. Execution remains separately opt-in.

## Retained artifact replay

The new evaluated-seed path can have no primitive initialization list. Its actual
seed/deformation NPZ, seed OBJ, retained OBJ and optional artist source hash are
checked independently. Tampering with the retained artist source fails replay.
The existing four-ellipsoid replay continues to check parameter roundoff against
stored seed coordinates. No deformed output is relabeled parameter-reproducible.

## Evidence and remaining limits

New fixtures establish cage affine reproduction and pullback, sparse decode and
pullback, relative regularizers, ray depth ambiguity, image/grid orientation,
partial coverage invariance, per-part union semantics, same-evidence seed binding
and mocked no-op seeded artifact replay. A mocked executor is not actual DVX
execution. Final pinned-runtime forward/backward, staged optimization, fold rates,
Blender scene integration and all reconstruction quality/performance remain
pending. No new dependency runtime, dataset, GPU or paid compute was acquired.
