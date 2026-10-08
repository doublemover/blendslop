# Oriented structure and evidence checkpoint

This is source implementation with bounded CPU correctness fixtures. New retained
reconstruction quality and performance have not been measured. Historical
06a7f02 / 8e7c15d measurements and ae377fe exact-contact/fold guards are preserved.

## Implemented

- Whole oriented box, ellipsoid, cylinder and frustum proposals fit directional
  supports directly from the calibrated masks. Pose and positive sizes are
  coupled through bounded CPU least squares. Numeric-Jacobian calls count toward
  the support allowance, and the best fully scored state survives interruption.
  Unknown farther pixels produce one-sided support bounds. Known holes exclude
  these convex hypotheses. Full actual mesh/render admission is still required.
- Shape-program search includes a one-box level and interleaves whole-shape
  hypotheses with grammar/cuboid/convex alternatives. ResFit gets a compatible
  whole seed alongside existing default/profile seeds. Superquadric box seeding
  is explicitly approximate and receives actual field/mesh scoring afterward.
- Strong-seed starts are deduplicated and screened with the same full objective;
  the best distinct seeds receive fitting work. A scored seed survives when
  refinement is unavailable. Preparation and objective evaluations are separate
  work categories; seed screening calls are charged to the shared allowance.
- `normalized_area_v1` divides squared geometric distances by a declared object
  length scale. Exact analytic surface samples get deterministic area weights
  derived from a small mesh quadrature; this is an approximation, not a claim of
  exact surface area. Reverse residuals use visible represented area. Directed
  overlap terms sum instead of being diluted by unrelated part-pair counts.
  `legacy_world_squared` remains an explicit comparison mode.
- A fixed-size residual vector squares to the complete weighted objective. New
  coupled part blocks reuse the existing local/log-size/Cholesky adapters and
  cached unchanged geometry. Every numeric proposal uses the shared count/time
  allowance. Coordinate descent remains available. Actual parameter visits are
  retained; scheduled-but-unvisited controls are not counted as visits.
- Residual ownership favors near-boundary explanations over the most negative
  field. Changed/new parts receive control priority, while untouched parts share
  coordinate coverage. Every proposal still needs a complete objective gain.
  These point residuals remain hull-derived; they are not relabeled observations.
- Revised extent penalties depend on actual primitive extents, not only centers.
  Unchanged extent work is cached. Input-only uncertainty/topology constants stay
  diagnostics rather than pretending to correct geometry.
- Local known-empty-hole checks accompany fresh final rendering. A contour-section
  loft is eligible when its effective representation can preserve a hole. A
  capped loft cannot inherit eligibility solely from a whole-image score. The
  contour alternative may be retained for required-feature gain even when the
  capped alternative has a higher aggregate score.
- Structure routing spends a slot on an actual numeric whole-shape challenger,
  reuses its prepared program and validates that challenger directly. It does
  not stop merely because a high-scoring hull appeared first. Construction-only
  failures release render reservations; real completed render work still counts.
- `structure_v1` is an explicit compactness prior among measured alternatives
  within .006 mean/worst/boundary evidence tolerance, with all required views and
  known empty features passing. `silhouette_only` remains available. This prior
  cannot identify hidden shape and has not been promoted by benchmark results.
- Structure-first hybrid retains a profile seed and grows only where an actual
  known foreground ray is unexplained. Distant hull bulges and unseen cavities
  are no longer treated as missing material in that mode. Hull-first is retained.
- The quality preset enables the coherent revised objective, whole support seed,
  coupled blocks, structure routing and structure-first hybrid. Each new option
  can be explicitly set to its legacy alternative for a matched future ablation.

## Remaining work and validation limits

The focused suite covers support algebra, tilted analytic fixture fitting, scale
invariance, area/overlap contracts, residual-vector identity, bounded coupled
updates, changed-part priority, duplicate seed handling, local holes, actual
router reservations and observed growth demand. Fake executor fixtures establish
routing control flow; they are not real backend executions or latency evidence.

No Blender invocation occurred, in accordance with repository instructions. The
actual scene/export/Boolean integration, target-specific retention and all quality
or speed claims remain pending. No new dataset or broad reconstruction campaign,
GPU rental, publication or paid compute occurred. A NumPy overflow warning from
existing extreme-input backend fixtures is retained in the passing test log.

Still open within the larger design: coherent local-frame profile fitting, a
fully fitted Gaussian proxy, confidence-weighted smooth boundary objectives,
conditioned seeded DVX, new sweep/extrusion/implicit representations, warm-helper
transport and qualification/toolchain reuse. They are not claimed as implemented
by this checkpoint.
