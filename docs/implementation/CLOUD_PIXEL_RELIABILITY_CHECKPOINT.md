# Spatial probability and boundary reliability source checkpoint

S4 now transports observed foreground probability, pixel confidence and boundary
reliability as spatial data. Unknown pixels, including arbitrary/NaN stored values,
make no assertion. Cropped validity also updates the retained uncertainty maps and
observed-only summaries. Map shapes/ranges are checked against the original camera;
implicit resizing of mismatched inputs is refused.

Area resampling carries observed fractions and weighted numerators separately.
Coarse proposals can use partially observed cells with their actual reliability
weight. Coarse hard diagnostics require fully observed cells rather than interpreting
unknown zero padding as background. Original-resolution hard masks/validity remain
the admission evidence; zero confidence does not hide a known empty hole.

The named pixel_reliability_v1 CPU mode reaches the analytic soft-gradient objective,
while view_mean_legacy remains explicit. Quality chooses the new mode only without
an explicit alternative. Opaque fitted proxies and exact ray targets also receive
spatial maps. Ray observed fractions remain separate from confidence/boundary weights.
The CPU adapter saves original masked probability, confidence, reliability, hard
mask and validity arrays in a separate numeric artifact. These are input evidence,
not optimized primitive opacity.

Same-evidence identity now includes all observed spatial maps in the precision used
by CPU objectives. It ignores unknown map values, preserves sub-float32 observed
changes and distinguishes legacy endpoint-center cameras from explicit cell-edge
cameras. A CPU-renderer viewport follow-through uses the same complete pixel-cell
camera contract.

The affected run records 97 passing CPU contracts and 576 parsed Python files.
Ten new fixtures establish unknown-value invariance, camera/map identity, strict
shape/range validation, conditional coarse moments, independent weighted-gradient
agreement, confidence versus observed-fraction semantics, input-map retention,
known-hole admission and actual CPU adapter/artifact transport. One old test still
expected adaptive diagnostic thresholds to drive primary hard admission; it now
checks the already implemented C10 contract instead of rewarding low-opacity geometry.
The eight bounded pinned DVX stages remain passing after reliability integration.

No reconstruction or speed improvement is claimed. Full nonellipse derivatives,
confidence calibration, thickness-independent mesh-ray projection, native renderer
agreement and matched original-resolution reconstructions remain open. Soft evidence
can propose geometry; it cannot manufacture a hard gate or boundary certificate.
