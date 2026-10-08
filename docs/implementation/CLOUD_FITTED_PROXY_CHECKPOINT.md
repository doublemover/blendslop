# Fitted opaque proxy source checkpoint

P8 now has an explicitly named `fitted_opaque_union_v1` path. The quality preset
chooses it only when no explicit alternative was given; `initializer_only` remains
available. The legacy initialization history is now labeled initialized, not fitted.

The fitter reuses the projected ellipse contribution/gradient cache and existing
local translation/rotation, positive-size and covariance-Cholesky adapters. Its
proposal residual balances observed foreground and background, preserves partial
cell coverage while downsampling and includes a smooth worst-view term. Analytic
loss pullbacks prioritize parts with actual evidence error; bounded coupled numeric
updates preserve independent objective-evaluation accounting.

Opacity, density and confidence are held out of geometric fitting. Proposal masks
use an opaque ellipsoid union matching the exported one-sigma mesh isosurface.
A nonmatching editable sigma is refused in this mode. Original-resolution mesh
checkpoints, final geometry and structural edits require nonregressing worst view,
boundary and known-empty evidence plus an explicit known-hole guard. A better soft
loss alone does not replace the seed. At most one negligible-unique-coverage prune
and one observed-missing-component reallocation are attempted within the cooperative
allowance. The actual fitted geometry, artist parameters and fit receipt are saved.
No single-solid certificate is inferred from an overlapping component set.

The Gaussian-to-ellipsoid conversion also now makes the eigenvector frame proper:
arbitrary eigenvector signs previously could invert mesh winding without changing
the covariance or implicit field. Two independent covariance fixtures establish
outward closed generated meshes after the repair.

`validation/fitted-proxy-focused.json` records 63 passing focused CPU contracts and
570 parsed Python files. Eleven new tests cover analytic local-gradient agreement,
confidence/opacity independence, partial-cell invariance, actual optimizer control
visits and safe retention, known-hole rejection, duplicate pruning, missing-region
support, proper Gaussian winding, editable isosurface consistency, quality-option
routing and backend artifact integration. Two existing Gaussian backend fixtures
remain passing. Synthetic soft-loss improvement is not a reconstruction quality
gain; strict hard evidence can and does retain the safe baseline. Blender/export
renderer parity, matched reconstructions, part-quality gains and timing comparisons
remain unmeasured.
