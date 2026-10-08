# Opt-in residual backend: same-evidence output acceptance and replay

The N2 numerical foundation now has an explicitly selectable `implicit_residual`
backend. It is not added to quality presets or default candidate lists. Numerical
execution stays opt-in and requires the existing pinned Torch 2.14.1+cpu runtime;
no automatic dependency installation occurs inside reconstruction.

## Source and evidence contracts

Seed selection requires actual retained geometry, matching camera/evidence
identity, passed rows for every observed input view, a closed/oriented volume
screen, and at most 512 source triangles. The original exact within-part boundary
predicate runs before field construction. Unsupported, clipped, unresolved,
multi-part or oversized sources decline without replacing the source. Original
source geometry is immutable; missing source ancestry is not reconstructed.

The field uses a fixed padded world cube and the original camera pixel-cell
viewports. Original probability, confidence, boundary reliability and validity
are integrated without turning unknown/outside-camera samples into foreground or
background. Only complete, wholly reliable zero-foreground footprints become
hard empty columns. The solver keeps the evaluated field snapshot and validates
returned evidence identity, constraint hash, finite scored checkpoint, residual
bounds, protected-empty values and inactive samples before extraction.

Both ensemble paths defer an explicitly requested residual until preceding
source candidates are available. The measured scheduler waits for pending source
jobs before attaching their results. Candidate allowances are propagated through
seed preparation, owned-helper startup/fit, exact guard and optional native check.
Existing contact and folded-part guards are unchanged.

## Proposal versus delivered output

The cell-center min-SDF/sigmoid objective remains a proposal surrogate. Extracted
triangles are independently scored using the existing original-size opaque
polygon projection and observed-pixel metrics. Admission requires required-view,
worst/mean area, boundary, known-empty-feature and new original-pixel empty
violation guards, plus exact output contacts and content-scoped native boundary
qualification. A smaller surrogate loss cannot substitute for these gates.

Unqualified or worse proposals retain the actual source. Source/artist geometry,
residual field, proposed topology-changing mesh and delivered output have distinct
artifacts/hashes. Saved-field replay is authoritative and reproduces exact
extraction geometry; OBJ and artist-source digests are separately checked.
Primitive/artist parameters do not claim to replay the final residual geometry.
Native render/export acceptance remains a separate, unrun stage.

## Runtime fixes found by actual execution

The restored pinned environment itself is new; historical runtime receipts remain
separate. Actual warm integration caught two transport problems: inherited host
PYTHONPATH/PYTHONHOME could load extension wheels built for another interpreter,
and TorchVersion pickle values could import Torch in the image/meshing host.
Owned helpers now isolate those Python path overrides, and implicit receipts
serialize the numerical version as an ordinary string. The selected venv
interpreter still preserves its symlink/prefix. Interrupted work publishes only
an evaluated field checkpoint, never an unscored last optimizer update.

A full/empty camera also exposed OpenCV's maximum-float absent-boundary distance
sentinel overflowing the SDF metric. Those two finite-image cases now use an
explicit image-diagonal saturation and float64 reduction; identical masks remain
zero. This is a metric-contract correction, not a revision of historical scores.

## Bounded new evidence

- 16 new pipeline contracts cover unknown/soft evidence, original non-square
cameras, source-view completeness, budgets, modified artifacts, artist/field/source
separation, original-size admission, and anisotropic 16/32 world cell centers.
- 133 host CPU tests pass across 15 focused suites, including original exact-contact
and folded-proposal regressions. Nine actual pinned numerical tests pass; one is
new scored-progress replay. These scopes overlap prior foundational checks.
- All 602 Python sources parse; registry and whitespace checks pass.
- Three actual source→owned pinned helper→field→extracted mesh→opaque score→artifact
replay fixtures share one helper process. Two use a real measured source
silhouette and one conditions a confirmed-empty through-hole in a thin plate.
The hole changes Euler characteristic 2 → 0 and passes the exact boundary guard,
but its actual top IoU drops 0.950617 → 0.940934; the source is correctly retained.
That negative fixture proves acceptance discipline, not reconstruction improvement.

No native Blender rendering, new reconstruction campaign, dataset acquisition,
Windows parity or measured shape/speed gain is claimed. Publication remains
parked after the prior cancellation. Larger/multi-part fields, sparse/adaptive
storage, thickness-independent field rendering, artist deformations and broader
feature/geometry acceptance remain open.
