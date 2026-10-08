# Observed-hole coverage weighting and bounded step-size follow-through

This follows the pixel-contract checkpoint, retaining the same fixed synthetic
observation, world domain, 16-update allowance and 0.0703125 world signed-field
residual bound. Public legacy ranking and all default step sizes are unchanged.
No new clearance constant, larger automatic residual budget, dataset or timing
campaign is introduced.

## What changed

An optional known-empty-feature term compares the actual extracted-mesh pixel
coverage against the original probability samples inside fully observed enclosed
hole regions. It retains the existing spatial confidence and boundary reliability;
unknown holes contribute nothing. The feature weights must be a subset of observed
reliability, and this term is supported only by the actual original-pixel mesh
operator. It cannot turn the min-SDF surrogate into an opaque coverage operator.

Its normalized feature loss is separate from the whole-image loss so a small
required empty feature is not implicitly treated as an independent observation
or left unexplained in the aggregate. The requested coefficient defaults to zero
and is recorded with the objective terms and evidence weight sum. This is
reweighting of existing evidence, not a new geometric observation or certificate.

## Mechanism isolated

At learning rate 0.03, adding a feature coefficient of one left the same
25%-filled known-hole core. Adam largely normalizes gradient rescaling, and the
bounded 16-update latent trajectory had not advanced far enough. A coefficient
increase alone is therefore not presented as the remedy.

The matched higher-step-size pair uses explicit learning_rate=0.1 with the same
world residual bound and update count. Without the feature term, the separately
identified half-area IoU is 0.992826; with it, 0.975469. Both clear the known-hole
core completely, pass the existing exact contact and closed/oriented-volume
checks, and add zero newly violated original empty pixels on this track. Maximum
actual field changes are 0.069143 and 0.069424, within the unchanged bound.

The unchanged legacy scores are 0.909449 and 0.916, respectively, versus the
legacy source score 0.950617. Scores must not be compared across identities. The
feature term was not the main improvement in this one fixture. No default was
tuned from it, and no held-out/native reconstruction improvement is claimed.

Both actual backend paths still return the immutable source, because native
qualification is unavailable. Public metrics continue to describe that source.
The proposed field/geometry, objective terms, canonical/legacy masks and continuous
coverage are saved separately for the pending input-renderer/filter check.

## Clearance and certification limits

A half-minimum-plane-cell radius is only a necessary center-distance lower bound
for a true SDF consistent with an empty rectangular footprint. It cannot certify
all corners. A farthest-point/half-diagonal bound would require a valid Lipschitz
premise and suitable depth/interpolation handling; the optimized residual field
has no such certificate. No heuristic of either kind replaces actual output
projection/known-empty checking or native geometry qualification.

The validator's explicit --include-bounded-step-pair option saves all seven stages
with the fixed original camera and observation. Existing five-stage diagnostics
remain available without that option. These executable proposals are candidates
for later native acceptance, not accepted user-facing reconstructions.

Focused final checks: 144 host CPU contracts, 17 actual pinned numerical tests,
607 Python parse checks, and both actual higher-step-size backend retention paths.
Their scopes overlap earlier checkpoints and are not quality-case counts.
