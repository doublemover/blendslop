# Quality follow-up backlog - 2026-10-07

Status: OPEN; documentation only. These tasks authorize no new measurement run.
Preserve the qualified repair commits `4d286fd` and `92f7ba26`, original fixtures,
and completed Windows validation receipts. Do not repeat the completed suite.

The current Blender5.2 render-IoU check passed front0.908, side0.905, top0.994
against0.700 per-view thresholds. Inspected front/side renders still show banding.
Silhouette agreement therefore remains a separate verdict from surface quality,
geometry, topology, editability and broader shape coverage.

## QUALITY-SURFACE1: diagnose and remove unintended surface banding

Goal: remove unintended rings, ridges and shading discontinuities on shapes whose
authored reference is smooth, while preserving intended corners and silhouette.

Deliverables:

- Retain the observed vase renders and exact evaluated mesh as the regression
  evidence. Record whether bands arise in geometry, normals, shading or rendering.
- Define an analytic smooth reference and an intentionally stepped negative
  control with closely matched silhouettes. Include a sharp-edge control so a
  global smoothing operation cannot qualify by erasing intended features.
- Measure deterministic area-weighted surface distances and normal-angle error
  against the reference, plus longitudinal radius/curvature continuity where the
  reference supports it. Declare world units, normalization, sampling density,
  normal orientation and treatment of seams before evaluating a candidate.
- Save front/side/top and oblique neutral-lit renders, a normal visualization,
  the exact rendered/exported mesh, per-case measurements and failure diagnosis.

Acceptance:

1. Report silhouette and surface verdicts independently. Preserve existing
   per-view area-IoU, boundary and signed-distance gates; a silhouette pass cannot
   override a surface failure or missing surface evidence.
2. Establish surface tolerances before candidate measurements from analytic
   references, tessellation error and independently qualified measurement noise.
   Record numeric tolerances and units in the fixture manifest; do not fit them to
   the observed candidate or relax an existing assertion.
3. The smooth reference passes and stepped control fails the surface gate despite
   similar silhouette scores. Intended sharp edges remain present in their control.
4. The repaired vase passes those frozen gates and shows no unintended bands in
   the retained inspection views. Explain any remaining geometry/shading defect;
   do not label it accepted solely because a mask matches.

## QUALITY-COVERAGE1: broaden deterministic shape coverage

Goal: qualify behavior beyond the current vase and existing narrow sample checks.

Prepare a bounded matrix with one frozen reference per family: sphere, anisotropic
ellipsoid, cylinder, tapered frustum, smooth vase, torus, capsule, rounded box,
thin plate, concave arch, asymmetric multipart solid, and the rounded triangular
case below. Distinguish intentional seams, sharp features, thickness and cavities.
Use fixed dimensions, orientation, cameras, masks, evaluated reference meshes,
seeds and configs. Include front/side/top and two specified oblique views.

Acceptance:

1. Freeze the exact workload manifest, source head, dependency requirements,
   time/memory limits and per-family metric contracts before an explicitly
   requested run. Declare held-out cases before tuning; no historical model A/B
   campaign is implied by this task.
2. For every required case retain independent silhouette, surface-distance/normal,
   topology and editability verdicts where those metrics apply. Record unsupported
   metrics and dependencies as explicit blockers, not zero errors or passes.
3. Verify exported/evaluated geometry is the same geometry used for rendering.
   Keep camera/frame normalization identical between reference and reconstruction.
4. A required-case failure blocks coverage acceptance even when matrix averages
   pass. Retain per-view/per-case receipts, images and exact mesh/config hashes.
5. The matrix detects controls with good silhouettes but incorrect depth,
   cavities, curvature or multipart geometry. Keep local numerical contracts and
   broader reconstruction acceptance distinct; do not weaken current tests.

## QUALITY-TRIANGLE1: rounded dot-shaped triangle fixture

Goal: add the owner's rounded dot-shaped triangle as a named quality case.
Working interpretation: a compact rounded triangular pebble/dot with three
distinct corners, rounded perimeter and finite depth. Freeze the authored reference
before execution; do not silently replace the requested shape with a sphere,
plain sharp triangle or a flat circular dot. Record any later owner refinement.

Deliverables:

- An authored reference specifying triangle vertices, corner radius, thickness,
  front/back dome or bevel profile, units and orientation. Retain a labeled preview
  that makes the intended three-corner outline and depth visible.
- Exact front/side/top/oblique reference masks and evaluated mesh, deterministic
  sampling, and a manifest tying them to the authored parameters.
- A smooth rounded candidate, a circularized/over-smoothed negative control and
  a banded/stepped negative control to qualify distinct failure modes.

Acceptance:

1. Freeze per-view silhouette and boundary tolerances and separate surface and
   thickness/curvature tolerances from analytic geometry and measurement error
   before evaluating the reconstruction.
2. Preserve all three intended corners, their rounded transitions, relative
   placement and finite depth. A circle-like projection cannot pass by a generous
   aggregate IoU; side/oblique evidence must detect a flattened reconstruction.
3. The circularized control fails shape preservation; the banded control fails
   surface quality even if its silhouette passes. The authored smooth reference
   passes both independent gates.
4. Retain the final evaluated mesh and inspection views, and add the fixture to
   the bounded coverage matrix without changing existing vase gates or claiming
   unmeasured families are accepted.

Execution remains pending a separately requested bounded workload. No rendering,
benchmarking, dependency installation, publication or cleanup accompanies this
backlog addition.
