# P5 bounded taper/bend checkpoint

An opt-in `deformed_superquadric` family now shares one local deformation across
the parametric mesh, surface samples, inverse-mapped signed field, contour
projection, serialized helper input, numeric proposal screening, and editable
shape-program compiler. Existing superquadric and default ranking paths remain
unchanged. Hollow variants are not implemented in this batch.

## Geometry contract

For base local coordinates `(x,y,z)`, axial radius `rz`, and `u=z/rz`, linear
taper maps `x` to `X=(1+tx*u)*x` and `y` to `Y=(1+ty*u)*y` on the base support.
Outside that support, taper factors saturate at the endpoint values. For signed
bend `b`, positive radius `R=rz/abs(b)`, sign `s`, and phase `a=abs(b)*z/rz`,
the circular map is

- `x' = s * (s*X*cos(a) + R*(1-cos(a)))`
- `y' = Y`
- `z' = (R-s*X)*sin(a)`

The zero-bend limit is implemented directly. `1-cos(a)` uses a stable
half-angle expression; inversion rationalizes `R-hypot(...)` near that limit.
The exact signed zero set and material sign follow inverse mapping. The signed
function is a distance proxy, not a Euclidean SDF or a 1-Lipschitz field.

Controls obey `abs(tx),abs(ty)<=0.35`, `abs(b)<=0.75`, and
`abs(b)*rx*(1+abs(tx))/rz<=0.45`. On the continuous base support, the determinant
is `(1+tx*u)*(1+ty*u)*(1-b*X/rz)`. A conservative lower bound is
`(1-abs(tx))*(1-abs(ty))*(1-ratio)`, strictly positive under these limits.
Positive radial coordinates and the restricted phase interval permit the
documented inverse. These are continuous-map bounds. They do not certify finite
triangulation contacts, native modifier output, silhouette quality, or clearance
of a rectangular pixel footprint. Exact contact and final native qualification
remain independent requirements.

## Integration and staged fitting

- Proper input pose and finite positive dimensions are validated before base
  normalization. Invalid dimensions cannot be silently clamped into admission.
- Numeric fields and screening use the world pose once. The compiler receives
  the same local deformed vertices and applies the same pose once.
- Reflection changes the local bend sign while retaining a proper pose.
- The shape-aware renderer projects the actual triangle union. Deformation is
  never assumed convex or replaced by an ellipse/convex hull.
- Primitive deformation parameters remain frozen until the explicit Boolean
  `deformation_fit_enabled` release. Existing center/size/pose fitting is still
  available in the coarse stage.
- Shape-program refinement exposes the three controls only with the explicit
  `deformation_refinement` configuration. Its default is false.
- Invalid coupled edits roll back atomically. A shape-program size/taper trial
  that invalidates the bend-radius limit is skipped without altering another
  control or aborting the search. Finite-difference pullbacks retain existing
  shared objective allowances and fail-closed reporting.

## Evidence and limits

15 new focused CPU checks cover the rigid limit, signed/tiny bend inverses,
posed surface/mesh field agreement, independently differenced Jacobians,
invalid controls, exact finite-mesh contacts, reflection, serialization,
compiler dispatch with mocked native APIs, real numeric screening, schema
validation, staged release, atomic rollback, triangle projection, cache
invalidation, and independently differenced deformation pullbacks.

129 focused host CPU checks passed together across this family and the existing
contour, pose, sweep, parameter adapter, shape-program, reconstruction, implicit
retention and pixel projection paths. Logs are retained in recovery artifacts.
These checks establish implementation behavior, not measured reconstruction
gains. Native Blender rendering/export has not run under repository execution
instructions. No held-out dataset, speed study, or broad timing campaign was
started. P5 remains experimental until final native outputs and real evidence
cases qualify it.
