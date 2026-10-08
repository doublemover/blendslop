# Continuous nonellipse contours and family-specific proposal derivatives

## Corrected contracts

The shape-aware cache previously rasterized nonellipse meshes onto integer
pixels before softening the mask. Its backward method returned zero nonellipse
world derivatives, after which the live optimizer nevertheless invoked an
ellipse-only parameter pullback. Unsupported frustum size/tilt controls could
therefore abort refinement instead of receiving an appropriate proposal route.

Nonellipse proposals now use continuous signed distance to projected mesh
contours at the actual pixel centers. Circular frusta and norm-convex
superquadrics use their actual finite projected-vertex hull; unknown or concave
families retain projected triangle unions. A concave superquadric negative
control prevents accidental global convexification. Real perforations, long
thin holes and disconnected loops remain represented above the declared
surrogate precision. Circular-frustum tessellation follows a projected chord
allowance and reports its analytic chord error, including cap exhaustion.
Generic finite mesh tessellation explicitly has no geometric error certificate.

Analytic ellipse pullbacks remain unchanged. Nonellipse controls use bounded
local-parameter differences of the continuous contour proposal, including
frustum taper/tilt and superquadric exponents. The numerical pairs consume the
shared forward-evaluation budget, reserve an admission slot, rotate across all
controls and report failures. The covariance-only backward interface now rejects
unsupported nonellipse use instead of silently returning misleading zeroes.
The mixed optimizer and renderer report the operator, approximation and numeric
fallback scope separately from ellipse-only steps.

## Precision and acceptance limits

GEOS floating unions can leave microscopic artificial sliver holes. A signed
distance surrogate would magnify their boundaries into pixel-wide craters.
Known convex families avoid redundant triangle unions altogether. Generic
proposal unions use an explicitly reported 1e-9-pixel precision grid and leave
holes or disconnected components of diameter at most 1e-7 pixels unresolved.
Large/long thin holes are not suppressed merely because their area is small.
Only derived 2D proposal contours use these approximations: original source mesh
coordinates, exact binary64 contact predicates and folded-part guards are
unchanged. Shapely's official precision behavior is documented in its
[union_all API](https://shapely.readthedocs.io/en/stable/reference/shapely.union_all.html).

This is a partial P10 implementation. It is not an exact area-filtered ray,
final opaque renderer or native qualification certificate. Generic curvature
error bounds, shared final tessellation policy, smooth worst-view staging and
periodic exported-render acceptance remain open. The existing strict score
retains only scored candidates; the proposal surrogate cannot replace the best
admitted state on an unsupported, failed or exhausted trial. Missing optional
Shapely fails clearly for nonellipse contours while the ellipse route remains
available.

## Actual bounded evidence

14 new contracts and 83 affected CPU tests passed; all 587 Python sources
parsed. The contracts cover subpixel continuity, independent local-parameter
derivative differences, mixed-family compositing/cache reuse, actual frustum
optimization under one shared allowance, full rotating control coverage,
optional-dependency failure, holes, a nonconvex negative control and
axis-aligned hidden-exponent invariance. The invariance fixture caught phantom
union boundaries that would otherwise fabricate an exponent gradient. A final
focused chord/invalid-dimension check also passed.

These are numerical/source fixtures, not measured reconstruction-quality or
speed gains. Native Blender render/export and larger actual-scene acceptance
remain unrun. Historical saved geometry and quality scores are unchanged.
