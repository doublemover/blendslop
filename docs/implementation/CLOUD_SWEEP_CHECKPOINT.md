# Editable local-frame sweep source checkpoint

N1 now has a first bounded, editable generalized sweep representation. Sections
share one proper rigid frame; a monotone local axial coordinate carries a
piecewise-linear centerline, positive x/y section scales and a convex superellipse
exponent. The coordinate map cannot fold its axial ordering. The mesh and implicit
field share all knots, centers, scales, exponent and caps. The field has the
correct sign and boundary zero set; it is explicitly not Euclidean distance.
Sampled face-centroid tessellation disagreement is reported separately from exact
vertex agreement and is not called a global Hausdorff certificate.

P6 is advanced, not fully completed: local pose, axial size, centerline bend,
section taper and exponent are available to actual geometry program refinement,
and all per-knot center/radius controls reach the existing ResFit local/log-size
adapters. Adaptive knots retain physical section/centerline changes. Initial
hypotheses still come from calibrated world-axial row evidence, with censored
completion labeled and unobserved camera dimensions refused. Automatic jointly
optimized full section splines, polygon/hollow sweep sections, transported/twisted
frames and branched or looped centerlines remain open. Axial section births/deaths
require another representation rather than silently connecting them.

Source/compiler/numeric screening use the same sweep generator. Validation
refuses malformed sections or improper pose before compiler scene mutation.
Complementary routing can screen three convex section hypotheses, and final
original-view render/known-hole gates still govern retention. Program controls
cycle through pose and section groups; two 14-control fixture windows reach all
28 controls for one sweep. Runtime deadlines can still stop any search early.

Follow-through found a calibrated legacy builder could fall back to pixel curves
when either axis had only censored rows. It now declines those exact curves and
labels its bounds-prior alternative. Row evidence also records whether the sampled
height actually lies in the camera viewport, preventing an extrapolated isolated
endpoint from being mistaken for a new supported axial component.

`validation/sweep-focused.json` records 57 passing focused CPU contracts and 573
parsed Python files. Ten new tests establish capped outward winding, implicit
vertex agreement across exponents, pose/serialization agreement, sampled chord
error reduction, fold/improper-frame rejection, safe parameter adapters, identical
screen geometry, fail-before-mutation behavior, cyclic bend/taper/exponent controls,
completed evidence proposals and censored legacy fallback. Existing profile,
contact/frustum/pixel, grammar, adapter and routing contracts remain passing.
Blender compilation/render/export, exact final boundary qualification and all
new reconstruction quality/performance measurements remain pending.
