# Frozen rounded triangular source certificate

The exact original authored source is continuously certified. Its bidirectional analytic-surface/facet distance upper bound is **0.00040500062340535856 world units**. Each actual oriented facet differs from every outward analytic normal in its corresponding parameter cell by at most **6.5098278988276554 degrees**. These are independent approximation certificates, not artist limits, candidate acceptance gates, nearest-normal uniqueness claims or camera certificates.

[The declaration](frozen-declaration.json) froze the exact binary64 author coordinates, symmetric depths, interpretation and five source/dependency byte hashes before implementation, test arithmetic or the retained-source calculation. Earlier source-only feasibility exposure was already disclosed. The strict public API recomputes actual array identity and admits only source indexed hash `2681272b491af3370176a3751e408ef36496ba592d230bce7c733b8c41cadc64`; the original NPZ byte SHA256 is `f6abc19a7ee7efbc62f3bc403b5e61ce6c426c64950604d0fa5cd7181c698cff`. It uses the actual original 6,239 vertices and 12,474 triangles, with no geometry replacement, equilateral substitution, construction tolerance, native process or surface resampling.

The new API is `evaluation.rounded_triangle_reference.rounded_triangle_reference_certificate(reference)`. Unsupported source/declaration/identity/cover/orientation conditions return an explicit reason and null artist limits. The public API never supplies candidate boundary qualification.

## Exact author angles and complete cover

Convert each supplied binary64 XY coordinate and radius to its exact rational value. The centroid is exactly zero for these particular supplied values. The exact isosceles symmetry is checked; equality to an ideal equilateral triangle is not assumed. Let `a` be the positive supplied X coordinate, `b = top_y - base_y` using exact rational subtraction, and `beta = atan(a/b)`. The supplied ratio squared differs from `1/3`.

With `t = (a/b - 1)/(a/b + 1)`, the proof encloses `beta = pi/4 + atan(t)` by a 40-term rational alternating series with its next-term remainder; `-1/3 < t < 0` is checked. Existing torus helpers provide a rational Machin enclosure for pi and directed Decimal50 Taylor sine/cosine enclosures. Interval angles use a rational anchor turn and the sine/cosine 1-Lipschitz bound. Exact outward rounding to a 100-bit dyadic interval grid bounds arithmetic work. Neither NumPy nor libm production rounding is assumed correct: its combined displacement is measured against the analytic interval vertices.

The three corner arcs, in cyclic outline order, have angles

- `theta0(t) = t*pi + (1-2*t)*beta`;
- `theta1(t) = (1+t/2)*pi + (-1+t)*beta`;
- `theta2(t) = (-1/2+t/2)*pi + t*beta`.

Each arc retains 33 endpoints. The following straight offset edge links its last endpoint to the next arc's first, giving 99 outline vertices and 99 edges: 96 circular intervals and three straight edges. The phi interval `[0,pi]` has 64 cells. Exact once-only oriented triangle inventory covers all 6,336 cells: 6,144 circular cells and 192 straight cells. Each of the 62 interior phi rows has two triangles per outline edge. Each of the two collapsed pole rows has one complete fan triangle per edge, for 198 pole triangles. Both cyclic outline seams and the original diagonal/winding are checked. Cyclic triangle rotations and face ordering are allowed by the cover proof; reversed winding, altered diagonals, duplicate/missing triangles are refused. The public indexed identity remains stricter.

## Continuous distance correspondence

Write `q(theta)` for the exact offset outline, `c` for its centroid and `h` for exact half-thickness. The analytic surface is

`F(theta,phi) = (c + sin(phi)*(q(theta)-c), h*cos(phi))`.

First interpolate the outline endpoints at fixed phi, then interpolate phi at a fixed outline chord position. Circular outline second derivative has norm `r`, so its linear interpolation error is at most `r*delta_theta^2/8`; straight outline interpolation is exact. A meridian at chord point `q` has second derivative norm at most `max(norm(q-c),h)`. The independent author bound `max_i norm(p_i-c)+r` bounds every chord point; the phi interpolation error is thus at most that radial/height maximum times `delta_phi^2/8`.

Every ideal outline chord lies on one XY line of positive constant support. For circular chords its outward midpoint support is the triangle support plus `r*cos(delta_theta/2)>0`; for straight chords it is edge support plus `r`. Scaling the two chord endpoints by each positive sin(phi) creates parallel horizontal segments at the two strictly ordered cos(phi) heights. Their bilinear image fills exactly the planar convex quad triangulated by the original two faces. At a pole one segment collapses, and the same image fills its single fan triangle. This is a set correspondence; it does not incorrectly identify triangle barycentric coordinates with bilinear parameter coordinates, and needs no mixed-derivative bound.

For either direction of correspondence, ideal-triangle barycentric weights applied to the corresponding actual vertices move the point by at most the maximum vertex construction displacement. Summing the sequential interpolation and construction bounds therefore covers the whole continuous analytic surface and every actual facet, including poles and seams.

The retained bound components are:

| Component | World upper bound |
|---|---:|
| Meridian interpolation | 0.0003192682087754737 |
| Circular outline interpolation | 0.00008567364931501181 |
| Straight outline interpolation | 0 |
| Actual vertex construction displacement | 0.00000005876531487310826 |

## Oriented normal correspondence

An outward analytic normal direction, extended continuously at the poles, is

`G(theta,phi) = (h*sin(phi)*n_outline(theta), cos(phi)*support_outline(theta))`.

On a straight edge the outline normal/support are constant; the meridian direction lies in the positive cone of the two phi endpoint normals. On a circular interval, `n(theta)=A*n0+B*n1`, with `A,B>=0` and `1<=A+B<=sec(delta_theta/2)`. Then `H=A*G(theta0,phi)+B*G(theta1,phi)` differs from `G` only by the vertical term `r*cos(phi)*(A+B-1)`. Positive phi interpolation coefficients put `H` in the cone of all four cell endpoint normals, including the collapsed-pole endpoints.

The vertical correction is bounded by `E=r*(sec(delta_theta/2)-1)`. Exact positive centroid-to-edge distances show every triangle support is at least its smallest such distance; hence `norm(G)>=mu=min(h,min_edge_distance+r)>0`. The circular correction angle is at most `E/(mu-E)` radians, with `E<mu` checked. Here its upper bound is `0.00035726048531963683` radians and the normal-length lower bound is `0.24`.

For every actual facet, its cross product is exact rational arithmetic on the original binary64 coordinates. Directed dot/cross intervals prove a positive dot with every endpoint normal and bound `tan(angle)` by `norm(cross(facet_normal,endpoint_normal))/dot(facet_normal,endpoint_normal)`. Positive sums preserve this cone bound by the triangle inequality. Use `atan(t)<=t`, add the circular correction only to circular cells, and verify the final angle is below pi/2. This proves oriented correspondence over each entire cell; no absolute-dot flip allowance is used.

## Validation, scope and remaining camera gaps

Nine focused pure checks passed in 12.157 seconds in the existing Blender 5.2 bundled Python. They cover complete pole/seam inventory, planar and collapsed-pole algebra, wrong winding/diagonal/duplicates/holes, non-equilateral exact coordinates, hostile global Decimal rounding, positive oriented cones and mirrored facets, and strict/forged source identity refusal without mutation. The one authorized actual retained-source certificate took 10.370148400004837 seconds. [Evidence](source-certificate-evidence.json) records the executable/version, exact implementation hashes and all five frozen input hashes unchanged afterward. No shared helper, test runner, historical receipt or original output was changed.

[Camera inventory](camera-inventory.json) locates the original source and candidate five-mask rows. The source declarations bind original matrices/ortho scale and PNG bytes, but lack full clipping, shift/raster/pass settings and producer geometry bindings. The candidate receipt has no actual camera snapshots or canonical manifest. The three legacy gallery shades cover front and the two obliques, without side/top shades or RGB normal passes; public asset bytes alone do not establish native camera provenance. The adaptive source has genuinely bound top and oblique145 linear-alpha acquisitions plus a modified top crop, not a five-view inventory. Changed adaptive candidate full five-view admission and canonical neutral/normal inspection remain explicitly unrun. New capture is a separate future action; this source-only certificate does not manufacture missing camera history or adopt historical owners.

During a broad read-only inventory search after declaration/module/test preparation, older candidate normal metric lines were inadvertently displayed. This exposure is disclosed in the evidence; those values were not used in any proof, code choice, threshold or calculation, and no candidate arrays were read. Artist surface limits remain null. Root owns future suite registration and report integration.

## Local policy integration

The source certificate is integrated into `family_surface_contracts.reference_facet_certificate` and registered as `pure_rounded_triangle_reference` in the custom runner. The API admits only the fixed authored parameters and exact original source. Two focused API guards passed in 0.230 seconds, including incomplete declaration and changed source refusal. [Integration evidence](integration-evidence.json) binds the changed files. This brings source-certificate support to eleven families; ten actual five-camera policies are retained. Triangle actual camera acquisition is a separate pending bounded job, so this commit does not claim an eleventh complete camera policy or candidate qualification.
