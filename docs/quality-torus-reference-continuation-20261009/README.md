# Frozen authored torus reference certificate

The actual retained authored torus certifies complete oriented parameter coverage and independent continuous distance/normal discretization bounds. This pure-Python report uses the original source arrays and frozen authored R=0.7/r=0.22,128-major/24-minor declaration. It takes no candidate geometry, reconstruction metric or acceptance threshold. No Blender process or acquisition ran.

| Independent source quantity | Certified upper bound |
|---|---:|
| Continuous source-facet/analytic-torus distance, world |0.0028693482129234916|
| Analytic interpolation contribution, world |0.002868728601282348|
| Actual numeric vertex construction shift, world |6.196116411432303e-7|
| Corresponding continuous facet/analytic normal angle |7.639090855364763 degrees|
| Face-to-cell-center tangent |0.00014649771788354232|
| Analytic normal half-cell radius |0.1331807895672215 radians|

`torus_reference_certificate(reference)` is restricted to original indexed source identity `3d1743e165e096fc7e413237523e2402a809a7e71c6209f95d1de9f705b1738a`. It recomputes the actual vertex/connectivity hash, so a declared/forged identity cannot admit changed arrays. Different source arrays are unsupported. A tentative generic4-float32-epsilon construction association guard would have excluded some real authored angle-construction shifts; that guard was removed rather than enlarged. The certificate measures every construction displacement directly under a strict original-source identity. This displacement is separate from tessellation, reconstruction engineering allowances and unknown artist tolerances.

The proof core verifies exactly3072 rectangular parameter cells and6144 outward oriented triangles, including both periodic seams. Every required triangle occurs exactly once with the authored00-to-11 diagonal. Missing/duplicate facets, changed diagonals and reversed winding are refused. The last row/column maps to unwrapped theta/phi endpoints on2pi; corresponding analytic values are periodic. This is complete facet/parameter coverage, not a sampled surface, vertex-on-surface or volume check. Public byte identity is strict; portable pure tests exercise the mathematical proof core on deterministic analytic lattice fixtures independently of local archives.

For T(theta,phi)=((R+r cos(phi))cos(theta),(R+r cos(phi))sin(theta),r sin(phi)), second derivative norms are bounded by R+r for theta/theta and r for mixed and phi/phi derivatives. Linear interpolation at any parameter-triangle barycenter cancels the first-order terms. The Taylor expectation remainder is bounded by

`[(R+r)*Var(theta)+2*r*sqrt(Var(theta)*Var(phi))+r*Var(phi)]/2`.

Each parameter variance is at most its rectangle span squared/4. The fixed cell spans2pi/128 and2pi/24 give the interpolation contribution above. Every ideal vertex coordinate is enclosed with directed arithmetic; the maximum interval-enclosed actual/ideal vertex displacement bounds every corresponding facet point by the same barycentric weights. Complete parameter triangulation gives the bidirectional continuous source-facet/analytic-surface distance correspondence. This does not claim closest-point correspondence or global reconstruction uniqueness.

For normals, every actual source facet cross product is computed exactly from rational representations of its source coordinates. Its dot product with the enclosed analytic cell-center normal must have a positive lower bound. The exact cross-to-center cross/dot ratio bounds the center-angle tangent; `atan(t)<=t` yields a conservative angle bound without libm transcendental error assumptions. The analytic unit normal n(theta,phi)=(cos(theta)cos(phi),sin(theta)cos(phi),sin(phi)) has orthogonal parameter derivatives of lengths |cos(phi)|<=1 and1. Straight parameter paths therefore have spherical arclength no larger than their Euclidean parameter length. Every point in the cell is within the half-cell diagonal radius above. Spherical triangle inequality yields the reported continuous face-to-analytic normal bound. The full cone must remain below90 degrees, preserving outward orientation throughout each cell. Edge/corner closest-normal uniqueness is not asserted.

Pi is enclosed by Machin's identity with exact40/12-term rational alternating-series remainders. Sine/cosine use directed50-digit Decimal interval arithmetic after exact quadrant reduction,24 Taylor terms and an explicit1e-45 remainder (argument<2; the first omitted terms are below that bound). Decimal negation preserves endpoints exactly. Resulting trig endpoints become Fractions; source crosses, dot/cone arithmetic and Hessian bounds are exact rational calculations. Float square-root and final upper endpoints are checked against rational squares/values. The certificate does not rely on one-nextafter(libm sin/cos/acos) as an error guarantee.

Proof work is fixed:3072 vertices,6144 triangles,3072 cells, at most304 cached grid/center trig turns and24 terms per trig evaluation. Different sizes are unsupported; no construction fallback or adaptive proof search exists. Eight focused checks pass in8.551 seconds on existing Blender-bundled Python3.13.13. They cover periodic barycentric correspondence, gaps/duplicates, seam reversal/diagonal changes, positive actual normal cones, changed/forged source refusal and trig enclosures under deliberately altered global Decimal precision/rounding.

[evidence.json](evidence.json) retains the exact source archive/hash, original declaration hashes, module/test hashes, rational pi endpoints and complete certificate. Artist surface limits remain null. No candidate boundary qualification, candidate acceptance or current-row relabeling is supplied. Root owns integration into source engineering contracts.
