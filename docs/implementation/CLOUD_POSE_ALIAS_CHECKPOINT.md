# Artist pose aliases and asymmetric reflection follow-through

The Gaussian editable program serialized its full eigenframe as rotation_row_major,
while the compiler only read rotation or Euler values. Local edits also only read
x/y/z, although these artist programs used location_x/y/z. Both paths could discard
an otherwise correct retained pose. The shared numeric/compile transforms now read
and synchronize those aliases; new proxy programs also emit the compiler's matrix
form explicitly. Numeric screening and compiler dispatch use the same frame.

Proper-frame reflection previously relied on local-X symmetry. Generalized sweeps
and concave polygons require an actual local reflection as well: centerline offsets,
profile offsets and outline/hole coordinates now follow it. Unsupported absolute
point-cloud reflection declines instead of faking an instance pose. Original
parameters remain untouched.

The receipt records 46 affected CPU contracts and 580 parsed Python files. Six new
fixtures cover artist alias preservation, numeric screen geometry, mocked compiler
matrix conversion, asymmetric sweep/polygon field reflection and unsupported-cloud
rejection. Existing pose/frustum/contact/profile/fitted-proxy contracts remain passing.
Mocked dispatch is not native Blender execution. Native artist/export agreement and
all reconstruction quality/performance gains remain unmeasured.

Absolute-tolerance follow-through: the common program pose helper now uses
`rtol=0` for its existing1e-8 orthonormal/determinant checks, matching the already
implemented P5 proper-frame contract. Relative tolerance previously admitted a
2e-6 unintended axis scale. Both matrix aliases and local edits now reject that
input; exact generated float64 rotations remain unchanged.77 affected host checks
pass across pose, P5, consistency, sweep, fitting adapters and contours. Native
compiler/export agreement and reconstruction quality remain unverified.
