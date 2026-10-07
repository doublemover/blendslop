# Narrow-band implicit residual numerical foundation

This is the first N2 foundation. It is separate from the existing fixed-mesh DVX
operator and is not yet a registered/promoted production reconstruction backend.
No new dependency was installed: numerical execution uses the existing exact
pinned CPU Torch runtime, and tiny mesh fixtures use host scikit-image 0.26.

## Implemented mechanism

A bounded numerical sampler constructs cell-centered Euclidean triangle/edge
distances and solid-angle winding from one actual closed indexed source part.
The source retains its original coordinates/hash. Closed topology, outward
volume and the existing exact within-part boundary guard are screened first;
clipped, open, unresolved or oversized inputs fail. Initial scope is at most
512 triangles and 16/32-cubed fields. Native vertex/manifold qualification remains
separate. Multi-part and cavity source admission is not implemented by this
first sampler.

Residual parameters occupy only a fixed narrow band around the source boundary.
A tanh coordinate bounds changes in signed-field values in world units. Explicit
known-empty samples and the fixed domain boundary remain positive. Conflicts
outside the admissible band fail rather than silently making unlimited cuts.
Origin, anisotropic voxel spacing, original field and protected constraints have
content identities. Dense base storage and full-field regularizer work remain
explicit; this is not a sparse/adaptive memory or speed claim. A signed-field
value bound is not itself a certified final surface-error bound.

The optional pinned-CPU routine optimizes an observed-ray surrogate with weak
world-space Eikonal regularity and a bounded-residual prior. It requires explicit
execution approval. It preserves the best finite evaluated field and its actual
parameter snapshot, including after a timeout leaves a last update unscored.
Unknown pixel values cannot enter the reliability-weighted objective. There is
no CUDA/HIP path and no DVX operator use. The retained field is authoritative for
mesh replay; parameter-only replay must use the declared numerical runtime.

The ray proposal is explicitly voxel-center minimum signed distance followed by
a sigmoid. It is not the exact opaque projected-mesh area operator, and it must
not supply final opaque admission or truth about invisible cavities. Source and
solver results carry no native qualification or automatic-admission claim.

## Actual bounded evidence

15 new contracts were added. The affected host run executed 37 CPU tests, with
six pinned-runtime tests skipped in that interpreter. Those six plus the lazy
execution guard then all passed in actual Torch2.14.1+cpu/CPython3.13.5. All 596
Python sources parsed. The actual double-precision forward/backward fixtures
match independent local differences and the NumPy coordinate reference.

A tiny screened cube's numerical field matches its analytic box distance at the
same physical cell centers. A confirmed-empty column creates a through-hole in
a thin plate: Euler characteristic changes from2 to0, outward volume and the
existing exact boundary guard pass, and saved-field mesh extraction replays
identical vertices/faces. A separate analytic field proves thin-bridge
representation capacity (two components to one); a confirmed-empty gap blocks
that bridge. These are capacity/contract fixtures, not inferred reconstruction
successes. Zero residual preserves unobserved interior without inventing a
cavity. A two-update solver fixture evaluates three states and returns an actual
scored, bounded checkpoint; unknown-data and partial-timeout fixtures pass.

## Open integration and acceptance

Production same-evidence seed selection, exact camera/ray preparation, warm
transport, separate coarse-artist/residual artifacts, actual output mesh gates,
component/feature acceptance and native final qualification still need a
separate backend integration. Larger or multi-part source fields are also open.
No new reconstruction campaign, native Blender rendering or speed comparison
ran. Historical saved geometry and quality scores remain unchanged.
