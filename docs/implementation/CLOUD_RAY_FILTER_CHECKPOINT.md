# Original-pixel box targets and thin-ray limits

D3/S4 now use an exact summed-area integral of declared original pixel cells for
observed ray targets, including partial validity, camera-external unknown regions,
non-square cameras and the original vertical orientation. One-voxel uniform box
support matches the pinned DVX closed-form filter. The previous deterministic
subcell target remains an explicit alternative. A shared complete-cell viewport
adapter also repairs legacy endpoint-center agreement for support/profile evidence,
polygon outlines, coarse screening and opaque proxy fitting.

The pinned-runtime fixture uses a closed axis-aligned pixel-cell box extending
through full depth cells. At resolutions 16/32, maximum disagreement is below
7.2e-15 in float64 and 7.7e-6 in float32. This is a bounded operator contract for
that geometry, not a claim of general renderer or reconstruction agreement. The
filter convention is supported by the installed dvx-python 0.1.1 source and the
[official implementation documentation](https://github.com/mworchel/differentiable-voxelization#conventions-for-coordinates-and-data-layout).

A separate thin-depth version of the same tiny fixture demonstrates an important
D2 limitation: maximum filtered volume along a depth ray is not true projected
coverage. Its gap is 0.90–0.95 despite the same planar silhouette. Solver receipts
now label that surrogate explicitly. Default routing omits globally unresolved
coarse levels; if every requested grid is globally too thin, it declines the DVX
candidate and preserves the existing seed. Explicit experimental opt-in is
available. The global PCA-envelope screen is only a diagnostic: it does not certify
local walls, branches, necks or exact projection. A thickness-independent mesh-ray
operator remains open. No boundary or single-solid certificate is inherited.

E5 follow-through: observed-ray jobs no longer backproject an unused solid-hull
occupancy target. They carry a clearly labeled unused zero shape buffer and the
actual ray evidence. Filtered/legacy occupancy modes keep their original target
preparation. The stage receipt explicitly records closed-form box support and ray
filter metadata, while physical transforms/connectivity remain fixed.

`validation/ray-filter-focused.json` records 74 passing focused CPU contracts and
574 parsed Python files. Seven new tests cover thin-strip preservation, independent
fractional area agreement, unknown/camera clipping, legacy camera centers, exact
legacy profile/support edges, invalid-mode refusal and rotation-invariant global
thin diagnostics. An affected retained-seed fixture verifies biased thin-ray work
is declined before optimization. The extended pinned validator passed eight
bounded stages, including the actual float32/float64 filter and thin-limit fixture.

Full original probability/boundary reliability propagation, exact filtered 3D
occupancy integration, thickness-independent mesh projection, Blender integration
and all new reconstruction quality/performance gains remain unmeasured or open.
