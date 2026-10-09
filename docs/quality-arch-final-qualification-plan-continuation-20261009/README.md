# Saved exterior arch final-gate feasibility

This is a documentation-only plan for the saved exterior arch update. It is **feasible but not launch-ready**: the narrow arch adapter and its runnable command do not exist. No shared code, tests, source/candidate geometry, historical owners or native processes were changed. Root prioritizes finishing the triangle packet first, then reviews any arch implementation and serial launch separately.

The existing candidate is indexed hash `e2b54f3511ccca10d456c3c52ac78ff2deb13915bf36905544e1943294b95e6c`. Source hash is `3efc8d0c7b67f6a229531ed3a7e9e5a51ae82dc03890a54d157a79af33a2d9a1`. Both retained archives were decoded during read-only feasibility and have 16 vertices /28 triangles. The existing arch producer remains succeeded/released; its actual read-only audit reports dry_run_ready, zero blockers/unknown files and 9,684,192 retained bytes. That audit performs no reclamation or adoption.

[feasibility-plan.json](feasibility-plan.json) freezes 60 existing authoritative/runtime inputs with full SHA256, including original source recipe/workload/results, all five legacy source masks and original camera declarations, candidate exact archives/program/results, actual measurement contracts and EXR/coverage bytes, semantic edited recipe and current compile/render/qualification/supervisor dependencies. Final adapter/argv/test-runner registration and current byte hashes still need a fresh launch preflight. The JSON deliberately records `runnable_command=null`, `adapter_exists=false` and `native_launch_authorized=false`.

| Authoritative file | SHA256 |
|---|---|
| Original source NPZ | `a8bb29e00cc7bdf47cb154c476ed29b8858a2809a9081403b2b3cae2afb90ce3` |
| Refined candidate NPZ | `5b6e91bb40e29702dc2ebd8ef381781d255203b634e50fb39eddbd2ea7228b57` |
| Refined candidate recipe | `4f1c6df58ca7405ab6500ded28bfdea886d06fb5c3a640f6403fce32a589697a` |
| Completed producer results | `4083179014a61f1cb95a750c2294ca19db6225b7fe3efbfe01f40a847fca021f` |
| Existing depth-edited recipe | `07e1970ff6f0313df62c80f9cbeb101422d198e9cdccca18c3b4284798629567` |

## Exact reuse and frame budget

Reuse exactly the existing oblique_145_40 source/refined candidate linear-alpha acquisitions. Their full physical contract digests are identical, `84d56d9dbf3197b00f68476dc701cde86e6a5b4ecd9d481c42ec28426861bfc1`. Both declare the exact respective geometry identities and unchanged geometry. Actual EXR SHA256 values match their measurement receipts; coverage NPY/PNG bytes and producer ownership digests are independently retained in the input inventory. Reuse must carry the original producer/receipt/artifact origin explicitly; it is neither a newly acquired pass nor ownership adoption. A new half-coverage hard mask derived from those retained alpha values is a diagnostic conversion, not another native frame.

| Retained physical acquisition | Measurement JSON SHA256 | EXR SHA256 |
|---|---|---|
| Authored source, oblique145 | `fc8e8ac0613df823dd60e12e8d431f0350cb4354dd73d1e95028fe0c59be5d0c` | `8a1a6e872a384e892ffe5ab6c46f0dfbb45ba952fdd9c7d7d04e09ca4b7108a1` |
| Refined candidate, oblique145 | `73464a49e6f3a433d0ad1cb79a7dbb5f70ef2f9c09a864d253591d5bb17b13c1` | `bffe00ac893600bc744cbb0515af43c2cd977acff2f37b47b2619512074b343c` |

The complete logical packet has five original views x source/candidate x mask/neutral/RGB shading normal =30 pass records, acquired as **28 new512-square frames plus two retained physical acquisitions**:

- Eight new linear-alpha frames: source and candidate at original front, side, top and oblique_35_28.
- Ten new neutral frames: source and candidate at all five original views.
- Ten new RGB shading-normal frames: source and candidate at all five original views.

Exactly one existing native boundary helper is allowed, with its unchanged15-second wait allowance. There are zero fits, raw surface comparisons, new semantic transactions, resampling, reference proof calculations, beauty frames or other-family campaign. If a replay/frame/style guard blocks the case, retain its concrete failure and do not automatically retry or substitute a mesh/camera.

## Why source neutrals cannot be reused

The actual five-view source-neutral packet at source-camera-01/capture/owned-58qbg8gm uses clip_start `0.10000000149011612` and clip_end `100.0`. The retained controlled oblique145 acquisitions use the same near clip but clip_end `1000.0`. The known clipping mismatch prevents their reuse in one fully matched three-pass packet. Preserve all old neutrals, declarations and receipts. Newly recaptured source neutrals/normals use the controlled packet's explicit clipping contract and retain their own new provenance.

Original frozen matrices and orthographic scales remain the per-pass replay inputs; do not replace them with the expanded fitting camera. Every new actual source/candidate frame, shift/raster/projection and full near/far clip must strictly agree across roles/passes. Oblique145 must additionally match the retained actual controlled camera exactly. Reset each pass from the same original declaration to avoid accumulating matrix decomposition changes. Any exact mismatch is a blocked row, never an excuse for a camera tolerance.

Measurement uses existing opaque float32 linear-alpha EXR, EEVEE64, filter1.5 and fixed occupancy alpha>=0.5, independent of display RGB/AgX. Existing area IoU>=0.7, boundary IoU>=0.8 and signed-distance loss<=0.05 apply independently to all five views. Alpha L1 is diagnostic; no new acceptance threshold is introduced. Original frame cropping, censored observed contours and prior held-out exposure remain explicit.

Neutral and RGB normal passes use the existing human-inspection settings separately. Preserve source authored polygons/modifiers/sharp/custom-normal state and saved candidate flat/sharp state; retain style differences and actual per-pass settings. RGB shading normals never supply a geometric-normal metric. Materials, render state and indexed geometry must be restored exactly after inspection.

## Concrete compiler and helper interfaces

Source: `synthetic.quality_references.build_quality_reference(source_case)`, capture actual `evaluated_arrays`, then require exact binary64 vertex/oriented-triangle equivalence with the retained original source before any new frame. The recipe remains the frozen two-box subtraction; no idealized or NPZ-replacement source is substituted. Retain the equivalence receipt and actual style snapshot.

Candidate: `reconstruction.frozen_family.retained_family_program(saved_wire)` followed by `compile_shape_program(...,lathe_segments=96,weighted_normals=False)`. `output_mesh_targets([compiled.root_object])` must return exactly one mesh descendant. Require `evaluated_arrays(root).content_hash` equal the retained e2b54f35 identity before and after all passes. Existing `_compile_exact(wire,captured)` provides this strict interface for the polygon extrusion; it offers no arch tolerance or face-order repair. A pure feasibility check cannot claim this future native replay has passed.

Boundary: `reconstruction.native_qualification.qualify_geometry(saved_arrays,python=existing_qualified_python,timeout_s=15.,ownership_root=fresh_children)`. Require exact `geometry_content_hash`, actual current toolchain identity and `single_solid_qualified`. The established existing qualification interpreter is `.venv312/Scripts/python.exe`; no runtime/install change is proposed. If20seconds of outer work budget do not remain before the helper, record unavailable instead of extending the budget. Its own full process-tree ownership remains subject to the fresh outer supervisor.

The existing semantic transaction in the bound producer's `cases.concave_arch.semantic_edit_restoration` already checks extrusion depth x1.05, unchanged local XY outline/triangle inventory, expected physical depth, the same native source pointer, original mesh/pose restoration and exact indexed restoration for this saved candidate. Reuse that actual receipt; do not relabel it as an edit on the future process's pointer or repeat the transaction.

## Remaining implementation and independent limits

The existing selected canonical runner admits only box/multipart, and `render_role` unconditionally renders every mask. A small standalone arch adapter therefore needs the exact arch case plus an explicit retained-ob145 mask-binding branch. It must check original measurement/owner/geometry/byte contracts, retain actual provenance, render only the eight missing physical acquisitions, and compare new preview actual frames to the reused record. This is the concrete code gap; no runnable argv is manufactured here. Appropriate focused reuse/identity/clip tests and final frozen adapter/source/command hashes are pending root approval.

Proposed limits are85seconds work plus five seconds bounded joins, two threads,8GiB sampled process-tree RSS and a distinct8GiB Job committed-memory guard, one fixed15-second helper. Use fresh OwnedRun output and a pre-resume Windows Job with actual creation HANDLE identity/kernel joins. Unconfirmed joins or primary failure retain active/failed history; no blanket deletion, lease rewrite or old-owner adoption is part of this plan.

The current checkpoint's raw P95 distance increased from0.002812504768371582 to0.002919286489world. Preserve that observation: there is no engineering P95-distance cutoff to pass or relax. Artist limits remain null. This job would supply missing view/solid/canonical evidence, not change the current twelve-family selection or claim aggregate artist acceptance.
