# Practical geometry and helper source checkpoint

## Implemented contracts

- N3: optional Shapely 2.1 constrained triangulation supports concave polygon
  extrusions and holes with matching signed distance, outward mesh, samples and
  serialized editable contours. Exact observed pixel-cell planar outlines enter
  complementary routing only with complete outline evidence and thin-axis support.
  Unknown boundaries do not silently become exact contour dimensions.
- E4: optional exact coplanar retessellation uses the existing integer geometry,
  edge-connected oriented plane patches, original boundary vertices and exact
  oriented area equality. It preserves plane junctions, holes and gaps and checks
  closed orientation/volume before and after. A deadline returns the original
  mesh. Frozen evaluated geometry retains the original artifact and emits a
  separate proposal and fresh qualification receipt, never an inherited operand
  certificate.
- E1: an owned numeric helper persists across probe/jobs, watches executable and
  source identity, uses atomic job files, reports progress, recovers failed jobs
  without stale output and has an explicit close and bounded orphan idle lifetime.
  Quality can opt into this session; the previous one-shot mode remains.
- E2: full toolchain hashes are reused only while executable, helper, native,
  predicate, producer and metadata file signatures plus Blender build identity
  remain unchanged. Mutations invalidate the hash. Actual geometry certificates
  still depend on exact geometry identity.
- Generated volume extraction now explicitly chooses scikit-image winding for
  high-inside occupancy versus negative-inside signed distance, avoids degenerate
  triangles and transforms normals correctly through anisotropic voxel scales.

## Measurements and retained evidence

The focused receipt `validation/output-focused.json` records 93 passing CPU tests
across 11 relevant suites and 567 parsed Python files. It includes actual
scikit-image 0.26.0 tiny occupancy/SDF extraction and outward closed-volume checks,
Shapely 2.1.2 polygon/retile fixtures, numeric subprocess lifecycle tests and
mocked file-identity invalidation tests. These are source/contract checks, not a
reconstruction quality or helper-performance campaign.

The historical saved box remains unchanged. Its 47,924 vertices / 95,844 faces
form a closed but inward-oriented shell (signed volume -3.3812390363311167).
A separately labeled outward-winding inspection copy, retaining all coordinates,
retessellated to 718 vertices / 1,432 faces within an 8-second bounded allowance.
Exact patch areas were preserved. The output is an unqualified inspection
proposal, not a replacement historical result or a matched speedup measurement.
Receipts and source/output hashes are in the recovery packet.

## Runtime and remaining acceptance

Optional official geometry dependencies are isolated outside the repository.
Pinned CPython 3.13 Torch 2.14.1+cpu / dvx-python 0.1.1 setup and bounded numerical
forward/backward fixtures are underway separately. No pinned DVX result is claimed
by this checkpoint. Blender execution is prohibited by repository instructions;
actual scene integration, rendering, signed-CSG qualification and reconstruction
quality remain pending. The ae377fe exact contact and folded-part guards remain.
