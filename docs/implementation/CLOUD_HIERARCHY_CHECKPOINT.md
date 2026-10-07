# Cell-aligned compact hierarchy and selective boundary extraction

## Implemented scope

- Adaptive hull queries and contour-section field samples now use exactly the
  cell centers represented by `VoxelTransform`, rather than endpoint samples
  paired with a different exported world transform. Conservative footprints
  and boundary displacement limits use the actual cell width.
- Legacy-camera adaptive hull queries and boundary corrections share the
  canonical pixel-cell viewport conversion.
- Accepted midpoint-partition occupancy boxes remain compact through indexed
  queries, volume serialization, six-neighbor surface-point diagnostics and
  boundary-tile marching cubes. A global dense occupancy array is not allocated
  in these paths. Caller chunk size is retained by adaptive dispatch.
- Tile seams share only exact identical global grid-coordinate vertices.
  No approximate weld, contact tolerance adjustment or vertex movement is used.
  Lewiner/Lorensen requests and aliases remain visible. Unsupported Boolean
  isovalues and excessive tile counts fail without hidden dense extraction.
- Invalid, fractional, out-of-range, overlapping or nonhierarchical accepted
  boxes are rejected. Existing exact-contact and folded-mesh guards are intact.
- The two new pure suites are registered in the existing CRLF test runner.

## Actual bounded evidence

66 affected CPU contracts passed. All 583 Python sources parsed. Tests compare
cell-center occupancy with independent evidence queries; check partial and
legacy camera semantics; round-trip compact NPZ; exercise adaptive dispatch;
compare actual scikit-image 0.26 tiled and dense mesh vertices, triangles and
oriented volume; preserve cavities and disconnected components; cover ambiguous
Boolean cells across tile seams; and compare selective surface points with the
ordinary six-neighbor dense boundary. Dense expansion is patched to fail in
compact-path fixtures.

For a 64-cubed logical grid containing one accepted 16-cubed box, actual selective
extraction visited 8 boundary tiles and 39,304 sampled nodes, producing 1,536
vertices and 3,068 faces. The actual fixture has zero boundary, nonmanifold or
degenerate edges/faces and a single closed component. These are fixture counts,
not a measured reconstruction or speed improvement. Boolean checkerboard
fixtures establish equivalence, not a claim that every occupancy is a valid
single solid.

## Still open

This is the first E6 source path. Full-scene end-to-end cost and quality
acceptance remain unmeasured. Boundary qualification is explicitly false;
ordinary final native/exact geometry qualification is still required. Explicit
`to_dense` and signed-distance conversion can still expand the full grid.
Boundary work can approach dense work for highly fragmented occupancy. The tile
allowance is a bounded fail-closed guard, not a universal efficiency guarantee.
Native Blender render/export was not run. Historical ae377fe-derived saved
meshes and 06a7f02/8e7c15d quality scores are unchanged.
