# Exact authored rectilinear source certificates

All three actual retained authored sources pass an independent complete facet-cover proof. This pure-Python report consumes only the frozen `quality_workload()` declarations and actual source arrays. It accepts no candidate geometry, reconstruction metric, candidate threshold or artist tolerance. No Blender process or new acquisition ran.

| Authored source | Actual source triangles | Proved boundary rectangles | Continuous source/analytic distance bound, world | Outward normal angle bound |
|---|---:|---:|---:|---:|
| thin plate |12|6|2.667099729e-8|0 degrees|
| concave arch |28|22|3.384893150e-8|0 degrees|
| asymmetric multipart |48|112|5.057621815e-8|0 degrees|

The independent recipe consists of fixed axis-aligned boxes, applied in authored union/subtraction order. Every box endpoint becomes an exact rational plane coordinate from its frozen binary64 declaration. Those planes partition space into cells on which analytic CSG membership is constant. Evaluating one exact interior representative classifies each whole homogeneous cell; it is not a sampled face/centroid acceptance test. Adjacent occupied/empty cells define the complete oriented analytic boundary rectangles.

Actual source vertices associate with that lattice under a fixed numerical construction budget of four binary32 epsilons times the authored coordinate scale. This budget is used only to choose a numerical grid association. The certificate's geometric bound is the actual maximum source-to-snapped vertex displacement, not that looser budget. Exact cached source identities are recomputed from arrays and must match the supplied content hash.

Every snapped source triangle must lie on one lattice plane and have the correct outward orientation. Exact rational convex clipping assigns its entire area to the corresponding analytic boundary rectangles. Every source triangle must be assigned fully, every boundary rectangle must have complete coverage, and every positive-area overlap between clipped source facets is forbidden. The proof therefore rejects missing surfaces, overlaps that compensate gaps, stray interior facets and reversed faces. It does not rely on vertices being on planes, matching total area/volume or a finite collection of facet samples.

Complete nonoverlapping coverage by closed triangles supplies every analytic boundary point, including rectangle edges by closure. For a source triangle and its snapped counterpart, the same barycentric weights give a continuous displacement no larger than the maximum corresponding vertex shift. This proves the stated bidirectional continuous facet/boundary distance bound. Actual cross-product normals must have a strictly positive exact dot product with the corresponding outward analytic plane. Their exact rational tangent cone supplies the normal bound. Edge/corner nearest-normal uniqueness is not asserted.

Geometry and cover calculations use `Fraction` throughout. Float square-root upper endpoints are checked against exact rational squares; the reported angular display has an outward numerical conversion allowance, and the exact tangent bound is retained. All three actual sources have tangent zero and therefore exact zero plane-normal angle. Construction limits are4096 vertices/triangles/cells/boundary patches,16384 triangle/patch clips and65536 exact overlap intersections; exhaustion returns an explicit unsupported reason. Actual proofs need only12/49/243 clips and6/34/108 overlap checks.

Seven focused checks pass in0.089 seconds on existing Blender-bundled Python3.13.13. One adversarial fixture duplicates one top triangle and removes the other while preserving the same source vertices, total facet area and signed volume; the exact overlap/coverage proof rejects it. Other checks cover missing boundary area, flipped orientation, off-lattice coordinates, continuous rounding/normal bounds, large drift, unsupported families and forged source hashes.

[evidence.json](evidence.json) binds each actual retained NPZ, exact source geometry hash, frozen recipe/workload hashes, runtime hashes and complete certificates. These are reference discretization certificates. They supply no candidate boundary qualification, no reconstruction admission and no independent artist surface limits. Root separately owns their integration into source engineering contracts.
