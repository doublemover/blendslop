# Spec 14: Topology-Aware Objectives And Geometry Sanity

## Scope

This expands idea 9. Silhouette scores alone can reward bad geometry. This spec defines topology-aware quality terms for candidate selection, optimization, meshing, and regression gates.

Related findings: F04, F05, F08, F11, F14, F15, F21, F22, F25, F26, F27.

## Current Code To Modify

- `blender_blocking/integration/blender_ops/profile_loft_mesh.py:31-51`: degenerate ring behavior.
- `blender_blocking/integration/blender_ops/profile_loft_mesh.py:82-114`: cap creation.
- `blender_blocking/placement/primitive_placement.py:314-326`: fallback can return simple join.
- `blender_blocking/placement/primitive_placement.py:427-460`: `bpy.ops.object.join()` creates one object but not one clean volume.
- `blender_blocking/test_profile_loft_mesh.py:72-82`: existing manifold-ish test.
- `blender_blocking/test_mesh_joiner_voxel_remesh.py:12-47`: voxel remesh test.
- `blender_blocking/test_e2e_validation.py:218-298`: validates silhouettes but not topology.
- `blender_blocking/validation/silhouette_iou.py:212-228`: only computes mask IoU.

## Mesh Quality Metrics

Implement or expand `MeshQualityReport` from Spec 03:

- vertex count,
- edge count,
- face count,
- loose vertices,
- boundary edges,
- non-manifold edges,
- self-intersection warnings,
- connected components,
- internal disconnected shells,
- watertightness,
- face normal consistency,
- zero-area faces,
- duplicate vertices,
- degenerate faces,
- bounding-box sanity,
- volume estimate,
- surface area,
- genus estimate when possible.

## Objective Terms

Add topology terms to candidate scoring and refinement:

```python
topology_score =
  watertight_bonus
  - loose_vertex_penalty
  - non_manifold_penalty
  - disconnected_component_penalty
  - degenerate_face_penalty
  - internal_overlap_penalty
  - bbox_violation_penalty
```

For blocking workflows, topology objectives should be configurable:

- strict watertight mesh,
- acceptable open shell,
- multi-component allowed,
- preserve separated parts,
- editable primitive set preferred.

## Join-Fallback Penalties

Simple join from `primitive_placement.py:427-460` must score lower unless the user explicitly requested disconnected editable components.

Fallback metadata must distinguish:

- boolean union success,
- voxel remesh success,
- simple join degraded,
- empty boolean failure,
- mesh cleanup failure.

## Optimization Integration

Primitive and differentiable refinement should use topology terms as outer-loop penalties:

- do not require differentiability for all topology metrics,
- evaluate topology every N iterations,
- reject updates that improve silhouette but create invalid topology,
- keep best topology-valid checkpoint.

## Topological Fixture Matrix

Synthetic shape factory should produce:

- watertight sphere/cube baseline,
- open cup/bowl where open topology is expected,
- holed torus where genus matters,
- table/chair where multiple thin supports matter,
- degenerate profile rings,
- overlapping primitives,
- disconnected components expected vs unexpected.

## Tests

- profile loft repeated zero-radius rings produce no loose vertices.
- simple join degraded status is scored lower than real union.
- candidate with higher IoU but non-manifold mesh loses to topology-valid candidate if policy is quality-first.
- open-shell fixture does not fail watertightness when configured open.
- topology metrics appear in manifest and candidate score JSON.

## References

- Blender BMesh API: https://docs.blender.org/api/master/bmesh.html.
- Blender BooleanModifier API: https://docs.blender.org/api/current/bpy.types.BooleanModifier.html.
- Lewiner marching cubes topology guarantees: https://docslib.org/doc/9682582/efficient-implementation-of-marching-cubes-cases-with-topological-guarantees.
- Poisson reconstruction: `temp/perf-quality-audit-20260509/lane-f-literature/paper-pdfs/kazhdan-bolitho-hoppe-poisson.pdf`.
