# Spec 12: Sparse Volume Interchange And OpenVDB-Oriented Design

## Scope

This expands idea 6. The visual-hull spec already asks for chunked/adaptive grids. This spec makes sparse volumes a first-class interchange layer so the project does not get trapped in dense NumPy cubes.

Related findings: F16, F17, F21, F26, F27.

## Current Code To Modify

- `blender_blocking/integration/multi_view/visual_hull.py:238-300`: dense `resolution^3` bool grid.
- `blender_blocking/integration/multi_view/visual_hull.py:342-455`: chunked projection exists but still writes into dense grid.
- `blender_blocking/integration/multi_view/visual_hull.py:457-475`: full voxel-center grid materialization.
- `blender_blocking/integration/multi_view/visual_hull.py:477-504`: point-cloud extraction from dense grid.
- `blender_blocking/integration/multi_view/visual_hull.py:506-567`: surface extraction over dense bool arrays.
- `blender_blocking/test_ground_truth_iou.py:299-304`: stores ground truth as a pickle file.

## Volume Abstraction

Create `blender_blocking/volume/`:

```text
volume/
  __init__.py
  contracts.py
  dense.py
  sparse_hash.py
  chunks.py
  openvdb_adapter.py
  serialization.py
  meshing.py
```

Contract:

```python
class VolumeGrid(Protocol):
    bounds: Bounds3D
    transform: VoxelTransform
    value_type: str
    def get_chunk(self, key: ChunkKey) -> np.ndarray: ...
    def iter_active_chunks(self) -> Iterator[Chunk]: ...
    def sample_world(self, points: np.ndarray) -> np.ndarray: ...
    def active_voxel_count(self) -> int: ...
    def to_dense(self, max_voxels: int | None = None) -> np.ndarray: ...
```

Supported values:

- `occupancy_bool`,
- `occupancy_prob`,
- `signed_distance`,
- `confidence`,
- `view_agreement`,
- `surface_flag`.

## Sparse Hash Backend

Implement a pure-Python sparse hash grid first:

- chunk key `(ix, iy, iz)`,
- fixed chunk size, e.g. 16 or 32,
- only store active/non-default chunks,
- support compression with `np.packbits` for bool grids,
- write `.npz` with metadata JSON.

This gives most of the architectural benefit without a hard OpenVDB dependency.

## OpenVDB Adapter

Design but keep optional:

- `openvdb_adapter.py` imports pyopenvdb or Blender VDB APIs only when available.
- Export to VDB if dependency exists.
- Import VDB as `VolumeGrid`.
- Preserve transform/world-space metadata.

Do not make OpenVDB required for tests.

## Serialization

Replace ad hoc pickles for volume artifacts:

- `volume.json`: metadata, bounds, transform, type, hashes.
- `volume.npz`: dense or sparse arrays.
- optional `.vdb`: sparse interchange.

Every volume artifact should record:

- resolution or voxel size,
- world bounds,
- active voxels,
- default value,
- value dtype,
- source candidate id,
- source masks/views,
- generation seed.

## Meshing

`volume/meshing.py` should consume `VolumeGrid`, not dense arrays only.

For dense-compatible methods:

- extract a chunk window,
- stitch mesh chunks,
- preserve world transforms,
- run topology QA.

For sparse/adaptive:

- maintain crack-free boundaries between chunks,
- add tests for chunk-boundary cracks.

## Benchmarks

Add to `benchmark_perf.py`:

- dense bool memory vs sparse hash memory,
- active chunk count by shape family,
- serialization size/time,
- surface extraction by backend,
- meshing by backend,
- random sample lookup throughput.

## Acceptance

- Dense visual hull can be represented as `DenseVolumeGrid`.
- Sparse hash grid round-trips through `.npz`.
- Empty-space-heavy visual hull uses less memory in sparse form.
- Mesh extraction produces consistent world-scale output.
- Missing OpenVDB dependency results in skipped adapter tests, not failure.

## References

- OpenVDB overview: https://www.openvdb.org/about/.
- OpenVDB documentation overview: https://www.openvdb.org/documentation/doxygen/overview.html.
- Marching Cubes: `temp/perf-quality-audit-20260509/lane-f-literature/paper-pdfs/lorensen-cline-marching-cubes.pdf`.
- Lewiner marching cubes: https://docslib.org/doc/9682582/efficient-implementation-of-marching-cubes-cases-with-topological-guarantees.
