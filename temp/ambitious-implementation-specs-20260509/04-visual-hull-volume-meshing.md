# Spec 04: Visual Hull, Volume Representation, And Topology-Safe Meshing

## Scope

Owns F08, F16, F17, F26, and F27. This spec promotes visual hull work from a pure-Python helper into a real reconstruction family with sparse/adaptive representations, mesh extraction, and optional post-processing. It intentionally goes beyond the old v1 scope that excluded full visual hull at `docs/IMPLEMENTATION_SPEC.md:15`.

## Current Code To Modify

- `blender_blocking/integration/multi_view/visual_hull.py:7-20`: module docstring already describes voxel visual hull and Marching Cubes but mesh extraction is not implemented.
- `blender_blocking/integration/multi_view/visual_hull.py:162-208`: `MultiViewVisualHull` initialization, resolution, bounds, chunk size.
- `blender_blocking/integration/multi_view/visual_hull.py:238-300`: reconstruct initializes dense `resolution^3` grid.
- `blender_blocking/integration/multi_view/visual_hull.py:342-455`: vectorized projection with z-slab chunking.
- `blender_blocking/integration/multi_view/visual_hull.py:457-475`: `_compute_voxel_centers` materializes a full coordinate grid.
- `blender_blocking/integration/multi_view/visual_hull.py:477-504`: extracts point-cloud points, not a mesh.
- `blender_blocking/integration/multi_view/visual_hull.py:506-567`: surface voxel extraction uses SciPy erosion or pure-Python triple loop fallback.
- `blender_blocking/benchmarks/benchmark_perf.py:194-269`: visual hull reconstruction benchmark.
- `blender_blocking/benchmarks/benchmark_perf.py:274-323`: surface voxel benchmark.
- `docs/IMPLEMENTATION_SPEC.md:15-16`: excludes full visual hull and ML reconstruction; update this when implementing ambitious modes.

## Representation Strategy

Do not scale dense grids blindly. Keep dense grids only as a baseline and fixture oracle.

Implement three interchangeable volume backends:

1. `DenseVoxelGrid`
   - Existing behavior.
   - Used for small resolutions, fixtures, and correctness tests.

2. `ChunkedVoxelGrid`
   - Stores z-slabs or cubic chunks.
   - Evaluates occupancy chunk-by-chunk without retaining large projection arrays.
   - Emits peak-memory metrics.

3. `AdaptiveVoxelTree`
   - Octree or hashed sparse chunks.
   - Refines near silhouette boundaries and occupied/free transitions.
   - Stores occupancy state: inside, outside, boundary, unknown.

Core API:

```python
class VolumeBackend(Protocol):
    def carve(self, constraints: Sequence[ViewConstraint]) -> CarveResult: ...
    def extract_surface_points(self) -> np.ndarray: ...
    def extract_mesh(self, method: str) -> MeshExtractionResult: ...
    def stats(self) -> Mapping[str, object]: ...
```

## Carving Algorithm

Use silhouette constraints from Spec 01 and Spec 02:

- Each `ViewConstraint` has a canonical mask, camera model, and projection transform.
- Orthographic front/side/top views should use direct axis projection when possible; do not pay general matrix cost for axis-aligned cases.
- Turntable/custom views use camera matrices and calibrated intrinsics/extrinsics.
- Occupancy predicate: a voxel is inside if every required view projects into foreground.
- Keep soft scores for boundary voxels if masks are anti-aliased or distance transforms are available.

For dense/chunked:

- Avoid `_compute_voxel_centers` full materialization from `visual_hull.py:457-475` for large grids.
- Generate coordinate planes per chunk.
- Reuse projected coordinate arrays where possible.
- Add early exit when a chunk is entirely outside after any view.

For adaptive:

- Start from a coarse grid.
- Refine cells whose projections straddle mask boundaries or whose neighboring occupancy differs.
- Cap refinement by memory and target voxel size.
- Emit refinement-depth histograms.

## Surface Extraction

Replace the fallback at `visual_hull.py:529-567`:

- Implement a vectorized NumPy fallback using shifted neighbor masks:
  - pad occupied grid by one cell of false,
  - compare six neighbors with slicing,
  - surface = occupied and not all six neighbors occupied.
- Keep SciPy erosion as optional fast path, but test exact equivalence on dense, hollow, and sparse grids.
- Benchmark both paths via `benchmark_perf.py:274-323`.

## Mesh Extraction

Add `integration/multi_view/volume_meshing.py`.

Methods:

- `marching_cubes_lewiner`: preferred when an available library provides topology guarantees.
- `marching_cubes_classic`: fallback for dense grids.
- `dual_contouring_research`: optional future experiment for sharper blocking edges.
- `point_cloud_only`: diagnostic fallback, not a production mesh.

Requirements:

- Preserve world bounds and scale in vertex positions.
- Return normals when available.
- Run `MeshQualityReport` from Spec 03.
- Save mesh extraction parameters and timing in the manifest.
- Include manifoldness, loose vertices, connected components, and face count in metrics.

## Poisson And Screened Poisson Postprocess

Do not make Poisson the core visual hull mesh path. Use it only after there are oriented points/normals:

- input: surface points plus normals from marching cubes or fitted implicit field,
- output: smoothed watertight mesh,
- record smoothing risk and compare against original silhouette IoU.

Acceptance gates:

- Poisson output must not reduce any required view IoU below configured thresholds.
- Poisson output must improve topology or smoothness metrics enough to justify use.
- Save both pre- and post-Poisson meshes.

## Configuration

Add visual-hull config:

```python
@dataclass
class VisualHullConfig:
    backend: Literal["dense", "chunked", "adaptive"]
    resolution: int
    max_resolution: int
    chunk_size: int | None
    adaptive_max_depth: int
    boundary_refine: bool
    mesh_method: Literal["marching_cubes", "lewiner", "dual_contouring", "points"]
    postprocess: Literal["none", "poisson", "screened_poisson"]
    memory_budget_mb: int | None
```

Integrate as a real `reconstruction_mode` from Spec 02.

## Benchmarks

Extend `benchmark_perf.py`:

- dense vs chunked vs adaptive at R=32/64/96/128/192/256,
- front/side/top vs 8-view vs 12-view,
- peak RSS or tracemalloc,
- voxels projected/sec,
- boundary voxels/sec,
- mesh vertices/faces/sec,
- output IoU vs synthetic fixtures.

The smoke result from the audit showed surface extraction at R=32 around 493 ms/iteration when the fallback path was used; this is the first perf regression target.

## Validation Fixtures

- cube, sphere, cylinder, vase, table-like thin supports,
- hollow torus-like silhouette where visual hull cannot recover hidden concavity,
- asymmetric front/side/top masks,
- noisy masks from Spec 01,
- high-resolution stress fixture,
- multi-view calibrated turntable fixture.

## Papers And References

- Laurentini visual hull: https://iris.polito.it/handle/11583/1401917.
- Kutulakos and Seitz space carving: `temp/perf-quality-audit-20260509/lane-f-literature/paper-pdfs/kutulakos-seitz-space-carving.pdf`.
- Matusik polyhedral visual hulls: `temp/perf-quality-audit-20260509/lane-b-reconstruction/paper-pdfs/matusik-2001-polyhedral-visual-hulls.pdf`.
- Image-based visual hulls: `temp/perf-quality-audit-20260509/lane-b-reconstruction/paper-pdfs/matusik-image-based-visual-hulls.pdf`.
- Exact visual hull from marching cubes: `temp/perf-quality-audit-20260509/lane-b-reconstruction/paper-pdfs/liang-wong-exact-visual-hull-from-marching-cubes.pdf`.
- Marching Cubes: `temp/perf-quality-audit-20260509/lane-f-literature/paper-pdfs/lorensen-cline-marching-cubes.pdf`.
- Lewiner topological marching cubes: https://docslib.org/doc/9682582/efficient-implementation-of-marching-cubes-cases-with-topological-guarantees.
- Poisson reconstruction: `temp/perf-quality-audit-20260509/lane-f-literature/paper-pdfs/kazhdan-bolitho-hoppe-poisson.pdf`.
- Screened Poisson: `temp/perf-quality-audit-20260509/lane-f-literature/paper-pdfs/kazhdan-hoppe-screened-poisson.pdf`.
- Generalized reprojection error: https://pmc.ncbi.nlm.nih.gov/articles/PMC4281271/.
