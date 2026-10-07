# Surface-quality implementation and bounded native handoff

Source base: exact qualified local repair head `92f7ba26c1eb778716a86e0b4ebc6bd6e30e5ad5`, descended from published `4d01e30`. Original documentation branch remains separate. The supplied exact `88cb2c1` backlog file was recovered without importing unrelated original-branch runtime changes.

## Implemented

- Profile loft has explicit `smooth`, `stepped`, and `sharp` geometry modes. Smooth uses bounded shape-preserving cubic interpolation through the supplied sections, rather than smoothing normals over a coarse staircase. Sharp retains authored section corners and flat faces. Stepped inserts connected planar shoulders, not individually capped overlapping cylinders. Crossing/shifted step shoulders fail explicitly.
- Flat caps and step shoulders remain flat shaded. The CLI exposes `--mesh-surface-mode` and `--mesh-surface-subdivisions`; refinement parameters, backend configuration and direct loft calls use the same controls.
- The default legacy reconstruction route remains unchanged. The retained historical vase is one indexed connected component, but has visible scallops/cracks. Legacy stacked-cylinder Boolean geometry followed by Catmull-Clark/radial refinement is the source path requiring a separate historical-input repair/qualification. Connected loft is the candidate repair path, not proof that the saved vase is fixed.
- A frozen 12-family workload specifies units, dimensions, seed, cameras, held-out cases, budgets and independent verdicts. All family reconstructions remain unmeasured. A reference declaration alone cannot establish broad acceptance.
- The rounded triangular dot is authored as an offset triangular outline with radius 0.16 and a symmetric lens dome of thickness 0.48 world units. Its three vertices, winding, dome formula and tessellation are frozen. The preview is a parameterized reference preview, not a reconstructed/native-qualified result. A 24-direction support/thickness check rejects circularized and flattened controls.
- A shared-world-frame area-weighted sample-to-triangle metric uses native Blender BVHs and oriented geometric normals. It avoids nearest-point sampling floors and does not hide flipped normals. Silhouette, topology/boundary and editability remain independent.

## Verification already completed

107 focused pure tests passed: loft plans/config/routing/refinement/E2E CLI tests plus new frozen-quality contracts. Python compilation and CRLF-aware diff checks passed. The supplied native validation packet's 200 entries and source supplement's listed files were checksum verified. The previous native suite is reused, not repeated; it does not validate these new source changes.

The pure triangle screen establishes closed indexed topology and positive oriented volume; it does not establish absence of geometric boundary intersections. Shape-preservation numerical controls pass/fail as intended. Native surface comparison and rendering remain pending.

## Exact native batch

Run once in the existing isolated qualified-source validator, with Blender 5.2.2 and its existing dependencies. No installs or existing-output cleanup.

`blender --background --factory-startup --python-exit-code 1 --python scripts/run_surface_quality_check.py -- --output temp/tasks/surface-quality-20261007`

Budget: 1,200 seconds, 8 GiB RSS. One new affected loft test; seven fixtures: analytic smooth vase, smooth loft candidate, stepped vase negative, authored sharp control, rounded triangular reference, circularized negative, banded negative. Each gets five fixed masks, five neutral-lit views, five normal visualizations, exact NPZ evaluated arrays, OBJ transport and geometry hashes. Results are saved per case. The historical full suite is not included.

Vase surface limits were declared before candidate measurements: symmetric area-weighted mean sample-to-triangle distance 0.003 world units; oriented normal-angle P95 2.5 degrees. Derivation is recorded in `frozen-workload.json` from radial/axial tessellation bounds. Identity reference must pass, smooth candidate must pass, stepped control must fail, while the sharp control remains unsmoothed. The rounded triangular circularized/flattened shape controls and banded surface control require independent failures.

Area-IoU observations use the historical 0.700 gate without changing it. This batch does not substitute area-only masks for the existing boundary and signed-distance reconstruction gates. Twelve-family acceptance, historical-vase repair acceptance, family-specific noise qualification, true solid-boundary qualification and artist edit-response receipts remain explicit next stages.
