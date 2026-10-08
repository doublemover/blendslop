# Primary-source formulation notes and comparison limits

Final decisions and executed evidence: [final-phase.md](final-phase.md). The shared spread initializer described below was isolated after a Gaussian regression; height seeding remains the shared default.

This pass uses three calibrated orthographic silhouettes, local CPU solvers and
editable Blender geometry. It does not reproduce a learned RGB method or turn
paper results into measurements on these inputs. See `docs/IMPLEMENTATION_SPEC.md`,
lines 8–21 and 122–125, for the existing orthographic-input/configuration boundary.

| Source | Useful formulation | Boundary for this pass |
|---|---|---|
| [PartGS, version 4](https://arxiv.org/html/2408.10789v4) | Calibrated multiview observations, joint primitive/appearance reasoning and regularized part fitting motivate measuring the projected assembly rather than a vertical profile confidence. | Uses RGB and masks, Gaussian appearance and a different optimization contract. No trained model, published score or SOTA ranking is reproduced here. |
| [Light-SQ](https://arxiv.org/html/2509.24986v1) | Superquadric abstraction motivates geometric part quality and a distinction between a useful assembly and a valid single solid. | A mesh abstraction task, with more geometric information than three silhouettes. Its reported accuracy is not a comparable baseline. |
| [SparseSurf](https://arxiv.org/html/2511.14633v1) | Sparse observations and surface ambiguity support explicit novel-view evaluation and preserving hidden-shape uncertainty. | Sparse RGB surface reconstruction differs from this mask-only blocking task. No training data or weights acquired. |
| [PrimitiveAnything](https://arxiv.org/html/2505.04622v1) | A primitive assembly must be assessed as geometry, not merely as a serialized or editable program. | A learned assembly method with a different input/training contract. Not an executed comparison. |
| [Open3D surface reconstruction](https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html) | Screened Poisson consumes oriented surface samples and normals. This pass derives both from the same reconstructed visual hull used by the local baseline. | A downstream surface reconstruction comparison, not an independent recovery of unseen concavity. Existing Open3D 0.19 CPU helper is isolated from Blender's ABI. |

The tested formulation changes are full mesh-union silhouette loss for primitive
fitting, spatially spread cluster initialization, negative-space-aware seeds,
coarse CPU fitting followed by full-resolution evaluation, and exporting the
fitted differentiable radii without the former unscored 1.6× enlargement. The
Gaussian two-sigma contour hypothesis is an explicit experiment; it is not a
general default or a claim that Gaussian splatting reconstructs a solid.

The measured ensemble uses only observed masks and recorded cameras. It snapshots
evaluated geometry before subsequent backends can clear a Blender scene, renders
shortlisted candidates, rejects required-view failures, and admits refinements
only for mean IoU gain above .002 with worst-view degradation at most .002. Its
45-second limit controls admission; an in-flight candidate can overrun. The caller
enforces a separate 100-second Blender-child cap. Skips, reasons and full routing
wall time are retained. A finite pool of candidates is not a proof of optimality.

All fitting/selection uses the three supplied views. The orbit-45 silhouette,
reference geometry, normalized Chamfer/F-score and volume diagnostics are used
only for evaluation. Known camera crop/scale changes are permissible capture
metadata; synthetic case names, latent parameters and reference meshes are not
solver inputs. Three silhouettes cannot identify every hidden cavity or surface.
