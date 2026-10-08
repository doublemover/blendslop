# Evaluation results and limits

[Open the evaluation gallery](https://doublemover.github.io/blendslop/) for retained synthetic references, actual reconstructed outputs, per-view metrics and explicit missing verdicts.

## Current snapshot: 2026-10-08

The frozen suite has twelve prepared reference families, twelve index/volume topology screens and twelve live object-scale edit responses. Three wrong-depth, filled-cavity and missing-part controls are caught. Reference preparation is not reconstructed-family acceptance. Two families have actual reconstruction evidence:

- **Smooth vase:** surface mean 0.00286860 world units and oriented normal P95 2.32051 degrees pass the frozen 0.003 / 2.5 limits. Actual boundary qualification passes. Four silhouette views pass; `oblique_145_40` boundary IoU 0.789001 fails the unchanged 0.8 requirement. Overall acceptance is false.
- **Rounded triangular dot:** all five visible-frame silhouette gates, support/thickness screens and actual boundary qualification pass. Mean surface distance 0.000923876 and normal P95 2.97078 degrees are raw observations. Independent triangle reference-noise and family surface limits remain unqualified; no vase limits are transferred. Frozen oblique frames touch the image boundary.
- **Other ten families:** authored reference preparation is retained; actual reconstructed outputs and complete independent acceptance are unavailable. The initial defective capsule reference is preserved in history; the gallery uses its corrected analytic replacement.

[Detailed implementation/evidence map](quality-implementation-20261008/README.md), [frozen workload](quality-implementation-20261008/coverage-workload.json), [vase receipt](quality-implementation-20261008/profile-repair-evidence.json), [triangle receipt](quality-implementation-20261008/triangle-support-evidence.json). Historical scalloped inputs still need an authored smooth intent. Other lifecycle producer integration remains open.

Blender 5.2.2 correctness CI on merged main `69eb23128a18006501b2a1f89e8df129bfdc048f` passed full 137/0/0, quick 135/0/2 intentional end-to-end skips, and adaptive policy. [Full/quick run](https://github.com/doublemover/blendslop/actions/runs/37851341020), [policy](https://github.com/doublemover/blendslop/actions/runs/37851341060). Correctness checks do not certify reconstruction quality. The public gallery contains synthetic PNGs with original encoded pixels and ancillary text/EXIF metadata removed and a curated numeric summary, rather than machine paths, raw run logs or personal source material.

## Independent metric contracts

Area IoU measures foreground overlap. Boundary IoU measures overlap of the configured boundary bands; signed-distance loss compares silhouette distance fields under the original evaluation normalization. Each required view must meet area >= 0.7, boundary >= 0.8 and signed-distance loss <= 0.05 for the two actual reconstructions. The preparation controls use their own recorded observation protocol.

Surface distance uses 4,096 deterministic area-weighted samples in each direction, seed 61007, and nearest native triangles in unchanged world coordinates. Normal angles use oriented geometric face normals; opposite normals are 180 degrees. Caps and authored corners are retained. Index topology, actual solid-boundary qualification and edit response are separate verdicts. Any required failure or unavailable metric blocks acceptance. [Gate implementation](../blender_blocking/evaluation/silhouette_eval.py), [surface protocol](../blender_blocking/evaluation/surface_quality.py), [quality/performance guidance](QUALITY_PERF_GATES.md).

### Historical native measurements

The [frozen 2026-10-06 campaign](branch-audit-20261006/quality-final-results.md) records implementation `06a7f0219c98612a03b6e7e127d698fd9cc6c541`: **247 of 252 driver cells completed, 5 process failures, and 171 completed cells passed the render gates.** It retains regressions and unavailable results. These measurements predate the later source changes in this branch.

Two examples show why both silhouette and geometry measurements matter. In that campaign's matched ordinary → quality comparison:

| Split / method | Matched cases | Mean image IoU | Surface F.02 |
|---|---:|---:|---:|
| External / profile loft | 8/8 | 0.852 → 0.906 | 0.533 → 0.621 |
| Held-out / primitive fitting | 3/3 | 0.817 → 0.754 | 0.357 → 0.322 |

F.02 here uses the report's independent-bbox normalization and 8,192 area-weighted surface samples; its shared-frame results are a separate protocol. These are matched workflow/preset comparisons, not a source-only ablation, universal ranking, or speed claim.

The later [saved-solid audit](branch-audit-20261006/solid-followthrough.md) accounts for all 247 saved cases and distinguishes numerical false reports, editable multipart overlap, folded parts, coincident surfaces, unavailable qualification, and prior qualified receipts. Its CSV rows are evidence records, not 247 qualified solids. Historical artifact identifiers under `temp/` do not mean the original meshes and renders ship in this checkout.

### Source checks and native acceptance

Focused CPU contracts, parse checks, numerical fixtures and artifact replay establish their specific source contracts. Native Blender 5.2.2 correctness CI also passes on the integrated implementation. The per-family rendered, surface, solid and editability verdicts above remain separate. Paper comparisons and larger human-editability studies remain unestablished. Preserve exact source, inputs, cameras, metrics and runtime when comparing results.
