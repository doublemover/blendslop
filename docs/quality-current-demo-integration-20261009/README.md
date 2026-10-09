# Current main demo and manual generation

The current main presentation is complete locally at source `85f50757924574cc7153c6791071a2abb88c5793`. It defaults to the accepted calibrated vase and presents all twelve actual retained bodies. The historical failed two-case display remains behind `historical.html`. No new fitting, Blender frames, surface samples or native qualifications were run for this integration.

The exact owner-approved 21-commit batch through `f5d7f41` was published to existing draft [PR7](https://github.com/doublemover/blendslop/pull/7). Remote and PR head match; main remains `d5e9816`. Its completed Blender5.2.2 CI passed full177/0/0 and quick175/0/2, plus the gallery build. Deployment was skipped. The later current-demo/manual-workflow commits remain local and need parent review before another push. No merge, retarget, workflow dispatch or Pages deployment occurred.

## Current family verdicts

All twelve have five passing original silhouette views, native solid-boundary/index-topology checks, retained semantic edit/restoration and independent surface-engineering passes: **60/60 view gates, 12/12 engineering-complete cases, zero required current geometry-gate failures in those scopes**.

| Current family | Selected body | Artist aggregate |
|---|---|---|
| Sphere | `833c2dcc` | Unqualified |
| Anisotropic ellipsoid | `b40ab45f` | Unqualified |
| Cylinder | `ab634394` | Unqualified |
| Tapered frustum | `41cc2701` | Unqualified |
| Smooth vase | `7447cc40` | Passed existing contract |
| Torus | `3afacdc3` | Unqualified |
| Capsule | `d436a592` | Unqualified |
| Rounded box | `9f6c8d4d` | Unqualified |
| Thin plate | `0b2cb766` | Unqualified |
| Concave arch | `e2b54f35` | Unqualified |
| Asymmetric multipart solid | `b2c937a9` | Unqualified |
| Rounded triangular dot | `bfe72738` | Unqualified |

Eleven artist contracts remain undefined. Engineering passes do not grant them. Vase acceptance already exists for its exact calibrated body and unchanged .003-world /2.5-degree surface limits; it does not define the historical scalloped input's intended shape. Original frozen matrix rows and earlier failed trials remain unchanged.

## What the main demo now shows

The selector, cards, counts, raw values and independent verdicts load from `site/latest.json`. Its339 display bindings link exact rendered values to source assertions and normalized retained JSON digests. The validator rejects plausible metric drift, changed bodies/receipts, relabeled pixels and invented non-vase artist acceptance.

Twenty-nine exact retained PNGs provide eleven neutral reference/candidate pairs, one accepted-vase candidate-only preview, and three normal pairs for box/multipart/torus. Each copy is bound to an actual producer SHA/body/view/style; public copies remove only text/Exif ancillary chunks and preserve IDAT. Missing roles/cameras/passes are explicit. The vase's source-neutral counterpart is unavailable; the current candidate still displays. Earlier capsule/triangle beauty bodies remain display-only on the secondary continuation page.

The final public artifact contains115 PNGs and15,539,702 bytes, below the unchanged15-MiB bound by188,938 bytes. Only the curated `site/` files enter it. [Build receipt](build-receipt.json), [browser receipt](browser-receipt.json), [full evidence](evidence.json) and [approved publication receipt](publication-receipt.json) retain exact source identities.

The existing Edge154 headless desktop check passed all historical exact-mask/difference controls, all twelve current body/verdict/metric selectors, actual neutral-to-normal pass selection, unretained-camera handling and absence of runtime exceptions under its60-second cap. Actual screenshot inspection found a hero grid-sizing crop; the final block slot and image-bounds assertion remove additional UI cropping while preserving the native frame. [Current main screenshot](screenshots/current-main.png) and [current normal-pair inspection](screenshots/current-paired-inspection.png) were inspected. Mobile remains skipped as requested. Earlier draft build/screenshots remain under ignored local `temp/`; they are not overwritten as final evidence.

## Manual generation

`evaluation-pages.yml` now has **only workflow_dispatch**. Pushes and ordinary PR updates do not regenerate or deploy the demo. An existing YAML parser and repository-wide source inspection verified one generation entry and no second workflow caller; the Pages API reports `build_type: workflow`, without an implicit branch build.

The publish boolean defaults false: manual build-only runs produce downloadable static-site/browser review artifacts. Explicit publish is accepted only on main; non-main publication is refused. The established SHA-pinned Pages upload/deploy actions remain gated by that option and main. Repository permissions/settings were not changed. No hosted job was dispatched to verify configuration. This job builds the curated current static demo; it does not launch a reconstruction/render campaign.

## Remaining work by kind

**Actual implementation:** persistent reconstruction workers need asynchronous complete-tree launch/poll/stop ownership; direct Popen/taskkill can miss descendants after primary exit. Warm/cold numeric helpers and the native qualifier still have primary-only or unbounded post-kill communicate/cleanup paths. Other producer ownership remains separately scoped. Recovery/reclamation execution is unimplemented and needs explicit bounded authorization; no blanket deletion or historical adoption occurred.

**Actual visual/metric observations:** multipart has a faint side seam absent from source; arch distanceP95 rises3.7967% while mean improves; triangle sampled maximum increases while mean/normalP95 improve. Torus/cylinder/frustum and the displayed sphere retain finite-facet shading in the authored/candidate pixels. Original oblique cropping remains. Sphere/ellipsoid trial updates were rejected for alpha regressions; their retained originals are the displayed candidates.

**Design choices:** independently authored artist tolerances for eleven families, or an explicit choice to keep engineering diagnostics; historical scalloped-input smooth intent. Neither blocks displaying the completed current engineering results.

**Optional qualification/history:** global hidden-surface uniqueness and broader studies are unproved; optional tail cutoffs are unauthored. Lost historical descendant HANDLEs/source receipt drift remain retained and cannot be retroactively repaired from PID absence. These are distinct from missing generator features.

The review decision is whether to publish the new local current-demo/manual-workflow batch to existing PR7. Merge and manual Pages publication remain separate decisions. The live page still serves main `d5e9816` until those steps occur.
