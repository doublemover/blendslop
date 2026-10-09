# Local quality and lifecycle follow-up

This is the separate unpublished `local/quality-followup-20261009` branch, descended from reviewed `fb2802a685bf57f18b44f7b23e984e875aa21477`. [Draft PR7](https://github.com/doublemover/blendslop/pull/7) remains open at that reviewed revision, targeting main. Its full Blender5.2.2, quick and gallery CI passed; deployment was skipped. Follow-up implementation revision and remaining verification are recorded in `pending-verification.json`. This document does not grant aggregate acceptance to diagnostic geometry.

## Useful changes retained

Nine generic adaptive family models now use actual primitive discretization, fixed pose/tessellation, candidate-only view sensitivity and bounded genuine physical-alpha source acquisition. Each measured update uses at most three physical controls,96 residual calls/one second, one genuine1024 crop, the original held-out view excluded from fitting/crop/view choice, raw4096/seed61007 world measurements, and a native physical edit plus exact indexed restoration. Authored source dimensions are acquisition inputs, never fitted control values. Prior old view exposure and clipped frame portions remain explicit.

| Family | Held-out alpha L1 reduction | Verdict for this bounded checkpoint |
|---|---:|---|
| Capsule |14.7004%|Improved diagnostic|
| Torus |51.2078%|Improved diagnostic|
| Rounded triangular dot |73.6218%|Improved diagnostic|
| Tapered frustum |42.8948%|Improved diagnostic|
| Arch inner stage |20.0216%|Improved diagnostic|
| Arch exterior stage, same family |87.5421%|Improved diagnostic; distance P95 regresses|
| Cylinder |71.2955%|Improved diagnostic|
| Thin plate |87.4998%|Improved diagnostic|
| Sphere |-2.2823%|Rejected; original retained|
| Anisotropic ellipsoid |-3.2329%|Rejected; original retained|

These numbers describe the same family-specific original held-out view at that checkpoint; they are not a family-averaged pass. Raw distance, geometric normal, silhouette, topology, editability, canonical appearance and artist acceptance remain independent. Triangle sampled maximum distance increases from0.002880168147 to0.002981405240world despite lower mean/normalP95. Capsule/torus/plate normalP95 increases are retained independently. Ellipsoid raw mean improves while alpha regresses, so its update is rejected. Current12 selected rows are unchanged. The separately retained final box and multipart repairs already passed their own five-view/native/editability checks; these follow-up diagnostics do not replace those identities.

A separate three-control arch exterior stage adjusts outer width/height/extrusion depth while fixing opening width and notch height above the bottom. The symmetric height edit moves bottom attachments and absolute roof together; the previous inner stage had fixed the bottom/exterior. Its one six-frame job lowers alpha error87.5421% and raw mean74.5968%, while distanceP95 rises3.7967%. Exact depth response/restoration and both fresh process joins pass. [The arch exterior packet](../quality-adaptive-arch-exterior-continuation-20261009/README.md) retains the uncensored fitting frame, censored held-out frame, geometry history and every independent verdict.

Source-only continuous certificate APIs cover sphere, ellipsoid, cylinder, frustum, capsule, exact rectilinear plate/arch/multipart, and the frozen128x24 torus. The rectilinear screen requires a complete nonoverlapping outward boundary cover, rather than area/volume equality. The torus certificate has a complete oriented periodic cover and directed construction/interpolation/normal bounds. Seven actual camera-bound engineering policies are frozen. Their source facet bounds plus sampling allowance do not invent artist tolerances. Old multipart failure remains in the unchanged policy alongside the passing final repair.

Fresh renderer, variant, backend OBJ and opt-in chunk-cache producers retain immutable stages and explicit receipts. `cache_owned_writes=False` is a strict production visual-hull option carried through the existing config/backend/sparse builders. Opted-in deterministic-key collisions preserve the old NPZ and retain a distinct failed attempt; the last constructor-local receipt is forwarded without discovering historical receipts. Cache-hit-only construction reports unrun. Default outputs, cache keys/read behavior and overwrite policy remain unchanged.

The reusable bounded-command launcher validates source/argv/resource provenance before launch. On Windows, it assigns a fresh no-breakaway Job before resuming the hidden primary process, retains actual creation HANDLEs, requires real kernel waits and zero active members before lifecycle completion, and releases only the freshly owned scope. Job committed-memory bounds and sampled tree RSS are separate observations. New native jobs use85 seconds work plus5 seconds joins,8GiB limits and explicit two-thread controls. A fresh successful run does not repair an older failed supervisor or adopt an older lease. Recovery inspection is read-only; no deletion or reclamation execution is implemented.

## Evidence and remaining verification

Latest code/tests/native-packet revision: `df922a96ab899ea7dcc552b73eb6c5255b7387f9`.

[Evidence index](evidence-index.json) gives exact retained packet hashes. [Pending verification](pending-verification.md) names the exact shell, working directory, command or reason no executable check exists, candidate revision and review URL for each item. No extra tests were executed for these metadata documents. Focused implementation checks and necessary native acquisitions are recorded by their original producer packets, without relabeling historical results as current.

The actual [axis held-out pixels](../quality-adaptive-axis-families-continuation-20261009/measured-axis-heldout-pixels.png), [triangle pixels](../quality-adaptive-triangle-continuation-20261009/measured-heldout-pixels.png), and [capsule/torus pixels](../quality-adaptive-families-continuation-20261009/measured-heldout-pixels.png) were inspected. Canonical box/multipart60-frame child output is retained, but its original outer supervision remains failed because a descendant creation HANDLE was lost. PID absence supplies no kernel wait or exit status. Old source-owner receipt drift also remains blocked and preserved.

Non-vase artist tolerances and the historical scalloped-input smooth target still require independent authored input. Changed diagnostic candidates require explicit selection and their full five-view/native/canonical gates before promotion. Global hidden-surface uniqueness remains unqualified. Desktop continuation layout is uninspected because the supported CUA inventory exposes no browser/app; mobile review was explicitly skipped by the owner. Parent review is required before publishing this follow-up, retargeting/creating a PR, merging or deploying Pages.

The original owner checkout remains `endblay_opslay` at88cb2c10 with a clean status. All implementation is isolated under `temp/tasks/latest-blender-only-20261008/source`; installed Blender5.2.2, Python3.13.13 and qualified packages are unchanged.
