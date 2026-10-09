# Authored input decisions

The missing owner input is a surface acceptance policy and an intended target for the historical scalloped vase. The frozen twelve-family fixture already defines geometry, measurement units and several presentation controls. Those values can be preserved without guessing new design intent. A third question is needed only if the owner wants a new smooth style/design rather than fidelity to that fixture.

This packet reviews clean follow-up `42e5fc890284bddebfb7da3463dd2cfb26e1e751` (implementation `df922a96ab899ea7dcc552b73eb6c5255b7387f9`). [decisions.json](decisions.json) contains the three unanswered questions, exact frozen target definitions, source/evidence hashes and remaining independent work. No owner answer, artist cutoff or candidate-derived target has been inferred. Preparation ran no native job or tests and changed only these two new documents.

## Three owner questions

1. **What should count as acceptable surface error for the eleven non-vase families?** Choose engineering diagnostics only, independently authored mean-distance plus normal-P95 limits, or those limits plus explicit tail-distance caps. For each family or explicitly named group, provide maximum symmetric area-weighted mean distance in **Blender world units** and maximum **oriented geometric-face normal P95 in degrees**. If tail defects must also block acceptance, supply a distance-P95 and/or sampled-maximum limit in world units. Leave an unspecified artist metric null; it stays unqualified. If the intended units are millimetres or another physical unit, supply the physical-units-per-world-unit mapping first. No normalization, alignment or unit conversion is inferred. A sampled maximum is not a continuous worst-case/Hausdorff guarantee.

   The existing analytic-vase contract remains mean <= `0.003` world / oriented normal P95 <= `2.5` degrees. It does not supply limits for other families. Source-facet certificates, the separately declared half-source-pixel distance allowance and additional one-degree normal allowance are **engineering policy**, not proposed artist cutoffs. Finer-mesh noise measurements are likewise not artistic acceptance. Freeze any new artist policy independently of candidate results; preserve old verdicts and apply new policies as distinct records.

2. **What is the intended target behind the historical scalloped ellipse-strip vase?** Choose preservation of the scalloped/stepped intent, an independently authored smooth target, or explicit adoption of the existing analytic fixture as a new target. For a custom smooth target, provide a formula or strictly increasing `(z, rx, ry, cx, cy)` source sections in world units, height/pose/scale, intended sharp/stepped features, and endpoint/cap intent. The known fixture is a closed solid with flat end caps: `0 <= z <= 2.6`, `rx = ry = 0.6 + 0.2*cos(2*pi*z/2.6)`, centered XY. Its source uses 257 sections /192 radial sectors, with smooth side shading and flat caps. Adopting it is an explicit new design decision, not recovered hidden intent from the old images. An open or hollow vessel additionally needs independently authored wall/bottom/rim dimensions; the closed fixture cannot establish them.

   The historical strip images and a regularization window do not determine a unique smooth profile. Its retained connected repair `086ea0cb...` still has the original 15-second boundary timeout and blocked surface acceptance. Neither a passing analytic vase nor a smoother-looking repair resolves those facts. [Historical receipt](../quality-implementation-20261008/profile-repair-evidence.json) preserves its exact input/archive hashes and failure.

3. **If a new smooth design is wanted, should it change shading, geometry, or both?** Choose fidelity to the frozen geometry/style, a named shading-only variant, or a separately authored geometric target. For shading-only changes, name the families and smooth/flat regions, caps/edges to retain, and applicable weighted-normal controls. For geometry changes, provide the independent profile/corner controls and features to retain; a loft also needs `smooth`, `sharp` or `stepped` intent and any permitted source-section radius displacement. Matching display shading never substitutes for geometric-normal acceptance. This question is conditional: existing fixture fidelity needs no blanket smooth/sharp override or new target approval.

## What is already specified

The following are **reference values**, not suggested fitted values or tolerances. Full box-part centers/sizes and triangle vertices are preserved verbatim in decisions.json and [quality_contracts.py](../../blender_blocking/synthetic/quality_contracts.py).

| Frozen family | Authored geometry in world units |
|---|---|
| Sphere | radius `0.8`; UV64/rings32 source |
| Anisotropic ellipsoid | radii `[0.9, 0.55, 0.7]`; UV64/rings32 source |
| Cylinder | radius `0.65`, height `1.6`; radial96 source |
| Tapered frustum | bottom radius `0.8`, top radius `0.35`, height `1.8`; radial96 source |
| Smooth vase | height `2.6`, mean radius `0.6`, cosine amplitude `0.2`; closed caps |
| Torus | major radius `0.7`, minor radius `0.22`; source128/24 lattice |
| Capsule | radius `0.4`, straight segment height `1.2`; connected corrected source |
| Rounded box | dimensions `[1.6, 1.1, 1.3]`, bevel radius `0.15`, eight bevel segments |
| Thin plate | dimensions `[1.6, 0.08, 1.2]`; authored sharp box |
| Concave arch | `[1.8, 0.4, 1.8]` box minus `[1.0, 0.8, 1.4]` box centered at Z `-0.3` |
| Asymmetric multipart solid | three exact authored box centers/sizes with live Exact UNION intent |
| Rounded triangular dot | offset-triangle corner radius `0.16`, thickness `0.48`, symmetric cosine lens; corner32/dome64 |

[Authored builders](../../blender_blocking/synthetic/quality_references.py) distinguish geometry from shading. The seven matched-source preview audit verifies flat sphere/ellipsoid/cylinder/frustum/plate and smooth capsule; the rounded box uses BEVEL8 plus custom weighted normals, with `weighted_normals=True`, `weighted_normals_keep_sharp=False`. The initially banded weighted-normal attempt remains preserved; its corrected paired preview is a style result, not a surface tolerance. Vase sides are explicitly smooth-shaded and its caps flat. The triangle source is explicitly smooth-shaded. Torus/compound source builders and their source hashes are known; missing complete source camera/style packets are not invented. See [source controls](../quality-source-controls-continuation-20261009/README.md), [rounded style](../quality-rounded-style-continuation-20261009/README.md) and [structured edits](../quality-structured-edits-continuation-20261009/README.md).

The loft controller has PCHIP `smooth` geometry, corner-preserving `sharp`, and connected-shoulder `stepped` modes. Source-section radius regularization is off by default (`window=0`, deviation `0`). Enabling it uses an odd window5..129, cubic Savitzky-Golay filtering, and a positive world-unit displacement bound; endpoint radii remain fixed and only RX/RY are regularized. Subdivisions are bounded1..16, default4. The historical attempt used 129 sections, window9, maximum source-section displacement `0.06` world, radial96 and subdivisions2. The analytic repair used65 sections, window9, displacement `0.003`, radial96 and subdivisions2. These retained processing settings are **not** authored target values or artist error limits; a section bound is not a whole interpolated-surface bound. [Controller](../../blender_blocking/geometry/loft_surface.py).

Meaningful retained edits establish narrower controls rather than blanket editability: coupled sphere radius, ellipsoid/plate depth, cylinder radius, frustum top radius, capsule straight height, box live bevel, torus tube radius, arch opening width and separate extrusion depth, multipart short depth, triangle corner support, and vase source-section radius. Each response is separate from exact geometry restoration and target selection; the physical response tolerance `1e-5` is not an artist allowance.

## Findings that remain independent

Existing strict silhouette gates remain area >=`0.7`, boundary >=`0.8`, SDF <=`0.05` for required views. Triangle support/thickness limits remain `0.005` world; oriented face normals still count opposite directions as180 degrees. Native solid boundary, source controls, presentation and artist acceptance stay separate.

The cylinder and plate bounded updates improve held-out alpha by71.2955% /87.4998%. Sphere is rejected (alpha2.2823% worse, raw mean3.6395% worse); ellipsoid is rejected (alpha3.2329% worse despite raw mean3.4865% better). The newest arch exterior checkpoint improves alpha87.5421% /raw mean74.5968%, while P95 distance rises `0.002812504768 -> 0.002919286489` world. The triangle improves mean/normal P95, while sampled maximum rises `0.002880168147 -> 0.002981405240` world. Capsule/torus normal P95 increases slightly despite mean/alpha improvements. These are observations, not boundaries for choosing acceptable error. [Axis evidence](../quality-adaptive-axis-families-continuation-20261009/native-evidence.json), [arch evidence](../quality-adaptive-arch-exterior-continuation-20261009/native-evidence.json), [triangle evidence](../quality-adaptive-triangle-continuation-20261009/evidence.json).

Changed diagnostic candidates still require their own complete five-view/native/canonical verdicts before selection; prior held-out exposure and censored views remain explicit. No new artist surface pass or aggregate acceptance follows from these documents.

## Work independent of the answers

Root can prepare missing bounded source-camera/pass acquisition for arch/torus, and defensible source-only continuous certificates for rounded box/triangle, without inventing artist intent. Nine analytic certificate APIs and seven complete camera-conditioned engineering packets already exist. Existing hash-bound raw observations can be classified under those frozen engineering policies while artist status stays null. An explicitly selected improved candidate can have its missing full five-view/native/canonical plan prepared for serial approval; original rows, failed updates and older lifecycle failures remain intact. None of this repairs absent provenance retroactively or supplies a historical smooth target. No recovery, cleanup or new native command is invented here.

## Independent engineering follow-up after the reviewed base

After the original42e5fc8/df922a9 review above, root completed the bounded arch/torus source-camera capture and executed the retained-raw engineering report for all nine supported camera-conditioned policies once. These are independent engineering/provenance additions, with original failures and artist status preserved. The three owner questions, frozen target values and unanswered numeric choices are unchanged. The family-surface-contract README digest in decisions.json is refreshed to its current bytes; other reviewed input digests and original review/base identities remain as recorded.
