# Family source controls and matched authored shading

Seven actual family source-control transactions passed in Blender 5.2.2, followed by 35 authored neutral previews in the same five original camera declarations as the retained candidate previews. Every actual source/candidate camera-frame hash and clipping pair matched exactly. This work establishes specific edit response and retains useful human inspection; independent family surface acceptance remains unqualified.

## Source control transactions

`scripts/run_family_source_edit_check.py` freezes saved recipe/OBJ/exact-NPZ digests before edits. All seven rows reproduced their original indexed binary64 geometry. Each edit operated on the same owned compiler root and its single mesh descendant, then restored the exact original mesh pointer, native pose controls, recipe presence and modifier controls. All restored indexed geometry hashes equal the saved originals, without tolerance. Temporary meshes/objects are strictly job-owned.

| Family | Actual source control | Observed response |
|---|---|---|
| sphere | coupled fitted radial dimensions ×1.05 | all three physical dimensions ×1.05 |
| anisotropic ellipsoid | local depth ×1.05 | local Y dimension ×1.05; other dimensions fixed |
| cylinder | radii and radial dimensions ×1.05 | local X/Y ×1.05; height fixed |
| tapered frustum | top radius ×1.10 | top radius ×1.10; bottom radius and height fixed |
| capsule | straight segment length ×1.05 | height increases by 0.060007693 world units; cap radius fixed |
| rounded box | live bevel width ×1.10 | corner cut ×1.099999921; dimensions and eight segments fixed |
| thin plate | local depth ×1.05 | plate thickness ×1.049999962; width/height fixed |

Six transactions use recipe parameter regeneration applied to the same source descendant. The rounded box uses its live BEVEL modifier. This is narrower than blanket editability. The physical-response tolerance is 1e-5; geometry identity and restoration use no tolerance.

The native source-control session exited 0 in 7.094 seconds with 229,842,944 peak working-set bytes. Both ordinary native world-bevel tests passed: a declared .12 world radius cuts equally across the anisotropic box axes, a live .18 edit changes the result and restores exactly, ordinary boxes remain unbeveled, and default rounded-box segments remain three. Explicit bevel_segments are strictly bounded integers 1..64. The shared compiler applies source scale before beveling, preserving position/rotation.

## Authored neutral previews

`scripts/run_reference_neutral_inspection.py` rebuilt the seven original authored sources, including the corrected capsule. Exact binary64 vertex-coordinate and oriented triangle-coordinate multisets match their retained reference NPZ arrays. This permits only vertex/face ordering and cyclic triangle indexing; it is geometric equivalence for surface comparisons, not topology identity qualification. All regenerated exact archives, source recipes and equivalence receipts are retained.

The same session replayed the seven saved candidates without renders, under their exact indexed geometry guards, to capture actual native shading state. It then rendered precisely 35 reference-neutral frames at 512², under the original camera declarations. Source and candidate actual frames and clips strictly match for every pair. Polygon flags, sharp edges, custom normals and native modifiers were preserved and recorded. Workbench/STUDIO/SINGLE .65/shadows on/cavity off and the existing display-managed preview contract were retained. These images are human previews, not linear coverage measurements.

The native session exited 0 in 11.965 seconds with 466,145,280 peak working-set bytes. No candidate frames, masks, normal passes, fits, raw surface comparisons or qualification children ran. Source inventories remain incomplete because source mask/normal passes are unrun and legacy declaration/binding gaps are still explicit. Old masks, matrices and admission evidence remain unchanged.

## Shading finding and focused correction

The paired side images in `paired-neutral/` show six matching source/candidate shading styles. Sphere, ellipsoid, cylinder, frustum and plate are authored flat. Capsule is authored smooth and its source/candidate previews both retain that intent. Blanket smoothing would change authored intent.

The rounded box differs: its authored source has BEVEL8 plus WEIGHTED_NORMAL with custom normals, while the old candidate has BEVEL8 only. The source looks smooth and the old candidate shows bevel bands. The compiler now accepts an explicit strict boolean `weighted_normals` node parameter; absent values preserve the existing global default, and explicit values override that default. The future rounded-box producer can preserve the authored weighted-normal intent without changing other families. A separate five-neutral exact-geometry verification is authorized and will be linked once retained; the old candidate receipts remain intact.

## Provenance and remaining blockers

[evidence.json](evidence.json) records exact receipt hashes, all edit verdicts, style comparisons, process bounds and read-only ownership audits. Original runs are below `temp/tasks/quality-continuation-20261008/family-source-edits-01/owned-5_o3o2jg` and `reference-neutral-01/owned-zhmkcyea`.

Both completed runs initially registered results.json before its final update. The final receipt bytes differ from their original ownership-manifest digest, so read-only reclamation planning correctly blocks both on receipt drift. No unknown files remain. These original receipts/manifests are preserved unchanged. Future publication now refreshes the owned digest after every receipt update; an ordinary pure released-run regression verifies that final publication is dry_run_ready. This correction does not retroactively qualify the old ownership receipts or authorize deletion.

Focused pure tests cover four source-control/publication contracts, two style contracts, and four compiler radius/style contracts; the two native compiler tests above were executed in the bounded source-control session. Non-vase reference tolerances remain independently unavailable, so none of these results makes the family aggregate accepted.
