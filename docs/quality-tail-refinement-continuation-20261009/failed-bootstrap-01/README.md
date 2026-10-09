# Observed-contour tail refinement, 2026-10-09

The two saved proposals are ready for one bounded native checkpoint. They are not selected in the current demo. No native work has been launched by this helper, and no historical score, source fixture, artist limit or accepted family selection is replaced by this packet.

`native-plan.json`, `preflight.json` and `launch-command.json` freeze the exact inputs and runnable argv. Root reviews and launches that argv serially after the multipart checkpoint. The plan SHA is `e2781e6b82bc6b1c7aaaec7f6f48447c36380502d9e52ae4763772f3b4eaea68`; launch SHA is `abfa6ca09135630d3688e9dfe380be273c6558ed33e1e47d406b6ccc65c785b8`; preflight SHA is `75de36772a7d149ee053ab55a7488742a6e4f2519415ddb81890b612b92a7556`. The checker SHA is `b63ea52217324a70e6ca84c41f736a3cb4663c5ae8c6a29045f28377a6ab2d72`. These exact runtime/JSON bytes remain stable through review and execution; this README is outside the frozen input inventory.

## Actual issue and bounded fix

The current exterior-arch checkpoint `e2b54f3511ccca10d456c3c52ac78ff2deb13915bf36905544e1943294b95e6c` improved overall distance but raised P95 from `0.002812504768` to `0.0029192864894866943` world units. Its mean is `0.00022957110319443697`; sampled maximum is also `0.0029192864894866943`. A bounded diagnostic of 256 deterministic source samples against every candidate triangle localized the largest residual to the flat cavity roof. The selected exterior stage had preserved the old opening/notch dimensions while moving the outer height and bottom; the roof retained that coupled residual. This diagnostic is separate from the existing 4096-per-direction raw metric and is not a new acceptance score.

The new arch stage varies only `notch_height_from_bottom_world`, the same physical dimension represented by `cavity_roof_height_world` (roof height above bottom). Outer width/height, opening width, extrusion depth, rigid frame and topology remain fixed at the selected e2b body. It fits the previously acquired, uncropped expanded ob35 controlled-alpha source view. The projected boundary is the actual concave triangle union; internal triangle edges do not become silhouette observations. It does not consume source mesh arrays or authored dimensions as fit controls.

The current triangle `bfe727380abc81d0215abec5687b77b83c251133605149a6f7ad2f714423a89d` lowered mean/P95 but raised sampled maximum to `0.0029814052395522594`, above the old original body's roughly `0.002880168`. Its current mean is `0.0006732319122740825` and P95 `0.0022511934163048863`. The same bounded diagnostic localized remaining residuals on the lower/front sloping dome rather than missing outline tessellation. The new stage varies world Y/Z center and front depth fraction only. X center, rigid orientation, total thickness, outline scales, corner radius and authored corner32/dome64 tessellation stay fixed. Its projected convex boundary uses all 6239 vertices of the actual discrete primitive. It fits only the retained front/top controlled-alpha source observations.

Source geometry was inspected for this independent residual diagnosis. That prior diagnostic exposure is explicit; the optimizers receive only source alpha, actual cameras and intervals frozen from the selected recipes/source pixel steps. Original ob145 was already exposed during earlier selection, metric and presentation review. It is excluded from this fit, bounds and ROI choice; this is a follow-up check with disclosed history, not a claim of an unseen test set.

## Observed fitting and retained history

Both stages use at most 96 residual calls and a one-second fit budget, with a reserved local-rank phase. Partial frames contribute only actual internal half-coverage crossings; artificial frame-border closure is forbidden. Equal-view L4 residuals are in source-pixel units. A changed recipe is retained only when observed mean absolute, P95 and maximum errors are all nonworse, local rank is full and no interval bound is active. This establishes local identifiability within the stated fixed-model scope, not global uniqueness.

| Stage | Selected physical controls | Mean absolute contour px, old → proposal | P95 px, old → proposal | Max px, old → proposal | Actual fitting |
| --- | --- | --- | --- | --- | --- |
| Arch roof | notch height `1.29960368904062` | 0.046998046 → 0.029674513 | 0.092575646 → 0.063109202 | 0.449354325 → 0.108373949 | 33 calls, 0.2413503 s, rank1 |
| Triangle dome | center Y `0.000022228765581030867`; center Z `-0.0005205612606806178`; front fraction `0.49975207897565155` | 0.150737190 → 0.114293507 | 0.421516946 → 0.211945255 | 0.524644109 → 0.322646894 | 65 calls, 0.6523627 s, rank3 |

All three fresh CPU owners are preserved under `temp/tasks/quality-continuation-20261009/tail-contour-cpu-01/`:

- `owned-j51vn8ro` failed and released after the arch proposal was saved. A scratch reader compared producer NPY file identity to the model's array-payload identity while loading triangle observations. That primary failure remains unchanged.
- `owned-nmdlk8dy` succeeded and released its measurement receipt, reusing the exact arch proposal without refitting. Its triangle attempt exhausted the fit allowance before rank admission, so the original exact controls were retained. It remains an explicit rejected trial.
- `owned-er25pj8u` succeeded and released. It reuses the same arch proposal and holds the admitted triangle proposal after eliminating redundant full-vertex sorting and reserving rank/time calls. These are the only two saved proposals in this native plan.

The proposed recipe file SHAs are arch `76dc835cd7d73e67426e2cedbdaa32ac8ec4bf9cf684a38008cb88f751af5d3d` and triangle `9150d659097fdaf33b6aa006541b7599ebaa536ed4db15cb6f4ac9544907a626`. Retained earlier stage metadata is scoped as historical to its exact baseline geometry/recipe. It does not describe new-body observations.

## Frozen native scope

The single fresh process has exactly 12 new 512² frames: five candidate linear-alpha views and one candidate ob35 neutral per family. Every alpha replays the same original frozen camera declaration, clipping, EEVEE64 sampling/filter1.5 and half-coverage threshold. The original area0.7/boundary0.8/SDF0.05 silhouette gates are unchanged. All five gates, including original ob145, are evaluated. No source acquisition, native fitting or control search is performed.

The two source ob35 neutral PNGs are reused from their exact successful/released final-qualification owners. Their original SHA, source body, pass settings, actual matching frame and clipping are bound. New candidate neutrals must use those same settings and actual frames; they are for human inspection, not geometric normals. Remaining new-body neutral/normal views are explicitly unrun.

Each case has one new 4096-per-direction/seed61007 raw comparison after its recipe/body is saved, one physical source-pointer edit/restoration transaction, and one existing known-package native boundary qualifier with the unchanged 15-second cap. The edits are arch notch height×1.005 and triangle front fraction×1.01. Native geometry changes, independent physical response and exact indexed restoration are checked. Old exact raw observations are reused as baselines; no old raw recomputation occurs.

A `bounded_checkpoint_improved` result requires every original silhouette gate, nonworse original ob145 alpha L1, nonworse raw mean/P95/sampled maximum, exact edit restoration and actual single-solid boundary qualification. All fields remain separately reported if that conjunction fails. Geometric normal observations are retained independently. Artist limits remain null and aggregate acceptance false. No derived engineering/source-facet allowance is promoted into artist approval.

The fresh supervisor work/join limits are85+5 seconds, two declared numerical/Blender threads and8GiB committed/RSS. It records complete-tree Windows Job/HANDLE lifecycle; each qualifier keeps its own fresh sibling owner. The producer reserves 96MiB for12 frame writes plus16MiB geometry/metadata allowance under its256MiB artifact cap. It freezes125 bounded small inputs totaling21,261,980 bytes plus two existing toolchain binaries totaling264,457,688 bytes. Large binary hashing is streamed, with512MiB/file and512MiB aggregate guards. Supervisor provenance uses only16 explicit critical inputs under its unchanged64-input/256MiB cap; the producer verifies every declared input before and after execution. No package installation/version change, helper fallback or historical lease adoption occurs.

Actual `validate_plan` passed in3.8099331 seconds with the real owned files. Eight focused model checks passed in2.6281118 seconds; two saved-checker guards passed in0.013 seconds. They cover actual mesh projection/controls, exact restore, rank/censored refusal, observed-tail selection, held-out isolation and scope expansion refusal. No broad suite or native run was used for preparation.

## Review and launch

The exact launch argv is in `launch-command.json`, with a prevalidated `prepared_command` and current-byte `prepared_provenance`. Root launches it after independent review and the multipart window joins. The following only illustrates how to invoke the frozen argv after that authorization:

```powershell
$packet = Get-Content -LiteralPath 'docs/quality-tail-refinement-continuation-20261009/launch-command.json' -Raw | ConvertFrom-Json
$launchArgs = @($packet.argv | Select-Object -Skip 1)
& $packet.argv[0] @launchArgs
```

Output parents are fresh `temp/tasks/quality-continuation-20261009/tail-supervisor-01` and `tail-checkpoint-01`. Previous owners, failed receipts and the current demo data remain untouched. The concrete preparation has no unresolved input blocker. Independent native/pixel results remain pending root's sole serial launch.
