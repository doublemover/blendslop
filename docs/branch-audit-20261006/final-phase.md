# Blendslop improvement phase: frozen final candidate

Historical checkpoint. Current implementation and validation status: [frozen final campaign results](quality-final-results.md). Measurements below retain their original source revisions and do not qualify the new candidate.

Current accepted continuation: [native performance and metric phase](native-phase.md). This document retains its earlier snapshot and measurements.


This continues the six authorized improvement areas on branch `endblay_opslay`, starting from camera-control commit `b4186fbbf447105ed1106e59e086a296d828170b`. Source edits, bounded local tests/benchmarks and local commits are authorized. No install, push or publication is part of this phase. Earlier audit restrictions are historical.

The final source is frozen in `temp/improvement-phase-20261006/final-candidate/validation.json`. That manifest records every Python source hash, the starting commit, dirty files, verified control snapshots, job commands, exit codes and logs. The final matrix writes into a new directory; the 82 preceding measurements and their failed/rejected experiments remain intact. The two preserved control manifests verify 458 training-source files and 471 held-out-source files byte for byte against the saved control checkout.

## Scope and implementation decisions

| Authorized area | Implementation/evidence delivered | Boundary still open |
|---|---|---|
| 1. Freeze matching inputs, baseline and failure map | Preserved camera-control snapshot, source/config/input/mesh hashes, training box/vase/torus seed 1234 and reserved cylinder/bottle/chair seed 77; per-case decisions retain regressions and gate failures. | Six synthetic cases and one timing sample per cell do not establish broad generalization or confidence intervals. |
| 2. Weak primitive/Gaussian/refinement methods | Full projected mesh-union primitive objective and explicit spread seeding; default shared height-based k-means restored. Gaussian two-sigma/guard/spread variant remains opt-in and is rejected from the final arm. Differentiable CPU fitting uses a coarse grid, full-resolution final evaluation, negative-space seed admission and fitted mesh scale 1.0. Confidence and soft-render diagnostics retain distinct namespaces. | Primitive mesh-union remains opt-in. Neither proxy assemblies nor silhouette improvement imply a valid solid or better unseen surfaces. No additional tuning on held-out cases. |
| 3. Cost-aware ensemble | Opt-in measured routing freezes actual geometry before scene replacement, uses fixed recorded-camera Blender renders, requires all observed-view gates, and admits mean-IoU gain >.002 with worst-view loss ≤.002. Ledger records skips, reasons, full wall time and selected backend. | 45 s is a soft candidate-admission cap; an in-flight candidate can overrun. Each benchmark child has a separate 100 s hard cap. No optimality claim for the finite pool or geometry-based selection using reference meshes. |
| 4. Camera/coordinates/units/crop | Canonical axes and pixel centers, recorded viewport edges and unequal known crops/scales, raw calibrated validation pixels, rejection of perspective/rotated axes/inconsistent extents/mixed unconverted units. Default extraction preserves holes and disconnected components. Held-out seed 77 contains prescribed crop/scale changes. | Uncalibrated views remain ambiguous; unknown cameras, arbitrary pose/perspective and unknown absolute scale are unresolved. |
| 5. Solids/editability/modifiers/exports | Structural mesh screens plus existing Open3D native intersection/containment checks; overlap, Boolean union, disconnected, nested and duplicate fixtures; parameter persistence and editing of actual saved primitive parts; Blender modifier stack preservation; actual reconstruction OBJ/GLB exports; transformed native OBJ/GLB/STL/BLEND fixtures at metre and centimetre scene scales. | Multiple closed part shells are not one Boolean solid. Internal/self-intersecting surfaces, failed screens and explicit native budgets remain failures or unverified. Semantic part usefulness, topology taste, UV/texture/animation fidelity and external DCC delivery require separate evaluation. |
| 6. Comparable external baseline / optional Poisson | Existing Open3D 0.19 Python 3.12 CPU helper runs in a capped isolated process using the same reconstructed hull surface and normals. Density filtering now precedes cropping to preserve vertex correspondence. One declared no-trim/no-crop follow-up tests the opening introduced by those operations. | This is a downstream hull postprocess, not an independent learned RGB baseline. No PartGS/Light-SQ/SparseSurf/PrimitiveAnything reproduction, downloaded data/weights, new dependency installation, GPU or OpenVDB certification. |

The repository scope is orthogonal silhouette blocking and configurable backends: `docs/IMPLEMENTATION_SPEC.md`, lines 8-21 and 122-125. This report does not claim completion of every historical acceptance checkbox.

## Comparison contract

Both arms use exactly the saved reference PNGs, camera metadata and evaluated ground-truth meshes. Images are 512×512 orthographic front/side/top. Surface metrics use 8192 area-weighted samples, seed 1234 for training and 77 for held-out, F-score tolerance .02, independent bbox centering and uniform longest-extent scaling, with no rotation or anisotropic alignment. Chamfer L1 sums the two directional mean distances. Single-component closed-mesh 24³ parity is a coarse conditional volume diagnostic; native solid screens are reported separately.

Observed-view gates require each area IoU ≥.7, boundary IoU ≥.1 and signed-distance loss ≤.1. The separate orbit-45 silhouette and reference geometry are evaluation only; they are never fitting or ensemble selection input. Training and held-out means remain separate. A process success, silhouette gate pass, single-solid screen and artist usability are different outcomes.

Workflow time includes reconstruction and observed-view render/validation, excluding Blender startup, novel-view and geometry evaluation. Children run serially with four configured threads and a 100 s hard cap. Optimizers use wall-clock budgets (primitive 8 s, differentiable 20 s), so their final iterations and geometry can vary with host load. Single-run speed differences are descriptive measurements.

## Reproduction and evidence

```powershell
& '.\.venv312\Scripts\python.exe' scripts\run_final_improvement_validation.py --blender 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --output temp\improvement-phase-20261006\fresh-final-candidate
& '.\.venv312\Scripts\python.exe' scripts\summarize_improvement_phase.py --phase temp\improvement-phase-20261006\fresh-final-candidate --output temp\improvement-phase-20261006\fresh-final-candidate\report
```

The driver requires a fresh destination, verifies saved controls and references, freezes source hashes, runs final training and held-out rows, the single Poisson follow-up, native tests, fixtures, exports and native solid analysis serially. A timeout terminates only its owned process tree. The driver refuses further jobs if source hashes change. Reports preserve every requested cell, including process failures.

Raw evidence remains under `temp/improvement-phase-20261006/`; generated PNG/OBJ/BLEND files are intentionally ignored by Git. Committed compact results and per-case CSVs will link to those local paths.

## Final outcome

[All eight technique means, separated by training/held-out](final-results.md), [all 48 paired cells](paired-cells.csv), [technique CSV](techniques.csv), and [compact machine-readable status](final-summary.json) are committed alongside this report. The paired CSV distinguishes silhouette decisions, geometry regressions and novel-view regressions.

The final training means are primitive IoU 70.7% -> 85.2% / F-score 17.1% -> 30.6%; differentiable IoU 50.0% -> 84.0% / F-score 24.2% -> 28.7%. Held-out means are primitive IoU 62.7% -> 83.6% / F-score 17.2% -> 31.1%; differentiable IoU 43.4% -> 82.3% / F-score 22.3% -> 22.1%. Primitive fitting passes one of three required-view gate sets in each split; refinement passes two. Chair remains a failure for both. Box and chair F-scores regress for both weak methods, even though observed and novel-view silhouettes improve. Primitive settings remain opt-in; broad geometric improvement is not established for refinement.

Profile, hull, hybrid, shape-program and restored Gaussian geometry/quality are unchanged. All six Gaussian output mesh hashes equal their controls; the earlier 75.0% training-IoU Gaussian experiment remains preserved and rejected. Ensemble selects visual hull in all six cases, with unchanged quality/geometry. Its measured workflow means are 71.1 -> 51.5 s training and 65.0 -> 39.5 s held-out (about 28% and 39% shorter in these single runs).

The full installed-Blender runner passes **86 groups, 0 failed, 0 skipped**, including procedural and E2E workflows. Eight representative reconstruction meshes pass both OBJ/GLB round trips with world vertex agreement below 1e-5. Eighteen saved primitive assemblies reload within 4.9914e-9 world units and respond to an actual parameter change. Native fixtures verify exact Boolean union, distinguish overlap/duplicate/nested/disconnected meshes, and preserve the editable modifier stack and source transforms. Metre and centimetre OBJ/GLB/STL/BLEND fixtures pass.

All **104** native solid records finish without analysis errors. A separate 150,000-face / 300-second bounded screen qualifies larger hulls; identical mesh hashes transfer that evidence to matching controls and ensemble outputs while preserving the original 60,000-face report. **27/48** final outputs pass the stated single-solid structural screen: profile/hybrid/hull/ensemble 6/6 each, shape-program 3/6, and primitive/Gaussian/refinement 0/6 each. All six visual hulls and their selected ensemble meshes have zero detected intersections. A structural screen does not establish correct shape or artist usability.

Untrimmed/uncropped Poisson produces watertight box/vase/torus meshes with IoU 98.9% / 96.9% / 96.0%. Box and torus pass the native solid screen; vase has one intersecting triangle pair and is rejected. Earlier trimmed/cropped nonwatertight outputs remain preserved. No more Poisson tuning or dependency acquisition is performed.

## Acceptance repair and provenance

The coordinated matrix ran once on its frozen source. The first full-suite invocation failed before tests because the launcher omitted Blender's `--` separator. Actual candidate export checks then exposed a saved Gaussian deserialization defect: reconditioning a valid covariance again could rotate a repeated eigenspace, so its mesh changed on reload. `from_dict` now retains the exact valid saved covariance; malformed covariance still receives constructor repair. A regression test covers exact replay and invalid covariance repair.

The fresh constructor, seeding and all mesh-generation methods are AST-identical to the matrix version. The measured Gaussian path initializes fresh primitives; measured refinement initializes ellipsoids. Gaussian saved-part deserialization is absent from those measured reconstruction paths, so the matrix remains valid. Only the native suite and affected export/replay checks were rerun, against a new frozen acceptance hash set. All now pass. The report labels were clarified to expose geometry regressions separately. The original failed logs, matrix hash set, changed-source list, AST-equivalence reason and acceptance hash set are retained in `validation.json`; no failed run was relabeled successful.

Owned children all exited or were waited on; no global process cleanup was performed. There were no installs, pushes, paid compute or publication.

## Done and remaining

All six bounded work areas in the scope table were implemented/evaluated, the held-out matrix is complete, and final local validation/reporting is complete. Remaining acceptance failures are the box/chair weak-method geometric regressions, chair required-view failures, non-solid proxy assemblies, three non-solid shape-program outputs and Poisson vase intersection. Averages do not waive them.

Deferred work is real photographs/unknown camera and unit recovery, broader shapes/seeds and repeat timing statistics, true solid-union delivery for proxy assemblies, calibrated 3D thresholds, artist semantic/editability studies, UV/texture/animation/external-DCC fidelity, learned external baseline reproduction, GPU/OpenVDB qualification, and fresh Blender 4.2/5.0 coverage. These require separate scope or unavailable/unapproved dependencies/data; this pass did not attempt to install them. The historical 1,344-checkbox refinement inventory remains a backlog snapshot, not 1,344 newly missing implementations or completed acceptances.

Local implementation commit: `974821ccf2c579ab2fcf629b666907c96c4961f0`. Validation tools and this report are saved in the following local evidence commit. No remote push is performed.
