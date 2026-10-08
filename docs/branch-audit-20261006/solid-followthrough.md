# Saved-solid corrective follow-through — 2026-10-06

This batch starts from `151408febcae21eb525126ffbc6202d80097753c`. It inspects saved geometry, corrects false intersection reporting and guards DVX against folded individual parts. No reconstruction or new quality/performance campaign ran. Original meshes, renders, scores and qualification receipts remain intact. The exact final HEAD and incremental Git bundle are recorded in the recovery packet. Local work is parked after this coherent batch at the user's budget request.

## Complete classification

All **247** saved export/qualification cases are accounted for. Primary case classes are exclusive; labels inside each case can overlap.

| Primary class | Cases | Interpretation |
|---|---:|---|
| Numerical false-positive reports only | 9 | Exact checks prove every reported pair disjoint. Both within-component surface checks and native topology/orientability checks pass geometrically. Actual execution qualification is separately recorded below. |
| Editable multipart overlap | 19 | Primitive cases whose confirmed reported crossings lie between parts. This is useful assembly evidence, not a qualified single-solid claim. |
| Within-part self-intersection | 14 | All saved CPU-DVX cases contain folded parts. Between-part overlap is also common, and is a different issue. |
| Duplicate/coincident surfaces | 7 | Default program cases with repeated/coincident surfaces and boundary contacts; assembly-versus-baked-boundary intent remains a choice. |
| Unqualified by guard/bound | 152 | All fail topology/volume guard; 86 also exceed the 60,000-triangle limit, 66 fail guard without exceeding it. None of these old unavailable cases is explained by a missing dependency. |
| Qualified original receipt | 46 | Earlier qualifications are preserved, not remeasured or recertified against the new source. |

All **152,921** Open3D-reported pairs from the 49 reported cases were independently classified on the decoded saved coordinates:

| Exact pair class | Count |
|---|---:|
| Disjoint native reports / numerical false positive | 483 |
| Proper crossing within an indexed component | 912 |
| Proper crossing between components | 8,248 |
| Duplicate triangles or coplanar area overlap | 33,172 |
| Point/segment boundary contact | 110,106 |

Boundary contacts are explicit evidence, not automatically called invalid overlap. **106 representative checks** agree under a second exact edge/triangle-plane clipping implementation, independent of separating-axis testing. All inspected mesh identities match their original qualification inputs. Per-case ledger: [solid-case-classifications.csv](solid-case-classifications.csv). Full receipts, coordinates and hashes: [solid-followthrough.json](solid-followthrough.json). The reproduction script `scripts/diagnose_saved_solids.py` provides stage/count/elapsed output and five-second heartbeats, and saves atomic partial results.

The nine false-only cases are box/default/profile loft; external ext02/quality/profile loft; vase/default and quality/program; torus/default and quality/program; ext01/default and quality/program; ext06/quality/program. Twenty additional cases mix false reports with actual contacts, so a raw count cannot be relabelled wholesale.

## Source fixes

1. Exact binary64 integer separating axes and rational plane cuts replace tolerance-based removal decisions. Raw native counts remain visible. Only pairs certified disjoint are removed; degenerate or unverified pairs remain blocking. Production reported-pair verification retains its bounded 1,000-pair diagnostic allowance; a remainder never qualifies.
2. Qualification also checks within-component candidate pairs, including indexed neighbors that the native reported list can omit. Ordinary shared-edge/shared-point adjacency uses exact geometric certificates. Toolchain identity includes the new predicate module hash, so older receipts cannot silently pass a different checker.
3. DVX admission checks individual parts before accepting the deformation. Within-part crossing/coincident defects or an exhausted remaining allowance reject the proposal and retain the initialization. Between-part overlap is permitted by this guard. The rejected proposal's exact vertices/hash are preserved alongside seed and retained state, and replay verifies proposal identity.
4. Structural diagnostics retain raw native counts and expose exact confirmed-contact evidence separately. Solid eligibility requires complete verification and the within-part guard; editable multipart and one qualified solid remain distinct contracts.

All 14 saved DVX states fail the new within-part guard, including additional indexed-neighbor defects not established by the native reported list. This demonstrates the admission correction; it does **not** claim new reconstruction outputs or remeasured scores. No saved shape was warped/welded, no default assembly was forced into a union, and optimizer presets, requested update counts, admission thresholds and global budgets retain their prior values.

## Focused validation and limits

- Initial affected/native Blender run: **25 passed, one documented skip**, zero failures/errors, 26 total tests. The skipped native Open3D test is unavailable inside Blender's interpreter; the existing Python 3.12 helper executes the actual saved native checks.
- After the exact adjacency optimization: **9 affected checks passed**, zero failures/errors/skips. Coverage overlaps the initial run and is not added as unique tests. Eight exact/adjacency/DVX regressions and the strict native receipt gate pass.
- Current-source, hash-matched actual qualifications: box/default/profile loft, vase/quality/program and ext01/default/program pass with nonzero raw reports and **zero confirmed contacts**, all one component. Saved multipart primitive and folded DVX fixtures remain rejected in the initial receipts.
- The ext02/quality loft is geometrically clear under the complete saved-coordinate check, but current qualification still reaches the **unchanged 15-second helper timeout**. It remains unavailable under that execution policy. There is no cap increase or inferred execution qualification.
- **538 Python files parse**, clean diff check, all final frozen source hashes match classification and passing affected receipts. The inspected original 201 meshes match their qualification input hashes; the other 46 historical qualified receipts were retained without reopening their meshes.

The initial and final source identities and logs are separate in the recovery packet. No repeated broad test suite, new held-out matrix, install, dataset download, subagent, push or publication occurred. All owned inspection, Blender and helper processes exited; unrelated processes were not touched.

## Concrete representation choice and remaining decisions

[Visual alternatives for the saved chair program](solid-visual-alternatives.html): retain 11 independently editable indexed parts, or preserve those artist sources alongside a separately evaluated Exact union preview with two indexed components. The preview's native boundary checks pass; its single-solid qualification is false. Bounds differ by at most `2.64e-8` metres. Neither alternative is applied to the source artifact, and containment/semantic intent for the remaining components is unestablished.

Remaining choices: assembly versus separately evaluated boundary output for the seven duplicate/coincident program cases; any explicitly larger qualification resource policy for large clear meshes; and the previously open longer-refinement policy. New-source reconstruction quality, convergence and held-out/performance measurements remain unrun. Cloud quality research continues independently; local follow-on execution is parked.

## Source/evidence handoff already saved in Library

- Full exact `151408f` committed source: `Blendslop-full-source-20261006-151408f.zip`, 605 files including all 535 Python modules, plus commit/file hashes and a separate empty uncommitted patch.
- `Blendslop-saved-hulls-and-evaluator-20261006-151408f.zip`: saved default box/cylinder hull OBJ, matching GT/capture specs and evaluator modules.
- `Blendslop-vase-hybrid-scene-export-evidence-20261006.zip`: ordinary vase hybrid/hull/profile result/configuration receipts, saved render and mesh references plus available OBJ geometry. No full scene .blend was present in those cell artifacts; this limitation is recorded.
