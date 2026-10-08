# Corrective batch after the frozen campaign — 2026-10-06

The corrective source batch starts from report commit `8e7c15d24221805cf6a14e6d2914d46f5e19816c` on `endblay_opslay`. It repairs failure handling, interrupted-work retention, DVX artifact replay, the three failing native suites and qualification reporting. The [frozen campaign results](quality-final-results.md) remain measurements of implementation `06a7f0219c98612a03b6e7e127d698fd9cc6c541`; they are not measurements of this corrected source. No new performance/held-out matrix ran. The exact corrective HEAD, source checksums, patch and incremental Git bundle are in the Library recovery packet.

## Done and still open

| Request | Completed in this batch | Remaining |
|---|---|---|
| R1: preserve failed cells | Required absent views produce null aggregate metrics and explicit unavailable reasons. The worker preserves primary/backend failure information and avoids dereferencing absent render paths/meshes. All five saved primary failed payloads were exercised. | The historical five failed cells remain failed; this batch did not rerun their reconstruction. |
| R2: fitting mechanics and work accounting | Atomic scored checkpoints preserve finite best work after interruption. Recorded objective counts are separate from conservative reservation charges; interrupted unacknowledged work is flagged as unknown. Queue, worker and total wall times and stop reasons are explicit. DVX's final requested Adam update is evaluated and eligible for retention within the same allowance. | Choose same-budget routing or a separately declared longer-refinement policy before new measurements. Twelve steps are not proved optimal. |
| R3: editable DVX initialization/deformation | Explicit initialization JSON at resolution 16, part ranges, seed OBJ, retained OBJ, deformation NPZ, fixed transform and content identities. Replay rejects tampered stored output and verifies connectivity/coordinates. The actual installed CPU helper pipeline produced and replayed these artifacts. | Editable parameters describe the seed; the NPZ describes retained deformation. Semantic editability and one connected qualified solid remain unproved. |
| R4: native correctness failures | Empty-camera surface-only fitting works; the differentiable mock implements the dependency interface. The receipt test expects the strict qualification rejection. Grammar bounds translation now requires a Mapping, fixing the actual swallowed `TypeError` from legacy `bounds=True`. | Installed Blender 5.2.2 coverage only; no new complete campaign or cross-version certification. |
| R5: solids and reported pairs | Specific guard/bound/toolchain/helper reasons; within/between-component intersection counts and sampled face provenance; explicit distinction between topology/volume screening and single-solid qualification. Actual cube passes, overlapping cubes fail. | Narrow-phase confirmation, the sampled loft issue and qualification resource policy remain open. No automatic geometry repair was justified by the sampled evidence. |

Default/quality presets, requested update counts, optimizer/search caps and acceptance thresholds retain their previous values. Checkpoint reservation still conservatively limits the existing search; it is not reported as measured objective work. A partial reconstructed artifact without fresh render validation stays failed/unadmitted. Strict toolchain/hash qualification requirements remain enforced.

## Validation receipts

- Consolidated affected correctness run: **133 passed, zero failures/errors/skips**, eleven modules, in actual Blender **5.2.2 LTS** (`d13f752e3b9c`). Includes the previously failing grammar, differentiable-render and native receipt suites, and real worker termination/recovery.
- After the final known-versus-reserved work-count correction: **34 affected checks passed**, zero failures/errors/skips. These overlap the 133 checks and must not be added as unique coverage. Saved five failed payloads, actual native solid fixtures and an actual CPU DVX artifact pipeline were also checked in this receipt.
- Actual DVX 16³ numeric fixture: one requested update, two objective observations, loss **0.0110338852 → 0.0106357010**, best observation 2, finite retained vertices and `final_update_evaluated=true`. The actual candidate/artifact fixture also records one update/two observations, admits its retained result under the existing projected guard and replays exact stored output identity. These tiny fixtures establish mechanics; they do not establish reconstruction quality or refinement convergence.
- **535 Python files parse**; `git diff --check` passes. Fixture-owned worker processes were joined; unrelated processes were not touched.
- The first consolidated run's **103 passing checks and three diagnostic errors** remain in `validation.log/json`. Two implementation errors (parameter rotation roundoff and a test executor without worker stack state) were repaired; one erroneous validation module name was corrected. Passing reruns are separate files.

Machine-readable details, module names, source hashes and actual fixture receipts: [corrective-batch.json](corrective-batch.json). Full local logs: `temp/corrective-batch-20261006/validation-repaired.log`, `validation-final.log`, `dvx-final-update.log`; matching JSON files are preserved. Validation scripts and logs are included in the recovery packet for reproduction.

## What the saved pair diagnosis establishes

Four saved candidates were inspected by indexed part/component provenance, with source OBJ hashes preserved. The sample is at most twenty reported pairs per candidate, not an exhaustive diagnosis of every old report.

| Saved candidate | Sampled reported pairs | Coordinate finding |
|---|---:|---|
| box/default/profile loft | 4 within a component | No simple separating-axis contact certificate; case-specific diagnosis remains. |
| box/default/primitive fit | 4 within, 16 between components | All four sampled within-component pairs are disjoint: their only possible contacts on a shared coordinate plane are distinct points. Between-part pairs retain their original report. |
| box/CPU DVX | 20 between components | Reported pairs belong to different initialized parts. Multipart semantics remain explicit; this is not qualified union evidence. |
| vase/quality/shape program | 8 within a component | All eight sampled pairs have distinct possible contact points on a shared coordinate plane, so their saved-coordinate triangles are disjoint. |

The coordinate certificates challenge these specific reported pairs. They do not certify every triangle or authorize globally filtering Open3D findings. Raw intersection counts and conservative rejection remain unchanged. No shapes were welded, warped or presented as repaired single solids. The old **49 intersection reports and 152 unavailable qualifications** were not rerun; the old combined unavailable reason cannot be retrospectively split into guard/bound/helper counts.

## Remaining choices and unperformed work

1. Pick the longer-refinement/routing policy, then gather source-specific performance and held-out measurements. The earlier falling DVX losses do not demonstrate a plateau or justify declaring twelve steps sufficient.
2. Preserve editable multipart as the output contract, or separately qualify an evaluated union while retaining artist sources. A seed parameter file cannot reproduce arbitrary vertex deformation.
3. Confirm questionable reported pairs with an appropriate narrow-phase check and diagnose the loft case before a case-specific shape repair. Choose any larger qualification resource allowance explicitly; this batch did not change the 60,000-triangle/15-second helper policy.
4. GPU, Blender 4.2/5.0, native external-paper data/weights and semantic editability studies remain unperformed. The earlier campaign's quality regressions and missing-data limitations are still reported in the frozen result documents.

No new installs, downloads, pushes or external code publication occurred. Current source edits, tests and this report form one local corrective commit. The recovery packet preserves the original report packet identity separately and includes the incremental Git bundle, patch, current changed sources, diagnostics and exact final HEAD.
