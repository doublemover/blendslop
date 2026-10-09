# Retained rounded triangle: final independent qualification plan

This packet is ready for root's serial review and launch. The helper has not launched Blender. It supplements the retained `bfe72738…` adaptive checkpoint with missing full-view observations and one existing solid-boundary helper. It preserves the original twelve-row selection, all earlier failures and raw regressions. Artist surface limits remain null and aggregate acceptance remains false.

The source was frozen before measurement: original authored `rounded_triangle_dot`, indexed source hash `2681272b491af3370176a3751e408ef36496ba592d230bce7c733b8c41cadc64`. The selected saved recipe/body is exactly `bfe727380abc81d0215abec5687b77b83c251133605149a6f7ad2f714423a89d`, with 6,239 vertices and 12,474 triangles. Source construction and saved candidate compilation must reproduce these exact indexed identities before the first frame and after all passes. There is no numerical fallback, fitting, new raw sampling or semantic control rerun.

## Frozen execution packet

- [native-plan.json](native-plan.json): SHA256 `0a1b781aeac7900cc5f7e5aeb66cdc9149e50fabb2c52f2cfea7902c24553c65`.
- [launch-command.json](launch-command.json): exact argv, prepared supervisor declaration and critical-input snapshots; SHA256 `c8d1a5587c13c35f9827531ae0ddaff217507a908c5b428e4d654875ec9ae5d2`.
- [preflight.json](preflight.json): read-only owner, body, package and acquisition audit; SHA256 `07dd0b15745db67918ee4c0442820b7283df82249dd0a92e08d7e8ccca9f6754`.
- [adapter](../../scripts/run_triangle_final_qualification.py): SHA256 `435f9860638102d01f269a85dacd204d601b3444da02c574a33fb44bd7854619`.
- [focused checks](../../blender_blocking/test_triangle_final_qualification.py): SHA256 `54bf842f7a17614f1a52843b301682a0f8838f5a21817a55d037c7d18606f760`.
- Root's registered test runner: SHA256 `f82fb657f472d55b9ebb209bf6aa8468ac942d6c99f4f3c0c7f32aebb4b40cd8`.

The plan was validated against 72 current source/input files totaling 273,529,813 bytes, below the adapter's 512 MiB input cap. The supervisor snapshots only the plan and eleven small critical files: twelve explicit inputs totaling 208,474 bytes, below its 64-file/256 MiB provenance cap. Large installed package binaries stay in place; their full hashes are frozen in the plan and checked by the adapter before and after execution. Frozen checkout HEAD is `e96191bc72041ea3414fd5c6df1e98dbcd0bca93`; the two new runtime files were uncommitted when frozen. Runtime and test CRLF bytes are preserved.

## Exact counts and limits

| Observation | Source | Candidate | New native frames |
| --- | --- | --- | --- |
| Linear alpha/mask | front, side, oblique35 new; top and oblique145 retained | front, side, top, oblique35 new; oblique145 retained | 7 |
| Neutral preview | original five views | original five views | 10 |
| RGB world-normal preview | original five views | original five views | 10 |
| Retained alpha slots | 2 | 1 | 0 |

The total is 27 new 512² frames plus three retained completed measurements, covering thirty logical passes. Each new mask uses controlled opaque linear float32 OpenEXR alpha, threshold 0.5, EEVEE 64 samples, filter 1.5, unit pixel aspect and the exact frozen original camera declaration. Neutral/normal passes use the existing shared inspection settings; actual pass settings, frame hash and geometry binding are retained. RGB normal previews describe visualization; existing raw normal metrics are geometric face-normal comparisons.

One existing known-package solid-boundary helper has an unchanged fifteen-second allowance and no fallback. The fresh Windows complete-tree supervisor allows 85 seconds of work plus five seconds of final joins, committed-memory and sampled RSS limits of 8 GiB, and two Blender/numerical-library threads. The child begins that helper only with at least twenty seconds remaining. Outer supervision still includes adapter preflight and startup. The total generated-artifact cap is 256 MiB: conservative thirty × 8 MiB logical frame allowances plus 8 MiB geometry exports and 4 MiB metadata total 252 MiB, leaving 4 MiB. Shared ownership registration enforces the total cap.

The exact executable argv is retained in `launch-command.json`; root can invoke it from PowerShell without rebuilding or shell-joining the arguments:

```powershell
$triangleLaunch = Get-Content -Raw -LiteralPath 'docs/quality-triangle-final-qualification-continuation-20261009/launch-command.json' | ConvertFrom-Json
$triangleArgv = @($triangleLaunch.argv)
& $triangleArgv[0] $triangleArgv[1..($triangleArgv.Count - 1)]
```

This is a prepared command, not an execution receipt. Fresh child evidence goes below `temp/tasks/quality-continuation-20261009/triangle-final-qualification-01`; the separate supervisor goes below `triangle-final-supervisor-01`. The boundary helper uses a sibling output parent, avoiding nested-owner adoption. Existing outputs are not overwritten, adopted or cleaned.

## Retained acquisition and body evidence

Original checkpoint owner: `temp/tasks/quality-continuation-20261009/adaptive-triangle-detail-01/owned-5w48rxfi`. Its manifest is succeeded, lease released, read-only ownership audit `dry_run_ready`, retained bytes 10,984,161, with no unknown files or blockers. Original result SHA256 is `81dd1d598d16b461d312cb389c2cefca7fbb1f517f48419df7a3fb68c2e7cce7`. Exact saved recipe, NPZ, OBJ, raw row, source equivalence and corner-radius response/restoration are bound to this owner and selected body.

The reused slots are source `source-fit` at top, source `source-heldout/oblique_145_40`, and candidate `refined-heldout/oblique_145_40`. Each retained measurement's original manifest/lease identity, registered file SHA, geometry identity/stability, contract signature, resolution and noncamera acquisition settings were verified. The original metadata has no coverage-file SHA field; the original owner manifest and the frozen NPY byte hash provide that binding. The adapter additionally checks that metadata field when present. A new black-occupied hard-mask PNG is explicitly derived from retained coverage >= 0.5 and does not count as a native frame. Reused EXR bytes keep their original ownership provenance even though copied into the new owned packet.

Original camera declarations omit clipping and actual geometry/pass binding. They remain legacy unavailable; this packet does not backfill those claims. Replay captures actual clipping (0.10000000149011612 to 1000), shifts, resolution, pixel aspect and camera frames for every pass. The top retained frame matches its original declaration hash. The retained oblique145 frames have the earlier documented native matrix-decomposition difference from the serialized declaration; source and candidate retained actual frames agree. Every new pass replays the same original declaration and requires strict actual frame/clip agreement with its partner and any reused slot, without a numeric tolerance. A discrepancy blocks that row rather than triggering extra rendering.

## Independent verdicts and exposure history

All five original-view silhouette gates remain area IoU >= 0.7, boundary IoU >= 0.8 and normalized signed-distance loss <= 0.05. The independently retained support/thickness limits remain 0.005 world units. These and the one actual solid-boundary verdict are recorded separately; a valid producer packet can contain failed required gates. Geometry, presentation and editability are independently reported. Paired pixels still require actual post-run inspection.

The source-only camera declaration output uses the existing report seam (`source-declarations.json`): authored parameters, original/regenerated exact source NPZ identities, exact oriented surface hash, actual five-view camera packet and source neutral files. Root may later evaluate its independently frozen engineering policy. Source certificates never consume selected candidate geometry or metrics, and source discretization allowances do not establish artist approval.

The old adaptive fit used genuine top acquisition plus a genuine 1024 detail crop. Original oblique145 was excluded from that fit and ROI selection, but its earlier display-managed silhouette was inspected; this packet makes no historical-blind claim. Original top crop and oblique clipping exposure are preserved. No camera, crop, tessellation, controls or threshold is tuned during this qualification.

Existing raw metrics are reused without resampling: symmetric mean 0.0006732319122740825 world, P95 0.0022511934163048863 world, sampled max 0.0029814052395522594 world, geometric normal P95 1.4659143045971141 degrees. The sampled max increased from baseline 0.0028801681473851204 and remains an explicit regression. Existing corner-radius ×1.1 response and exact indexed restoration pass for this exact retained body; this run does not repeat it. Artist surface thresholds remain unresolved.

## Focused verification and remaining work

Six meaningful pure checks passed on existing Blender bundled Python 3.13.13 in 0.890 seconds. They cover original-owner coverage reuse, active lease/file drift refusal, self-consistent altered geometry/settings/coverage metadata refusal, exact camera/clip disagreement, scope/budget refusal and finite NumPy scalar receipt transport. Fixtures use real fresh files and leases; no Blender test, broad suite or benchmark was run by this helper.

The pure plan validation and existing supervisor's read-only `prepare_command` succeeded. There are no current preflight blockers. Native replay, new pass acquisition, one boundary verdict and real paired pixel inspection remain unrun until root launches the frozen argv. Retained source/candidate inputs, current row selections, failures and owner histories are preserved.

Existing known-helper installation identity (full hashes, no install or package modification):

| File | Bytes | SHA256 |
| --- | --- | --- |
| METADATA | 4193 | `7020e4ea25bf1195edece73e72120cd2b86ead8ef3e8f9db10bafb05c9c2e572` |
| pybind.cp312-win_amd64.pyd | 151434752 | `ab1b2f4a1126a11886a2a9f3eeab5ceaeac0bab1b53ee319eb363df7ef629d88` |
| METADATA | 6608 | `8f9e04338ffa930868a2d157ce44f6e4cb7ac524b968fb13d04187a6ac575282` |
| _multiarray_umath.cp312-win_amd64.pyd | 3714560 | `f4523c2dfe641a904605029ce4a5046388bd61b18afc0594d34018b5e6d848f8` |
| pyvenv.cfg | 330 | `7629eb68ddb84edf4972de38284488d9c6e37a4badef61cecb67b945efe17d74` |

## Completed native acquisition

[Actual evidence](native-evidence.json) supersedes the preparation-only execution status above. The one job completed all 27 new frames and verified three original-owner alpha passes in 29.763 seconds. All five unchanged silhouette gates and the .005-world support/thickness checks pass for exact bfe72738 geometry. The one uncached native boundary helper qualified the actual single solid; total qualification cost including identity/transfer was 14.700 seconds under the unchanged 15-second helper wait. The full outer scope finished in 39.313 seconds. All five observed creation HANDLEs were kernel joined at exit0; Job active membership is zero, sampled peak tree RSS730,996,736 bytes and separate peak Job committed memory2,784,456,704 bytes. Producer, helper and supervisor leases are released and their read-only ownership audits are ready with no blockers/unknown files. All72 frozen input hashes remained unchanged; no deletion or old-owner adoption occurred.

Root inspected actual source/candidate top neutrals and oblique145 neutral/RGB shading normals. Their rounded outline and domed gradients correspond closely; the original right-edge crop is retained. This supplies canonical provenance and separate required engineering gates, while artist surface acceptance remains null, the sampled-max regression remains visible, and the current twelve-row selection and aggregate acceptance remain unchanged.
