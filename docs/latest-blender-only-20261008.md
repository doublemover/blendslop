# Latest stable Blender only

Owner correction: support the latest official stable Blender release only.
On 2026-10-08 the [official download](https://www.blender.org/download/) and
[5.2 release page](https://www.blender.org/releases/5-2/) identify **5.2.2 LTS**.
The [release archive](https://download.blender.org/release/Blender5.2/) and two
Blender-listed mirror manifests agree on Linux x64 archive SHA-256
`84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168`.
5.3 preview and pull-request builds do not change the stable release choice.

This focused correction starts from merged main
`8a1de973aeeae3fb22d9940cabf22b4f00e924ce` on
`correction/latest-blender-only-20261008`. PR/push full and quick jobs and the
scheduled full job now use only the checksum-pinned 5.2.2 archive. Stable cycle,
exact version, bundled Python and scheduled source SHA checks remain explicit.
Cadence, main source selection and historical result packets are unchanged.

Old solver version branches, FAST configuration/CLI support, older native-test
skips, EEVEE_NEXT aliases, the multi-version suite and historical comparison-only
tool are retired. Ordinary solver, mesh joining, query, SDF, renderer, edit,
geometry, image, config and end-to-end checks remain. Current runtime checks
reject another release or preview before native workflow work; invalid explicit
solver choices fail rather than silently choosing a substitute. Active setup,
README, agent and CI docs describe the supported current installation.

Validation uses the existing local **5.2.2 LTS release / Python 3.13.13**.
All 73 intended affected pure cases, 28 cadence checks and ten native checks pass.
The native batch also constructs the production workflow through its runtime
gate. It takes 6.821 seconds, peaks at 397.73 MiB RSS and uses two threads with
60-second/2-GiB caps, factory startup, background mode and no renders. One pure
harness loader name was corrected and only that single case rerun; raw logs
retain both invocations. Static YAML/Bash/checksum/source-contract checks pass.
No local install, package change, old full-suite repeat, historical deletion or
quality/performance campaign occurs. [Source-scoped receipt](latest-blender-only-20261008.json)
records checks and raw log hashes.

Proposed integration: publish a draft PR from this correction branch to main,
run the single-release hosted full/quick/policy checks on its final head, then
obtain parent approval for a normal merge. No new PR or merge has been performed.
Hosted 5.2.2 provisioning is prepared and statically checked; that Linux run is
pending draft PR publication approval. The existing Windows installation already
matches the target, so local upgrade approval is unnecessary.

Existing visual-quality and lifecycle blockers in
[the quality task map](quality-implementation-20261008/README.md) remain independent.
Current-version correctness does not complete those missing or failed verdicts.
