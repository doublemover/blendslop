# Adaptive CI maintenance

Scheduled full suites run at most once per calendar month by default. Weekly
UTC evaluation slots allocate only the small cadence job when no suite is due.
Relevant uncovered code, test, dependency or toolchain activity permits another
pass after 14 local calendar days. Unrelated prose does not promote a suite;
protected machine-consumed documents do. Activity expires after 28 local days.
Default timezone is America/Detroit; DST does not turn 14 days into an hour count.

The first weekly slot on/after monthlyDay is the monthly opportunity. Active
cadence continues from the previous actual attempt across month boundaries;
a third fortnight in a long month is possible. Failed attempts also consume a
cadence slot; they never become a complete-success baseline. No-output or skipped
work is not full coverage. API uncertainty fails the decision job instead of
claiming validation passed. Large changed-file sets conservatively select full
scope. Original PR/push required checks remain; this is scheduled-work control,
not a replacement for merge/release validation.

The dispatcher and its called workflows must reach the actual default branch
before cron is active. A commit on a development branch is only prepared work.
Manual dispatcher invocation still respects cadence and default-branch identity.
No token, repository setting, billing plan or paid runner is provisioned.

Budget reservations are planning allocations, not account-wide spend enforcement.
Per-repository GITHUB_TOKEN cannot establish the account's remaining allowance.
Keep the account's zero-paid-overrun controls. Public compute exemption does not
remove artifact-storage limits. Dispatcher overhead belongs in contingency;
reserve up to 100 private allowance units for its bounded weekly checks until
actual setup/API timings are measured, rather than assuming they are free.

For private SSR, all scheduled allocation is paused by default. Do not set
CI_PRIVATE_SCHEDULES_ENABLED=true before November 1, 2026, and only after checking
the reset allowance and zero-paid-overrun setting. The source date gate provides
an additional pause, but checking that gate still requires an allocated runner
if the repository variable is enabled prematurely. No activation is implied by
committing these files. Existing PR jobs are not disabled by this variable.

Verification: the adjacent node:test suite exercises monthly cadence, relevant
activity, docs-only changes, 14/28-day rules, month boundaries, DST, third
fortnights, incomplete successes, skipped jobs, duplicate suppression and API
failures. Run it with the repository's normal Node command (SSR uses pnpm exec).

## Source identity

The dispatcher runs from the default branch but validates `main`.
It resolves that trusted configuration to one immutable commit, records the
source ref/SHA in the decision summary and job outputs, and passes that same SHA
to every source checkout. Source refs are not workflow_dispatch inputs.
Schedule-only reusable definitions live alongside the dispatcher; existing
push/PR/manual workflow definitions remain unchanged except retirement of old
recurring cron entries. Scheduled checks do not validate an unmerged PR head.

## Blender runtime provisioning

PR/push and scheduled full jobs share `.github/actions/setup-blender` on hosted
Ubuntu 24.04. The existing 5.0 and 4.2 targets use exact official Linux x64
releases 5.0.0 and 4.2.0. `.github/ci/setup-blender.sh` pins SHA-256 values from
the release manifests, cross-checked byte-for-byte at Blender-listed RWTH Aachen
and NLUUG mirrors. It tries the official download host, then the listed Aachen
mirror for transfer failures. A checksum mismatch fails before extraction or
execution. Each job uses its own runner temporary directory and the archive's
bundled Python; no developer-machine runtime changes are needed.

The shared action installs the repository requirements into bundled Python.
All test invocations use factory startup, a virtual X display, background mode
and an explicit Python failure exit code. Full jobs remain full; quick mode
remains quick. Job names and scheduled immutable source checkout checks remain
unchanged. Each job is bounded to 45 minutes. Original missing Docker tags caused
run 37845681445 to fail before checkout, so that run contains no test evidence.

Release manifests:
- <https://download.blender.org/release/Blender5.0/blender-5.0.0.sha256>
- <https://download.blender.org/release/Blender4.2/blender-4.2.0.sha256>
- Listed mirrors: <https://mirror.blender.org/source/blender-3.6.14.tar.xz?mirrorlist=>
