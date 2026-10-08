# Blender correctness CI

Only the latest official stable release is supported: **Blender 5.2.2 LTS**,
verified 2026-10-08 against the [official download](https://www.blender.org/download/)
and [release page](https://www.blender.org/releases/5-2/). Older versions and
previews are unsupported. Historical version comparison packets remain evidence
of their original sources; they are not active CI/support requirements.

`.github/workflows/blender-tests.yml` runs one full stable job and one quick job
for relevant pushes and PRs. Both use the shared setup action on hosted Ubuntu
24.04. It downloads the official Linux x64 archive, checks the pinned SHA-256
before extraction, verifies the exact release and release cycle, and installs
repository requirements into bundled Python. Transfer fallback is a Blender-listed
mirror; mismatched bytes fail without execution. No Docker image tags are used.

The custom runner retains pure geometry/config/image tests, native queries and
solid checks, editable geometry, booleans, procedural generation, rendered
silhouettes, end-to-end validation and dependency diagnostics. Quick mode skips
only the two existing slow procedural/end-to-end suites. Current native API
checks no longer skip merely to accommodate an older Blender version.

Each job is capped at 45 minutes and runs under a virtual display with background
mode, factory startup and an explicit Python failure exit code. Failure logs are
retained for seven days. These are ordinary CPU correctness jobs, not performance
or visual-quality release campaigns.

From `blender_blocking/`, use the supported installation:

```bash
blender --background --factory-startup --python-exit-code 1 --python test_runner.py
blender --background --factory-startup --python-exit-code 1 --python test_runner.py -- --quick
```

The adaptive maintenance workflow calls one stable full job. Cadence and trusted
source selection remain in `.github/ci/adaptive-ci.json`; the dispatcher resolves
main to one immutable SHA and the called job verifies its checkout matches.
See [maintenance policy](../.github/ci/README.md). Scheduled checks do not validate
an unmerged PR head.

When a new official stable release becomes current, verify the official download
and release manifest, update the exact runtime pin/checksum and runtime guard,
and validate the current release. Do not follow alpha, beta, release-candidate
or pull-request builds. Updating repository support policy never authorizes
replacing an owner's installed Blender or its packages.

Before integration, require full/quick/policy checks on the final PR head.
Keep silhouette, surface, topology and editability verdicts independent:
a green correctness suite does not override a failed or missing quality gate.
