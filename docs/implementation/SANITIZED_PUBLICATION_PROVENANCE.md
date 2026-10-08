# Sanitized isolated-branch publication

This complete source tree preserves cloud source checkpoint
`963c474e6df4c99890077201341a9eb1a4a53c50` (original tree
`34e3049f671560b9712454da4dd428adc78c1ac9`) with the four privacy/portability edits
listed below. Its publication commit has the genuine published
`endblay_opslay` ancestor `fbde785b4f112e3f49151e924c17279ef5319c73` as its only
parent. The intended destination is `cloud/quality-consistency-20261006`.
No main integration or PR is implied.

The 30 unpublished cloud commits and their original bytes remain in the private
recovery. They are consolidated for publication because the restored audit
contained machine-specific home paths in unpublished ancestors. Those ancestors
are not reachable from this publication commit. Original missing local history
is still recorded separately in `SOURCE_RECOVERY_PROVENANCE.json`; consolidation
does not reconstruct that unavailable ancestry.

## Data-preserving path changes

- `solid-case-classifications.csv`: all 247 data rows and 17 named columns are
  retained. Only the `mesh` values change to normalized repository-relative
  artifact identifiers. Every other field, measurement and digest is unchanged.
- `solid-followthrough.json`: the 247 corresponding mesh values and two additional
  artifact references use the same stable identifiers. All other JSON values
  remain unchanged. CSV/JSON joins and original path equality are preserved;
  the 248 distinct identifiers have no collisions.
- `open3d313-proposal.md`: staged installation examples are relative to the
  repository root. This historical proposal does not report a new installation.
- `scripts/diagnose_saved_solids.py`: the default root derives from the script
  location. A deliberately relocated copy can use `BLENDSLOP_REPO_ROOT` instead
  of a hardcoded personal directory.

Identifiers under `temp/` identify the historical repository-relative campaign
artifacts. The external preview is represented by
`audit-artifacts/branch-audit-20261006/chair-evaluated-union-preview.obj`.
These are audit identifiers, not a claim that the original mesh files are shipped
in this repository. Original private paths and files remain in the private
recovery; none of the saved meshes was modified by sanitization.

The historical input/classifier/script hashes and measurements remain tied to
that original execution. The portability edit does not retroactively declare
that the edited diagnostic script produced the historical receipt.

## Validation and acceptance limits

The sanitized tree is checked against every tracked source path, with all
published ancestor paths retained. CSV/JSON equivalence and joins are verified,
private-home paths are scanned in new reachable Git objects, and the complete
publication parent/tree is recorded before pushing. The PNG metadata contains
no private source path; image bytes remain unchanged.

The source checkpoint includes bounded residual checkpoint selection/replay,
source retention on optional failures, exact fold/contact eligibility, separated
assembly support and experimental taper/bend. Its existing scoped CPU/pinned
helper receipts remain source-identified; sanitization is not a new quality or
performance evaluation. Native renderer/filter, scene/export/CSG and real
reconstruction acceptance remain open.
