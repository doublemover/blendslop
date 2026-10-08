# Cloud feature-branch validation and provenance

The source-restoration commit is parented to the actual published
`endblay_opslay` revision fbde785b4f112e3f49151e924c17279ef5319c73. It restores the
verified parked ae377fe source snapshot and corrective overlay; it does not
recreate unavailable original unpublished Git ancestry. All 510 previously
published paths remain. Two unpublished handoff documents containing private
Library identifiers are omitted; their originals remain in recovery. Detailed
file hashes and original revision boundaries are recorded in
`SOURCE_RECOVERY_PROVENANCE.json`.

Seven coherent implementation batches were transplanted with byte-identical
source to the separate `cloud/quality-consistency-20261006` branch. The
consolidated receipt records 157 passing CPU contracts across 18 focused suites,
573 Python parse checks, exact source comparison and a passed private-reference
scan. These include the original exact triangle-contact and folded-proposal
regressions. No source file was changed during validation.

The separately pinned Linux CPython 3.13.5 / Torch 2.14.1+cpu / DVX 0.1.1 runtime
passed seven bounded synthetic numerical stages, including actual forward/backward,
three coordinate pullbacks, one-update filtered modes, 16-to-32 observed-ray staging
and two jobs sharing one owned helper. Actual scikit-image 0.26.0 occupancy/SDF
extraction winding also passed tiny fixtures. These establish runtime contracts,
not a reconstruction or performance campaign. Dependency trees are not published.

Blender compilation/render/export and integrated final native qualification remain
unrun here under repository instructions. No new reconstruction quality or speed
gain is claimed. Original 06a7f02/8e7c15d quality measurements remain historical.
Publication does not imply acceptance, merging or deployment; subsequent CI results
must be evaluated independently for this exact feature revision.
