# Cloud consistency checkpoint

Source provenance: verified complete included source/config/docs from 151408f
with all 16 changed source files from ae377fe526d93cc9cfa7fb4d7d4ab0470748ac46.
The restored cloud source starts at ae377fe on endblay_opslay. The cloud Git root
is a source restoration commit, not a fabricated historical commit. 31 omitted
historical output tables stay in the original evidence packets. Original bundles,
meshes, renders, classification and quality scores are unchanged.

## Implemented contracts

- C1: one output-mesh selector for numeric evaluation, camera framing,
  validation rendering and export QA. Artist exclusions apply to descendants;
  legitimate multipart outputs remain included. Render sessions restore viewport
  and render visibility. Actual Blender render/export regression is pending.
- C2: centered capped-frustum distance uses local z and half-height consistently.
  Scalar evaluation delegates to the batch implementation. Independent cylinder
  cap/distance controls and translated/tilted unequal-radius zero sets pass.
- C3: local matrix-preserving pose edits, independent translation/rotation moves,
  cyclic control enumeration across nodes, local-axis structural splitting and
  full reflected proper frames for symmetric primitives. Search iteration cursor
  advances instead of restarting the same prefix. Finite search allowances can
  still end before every control is visited; no coverage guarantee is claimed
  under insufficient work.
- C4: optional output-resolution projection uses pixel-center resize mapping.
- C5: lower pole fan winding fixed at the analytic generator. All three analytic
  families pass directed-edge and outward-volume/pole-normal checks.
- C6: touching AABBs enter actual Boolean grouping with binary64 roundoff bounds.
  Grouping does not weld or inflate geometry. Actual face/edge/point-touch Boolean
  outcomes require Blender validation and remain pending.
- C7: known foreground extents and exact/censored row edges are distinct. Unknown
  pixel contents cannot change profile measurements. Calibrated loft completion
  is an explicitly labeled interpolation hypothesis, or unavailable when no fully
  measured foreground section exists. Partial uncalibrated inputs use the same
  evidence-aware path. Censored ResFit rows cannot seed exact profile knots.
- C8: compiler capability validation fails before any scene mutation for capsule,
  unsupported operations and unresolved child semantics. General authoring schema
  is retained separately; unsupported labels no longer produce silent Empty nodes.
- C9: actual final retained signed CSG geometry receives a content-scoped receipt.
  Operand receipts do not certify output. Missing qualification tools and rejected
  output remain unchecked candidates. Surface components are not relabeled as
  material bodies; cavity/body semantics remain an explicit qualification limit.
- C10: admission uses fixed 0.5 hard masks. Target-area adaptive hardening stays
  separately labeled diagnostics. Export proxy scaling was inspected: the existing
  adapter already reprojects the actual exported mesh for final metrics, so that
  correct final-score behavior is preserved rather than duplicated.
- C11: all profile knots reach one adjacent-interval partition with shared
  endpoints and no dropped tail. Heights/axes come from endpoint displacement.
  Single-row evidence no longer invents height from diameter. Paired front/side
  rows contribute both centers/radii; strongly elliptical sections decline a
  circular profile seed and leave the existing other-family path available.

## Validation and boundaries

The focused receipts include new deterministic regression fixtures and affected
existing tests, with exact-contact and folded-part guard suites preserved. These
are CPU/source correctness checks, not a reconstruction or performance campaign.
Final per-suite counts and status live in the checkpoint validation receipts.

The repository AGENTS.md prohibits agents running Blender here. Therefore no
Blender invocation, new campaign, GPU rental, new dataset, publication, merge or
hardware check was performed. Actual scene visibility, export roundtrip, native
Boolean outcomes and helper qualification remain pending. Quality and speed gains
are unmeasured. Historical metrics belong to 06a7f02 / 8e7c15d only.

Next coherent work: oriented whole-object support fitting, scale-consistent
objectives, changed-part control priority and complementary candidate routing.
