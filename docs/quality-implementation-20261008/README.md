# Editable surface implementation — 2026-10-08

Base: published PR2 candidate `c3005d493011078adfbf9fbe63a512cfc1672c8b`, verified against current origin before checkout. Work is isolated on `local/quality-implementation-20261008`; the clean original `endblay_opslay` at `88cb2c1` is preserved.

Completed batch 1:

- Promote the frozen rounded triangular lens into a production primitive and shape-program compiler/numeric-screening capability. Corner radius, thickness, pose, XY scale and asymmetric front/back dome depths serialize independently. The field has an exact signed zero set; it is explicitly not Euclidean distance. The default preserves the retained authored reference exactly.
- Save source sections and options on loft objects. `rebuild_loft_mesh` regenerates their geometry after artist edits, preserving the object, transforms, parent, materials, tags and modifier stack. Invalid input fails before geometry replacement.
- Add a circular analytic-vase radius/planar-shoulder diagnostic. This diagnoses geometric bands without confusing shading normals, a good mask, or an intended sharp corner with a surface pass.

Validation: 22 focused pure geometry/schema/loft checks passed. Two new native artist-edit checks passed in existing Blender 5.2.2 LTS, Python 3.13.13; no dependency installation or runtime change. Native log: `temp/tasks/native-edit-check/log.txt`. The new default triangular mesh coordinates and indexed triangles exactly equal the retained evaluated NPZ; content hash `2681272b491af3370176a3751e408ef36496ba592d230bce7c733b8c41cadc64`. Patch whitespace check passed.

The previously passed full Blender suite and seven-control/105-render batch are retained evidence, not rerun. Inspected retained smooth-vase oblique pixels show a smooth wall and flat cap. That analytic control does not qualify the historical legacy vase reconstruction.

Remaining: executable twelve-family reference/negative coverage, independent family surface qualification, historical-vase input repair and acceptance, and lifecycle ownership design. Required failures and unavailable evidence continue to block broad acceptance. Native edits are checked only in installed Blender 5.2.2, not cross-version certified. No PR update, publication, historical cleanup, ACL change or merge.
