# Observed arch and torus source controls

Both saved-family source edits passed in Blender 5.2.2. The two-case session exited successfully in 4.116 seconds, with peak process RSS 217,817,088 bytes, under the approved 60-second/8-GiB/two-thread bounds. It performed zero fits, renders, raw surface comparisons or qualification helpers. The native receipt is `temp/tasks/quality-continuation-20261008/structured-source-edits-01/owned-7_k13149/results.json`; [evidence.json](evidence.json) retains its digest, physical measurements, guards, source hashes and process bounds.

The standalone producer `scripts/run_structured_family_source_edits.py` uses saved observed recipes from family-inspection04/owned-t44nr_f0. It freezes the recipes and NPZ/OBJ archives before compilation. The original seven-control producer and all prior receipts remain unchanged. This establishes recipe parameter regeneration on the same owned source/root pointers; it does not claim every native source control is live-editable.

| Saved family | Actual edit response | Fixed controls | Exact restoration |
| --- | --- | --- | --- |
| Concave arch | Cavity opening 1.0032187700 → 1.1035406590 world units; ratio 1.1000000119 | All three outer extents and inner roof unchanged | Saved indexed hash `a1e2a84a728d7d913f9bf454ac6cbd731e0ddb8dd837271d02bc47efb6308177` |
| Torus | Tube radius 0.2200844089 → 0.2310886593; ratio 1.0500001362; tube height grows 5%; outer ring expands and hole narrows | Major radius changes only 5.23e-9 world units from native arithmetic | Saved indexed hash `7a68da759b8536b864df8e00ca6fe3e322391ea05a6c98a82d2a4132fd89f8be` |

The arch is the observed eight-point polygon extrusion. Multiplying only the X coordinates of inner-wall vertices 4–7 by 1.10 keeps the four outer corners, inner roof, other outline coordinates and extrusion thickness fixed. Native observations measure the evaluated cavity-wall spacing, roof and all three physical extents in the source rotation frame.

The torus multiplies the saved minor/tube radius by 1.05 while preserving the major radius. Native observations measure inner and outer radial extrema, derive the actual major/tube radii, and require the axial tube height to grow 5%. A changed major radius fails the semantic response even when the tube expands.

Each transaction regenerates a temporary source mesh from the edited recipe and applies it to the existing owned source. The original mesh pointer, native pose controls and original recipe-property presence are then restored. Both receipts require the same root/source pointers, a changed edited geometry hash, and an exactly equal restored indexed binary64 hash. The physical-response tolerance is 1e-5; it never enters geometry identity. Temporary objects and meshes are restricted to the job's compiled sources.

Three pure regressions passed: inner-only arch edits preserve outer/roof controls; outer scaling or roof motion cannot pass the opening response; tube edits preserve major radius and a changed major radius cannot pass. The read-only ownership audit reports `dry_run_ready`, with zero unknown files or blockers and 230,817 retained bytes. No reclamation was performed.

These checks add actual semantic-edit evidence for the observed arch and torus. Independent family surface acceptance remains unqualified because no independently justified surface tolerance is supplied by this stage; the receipt keeps `aggregate_accepted` false. Existing mask, shaded and boundary receipts are preserved without modification.
