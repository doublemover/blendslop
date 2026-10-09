# Rounded-box source style verification

The corrected explicit weighted-normal replay restores the authored rounded-box bevel shading while preserving its original indexed geometry. Five corrected neutral frames strictly match the retained source camera/clip pairs. The first weighted-normal attempt remains preserved as a structural pass with unresolved visible banding.

## Exact evidence

`scripts/run_rounded_box_style_check.py` rebuilt the saved rounded-box recipe with explicit eight bevel segments, then with `weighted_normals=True`. Both evaluated arrays retain the original indexed hash `aa683c0899ba9b98c077a6164ef87fca69c66ec60fad2cba2aa5e62b8bf14753`. Custom-normal state changed from absent to present. All five actual camera-frame and clipping pairs strictly match the retained authored source neutrals. There were no fits, masks, normal passes, raw metrics or qualification children.

The native child joined successfully in 5.846 seconds with 452,407,296 peak working-set bytes, below the approved 60-second/8-GiB/two-thread bounds. `rounded-style-01/owned-wz8m1bxm/results.json` and all five previews remain intact. Unlike the two earlier source-control/reference-neutral runs, this final receipt was re-registered after publication: read-only ownership planning reports dry_run_ready with no unknown files or blockers. No reclamation or deletion ran.

## First attempt: visual mismatch

[Side comparison](side-comparison.png) and [oblique comparison](oblique_35_28-comparison.png) place the authored source, old candidate, and first styled candidate left to right. The source remains visibly smoother than both candidate versions. The modifier records identify an additional authored control: source WEIGHTED_NORMAL.keep_sharp is false, while the compiler uses true. Matching modifier type and custom-normal presence does not establish matching shading behavior.

This finding led to an explicit validated boolean `weighted_normals_keep_sharp` recipe parameter. Absence preserves the existing compiler default; the corrected rounded-box recipe declares false together with weighted_normals true. Other authored flat/sharp families retain their existing styles. The continuation below preserves this first attempt and the original mask/metric evidence, requires the same exact indexed geometry, and compares five neutral-only frames against the existing source frames.

[evidence.json](evidence.json) records exact receipt hashes, style state, process bounds and the read-only ownership audit. The source inventory remains incomplete (no new masks or normal passes), and independent family surface acceptance remains unqualified.

## Corrected keep-sharp continuation

The compiler now accepts strict boolean `weighted_normals_keep_sharp`; absence preserves the previous true default. A fresh rounded-box recipe explicitly declares weighted_normals true and weighted_normals_keep_sharp false. Pure tests verify explicit controls, invalid values before geometry creation, unchanged defaults, and that matching custom-normal existence alone does not hide a keep-sharp mismatch.

The failed-only continuation at `rounded-style-02/owned-n9i0pxq1` retained precisely five neutral frames in 5.478 seconds, peak 439,607,296 working-set bytes. Baseline/styled indexed hashes remain exactly `aa683c0899ba9b98c077a6164ef87fca69c66ec60fad2cba2aa5e62b8bf14753`. All five frame and clipping pairs match strictly. Actual weighted-normal controls, modifier sequence and custom-normal state now match the authored source. No fitting, masks, normal passes, raw comparisons or qualification helpers ran.

[Corrected side comparison](side-corrected-comparison.png) and [corrected oblique comparison](oblique_35_28-corrected-comparison.png) show source, first attempt, and corrected candidate left to right. Direct pixel inspection confirms the banding is removed. This fixes a display style defect; it does not create an independent surface tolerance or change existing metric admission. [corrected-evidence.json](corrected-evidence.json) records the distinct continuation and its dry_run_ready ownership audit without replacing the original attempt.
