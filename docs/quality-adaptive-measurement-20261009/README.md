# Controlled adaptive measurement

The selected rounded box improves untouched oblique coverage error by 23.886385% with one genuinely rendered 1024-square detail crop of the original 512-square top viewport. Both strict held-out silhouette gates pass and improve. No cutoff was relaxed. The recipe is frozen before raw reference proximity metrics. A separate follow-up passes all five silhouettes, exact native boundary qualification and a live corner-radius edit with exact indexed restoration. Independent family surface acceptance is still unavailable.

| Observation | Three-view512 baseline | Baseline plus1024 crop |
|---|---:|---:|
| oblique35 boundary IoU |0.968153561|0.974398249|
| oblique145 boundary IoU |0.970761613|0.977514793|
| mean held-out alpha L1 |0.000642503117|0.000489032347|
| raw symmetric mean surface distance, world |0.000253135374|0.000189522010|
| exact native boundary |not requested|qualified separately|
| specific radius edit/restore |not requested|passed separately|
| qualified family surface limits |unavailable|unavailable|

Measurements use the dedicated opaque-linear-alpha contract: transparent film, opaque emission override, recorded orthographic camera, EEVEE64 samples and 32-bit ZIP OpenEXR alpha. The half-coverage mask feeds the unchanged area0.7 / boundary0.8 / distance0.05 gates. Display RGB, AgX, color, texture and lighting are excluded. Source/candidate acquisition signatures must match exactly. Existing display-PNG receipts remain their original protocol.

Only front/side/top measurements are passed to fitting. The top residual weighted by diagonal source-edge gradients selects the160-pixel-cell region [0,37,160,197]. A new1024-square crop render samples that source geometry; it is not a zoom of old pixels. Oblique35/145 are excluded from fit and ROI selection. The measured23.9% improvement met the predeclared10% experimental win criterion; that criterion is not a family-acceptance threshold.

The corrected comparison took7.122 seconds and598,409,216 peak process-tree RSS bytes: eight new frames, five source measurements reused after owned-byte verification, two256-call /3-second fit allowances. Measured source-crop acquisition cost was0.179 seconds and additional fit0.075 seconds. There is no general16-times render-time assertion. The final saved-only follow-up took10.756 seconds /675,442,688 RSS: three missing axis measurements, one unchanged15-second helper and one live radius transaction, with no fitting or raw-surface rerun.

The first three-segment prototype remains retained and unaccepted: it improved alpha error by4.7% but failed both boundary gates. Root found that its adapter omitted the frozen family's eight-segment bevel declaration. The corrected recipe explicitly declares eight segments. Neither prototype receipt was rewritten and no failed result was promoted.

The selected exact geometry is f5dd0942272c6060247e530d9faf71f3ca311b496c0e10361677b5cfa2ca9f50. The source radius changes by1.10 on the same native mesh descendant, dimensions remain fixed and the original indexed arrays restore exactly. Geometric-normal P95 changed from0.00008995 to0.00012283 degrees; both are raw observations with no qualified rounded-box cutoff.

Existing CPU soft-silhouette fitting already uses bounded coarse-to-fine128-square optimization, normally70% of the global allowance coarse and optional remaining30% fine. Final original-resolution fit-camera admission is separate from independent held-out validation. Existing active-view planning suggests cameras but does not generate ground truth. For fixed external images, a crop can reframe existing samples; it cannot supply newly observed detail or a novel reference view. Controlled source geometry supports those acquisitions.

[evidence.json](evidence.json) binds every receipt/resource result. [Presentation assets](../quality-presentation-continuation-20261009/README.md) retain three1536-square color/lit frames as display-only output. The [remaining-work map](../quality-continuation-20261008/remaining-work.json) keeps surface, hidden-depth, provenance, lifecycle and publication gaps explicit.

## Quality within the remaining budget

Owner clarified that saved time can be reinvested in better geometry. One additional2048 detail crop was tested under the same120-second /8-GiB /two-thread checkpoint ceiling. It reused five verified source and five baseline measurements, made exactly three new acquisitions and performed one256-call /3-second fit plus one raw4096-sample comparison. No larger global-resolution campaign ran.

The additional crop lowers held-out alpha L1 a further11.219656% relative to1024 and32.426071% relative to the512 baseline. Oblique boundary IoU improves to0.975911530 /0.978685613; mean raw surface distance improves slightly from0.000189522010 to0.000189279811 world. Normal P95 improves relative to1024, from0.000122831 to0.000115541 degrees, but remains microscopically above the512 baseline0.000089951. These are independent raw observations, not a family-tolerance claim.

| Checkpoint | Held-out mean alpha L1 | Minimum boundary IoU | Raw mean distance, world | New crop time | Fit time |
|---|---:|---:|---:|---:|---:|
|512global baseline|0.000642503117|0.968153561|0.000253135374|0.000 s reused source|0.086914 s|
|1024detail|0.000489032347|0.974398249|0.000189522010|0.179371 s|0.075362 s|
|2048detail|0.000434164598|0.975911530|0.000189279811|1.652720 s|0.160214 s|

These phase times omit setup, held-out rendering, raw comparisons and native qualification; the complete resource receipts are retained. The successful2048 checkpoint took7.779 seconds /683,593,728 peak RSS. Its actual selected9f6c8d... saved-only follow-up took12.497 seconds /678,191,104 RSS, passing all five strict silhouettes, unchanged15-second exact native boundary and live radius edit/restore. A single final1536-square beauty render retained this exact geometry.

Two implementation mistakes remain explicit: a pre-render continuation used a nonexistent parser and exited before any acquisition; a follow-up wrapper initially selected the previous1024 geometry, so it produced a duplicate old-candidate receipt. Both histories are retained. The existing retained-program parser and structured command-vector update correct the final runs. No prior result is relabeled.

[quality-time-checkpoints.csv](quality-time-checkpoints.csv) provides the three discrete quality/cost checkpoints. Higher-than2048 refinement is not proposed: the useful next observation is the multipart complementary view. The final family surface limits remain unavailable.
