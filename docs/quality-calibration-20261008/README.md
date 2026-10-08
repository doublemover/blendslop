# Calibrated vase fitting — 2026-10-08

This local batch follows main `69eb231` on `local/quality-continuation-20261008`. The README/Pages release was integrated separately as PR #5 at `ed905f6`. This quality change has not been published. [Independent measured evidence](evidence.json) preserves the remaining failures.

The retained black/white PNGs use AgX. Their 128/255 hard-mask cutoff selects about 78.3% foreground coverage, causing systematic inset before geometry interpolation. Explicit linear foreground coverage now supplies half-coverage edge estimates for calibrated extents and profile fitting. Segmentation probabilities are not interpreted as physical coverage. Hard masks and acceptance thresholds retain their existing contracts.

`build_target_from_images(..., coverage_masks={view: array})` accepts finite observed values in `[0, 1]` at original image dimensions. Unknown values are omitted/zeroed in artifacts and hash identity; changing an observed float64 value invalidates identity. Coverage NPY files remain separate from uncertainty probabilities. Coverage accompanying hard-mask constraint edits is refused because those edits invalidate the original measurement. Partial views retain search-domain semantics; complete views can still supply calibrated coverage extents.

Partially covered terminal rows mix vertical cap coverage into horizontal edges. Their filtered width remains recorded, and a labeled flat-cap completion uses reliable interior sections. This is a proposal prior, not an exact hidden contour. Known empty interior rows remain zero; censored edges remain unavailable.

The replay helper now sets orthographic projection explicitly. Previously, reusing the backend's perspective camera ignored the saved orthographic scale and rendered at a different zoom. The first failed comparison remains retained. Frozen reference pixels, camera matrices, clipping and gates are unchanged.

| Independent observation | Before | After | Frozen limit |
| --- | ---: | ---: | ---: |
| Mean symmetric surface distance, world units | 0.00286860 | 0.000377357 | ≤ 0.003 |
| Oriented geometric normal p95 | 2.32051° | 1.10240° | ≤ 2.5° |
| Held-out `oblique_145_40` boundary IoU | 0.789001 | 0.979543 | ≥ 0.80 |

All five silhouette views pass area ≥ 0.70, boundary ≥ 0.80 and signed-distance loss ≤ 0.05. Surface and actual source-section radius edit/exact regeneration pass separately. Both frozen obliques touch the image edge, so these remain visible-frame verdicts. Inspected output pixels show the improved outline and smooth wall/flat cap.

The first new output passed native boundary qualification. The corrected-camera run has exactly identical decoded vertices/connectivity, geometry hash and toolchain identity, but its helper hit the unchanged 15-second allowance. Both receipts remain. **Aggregate acceptance is false**; the later timeout is not replaced with a pass. No extra qualification retry or timeout increase was performed.

Validation used existing Blender 5.2.2 LTS/Python 3.13.13 and the retained Open3D interpreter. All 43 affected pure checks passed; after mixed-visibility changes, all 26 affected coverage/identity/builder checks passed. Two native camera regressions passed without rendering. A zero-render transfer reproduction matches every encoded sample and restores render settings/image/object counts. The final one-case run took 28.65 seconds, peaked at 743944192 tree RSS bytes and stayed within two threads, 300 seconds and 8 GiB. No full suite or prior seven-control campaign was repeated.

To reproduce the declared grayscale transfer from a fresh checkout using the existing runtime:

```powershell
& $blender --background --factory-startup --threads 2 --python-exit-code 1 --python scripts/measure_grayscale_transfer.py -- --output temp/tasks/declared-transfer
```

This saves 4097 linear samples as a 16-bit grayscale PNG and records the actual view transform/look/exposure/gamma. It performs no 3D render. Supply that matching LUT to `run_profile_surface_repair.py --coverage-transfer .../transfer-lut.png --skip-historical`, together with the retained reference directory, existing qualification interpreter, historical-input path and a fresh output directory. Transfers must match the rendered source's black/white display settings; arbitrary photographs require an independently established coverage model.

Remaining: native helper deadline reliability; ten actual family rows; independent non-vase reference noise/surface limits, including the triangular dot; authored smooth intent for historical scalloped inputs; cold DVX/render/variant ownership and promotion/crash/locked-file lifecycle integration. No deletion executor, historical cleanup, ACL change or quality PR/merge/publication occurred. The public gallery retains its earlier dated evidence snapshot.
