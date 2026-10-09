# Evaluation gallery publishing

The static [GitHub Pages gallery](https://doublemover.github.io/blendslop/) publishes only `site/`. It has no external fonts, analytics, account connections, paid services or runtime dependencies. The README points to focused setup, reconstruction/API, evaluation and development guides.

## Evidence scope

The 2026-10-08 snapshot contains 82 project-authored synthetic PNGs: 79 retained assets and three new shaded triangle inspections and a curated numeric summary. Original evaluated OBJ identities and mask hashes were verified against retained receipts before asset selection. PNG text/EXIF chunks were removed; encoded image pixels were preserved. `site/assets.json` records original and published hashes. Source meshes, machine paths, usernames, caches, personal images, raw logs and external sample datasets are excluded from the public site artifact. The three new 512×512 Workbench passes replay the retained triangle program and frozen cameras; evaluated world-coordinate geometry hashes match exactly before/after rendering. They add no reconstruction, metric or benchmark rerun.

Twelve authored references are prepared; two actual reconstructions have retained evidence. The vase fails one required oblique boundary gate. Triangle surface/noise qualification remains unavailable. Its additional shaded inspections do not change that verdict. The gallery never substitutes an authored reference for a reconstructed output or copies vase surface limits to another family. Frozen oblique framing limitations are shown explicitly. Correctness CI is labeled separately from geometry acceptance.

## Build and checks

Use an existing Node.js installation; no npm install is needed:

```bash
node scripts/build-evaluation-site.mjs --out temp/pages-dist
```

Choose a fresh output directory. The builder checks file allowlists, byte bounds, symlink exclusion, sensitive-path/credential patterns, PNG metadata, asset identities, local links, case counts and metric/verdict consistency. `diagnostics.json` stores exact missing/extra foreground and radius-2 elliptical boundary-band row runs, tied to published mask hashes. Counts and overlap ratios reproduce all ten retained case/view metrics; overlays use those original runs without alignment or exaggerated displacement. Build provenance is saved as `build.json` with the exact site source commit, evidence source commit and evidence date.

On the hosted Ubuntu runner, `scripts/check-evaluation-browser.mjs` uses its existing sandboxed headless Chrome, a localhost-only server and a 60-second cap. It checks desktop screenshots, actual shaded defaults, exact vase failure explanations, scored boundary/filled difference overlays, triangle unrun qualification, unavailable camera passes and actual/reference filters. It installs no local packages. The four public-safe screenshots and a compact result are retained for seven days as a CI artifact.

## Deployment

[Evaluation Pages workflow](../.github/workflows/evaluation-pages.yml) builds and checks pull requests without publishing them. Main pushes affecting the gallery, builder or workflow deploy through official SHA-pinned Pages actions. Manual dispatch also deploys only from `main`. Default permissions are `contents: read`; `pages: write` and `id-token: write` belong only to the deployment job and `github-pages` environment. The repository remains public, with no custom domain, protection bypass or credentials added.

Pages must use the repository's GitHub Actions build mode. Enable that destination only as part of an authorized publication. Verify the deployment run, published URL and live `build.json` before claiming a release is live. Existing historical reports remain unchanged; update only curated observations whose source identities and independent verdicts are known.

## Presentation and failure explanation

Actual families open on shaded oblique objects. Silhouettes and exact difference overlays remain secondary controls, with a 4× nearest-pixel crop. Reference-only families are labeled not reconstructed. The vase explanation identifies the 145°/40° view, boundary IoU 0.789001 versus ≥0.800000, 99.08% filled overlap, 998 missing / zero extra foreground pixels, and 7819 / 9910 boundary-band overlap. Triangle surface values are raw measurements without an established family threshold; qualification is incomplete rather than failed. Completed solid-boundary checks are distinguished from unrun checks. All original acceptance values, source geometry and reference cameras stay fixed. The separately committed local calibration work is excluded from this publication.

## Quality continuation under review

The separate `site/continuation.html` page retains calibrated linear-alpha, rounded-box detail-crop and multipart endpoint results plus1536-pixel presentation renders. The original gallery/data/diagnostics stay bound to their historical source snapshot. Continuation silhouettes, raw surface observations, native boundary and semantic edits remain separate verdicts; unavailable family surface limits and old provenance gaps remain explicit. The builder validates both manifests under the existing15MiB artifact bound and preserves PR6 exact diagnostic/thumbnail checks. Native PNG originals remain in documentation; only reviewed text/Exif chunks are removed from site copies, with exact IDAT payload hashes retained.

This integration prepares a draft PR. The new continuation has not been deployed; merging or publishing it requires separate owner approval.
