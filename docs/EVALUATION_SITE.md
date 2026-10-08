# Evaluation gallery publishing

The static [GitHub Pages gallery](https://doublemover.github.io/blendslop/) publishes only `site/`. It has no external fonts, analytics, account connections, paid services or runtime dependencies. The README points to focused setup, reconstruction/API, evaluation and development guides.

## Evidence scope

The 2026-10-08 snapshot reuses 79 project-authored synthetic PNGs and a curated numeric summary. Original evaluated OBJ identities and mask hashes were verified against retained receipts before asset selection. PNG text/EXIF chunks were removed; encoded image pixels were preserved. `site/assets.json` records original and published hashes. Source meshes, machine paths, usernames, caches, personal images, raw logs and external sample datasets are excluded from the public site artifact. No render or benchmark campaign was run to populate the gallery.

Twelve authored references are prepared; two actual reconstructions have retained evidence. The vase fails one required oblique boundary gate. Triangle surface/noise qualification remains unavailable, and its actual neutral output was not retained. The gallery never substitutes an authored reference for a reconstructed output or copies vase surface limits to another family. Frozen oblique framing limitations are shown explicitly. Correctness CI is labeled separately from geometry acceptance.

## Build and checks

Use an existing Node.js installation; no npm install is needed:

```bash
node scripts/build-evaluation-site.mjs --out temp/pages-dist
```

Choose a fresh output directory. The builder checks file allowlists, byte bounds, symlink exclusion, sensitive-path/credential patterns, PNG metadata, asset identities, local links, case counts and metric/verdict consistency. Build provenance is saved as `build.json` with the exact site source commit, evidence source commit and evidence date.

On the hosted Ubuntu runner, `scripts/check-evaluation-browser.mjs` uses its existing sandboxed headless Chrome, a localhost-only server and a 60-second cap. It checks desktop/mobile screenshots, view selection, vase failure, triangle unqualified surface, unavailable neutral output, actual/reference filters and mobile overflow. It installs no local packages. The four public-safe screenshots and a compact result are retained for seven days as a CI artifact.

## Deployment

[Evaluation Pages workflow](../.github/workflows/evaluation-pages.yml) builds and checks pull requests without publishing them. Main pushes affecting the gallery, builder or workflow deploy through official SHA-pinned Pages actions. Manual dispatch also deploys only from `main`. Default permissions are `contents: read`; `pages: write` and `id-token: write` belong only to the deployment job and `github-pages` environment. The repository remains public, with no custom domain, protection bypass or credentials added.

Pages must use the repository's GitHub Actions build mode. Enable that destination only as part of an authorized publication. Verify the deployment run, published URL and live `build.json` before claiming a release is live. Existing historical reports remain unchanged; update only curated observations whose source identities and independent verdicts are known.
