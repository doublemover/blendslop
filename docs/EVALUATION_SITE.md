# Evaluation gallery generation and publishing

The static [GitHub Pages demo](https://doublemover.github.io/blendslop/) publishes only the curated `site/` artifact. It has no external fonts, analytics, account connections, paid services or runtime packages.

## Current and historical results

The main `site/index.html` presents twelve actual retained reconstructions from `site/latest.json`. Each result identifies its exact candidate body, five original silhouette measurements, solid-boundary/topology checks, semantic editability and source-conditioned surface engineering separately. The calibrated smooth vase has an existing authored acceptance contract. Eleven other families have no authored artist acceptance limits; their engineering results do not fabricate those limits or an aggregate pass.

The current view uses existing measured packets and retained pixel pairs. It performs no Blender renders, fitting, resampling or qualification jobs. `latest-assets.json` ties each curated image to the depicted reference or candidate mesh and native pass. Original PNG copies remain under documentation; only text/Exif metadata is removed from public copies, preserving the exact encoded pixel payload. When a current pair is unavailable, the UI says so rather than displaying a different candidate.

The older two-case display remains at `site/historical.html`, bound to its original `data.json`, `diagnostics.json` and 2026-10-08 source snapshot. It retains the old vase's oblique boundary failure and the original triangle's pending surface verdict. The continuation page retains its earlier improvement comparisons and presentation renders. Historical observations, acceptance limits and failed trial updates remain unchanged.

## Local build and checks

Use the existing Node.js installation; no npm install is needed:

```bash
node scripts/build-evaluation-site.mjs --out temp/pages-dist
node scripts/check-evaluation-browser.mjs
```

Choose a fresh build directory. The browser check accepts `--dir <built-directory>` and `--out <evidence-directory>` for an isolated local preview. `CHROME_BINARY` may name an existing Chrome-compatible browser; the hosted runner uses its existing headless Chrome. No browser is downloaded.

The builder checks the public file allowlist, the existing 15-MiB bound, symlink exclusion, private-path/credential patterns, PNG metadata, exact asset hashes and local links. Current results are checked against normalized retained JSON evidence digests and exact JSON-pointer assertions, unchanged silhouette limits and body/pass image identities. Non-vase artist gaps cannot become aggregate passes. Historical diagnostic overlays continue to reproduce the original counts and metrics.

`build.json` records both the site source commit and the historical/current evidence source commits. The bounded browser check exercises the main current family selector, independent verdicts, current pixel pairs and unavailable passes, plus the historical exact failure explanations and difference overlays. Its screenshots and compact result are review artifacts, not new geometry measurements.

## Manual workflow

[Evaluation Pages](../.github/workflows/evaluation-pages.yml) has only a `workflow_dispatch` trigger. Pushes and ordinary pull-request updates do not regenerate or deploy the demo. No other repository workflow invokes its builder or calls this workflow.

In GitHub Actions, select **Evaluation Pages**, choose the ref and run the workflow. The **publish** option defaults to **false**. A build-only run validates the current demo and uploads the generated static site and browser evidence as seven-day review artifacts. It does not publish Pages.

To publish an authorized result, choose **main** and explicitly enable **publish**. Publication from another ref is refused. Pages artifact upload and deployment are both gated by that option and the main ref, through the existing SHA-pinned official actions. Default permissions remain `contents: read`; `pages: write` and `id-token: write` belong only to the deployment job and `github-pages` environment.

The workflow rebuilds the curated static demo from the chosen checked-out source. It does not run a reconstruction or render campaign. Dispatching a build, merging a PR and publishing Pages are separate actions; changing this configuration does not perform them. Verify the deployment run, published URL and live `build.json` before claiming a release is live.
