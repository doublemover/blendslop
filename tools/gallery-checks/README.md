# Gallery contracts

Install the exact tools with `npm ci --prefix tools/gallery-checks --ignore-scripts`.
Build with `node scripts/build-evaluation-site.mjs --out temp/pages-dist`, then run:

```
node tools/gallery-checks/check-outline.mjs
node --experimental-vm-modules tools/gallery-checks/check-dom.mjs temp/pages-dist
node --experimental-vm-modules tools/gallery-checks/check-current-dom.mjs temp/pages-dist
```

These exercise the actual page modules, retained images and native Canvas2D output in jsdom. They do not replace the sandboxed real-browser layout/screenshot checker (`node scripts/check-evaluation-browser.mjs`) or native Blender qualification. The lockfile pins the tooling; installed packages are ignored and not published as gallery assets.
