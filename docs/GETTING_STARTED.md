# Getting started

Run the commands in this guide from the repository root unless noted. Only Blender 5.2.2 LTS is supported.

## What you can do

- Build a blockout from high-contrast orthogonal reference images, using a mesh-oriented method or an editable primitive/shape-program representation.
- Compare candidates with rendered silhouette IoU, boundary and distance metrics; use synthetic ground truth and held-out views where available.
- Run bounded parameter sweeps, rank results, inspect overlays and failure reports, and preserve the configuration and provenance for each candidate.
- Keep editable source parts separate from evaluated render/export geometry. A multipart assembly and a qualified single solid have different acceptance requirements.
- Explore opt-in research paths for calibrated/partial evidence, geometry proposals, differentiable refinement, and implicit residuals without treating proxy scores as native-render validation.

There is no trained image-to-3D model or packaged Blender add-on to install. The main entry points are Python scripts and the `BlockingWorkflow` API.

## Quick start

### 1. Get the project

```bash
git clone --branch main https://github.com/doublemover/blendslop.git
cd blendslop
```

The commands below run from the repository root. The reconstruction framework and completed implementation are integrated in `main`.

### 2. Configure Blender's Python

Use the Python interpreter bundled with your Blender installation. NumPy, OpenCV, Pillow, SciPy, and other compiled packages must match that interpreter's ABI. Do not attach an unrelated project's virtual environment to Blender.

Only the latest official stable release is supported: **Blender 5.2.2 LTS**, verified on 2026-10-08 against the [official download](https://www.blender.org/download/) and [5.2 release page](https://www.blender.org/releases/5-2/). CI pins its official archive and runs full and quick correctness checks; older releases and previews are unsupported. Previous cross-version comparisons remain [historical evidence](branch-audit-20261006/blender52-verification.md), not current support targets.

**Windows / PowerShell** — adjust both paths to the same installation:

```powershell
$blender = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
$py = 'C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe'
& $py -m pip install --user -r blender_blocking/requirements.txt
& $blender --background --python-exit-code 1 --python blender_blocking/verify_setup.py
```

**macOS / bash or zsh** — example paths for Blender 5.2:

```bash
BLENDER='/Applications/Blender.app/Contents/MacOS/Blender'
PY='/Applications/Blender.app/Contents/Resources/5.2/python/bin/python3.13'
"$PY" -m pip install --user -r blender_blocking/requirements.txt
"$BLENDER" --background --python-exit-code 1 --python blender_blocking/verify_setup.py
```

**Linux** — use the same shell commands after setting `BLENDER` and `PY` to the executable and bundled `python/bin/python3.x` in your Blender installation. Paths and Python versions vary by distribution and build.

The verifier exposes the matching Python user site and an existing `~/blender_python_packages` directory. Keep that shared directory free of packages from a different Python ABI. For path discovery and import troubleshooting, see [Blender setup](../blender_blocking/BLENDER_SETUP.md); use the exact supported installation.

The [requirements file](../blender_blocking/requirements.txt) includes the scientific/image stack, `tqdm`, `scikit-image`, and Open3D. Marching-cubes extraction uses scikit-image; Open3D is used by optional Poisson/native-qualification paths. If an optional wheel or runtime is unavailable, inspect the reported skip and avoid claiming that capability is validated. Shapely 2.x is required for continuous contours and canonical pixel coverage. Torch/LPIPS, DVX, OpenVDB, and NVIDIA-only nvdiffrast serve additional research paths and are not all installed by this command. Do not install them just to try the basic loft example.

### 3. Generate the tiny example inputs

Sample images are generated locally and ignored by Git. Run the generator from `blender_blocking/` so both the validator and refinement lab find them.

```powershell
# PowerShell
Push-Location blender_blocking
& $py create_test_images.py
Pop-Location
```

```bash
# bash / zsh
(cd blender_blocking && "$PY" create_test_images.py)
```

This writes nine PNGs: front, side, and top silhouettes for a vase, bottle, and cube.

### 4. Reconstruct and validate the vase

**Start in a new Blender process or a disposable scene. Reconstruction paths can clear existing scene objects. Save your work first.**

```powershell
& $blender --background --python-exit-code 1 --python blender_blocking/test_e2e_validation.py -- `
  --reconstruction-mode profile_loft --num-slices 24 `
  --front blender_blocking/test_images/vase_front.png `
  --side blender_blocking/test_images/vase_side.png `
  --top blender_blocking/test_images/vase_top.png `
  --validation-mode render-iou `
  --render-output-dir temp/quickstart/renders `
  --artifact-output-root temp/quickstart/artifacts `
  --result-json temp/quickstart/result.json --no-progress
```

```bash
"$BLENDER" --background --python-exit-code 1 --python blender_blocking/test_e2e_validation.py -- \
  --reconstruction-mode profile_loft --num-slices 24 \
  --front blender_blocking/test_images/vase_front.png \
  --side blender_blocking/test_images/vase_side.png \
  --top blender_blocking/test_images/vase_top.png \
  --validation-mode render-iou \
  --render-output-dir temp/quickstart/renders \
  --artifact-output-root temp/quickstart/artifacts \
  --result-json temp/quickstart/result.json --no-progress
```

Inspect the render PNGs and `temp/quickstart/result.json`, including each required view, warnings, and failure reasons. This command exercises reconstruction and image validation; it does not promise a passing score or save a `.blend` scene. Use the [Python example](RECONSTRUCTION.md#create-and-save-a-blockout-in-blender) to save one. Blender arguments come before `--`; script arguments come after it.

## Download the small comparison pack

From the repository root in PowerShell 7.2 or later:

```powershell
.\scripts\Get-BlendslopSamples.ps1
```

The script saves to `temp/sample-pack` in this checkout, resolved from the script's location even when called from another working directory. The fixed pack contains 16 source meshes, their 16 matching SuperFit fitted outputs, and 12 PrimitiveAnything point clouds, plus configurations and notices. A fresh download is about **36 MB**, using about **44 MB** on disk; verified existing copies inside the destination are reused.

The downloader uses PowerShell/.NET only, fetches selected ZIP entries rather than whole collections, and checks sizes, CRCs and available SHA256 pins. Use `-Folder` for another absolute destination or `-WhatIf` for a no-download preview. It doesn't install tools or execute downloaded models/pickle files. Source and license notices remain with the data. Point clouds are not solid-mesh ground truth; fitted outputs are not additional test subjects. DTU is excluded from this small pack.

## Prepare your own references

- Use orthogonal front, side, and top views of the same object with consistent scale, orientation, and framing. Perspective photographs are not interchangeable with these views.
- Begin with dark silhouettes on a light background, or meaningful foreground alpha. PNG and JPG are accepted; alpha requires an appropriate image format.
- Supply all three `--front`, `--side`, and `--top` paths together in the E2E CLI. The lower-level API can accept fewer views, but support and fallback behavior depend on the backend.
- Inspect mask extraction before tuning geometry. Hole filling, largest-component filtering, thresholds, and morphology can remove real features.
- Treat cropped or unobserved regions as unknown evidence when using the calibrated/validity-mask paths. They should not silently become known-empty constraints.

Replace the three vase paths in the quick-start command with your references. Use a new output directory for each comparison so earlier evidence remains available. Three silhouettes cannot uniquely recover hidden concavities, cavities, or surfaces that do not affect the outlines.
