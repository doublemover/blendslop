# blendslop

Turn front, side and top silhouettes into editable Blender blockouts, then inspect what the reconstruction preserves.

Blendslop is a Python research workbench with profile lofts, visual hulls, primitive fitting and editable shape programs. It runs inside **Blender 5.2.2 LTS**, the supported latest stable release verified on 2026-10-08. Older versions and previews are unsupported.

**[Evaluation gallery](https://doublemover.github.io/blendslop/)** · [Setup](docs/GETTING_STARTED.md) · [Methods & API](docs/RECONSTRUCTION.md) · [Results & limits](docs/EVALUATION.md) · [Development](docs/DEVELOPMENT.md)

## Quick start

Use Blender's bundled Python for dependencies. The PowerShell example below runs from the repository root; [the setup guide](docs/GETTING_STARTED.md#2-configure-blenders-python) includes macOS/Linux paths and troubleshooting.

```powershell
$blender = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
$py = 'C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe'
& $py -m pip install --user -r blender_blocking/requirements.txt
Push-Location blender_blocking
& $py create_test_images.py
& $blender --background --factory-startup --python-exit-code 1 --python test_e2e_validation.py -- `
  --reconstruction-mode profile_loft --num-slices 24 `
  --front test_images/vase_front.png --side test_images/vase_side.png --top test_images/vase_top.png `
  --validation-mode render-iou --render-output-dir temp/quickstart/renders `
  --result-json temp/quickstart/result.json --no-progress
Pop-Location
```

Start in a fresh Blender process: reconstruction can clear scene objects. Inspect the generated images and result JSON. This sample exercises the workflow; it does not promise a passing result or save a `.blend` file. [Save a blockout through the API](docs/RECONSTRUCTION.md#create-and-save-a-blockout-in-blender).

## Core workflow

1. Prepare consistently framed orthogonal references; inspect their extracted masks.
2. Choose one explicit reconstruction method and preserve its configuration.
3. Compare source and output in fixed cameras, including held-out views where available.
4. Check silhouette, surface, actual solid boundary and editability independently before using the asset.

Three silhouettes cannot uniquely recover hidden geometry. A good outline does not establish a correct surface or a qualified solid. The current twelve-family suite has two actual reconstructions: the vase still fails one oblique boundary gate, and triangle surface qualification remains open. [See the evidence and missing verdicts](docs/EVALUATION.md).

No trained image-to-3D model or packaged add-on is required. Use the scripts or `BlockingWorkflow` API. Detailed [reference preparation and sample pack](docs/GETTING_STARTED.md), [configuration and refinement](docs/RECONSTRUCTION.md), [testing and architecture](docs/DEVELOPMENT.md), and [documentation index](docs/INDEX.md) live in focused guides.

## Credits and licensing

This fork builds on [zbuc/blender_experiments](https://github.com/zbuc/blender_experiments), with continued development in [doublemover/blendslop](https://github.com/doublemover/blendslop).

The original blocking-tool documentation credits Gas Town's multi-agent workflow: **chrome** (primitives), **nitro** (shape matching), **guzzle** (placement), **witness** (integration framework), and **sculptor** (final integration/testing), convoy `hq-cv-7sdbk`. Those credits are retained.

This checkout has no root project license file. No new license is implied by this README; dependency and dataset notices, including the [DTU reference notice](blender_blocking/evaluation/protocols/DTU_REFERENCE_LICENSE.txt), remain separate.
