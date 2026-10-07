# blendslop

**Turn orthogonal silhouettes into editable Blender blockouts, then measure what the reconstruction actually preserves.**

Blendslop is a Python reconstruction workbench for sculpting bases and geometry experiments. Give it front, side, and top reference images; compare profile lofts, visual hulls, primitive assemblies, and other candidates; inspect the resulting geometry and per-view evidence before choosing an output.

This branch combines Blender workflows with an experiment and validation toolkit. It is research software: a good silhouette match does not guarantee correct hidden geometry, a watertight solid, or an artist-ready asset. Newer numerical and geometry contracts have focused CPU tests; their full native Blender qualification remains open.

[Quick start](#quick-start) · [Choose a method](#choose-a-reconstruction-method) · [Python example](#create-and-save-a-blockout-in-blender) · [Results and limits](#results-and-limits) · [Development](#development)

## What you can do

- Build a blockout from high-contrast orthogonal reference images, using a mesh-oriented method or an editable primitive/shape-program representation.
- Compare candidates with rendered silhouette IoU, boundary and distance metrics; use synthetic ground truth and held-out views where available.
- Run bounded parameter sweeps, rank results, inspect overlays and failure reports, and preserve the configuration and provenance for each candidate.
- Keep editable source parts separate from evaluated render/export geometry. A multipart assembly and a qualified single solid have different acceptance requirements.
- Explore opt-in research paths for calibrated/partial evidence, geometry proposals, differentiable refinement, and implicit residuals without treating proxy scores as native-render validation.

There is no trained image-to-3D model or packaged Blender add-on to install. The main entry points are Python scripts and the `BlockingWorkflow` API.

## Quick start

### 1. Get this branch

```bash
git clone --branch cloud/quality-consistency-20261006 https://github.com/doublemover/blendslop.git
cd blendslop
```

The commands below run from the repository root. This README describes the feature branch above; `main` has a smaller, older implementation.

### 2. Configure Blender's Python

Use the Python interpreter bundled with your Blender installation. NumPy, OpenCV, Pillow, SciPy, and other compiled packages must match that interpreter's ABI. Do not attach an unrelated project's virtual environment to Blender.

The recorded native comparisons used Blender **5.0.1** and **5.2.2 LTS**. The existing CI workflow declares **4.2 LTS** and **5.0** jobs. These are bounded historical checks and configured CI targets, not a compatibility certification for every path in this branch. See the [version verification report](docs/branch-audit-20261006/blender52-verification.md).

**Windows / PowerShell** — adjust both paths to the same installation:

```powershell
$blender = 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe'
$py = 'C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe'
& $py -m pip install --user -r blender_blocking/requirements.txt
& $blender --background --python-exit-code 1 --python blender_blocking/verify_setup.py
```

**macOS / bash or zsh** — example paths for Blender 5.0:

```bash
BLENDER='/Applications/Blender.app/Contents/MacOS/Blender'
PY='/Applications/Blender.app/Contents/Resources/5.0/python/bin/python3.11'
"$PY" -m pip install --user -r blender_blocking/requirements.txt
"$BLENDER" --background --python-exit-code 1 --python blender_blocking/verify_setup.py
```

**Linux** — use the same shell commands after setting `BLENDER` and `PY` to the executable and bundled `python/bin/python3.x` in your Blender installation. Paths and Python versions vary by distribution and build.

The verifier exposes the matching Python user site and an existing `~/blender_python_packages` directory. Keep that shared directory free of packages from a different Python ABI. For path discovery and import troubleshooting, see [Blender setup](blender_blocking/BLENDER_SETUP.md); older path examples there need adjustment to your build.

The [requirements file](blender_blocking/requirements.txt) includes the scientific/image stack, `tqdm`, `scikit-image`, and Open3D. Marching-cubes extraction uses scikit-image; Open3D is used by optional Poisson/native-qualification paths. If an optional wheel or runtime is unavailable, inspect the reported skip and avoid claiming that capability is validated. Torch/LPIPS, DVX, Shapely, OpenVDB, and NVIDIA-only nvdiffrast serve additional research paths and are not all installed by this command. Do not install them just to try the basic loft example.

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

Inspect the render PNGs and `temp/quickstart/result.json`, including each required view, warnings, and failure reasons. This command exercises reconstruction and image validation; it does not promise a passing score or save a `.blend` scene. Use the [Python example](#create-and-save-a-blockout-in-blender) to save one. Blender arguments come before `--`; script arguments come after it.

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

## Choose a reconstruction method

These names are accepted by `--reconstruction-mode`. Start with a single explicit method before running an ensemble.

| Mode | Representation and useful starting point | Main limitation |
|---|---|---|
| `profile_loft` / `loft_profile` | Mesh lofted from front/side profiles; useful for vase-like forms | Cross-section assumptions can miss non-profile geometry |
| `silhouette_intersection` | Boolean intersection of extruded silhouette contours | Sensitive to contour quality and Boolean validity |
| `visual_hull_voxel` | Silhouette-constrained volume and extracted mesh | Resolution, memory, thin features, and unseen concavities |
| `hybrid_loft_hull` | Profile/hull candidate route | Inspect the selected geometry and its evidence; the name alone does not describe the output |
| `primitive_fit_refine` | Fitted primitive assembly | Budget-sensitive fitting; overlapping editable parts are not automatically one solid |
| `gaussian_ellipsoid_proxy` | Ellipsoid proxy assembly | Proxy representation and projected scores need independent output validation |
| `shape_program` | Structured editable shape program with a Blender compiler | Compiler capabilities and geometry qualification are narrower than the authoring schema |
| `differentiable_refine` | Refinement adapter with CPU and optional research paths | Backend availability and proposal admission must be checked separately |
| `ensemble` | Bounded candidate evaluation and selection | Selection can only use available evidence; it does not guarantee the best 3D geometry |
| `legacy` | Slice-based primitive placement and joining | Retained baseline; sequential operations and heuristics have known limits |

The API and CLI still default to `legacy`; the quick start deliberately selects `profile_loft`. The registered experimental `implicit_residual` backend is a lower-level opt-in path, **not** an accepted top-level CLI mode. Its [checkpoint selection and replay notes](docs/implementation/CLOUD_IMPLICIT_OUTPUT_CHECKPOINT_SELECTION.md) describe the exact contracts and remaining acceptance work.

### Configuration and validation

The [config directory](configs/) contains the legacy, loft, and silhouette-intersection `default`, `higher`, `ultra`, and `extreme-ultra` files, plus scoped quality presets and gate budgets. A larger tier is a parameter choice, not a measured quality guarantee.

Inspect a resolved configuration without reconstructing anything, using a regular Python environment with the core dependencies available:

```bash
python blender_blocking/test_e2e_validation.py --reconstruction-mode profile_loft --config-path configs/loft_profile-default.json --print-config --dry-run
python blender_blocking/test_e2e_validation.py --list-modes
```

Configuration precedence is defaults → CLI settings → config-file overrides → inline JSON overrides. Inline JSON replaces matching top-level sections of the file's override dictionary; it is not a recursive merge of the two JSON documents. `--print-config` shows the resolved configuration. For slice-count comparisons, keep `--num-slices` and any `reconstruction.num_slices` JSON override equal: the harness passes the flag separately, while backend dispatch reads the resolved configuration.

Choose the evidence you want:

- `render-iou`: render the candidate and compare required silhouettes. Useful for seeing whether actual output agrees with references.
- `backend-status`: inspect structured backend execution/artifact results. Successful execution alone is not a render-quality pass.
- `novel-view`: compare supplied held-out references using image metrics; LPIPS is optional.
- `auto`: dispatch according to mode. Do not assume it always runs native render validation.

## Create and save a blockout in Blender

After generating the sample images, open a **new** Blender scene and run this in the Scripting workspace. Set `repo` to your checkout. The save path is explicit and will be overwritten on repeated runs.

```python
from pathlib import Path
import sys
import bpy

repo = Path('/absolute/path/to/blendslop')
sys.path.insert(0, str(repo))

from blender_blocking.verify_setup import configure_dependency_paths
configure_dependency_paths()
from blender_blocking.config import BlockingConfig
from blender_blocking.main_integration import BlockingWorkflow

config = BlockingConfig()
config.reconstruction.reconstruction_mode = 'profile_loft'
config.reconstruction.num_slices = 24
config.validate()

images = repo / 'blender_blocking' / 'test_images'
workflow = BlockingWorkflow(
    front_path=str(images / 'vase_front.png'),
    side_path=str(images / 'vase_side.png'),
    top_path=str(images / 'vase_top.png'),
    config=config,
)
mesh = workflow.run_full_workflow()
if mesh is None:
    raise RuntimeError(f'No mesh produced: {workflow.reconstruction_result}')

output = repo / 'temp' / 'examples' / 'vase.blend'
output.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(output))
print(f'Saved {output}')
```

This example uses a mesh-returning backend. Other backends may return structured payloads or editable hierarchies; inspect `workflow.reconstruction_result` rather than assuming every return value is one `bpy.types.Object`. Saving a scene is not a geometry-quality check.

## Compare and refine candidates

The refinement lab separates planning, execution, ranking, and review. Start small:

```bash
# No Blender execution: inspect tracks and write a bounded plan.
python -m blender_blocking.refinement_lab.cli list-tracks
python -m blender_blocking.refinement_lab.cli plan --suite default-vase --track profile-loft-refinement --max-runs 4 --out temp/refinement-runs/plan.json
```

To execute one candidate in Blender, use the configured executable:

```powershell
& $blender --background --python-exit-code 1 --python blender_blocking/test_e2e_validation.py -- `
  --refinement-suite default-vase --refinement-track profile-loft-refinement `
  --refinement-max-runs 1 --refinement-result-root temp/refinement-runs/vase `
  --refinement-html-report --refinement-autopsy --no-progress
```

On bash/zsh, use `"$BLENDER"` in place of `& $blender` and `\` for continued lines. The run directory holds its plan, per-candidate configuration/results, leaderboard, and `report.html`. Review required views, topology/editability blockers, selected backend, resource use, and failed/skipped candidates before promoting a preset. See the [refinement lab guide](blender_blocking/REFINEMENT_LAB.md) for subprocess execution, adaptive loops, labels, and promotion rules.

## Results and limits

### Historical native measurements

The [frozen 2026-10-06 campaign](docs/branch-audit-20261006/quality-final-results.md) records implementation `06a7f0219c98612a03b6e7e127d698fd9cc6c541`: **247 of 252 driver cells completed, 5 process failures, and 171 completed cells passed the render gates.** It retains regressions and unavailable results. These measurements predate the later source changes in this branch.

Two examples show why both silhouette and geometry measurements matter. In that campaign's matched ordinary → quality comparison:

| Split / method | Matched cases | Mean image IoU | Surface F.02 |
|---|---:|---:|---:|
| External / profile loft | 8/8 | 0.852 → 0.906 | 0.533 → 0.621 |
| Held-out / primitive fitting | 3/3 | 0.817 → 0.754 | 0.357 → 0.322 |

F.02 here uses the report's independent-bbox normalization and 8,192 area-weighted surface samples; its shared-frame results are a separate protocol. These are matched workflow/preset comparisons, not a source-only ablation, universal ranking, or speed claim.

The later [saved-solid audit](docs/branch-audit-20261006/solid-followthrough.md) accounts for all 247 saved cases and distinguishes numerical false reports, editable multipart overlap, folded parts, coincident surfaces, unavailable qualification, and prior qualified receipts. Its CSV rows are evidence records, not 247 qualified solids. Historical artifact identifiers under `temp/` do not mean the original meshes and renders ship in this checkout.

### What the current branch establishes

The source adds focused contracts around output identity, partial evidence, exact contact/fold guards, helper lifecycle, and bounded residual proposals. The [publication validation](docs/implementation/CLOUD_PUBLICATION_VALIDATION.md), [consistency checkpoint](docs/implementation/CLOUD_CONSISTENCY_CHECKPOINT.md), and [implicit checkpoint notes](docs/implementation/CLOUD_IMPLICIT_OUTPUT_CHECKPOINT_SELECTION.md) identify their scopes.

CPU contracts, Python parse checks, numerical forward/backward fixtures, and artifact replay do **not** establish native Blender rendering/export acceptance or a new reconstruction-quality gain. Native renderer/filter agreement, integrated scene/export/CSG checks, and real reconstruction acceptance for the newer paths remain open. Paper comparisons and larger human-editability studies remain unestablished. Preserve the exact source, inputs, cameras, metrics, and runtime when comparing results.

## Development

### Tests and benchmarks

The project uses its custom runner rather than a blanket `pytest` invocation. From the repository root:

```bash
# Pure-Python phase; requires the core dependencies in this Python environment.
python blender_blocking/test_runner.py --phase pure

# Blender examples below assume its executable is on PATH.
blender --background --python-exit-code 1 --python blender_blocking/test_runner.py -- --quick
blender --background --python-exit-code 1 --python blender_blocking/test_runner.py

# Inspect microbenchmark options before choosing a workload.
python blender_blocking/benchmarks/benchmark_perf.py --help
```

The quick phase skips slow end-to-end work; it is not the full suite. Read [quality/performance gates](docs/QUALITY_PERF_GATES.md) for synthetic matrices, budgets, and metric-specific acceptance. Microbenchmark timings are not end-to-end reconstruction speedups.

The [existing CI workflow](.github/workflows/blender-tests.yml) runs on filtered pushes to `main`/`develop` and PRs targeting `main`; it does not automatically validate every feature-branch or documentation push. Check the workflow and results for the exact commit before calling a branch validated. Follow [AGENTS.md](AGENTS.md) for repository-specific development and execution constraints.

### Where to work

| Path | Responsibility |
|---|---|
| [`main_integration.py`](blender_blocking/main_integration.py) | Image-to-blockout orchestration and API |
| [`config_models/`](blender_blocking/config_models/) | Typed configuration groups and validation |
| [`reconstruction/`](blender_blocking/reconstruction/) | Targets, backend registry, candidate selection, geometry/evidence contracts |
| [`integration/`](blender_blocking/integration/) | Image processing and native Blender scene/mesh/render operations |
| [`geometry/`](blender_blocking/geometry/), [`primitives/`](blender_blocking/primitives/), [`placement/`](blender_blocking/placement/) | Geometry building blocks and fitting |
| [`e2e/`](blender_blocking/e2e/), [`evaluation/`](blender_blocking/evaluation/), [`validation/`](blender_blocking/validation/) | CLI, rendered/geometry evidence, and acceptance checks |
| [`refinement_lab/`](blender_blocking/refinement_lab/), [`synthetic/`](blender_blocking/synthetic/) | Experiment plans, reports, and deterministic fixtures |
| [`scripts/`](scripts/), [`configs/`](configs/) | Focused utilities, presets, and budgets |

To add a reconstruction method, implement the [backend protocol](blender_blocking/reconstruction/backend.py), return a structured `CandidateResult`, and register it in the [registry](blender_blocking/reconstruction/registry.py). Registry registration alone does not expose a top-level workflow/CLI mode: configuration, dispatch, and validation also need wiring. Keep Blender imports and optional numerical dependencies at the appropriate boundary, declare skips/failures, and add focused contract tests before native integration checks.

Commit source, documentation, and small deterministic fixtures. Keep generated meshes, renders, volumes, and run reports under ignored output roots such as `temp/`. Preserve failed cases and provenance when sharing comparison evidence.

## More documentation

- [Blender setup](blender_blocking/BLENDER_SETUP.md): interpreter/dependency troubleshooting and optional research environments
- [Integration guide](blender_blocking/INTEGRATION.md): original workflow API and module examples
- [Refinement lab](blender_blocking/REFINEMENT_LAB.md): experiment planning, diagnostics, review, and preset promotion
- [Quality and performance gates](docs/QUALITY_PERF_GATES.md): evaluation and budget contracts
- [Frozen native campaign](docs/branch-audit-20261006/quality-final-results.md) and [saved-solid follow-through](docs/branch-audit-20261006/solid-followthrough.md): measured results and their limits
- [Source checkpoints](docs/implementation/): implementation-specific contracts and pending native validation

Older guides and audit reports describe their own snapshots; use this branch's source and CLI help when a default or path differs. Some historical reports refer to external run artifacts that are intentionally absent from Git.

## Credits and licensing

This fork builds on [zbuc/blender_experiments](https://github.com/zbuc/blender_experiments), with continued development in [doublemover/blendslop](https://github.com/doublemover/blendslop).

The original blocking-tool documentation credits Gas Town's multi-agent workflow: **chrome** (primitives), **nitro** (shape matching), **guzzle** (placement), **witness** (integration framework), and **sculptor** (final integration/testing), convoy `hq-cv-7sdbk`. Those credits are retained.

This checkout has no root project license file. No new license is implied by this README; dependency and dataset notices, including the [DTU reference notice](blender_blocking/evaluation/protocols/DTU_REFERENCE_LICENSE.txt), remain separate.
