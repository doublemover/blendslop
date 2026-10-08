# Reconstruction and API guide

Run CLI examples from the repository root. [Setup and sample inputs](GETTING_STARTED.md) come first.

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

The API and CLI still default to `legacy`; the quick start deliberately selects `profile_loft`. The registered experimental `implicit_residual` backend is a lower-level opt-in path, **not** an accepted top-level CLI mode. Its [checkpoint selection and replay notes](implementation/CLOUD_IMPLICIT_OUTPUT_CHECKPOINT_SELECTION.md) describe the exact contracts and remaining acceptance work.

### Configuration and validation

The [config directory](../configs/) contains the legacy, loft, and silhouette-intersection `default`, `higher`, `ultra`, and `extreme-ultra` files, plus scoped quality presets and gate budgets. A larger tier is a parameter choice, not a measured quality guarantee.

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

On bash/zsh, use `"$BLENDER"` in place of `& $blender` and `\` for continued lines. The run directory holds its plan, per-candidate configuration/results, leaderboard, and `report.html`. Review required views, topology/editability blockers, selected backend, resource use, and failed/skipped candidates before promoting a preset. See the [refinement lab guide](../blender_blocking/REFINEMENT_LAB.md) for subprocess execution, adaptive loops, labels, and promotion rules.
