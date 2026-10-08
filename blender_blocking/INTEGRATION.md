# Blender Blocking Integration Guide

This document describes the complete integration of the Blender automated blocking tool and how to use it.

## Overview

The integration combines image processing, shape analysis, and 3D primitive placement to create rough blockouts from orthogonal reference images for sculpting workflows.

## Components

### 1. Image Processing (`integration/image_processing/`)
- **image_loader.py** - Loads orthogonal reference images
- **image_processor.py** - Edge detection and normalization using OpenCV

### 2. Shape Analysis (`integration/shape_matching/`)
- **contour_analyzer.py** - Contour detection and shape property extraction
- **shape_matcher.py** - Advanced shape matching for primitive selection

### 3. Blender Operations (`integration/blender_ops/`)
- **scene_setup.py** - Scene initialization, camera, and lighting
- **mesh_generator.py** - Mesh creation from contour data
- **render_utils.py** - Rendering utilities

### 4. Primitives Library (`primitives/`)
- **primitives.py** - Blender primitive spawning (cube, sphere, cylinder, cone, torus)

### 5. Placement System (`placement/`)
- **primitive_placement.py** - Slice-based analysis and primitive positioning

### 6. Main Integration (`main_integration.py`)
- **BlockingWorkflow** - Main workflow class that orchestrates the entire pipeline

### 7. Refinement Lab (`refinement_lab/`)
- **contracts.py** - Typed experiment cases, variants, results, and run manifests
- **presets.py** - Built-in suites and parameter tracks
- **matrix.py** - Deterministic grid, random, coordinate, and successive-halving plan construction
- **runner.py** - In-process and subprocess execution against the E2E CLI
- **parameter_search.py** - Objective scoring and ranking
- **candidate_autopsy.py** - Structured failure diagnosis
- **bounds_debug.py** - Visual-hull transform, projection, and bounds diagnostics
- **artifact_report.py** - Static HTML reports and overlay/diff assets
- **human_labels.py** - Reviewer labels that influence human-adjusted ranking
- **cli.py** - Pure command line interface for list/plan/run/report/rank/autopsy/label/promote

## Dependencies

### Python Packages
```
numpy
opencv-python (cv2)
Pillow (PIL)
scipy
```

### Blender
- Blender 5.2.2 LTS stable with its bundled Python API (bpy); older releases and previews are unsupported

Install Python dependencies:
```bash
pip install -r requirements.txt
```

For Blender's Python environment:
```bash
# macOS example (adjust path for your Blender version)
/Applications/Blender.app/Contents/Resources/5.2/python/bin/python3.13 -m pip install -r /path/to/blendslop/blender_blocking/requirements.txt
```

## Usage

### Creating Test Images

First, create sample reference images:

```bash
cd blender_blocking
python3 create_test_images.py
```

This creates test images in `test_images/` directory with bottle, vase, and cube shapes.

### Running in Blender

#### Method 1: With Reference Images

1. Open Blender
2. Switch to Scripting workspace
3. Run the following code:

```python
import sys
from pathlib import Path

# Add this repository to Python path
repo_path = "/path/to/blendslop"
sys.path.insert(0, repo_path)

# Import and run workflow
from blender_blocking.main_integration import example_workflow_with_images

# Run with test images
workflow = example_workflow_with_images(
    front_path=f"{repo_path}/blender_blocking/test_images/vase_front.png",
    side_path=f"{repo_path}/blender_blocking/test_images/vase_side.png",
    top_path=f"{repo_path}/blender_blocking/test_images/vase_top.png"
)
```

#### Method 2: Procedural Generation (No Images)

For testing the 3D generation pipeline without images:

```python
import sys
sys.path.insert(0, "/path/to/blendslop")

from blender_blocking.main_integration import example_workflow_no_images

# Create procedural blockout
mesh = example_workflow_no_images()
```

### Custom Workflow

For custom workflows:

```python
from blender_blocking.main_integration import BlockingWorkflow

# Initialize workflow
workflow = BlockingWorkflow(
    front_path="path/to/front.png",
    side_path="path/to/side.png",
    top_path="path/to/top.png"
)

# Run complete workflow
result = workflow.run_full_workflow(num_slices=12)

# Or run individual steps:
workflow.load_images()
workflow.process_images()
workflow.analyze_shapes()
workflow.create_3d_blockout(num_slices=15, primitive_type='CYLINDER')
```

### Refinement Lab Workflow

Use the lab when comparing reconstruction strategies or diagnosing a bad output. The lab builds a plan, executes variants, writes structured results, and ranks them by objective:

```bash
python -m blender_blocking.refinement_lab.cli list-suites
python -m blender_blocking.refinement_lab.cli list-tracks
python -m blender_blocking.refinement_lab.cli plan --suite default-vase --track visual-hull-transform --max-runs 4 --out temp/refinement-runs/plan-smoke.json
```

Run through Blender:

```bash
blender --background --python blender_blocking/test_e2e_validation.py -- --refinement-suite default-vase --refinement-track visual-hull-transform --refinement-search coordinate --refinement-max-runs 48 --refinement-result-root temp/refinement-runs/default-vase-vh-transform --refinement-bounds-debug --refinement-autopsy --no-progress
```

The run root includes `manifest.json`, `plan.json`, `index.jsonl`, `leaderboard.md`, `report.html`, per-variant `result.json`, optional `autopsy.json`, and optional `bounds-debug.json`. Generated run roots are ignored and must not be committed.

## Workflow Steps

The integration follows these steps:

1. **Load Images** - Load front, side, and top orthogonal reference images
2. **Process Images** - Normalize and extract edges using Canny edge detection
3. **Analyze Shapes** - Find contours and analyze shape properties:
   - Area and perimeter
   - Circularity (4π × area / perimeter²)
   - Aspect ratio
   - Bounding box
4. **Determine Primitives** - Select best-fit primitive types based on shape analysis:
   - High circularity (>0.8) → Sphere
   - Medium circularity (>0.5) → Cylinder
   - Low circularity + aspect ratio near 1.0 → Cube
   - Otherwise → Cylinder (default)
5. **Calculate Bounds** - Estimate 3D bounds from 2D silhouettes
6. **Slice Analysis** - Divide volume into horizontal slices
7. **Place Primitives** - Position and scale primitives for each slice
8. **Boolean Union** - Join all primitives into single mesh
9. **Scene Setup** - Add camera and lighting for rendering
10. **Optional Refinement Loop** - Sweep candidate parameters, inspect report/autopsy output, label winners, and promote the best reproducible configuration

## Example Output

After running the workflow, you'll have:
- A joined mesh object named "Blockout_Mesh"
- Scene camera positioned for 3D view
- Lighting setup for rendering
- Individual slice primitives (before joining)

## Testing

### Without Blender
The test suite can validate image processing without Blender:

```bash
cd blender_blocking
python3 test_integration.py
```

This tests:
- Image loading
- Edge detection
- Shape analysis

### With Blender
Run the test suite inside Blender to test the complete workflow:

```python
import sys
sys.path.insert(0, "/path/to/blendslop")

from blender_blocking.test_integration import run_all_tests
run_all_tests()
```

This tests:
- Image processing pipeline
- Full workflow with sample images
- Procedural generation

## Technical Details

### Slice-Based Reconstruction

The workflow uses a slice-based approach:

1. Divide the 3D volume into N horizontal slices (default: 10-12)
2. For each slice:
   - Calculate center position
   - Determine radius/scale from image analysis
   - Create primitive at that location
3. Join all primitives using boolean union

### Shape Analysis

Each contour is analyzed for:
- **Area** - Total pixel area
- **Perimeter** - Boundary length
- **Circularity** - How circular the shape is (1.0 = perfect circle)
- **Aspect Ratio** - Width / height ratio
- **Centroid** - Center point

These metrics determine which primitive best matches the shape.

### Coordinate Systems

- **Image coordinates**: Pixels (0-512 typically)
- **Blender coordinates**: World units (scaled by factor, default 0.01)
- **Views**:
  - Front: X-Z plane
  - Side: Y-Z plane
  - Top: X-Y plane

## Limitations and Future Work

### Current Limitations
1. Simple primitive selection (could be more sophisticated)
2. Fixed slice spacing (could adapt to shape complexity)
3. No handling of concave shapes
4. Boolean operations can be slow for many primitives

### Potential Improvements
1. Adaptive slicing based on shape variation
2. Multiple primitive types per slice
3. Machine learning for better primitive selection
4. Support for non-orthogonal views
5. Interactive parameter tuning
6. Export to other 3D formats

## File Structure

```
blender_blocking/
├── INTEGRATION.md              # This file
├── README.md                   # Project overview
├── main_integration.py         # Main integration script
├── test_integration.py         # Test suite
├── create_test_images.py       # Sample image generator
├── primitives/
│   ├── primitives.py           # Primitive spawning
│   └── example_primitives.py
├── shape_matching/
│   └── slice_shape_matcher.py  # Advanced shape matching
├── placement/
│   └── primitive_placement.py  # Slice analysis and placement
└── integration/
    ├── image_processing/
    │   ├── image_loader.py
    │   └── image_processor.py
    ├── shape_matching/
    │   ├── contour_analyzer.py
    │   └── shape_matcher.py
    └── blender_ops/
        ├── scene_setup.py
        ├── mesh_generator.py
        └── render_utils.py
```

## Troubleshooting

### Import Errors
- Ensure all dependencies are installed in Blender's Python
- Check that repository path is in `sys.path`

### No Shapes Detected
- Check image contrast (silhouettes should be black on white)
- Adjust edge detection thresholds
- Ensure images are grayscale or will convert properly

### Boolean Union Failures
- Too many primitives can cause issues (reduce num_slices)
- Try `join_simple()` instead of `join_with_boolean_union()`
- Check for degenerate primitives (zero scale)

### Performance Issues
- Reduce `num_slices` (default: 10-12)
- Use lower resolution images
- Simplify primitive geometry (fewer vertices)

## Support

For issues or questions:
- Check existing beads: `bd list`
- Review README.md for project status
- Check code comments for implementation details
