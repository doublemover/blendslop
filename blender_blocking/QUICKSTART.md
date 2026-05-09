# Blender Blocking Tool - Quick Start

Get started with the Blender automated blocking tool in minutes.

## Prerequisites

- Blender 4.2 LTS or 5.0 (tested)
- Python 3.8+

## REQUIRED: Blender Python Setup

**⚠️ YOU MUST COMPLETE THIS STEP BEFORE USING THE TOOL**

The tool requires dependencies (see `requirements.txt`) to be installed **directly into Blender's Python**. Virtual environment approaches DO NOT work due to binary compatibility issues.

### Step-by-Step Setup:

1. **Find Blender's Python path:**
   - Open Blender
   - Switch to **Scripting** workspace
   - In the Python Console, run:
   ```python
   import sys
   print(sys.executable)
   ```
   This will print something like: `/Applications/Blender.app/Contents/Resources/4.2/python/bin/python3.11`

2. **Install dependencies:**
   ```bash
   # Use the path from step 1
   /path/to/blender/python -m pip install -r /path/to/blendslop/blender_blocking/requirements.txt
   ```

3. **Verify setup:**
   ```python
   # In Blender's Python Console:
   import sys
   sys.path.insert(0, "/path/to/blendslop")
   exec(open("/path/to/blendslop/blender_blocking/verify_setup.py").read())
   ```

   You should see "✅ SETUP COMPLETE" if everything is installed correctly.

👉 **See [BLENDER_SETUP.md](BLENDER_SETUP.md) for platform-specific details and troubleshooting**

## Step 1: Install Python Dependencies (For Testing Outside Blender)

From the `blender_blocking` directory:

```bash
cd blender_blocking
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Run Tests (Optional)

Verify everything works:

```bash
# Create test images
./venv/bin/python3 create_test_images.py

# Run tests (without Blender)
./venv/bin/python3 test_integration.py
```

You should see:
```
✓ Image loading successful
✓ Image processing successful
✓ Shape analysis successful
```

## Step 3: Use in Blender

### Quick Test (Procedural - No Images)

1. Open Blender
2. Switch to **Scripting** workspace (top menu bar)
3. In the Python console at bottom:

```python
import sys
sys.path.insert(0, "/path/to/blendslop")

from blender_blocking.main_integration import example_workflow_no_images
mesh = example_workflow_no_images()
```

You should see a mesh named "Procedural_Blockout" in your scene.

### With Reference Images

1. Create test images (if you haven't):
```bash
cd blender_blocking
./venv/bin/python3 create_test_images.py
```

2. Open Blender and run:

```python
import sys
sys.path.insert(0, "/path/to/blendslop")

from blender_blocking.main_integration import example_workflow_with_images

# Replace paths with your actual paths
workflow = example_workflow_with_images(
    front_path="/path/to/test_images/vase_front.png",
    side_path="/path/to/test_images/vase_side.png",
    top_path="/path/to/test_images/vase_top.png"
)
```

### With Your Own Images

Your images should:
- Be orthogonal views (front, side, top)
- Have black shapes on white background
- Be PNG or JPG format

```python
import sys
sys.path.insert(0, "/path/to/blendslop")

from blender_blocking.main_integration import BlockingWorkflow

workflow = BlockingWorkflow(
    front_path="/path/to/your/front.png",
    side_path="/path/to/your/side.png",
    top_path="/path/to/your/top.png"
)

# Run the workflow
mesh = workflow.run_full_workflow(num_slices=12)
```

## What You Get

After running the workflow:
- **Mesh**: "Blockout_Mesh" object in your scene
- **Camera**: Positioned for 3D view
- **Lighting**: Basic lighting setup

The mesh is ready for sculpting!

## Adjusting Parameters

Control the detail level with `num_slices`:

```python
# More slices = more detail (but slower)
mesh = workflow.run_full_workflow(num_slices=20)

# Fewer slices = faster (but less detail)
mesh = workflow.run_full_workflow(num_slices=8)
```

## Refining Reconstruction Quality

When a result is close but not good enough, use the refinement lab instead of hand-editing one knob at a time. It sweeps reconstruction modes and parameters, ranks candidates, writes autopsies, and produces an HTML report under ignored `temp/refinement-runs/`.

Plan a small run without opening Blender:

```bash
python -m blender_blocking.refinement_lab.cli plan --suite default-vase --track profile-loft-refinement --max-runs 4 --out temp/refinement-runs/plan-smoke.json
```

Run one actual Blender refinement smoke:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python blender_blocking\test_e2e_validation.py -- `
  --refinement-suite default-vase `
  --refinement-track profile-loft-refinement `
  --refinement-max-runs 1 `
  --refinement-result-root temp\refinement-runs\smoke-profile `
  --refinement-html-report `
  --no-progress
```

See [REFINEMENT_LAB.md](REFINEMENT_LAB.md) for visual-hull diagnostics, ranking, labels, and promotion commands.

## Troubleshooting

### "No module named numpy" or dependency errors

**This means you haven't completed the required setup step.**

Run the verification script to diagnose:
```python
# In Blender's Python Console:
import sys
sys.path.insert(0, "/path/to/blendslop")
exec(open("/path/to/blendslop/blender_blocking/verify_setup.py").read())
```

The script will tell you exactly what's missing. Then follow the setup instructions at the top of this file.

See [BLENDER_SETUP.md](BLENDER_SETUP.md) for complete troubleshooting guide.

### "ImportError: cannot import name '_imaging' from 'PIL'"

**This means you installed Pillow in a venv instead of into Blender's Python.**

Pillow has compiled C extensions that must match Blender's Python version exactly. Virtual environment packages won't work.

**Fix:** Install directly into Blender's Python as described in the REQUIRED setup section above.

### "No shapes detected"
- Ensure images have good contrast (black on white)
- Try inverting your images if they're white on black

### Boolean union fails
- Reduce `num_slices` to 8 or 10
- Check that your images have clear silhouettes

### Visual hull mesh extraction is skipped
- Install `scikit-image` into Blender's Python; visual hull marching cubes needs `skimage.measure`.
- If Poisson postprocess is skipped, install `open3d` into Blender's Python.
- Run `verify_setup.py` from Blender's Python console to confirm both packages are visible to Blender.

### Import errors

Make sure you:
1. Completed the REQUIRED setup (installed dependencies into Blender's Python)
2. Use the correct path in `sys.path.insert(0, "...")` - it should point to the directory containing `blender_blocking/`
3. Run the verification script to confirm setup: `verify_setup.py`

## Next Steps

- See **[BLENDER_SETUP.md](BLENDER_SETUP.md)** for Blender Python configuration options
- See **[INTEGRATION.md](INTEGRATION.md)** for detailed API documentation
- See **[README.md](README.md#testing)** for testing commands and CI notes
- Experiment with different reference images
- Adjust `num_slices` for your needs

## Example Session

Complete example from scratch:

```bash
# 1. Setup
cd /path/to/blendslop/blender_blocking
python3 -m venv venv
./venv/bin/python3 -m pip install -r requirements.txt

# 2. Create test images
./venv/bin/python3 create_test_images.py

# 3. Open Blender, then in Python console:
```

```python
import sys
sys.path.insert(0, "/path/to/blendslop")

from blender_blocking.main_integration import example_workflow_with_images

workflow = example_workflow_with_images(
    front_path="/path/to/blender_blocking/test_images/vase_front.png",
    side_path="/path/to/blender_blocking/test_images/vase_side.png",
    top_path="/path/to/blender_blocking/test_images/vase_top.png"
)
```

Done! You now have a blockout mesh ready for sculpting.
