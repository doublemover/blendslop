# Blender Python Setup Guide

Complete guide to configuring Blender's Python environment to use the blocking tool dependencies.

## The Challenge

Blender bundles its own Python interpreter, which doesn't have access to your virtual environment by default. We need to make the dependencies (numpy, opencv-python, Pillow, scipy, scikit-image, and Open3D) available to Blender's Python.

## REQUIRED Setup: Install to Blender's Python

**⚠️ IMPORTANT: You MUST install dependencies directly into Blender's Python interpreter.**

The virtual environment approach (using your project's venv) does NOT work reliably due to binary compatibility issues. Packages like Pillow include compiled C extensions that are Python version-specific. When your venv uses Python 3.13 but Blender uses Python 3.11, you'll get errors like:

```
ImportError: cannot import name '_imaging' from 'PIL'
```

**There is only one supported installation method:**

### Install Dependencies into Blender's Python (REQUIRED)

Install packages directly into Blender's bundled Python interpreter.

**macOS:**
```bash
# Find Blender's Python (adjust version numbers as needed)
BLENDER_PYTHON="/Applications/Blender.app/Contents/Resources/4.2/python/bin/python3.11"

# Install dependencies
$BLENDER_PYTHON -m pip install -r /path/to/blendslop/blender_blocking/requirements.txt
```

**Linux:**
```bash
# Common Blender Python location
BLENDER_PYTHON="/usr/share/blender/4.2/python/bin/python3.11"

# Or if installed via snap
BLENDER_PYTHON="/snap/blender/current/4.2/python/bin/python3.11"

# Install dependencies
$BLENDER_PYTHON -m pip install -r /path/to/blendslop/blender_blocking/requirements.txt
```

**Windows:**
```powershell
# Common Blender Python location
$BLENDER_PYTHON = "C:\Program Files\Blender Foundation\Blender 4.2\4.2\python\bin\python.exe"

# Install dependencies
& $BLENDER_PYTHON -m pip install -r "C:\path\to\blendslop\blender_blocking\requirements.txt"
```

**Finding Blender's Python:**
If the paths above don't work, open Blender's Python console and run:
```python
import sys
print(sys.executable)
```

## Verification

Test your setup in Blender's Python console:

```python
# Test imports
import numpy as np
import cv2
from PIL import Image
import scipy
import skimage
import open3d

print("✓ All dependencies available!")

# Test blocking tool import
import sys
sys.path.insert(0, "/path/to/blendslop")  # If not using startup script
from blender_blocking.main_integration import BlockingWorkflow

print("✓ Blocking tool ready!")
```

## Quick Start After Setup

Once dependencies are available, using the tool is simple:

```python
import sys
sys.path.insert(0, "/path/to/blendslop")  # If needed

from blender_blocking.main_integration import example_workflow_with_images

# Run with test images
workflow = example_workflow_with_images(
    front_path="/path/to/test_images/vase_front.png",
    side_path="/path/to/test_images/vase_side.png",
    top_path="/path/to/test_images/vase_top.png"
)
```

## Troubleshooting

### "No module named 'cv2'" or "No module named 'numpy'"

Your Blender Python doesn't have access to the dependencies.

**Unsupported workaround:** Do not add your venv `site-packages` to Blender. Install dependencies directly into Blender's Python as described above.

### "Permission denied" when installing to Blender's Python

On macOS/Linux, you may need admin permissions:
```bash
sudo /Applications/Blender.app/Contents/Resources/4.2/python/bin/python3.11 -m pip install -r /path/to/blendslop/blender_blocking/requirements.txt
```

Or install for user only:
```bash
/Applications/Blender.app/Contents/Resources/4.2/python/bin/python3.11 -m pip install --user -r /path/to/blendslop/blender_blocking/requirements.txt
```

### "ImportError: numpy.core.multiarray failed to import"

Version mismatch between numpy and Blender's Python. Install a compatible version:
```bash
$BLENDER_PYTHON -m pip install -r /path/to/blendslop/blender_blocking/requirements.txt
```

### Visual hull only emits point clouds instead of meshes

Install or repair `scikit-image` in Blender's Python. Visual hull marching-cubes extraction requires `skimage.measure.marching_cubes`.

```bash
$BLENDER_PYTHON -m pip install scikit-image
```

### Poisson postprocess is skipped

Install or repair `open3d` in Blender's Python. Poisson and screened-Poisson postprocess modes are explicit optional stages and will report a structured skip when Open3D is unavailable.

```bash
$BLENDER_PYTHON -m pip install open3d
```

### Different Python versions

If you see errors about missing extensions or binary incompatibility, your venv and Blender are using different Python versions. This is why you MUST install directly into Blender's Python - the compiled extensions must match Blender's Python version exactly.

### opencv-python build errors on macOS ARM (M1/M2/M3)

Use prebuilt wheels:
```bash
$BLENDER_PYTHON -m pip install --only-binary :all: opencv-python
```

## Why Other Approaches Don't Work

You may find guides online suggesting virtual environment integration (sys.path manipulation, PYTHONPATH, etc.). **These approaches are NOT supported** for this tool because:

1. **Binary compatibility issues**: Pillow, opencv-python, and numpy include compiled C extensions that must match your Python version exactly
2. **Unreliable imports**: Even if pure-Python parts load, C extensions will fail with cryptic errors
3. **Version mismatches**: Your venv likely uses a different Python version than Blender

**The only reliable approach is installing directly into Blender's Python.**

## Creating a Helper Script

Create a convenience script that handles everything:

```bash
cat > setup_blender_blocking.sh << 'EOF'
#!/bin/bash

# Configuration
BLENDER_APP="/Applications/Blender.app"
BLENDER_PYTHON="$BLENDER_APP/Contents/Resources/4.2/python/bin/python3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔧 Setting up Blender Blocking Tool"
echo "=================================="

# Check if Blender exists
if [ ! -f "$BLENDER_PYTHON" ]; then
    echo "❌ Blender Python not found at: $BLENDER_PYTHON"
    echo "Please update BLENDER_PYTHON in this script"
    exit 1
fi

echo "📦 Installing dependencies to Blender's Python..."
$BLENDER_PYTHON -m pip install -r /path/to/blendslop/blender_blocking/requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To use in Blender:"
echo "  import sys"
echo "  sys.path.insert(0, \"$SCRIPT_DIR/..\")"
echo "  from blender_blocking.main_integration import BlockingWorkflow"
EOF

chmod +x setup_blender_blocking.sh
./setup_blender_blocking.sh
```

## Platform-Specific Notes

### macOS
- Blender app is typically at `/Applications/Blender.app`
- Python is in `Blender.app/Contents/Resources/[version]/python/bin/`
- May need to allow Blender in Security & Privacy settings

### Linux
- Installed via package manager: `/usr/share/blender/`
- Installed via snap: `/snap/blender/current/`
- Installed manually: Check `~/blender/` or `/opt/blender/`

### Windows
- Default: `C:\Program Files\Blender Foundation\Blender [version]\`
- Python: `[blender]\[version]\python\bin\python.exe`
- Use PowerShell (not cmd) for better path handling

## See Also

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide for using the tool
- [INTEGRATION.md](INTEGRATION.md) - Full API documentation
- [README.md](README.md#testing) - Testing commands and validation notes
