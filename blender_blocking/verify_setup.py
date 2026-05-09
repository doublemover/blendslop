#!/usr/bin/env python3
"""
Setup Verification Script for Blender Blocking Tool

Run this script in Blender's Python console to verify your setup is correct.

Usage in Blender:
    import sys
    sys.path.insert(0, "/path/to/blendslop")
    exec(open("/path/to/blendslop/blender_blocking/verify_setup.py").read())
"""

from __future__ import annotations

import sys
import site
from pathlib import Path


def _add_dependency_path(path: Path) -> None:
    """Expose user-installed packages to Blender without shadowing bundled libs."""
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.append(path_str)


def configure_dependency_paths() -> None:
    """Add supported external dependency install locations to sys.path."""
    _add_dependency_path(Path.home() / "blender_python_packages")
    try:
        _add_dependency_path(Path(site.getusersitepackages()))
    except Exception:
        pass
    _add_dependency_path(
        Path.home()
        / "AppData"
        / "Roaming"
        / "Python"
        / f"Python{sys.version_info.major}{sys.version_info.minor}"
        / "site-packages"
    )


def verify_setup() -> bool:
    """Verify that all dependencies are properly installed and compatible."""
    configure_dependency_paths()

    print("\n" + "=" * 70)
    print("Blender Blocking Tool - Setup Verification")
    print("=" * 70 + "\n")

    # Check Python version
    print(f"OK: Python version: {sys.version}")
    print(f"OK: Python executable: {sys.executable}\n")

    errors = []
    warnings = []

    # Check numpy
    print("Checking numpy...")
    try:
        import numpy as np

        print(f"  OK: numpy {np.__version__} installed")
        print(f"    Location: {np.__file__}")
    except ImportError as e:
        errors.append(f"numpy: {e}")
        print("  FAIL: numpy not found")

    # Check opencv-python
    print("\nChecking opencv-python...")
    try:
        import cv2

        print(f"  OK: opencv-python {cv2.__version__} installed")
        print(f"    Location: {cv2.__file__}")
    except ImportError as e:
        errors.append(f"opencv-python: {e}")
        print("  FAIL: opencv-python not found")

    # Check Pillow (with C extension verification)
    print("\nChecking Pillow...")
    try:
        from PIL import Image

        print(f"  OK: Pillow {Image.__version__} installed")
        print(f"    Location: {Image.__file__}")

        # Critical: check C extensions
        try:
            from PIL import _imaging

            print("  OK: Pillow C extensions (_imaging) working")
        except ImportError as e:
            errors.append(f"Pillow C extensions: {e}")
            print("  FAIL: Pillow C extensions not compatible!")
            print(
                f"     This means Pillow was installed for a different Python version."
            )
            warnings.append("Pillow C extension incompatibility detected")

    except ImportError as e:
        errors.append(f"Pillow: {e}")
        print("  FAIL: Pillow not found")

    # Check scipy
    print("\nChecking scipy...")
    try:
        import scipy

        print(f"  OK: scipy {scipy.__version__} installed")
        print(f"    Location: {scipy.__file__}")
    except ImportError as e:
        errors.append(f"scipy: {e}")
        print("  FAIL: scipy not found")

    # Check scikit-image for visual hull mesh extraction
    print("\nChecking scikit-image...")
    try:
        import skimage
        from skimage import measure

        print(f"  OK: scikit-image {skimage.__version__} installed")
        print(f"    Location: {skimage.__file__}")
        print("  OK: skimage.measure.marching_cubes available")
    except ImportError as e:
        errors.append(f"scikit-image: {e}")
        print("  FAIL: scikit-image not found")
    except AttributeError as e:
        errors.append(f"scikit-image marching_cubes: {e}")
        print("  FAIL: scikit-image marching_cubes not available")

    # Check Open3D for optional Poisson visual hull postprocess
    print("\nChecking Open3D...")
    try:
        import open3d

        print(f"  OK: open3d {open3d.__version__} installed")
        print(f"    Location: {open3d.__file__}")
    except ImportError as e:
        warnings.append(f"open3d: {e}")
        print("  WARN: open3d not found; Poisson postprocess will skip")

    # Try importing Blender (if available)
    print("\nChecking Blender availability...")
    try:
        import bpy

        print(f"  OK: Running in Blender {bpy.app.version_string}")
    except ImportError:
        warnings.append("Not running in Blender")
        print("  WARN: Not running in Blender (this is OK if testing outside Blender)")

    # Summary
    print("\n" + "=" * 70)
    if not errors and not warnings:
        print("SETUP COMPLETE - All dependencies verified!")
        print("\nYou're ready to use the Blender Blocking Tool.")
        print("\nQuick start:")
        print(
            "  from blender_blocking.main_integration import example_workflow_no_images"
        )
        print("  example_workflow_no_images()")
    elif errors:
        print("SETUP INCOMPLETE - Missing or incompatible dependencies")
        print("\nErrors found:")
        for error in errors:
            print(f"  - {error}")

        print("\nFIX:")
        print("Install dependencies into Blender's Python:")
        print("\n  # Find Blender's Python:")
        print("  # In Blender console: import sys; print(sys.executable)")
        print("\n  # Then run:")
        print(
            "  /path/to/blender/python -m pip install -r blender_blocking/requirements.txt"
        )
        print("\n📖 See BLENDER_SETUP.md for complete instructions")

        if "Pillow C extension" in str(errors):
            print("\nWARN: PILLOW C EXTENSION ERROR DETECTED")
            print("This means you installed Pillow in a venv with a different Python")
            print("version than Blender uses. You MUST install directly into Blender's")
            print("Python for C extensions to work.")
    else:
        print("SETUP MOSTLY COMPLETE - Minor warnings")
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

        print("\nYou should be able to proceed, but check the warnings above.")

    print("=" * 70 + "\n")

    return len(errors) == 0


if __name__ == "__main__":
    # When run directly (or exec'd in Blender console)
    verify_setup()
