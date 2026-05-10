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

import argparse
import sys
import site
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_PACKAGE_ROOT = _THIS_FILE.parent
_REPO_ROOT = _PACKAGE_ROOT.parent
for _path in (str(_REPO_ROOT), str(_PACKAGE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


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


def install_research_dependencies(
    *,
    dry_run: bool = False,
    user: bool = True,
    force_reinstall: bool = True,
) -> int:
    """Install/repair optional CPU research dependencies for this Python."""

    configure_dependency_paths()
    try:
        from utils.dependency_installer import (
            blender_research_cpu_plan,
            execute_install_plan,
            print_install_plan,
        )
        from utils.optional_deps import clear_dependency_cache
    except Exception:
        from blender_blocking.utils.dependency_installer import (
            blender_research_cpu_plan,
            execute_install_plan,
            print_install_plan,
        )
        from blender_blocking.utils.optional_deps import clear_dependency_cache

    plan = blender_research_cpu_plan(
        python_executable=sys.executable,
        user=user,
        force_reinstall=force_reinstall,
    )
    if dry_run:
        print_install_plan(plan)
        return 0
    exit_code = execute_install_plan(plan)
    clear_dependency_cache()
    return exit_code


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

    # Check research-only optional dependencies without making setup fail.
    print("\nChecking research-only optional dependencies...")
    try:
        from utils.optional_deps import probe_dependency
    except Exception:
        try:
            from blender_blocking.utils.optional_deps import probe_dependency
        except Exception as e:
            probe_dependency = None
            warnings.append(f"optional dependency probe unavailable: {e}")
            print(f"  WARN: optional dependency probe unavailable: {e}")

    if probe_dependency is not None:
        for dep_name in ("trimesh", "torch", "torchvision", "lpips", "openvdb", "nvdiffrast"):
            dep = probe_dependency(
                dep_name,
                cache=False,
                isolated=dep_name in {"torch", "torchvision", "lpips"},
            )
            payload = dep.to_dict()
            if dep.available:
                version = payload.get("module_version") or "unknown version"
                resolved = payload.get("resolved_module_name") or payload.get("import_name") or dep_name
                suffix = " [isolated probe]" if payload.get("details", {}).get("isolated") else ""
                print(f"  OK: {dep_name} available as {resolved} ({version}){suffix}")
                continue
            print(f"  WARN: {dep_name} unavailable")
            diagnostic = payload.get("details", {}).get("diagnostic", {})
            if isinstance(diagnostic, dict) and diagnostic:
                category = diagnostic.get("category")
                likely_cause = diagnostic.get("likely_cause")
                remediation = diagnostic.get("remediation")
                if category:
                    print(f"    Diagnostic: {category}")
                if likely_cause:
                    print(f"    Likely cause: {likely_cause}")
                if remediation:
                    print(f"    Remediation: {remediation}")
            install_hint = payload.get("install_hint")
            if install_hint:
                print(f"    Install hint: {install_hint}")
            if payload.get("supports_rocm") is False:
                print("    ROCm/AMD: unsupported by this dependency")
                alternative = payload.get("rocm_alternative")
                if alternative:
                    print(f"    Alternative: {alternative}")
            warnings.append(f"{dep_name}: {dep.skip_reason}")

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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and optionally repair Blender Blocking dependencies."
    )
    parser.add_argument(
        "--install-research-deps",
        "--repair-torch",
        action="store_true",
        help=(
            "Install or repair the CPU torch/torchvision/lpips stack for the "
            "active Python, then run verification."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print dependency install commands without running pip.",
    )
    parser.add_argument(
        "--no-user",
        action="store_true",
        help="Install into the active Python environment instead of --user site.",
    )
    parser.add_argument(
        "--no-force-reinstall",
        action="store_true",
        help="Do not force reinstall pinned torch/torchvision/MKL packages.",
    )
    return parser.parse_args(_script_args(argv))


def _script_args(argv: list[str] | None = None) -> list[str]:
    if argv is not None:
        return list(argv)
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    script_name = Path(__file__).name.lower()
    for index, value in enumerate(sys.argv):
        if Path(str(value)).name.lower() == script_name:
            return sys.argv[index + 1 :]
    return sys.argv[1:]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.install_research_deps:
        exit_code = install_research_dependencies(
            dry_run=bool(args.dry_run),
            user=not bool(args.no_user),
            force_reinstall=not bool(args.no_force_reinstall),
        )
        if exit_code != 0 or args.dry_run:
            return exit_code
    return 0 if verify_setup() else 1


if __name__ == "__main__":
    # When run directly (or exec'd in Blender console)
    raise SystemExit(main())
