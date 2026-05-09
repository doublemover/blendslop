#!/usr/bin/env python3
"""
Blender Test Runner - Runs pure-Python tests anywhere and Blender tests in headless Blender.

This is the main test runner for CI/CD and local testing.
Blender-only suites require Blender; pure-Python suites can run outside Blender.

Usage:
    blender --background --python test_runner.py
    blender --background --python test_runner.py -- --verbose
    blender --background --python test_runner.py -- --quick  # Skip slow tests
    python test_runner.py  # Pure-Python tests + dependency check (no Blender)

Exit codes:
    0: All required tests passed
    1: One or more tests failed
    2: Test runner error (unexpected crash)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add blender_blocking directory to path for test module imports
sys.path.insert(0, str(Path(__file__).parent))

# Add ~/blender_python_packages for user-installed dependencies (numpy, opencv-python, Pillow, scipy)
sys.path.insert(0, str(Path.home() / "blender_python_packages"))

from utils.progress import iter_progress, progress_print

PURE_PYTHON_TESTS: List[Tuple[str, str]] = [
    ("pure_generation_context", "utils.test_generation_context"),
    ("pure_manifest_schema", "utils.test_manifest_schema"),
    ("pure_config_defaults", "test_config_defaults"),
    ("pure_config_validation", "test_config_validation"),
    ("pure_profile_models", "test_profile_models"),
    ("pure_primitive_placement_math", "placement.test_primitive_placement_math"),
    ("pure_image_processor_rgba", "test_image_processor_rgba"),
    ("pure_silhouette_extraction", "test_silhouette_extraction"),
    ("pure_profile_sampling", "test_profile_sampling"),
    ("pure_elliptical_profile", "test_elliptical_profile"),
    ("pure_slice_sampling", "test_slice_sampling"),
    ("pure_silhouette_iou", "test_silhouette_iou"),
    ("pure_contour_analyzer", "test_contour_analyzer"),
    ("pure_shape_matcher", "test_shape_matcher"),
    ("pure_profile_combination", "test_profile_combination"),
    ("pure_slice_shape_metrics", "test_slice_shape_metrics"),
    ("pure_resfitting_metrics", "test_resfitting_metrics"),
    ("pure_visual_hull", "integration.multi_view.test_visual_hull"),
    ("pure_silhouette_pipeline", "test_silhouette_pipeline"),
    ("pure_constraints_package", "test_constraints_package"),
    ("pure_volume", "test_volume"),
    ("pure_synthetic_factory", "test_synthetic_factory"),
    ("pure_metrics_foundation", "test_metrics_foundation"),
    ("pure_reconstruction_contracts", "test_reconstruction_contracts"),
]


def check_blender_available() -> Tuple[bool, Optional[str]]:
    """Verify we're running inside Blender."""
    try:
        import bpy

        return True, bpy.app.version_string
    except ImportError:
        return False, None


def run_unittest_modules(
    modules: List[Tuple[str, str]],
    verbose: bool = False,
    progress: bool = False,
) -> Dict[str, bool]:
    """Run unittest modules by dotted name."""
    results: Dict[str, bool] = {}
    for label, module_name in iter_progress(
        modules,
        desc="pure_tests",
        total=len(modules),
        enabled=progress and len(modules) > 1,
    ):
        progress_print("\n" + "-" * 70, enabled=progress)
        progress_print(f"[Pure] {label} ({module_name})", enabled=progress)
        progress_print("-" * 70, enabled=progress)
        try:
            suite = unittest.defaultTestLoader.loadTestsFromName(module_name)
            if suite.countTestCases() == 0:
                progress_print(
                    f"WARN: No tests discovered in {module_name}", enabled=progress
                )
                results[label] = False
                continue
            runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
            result = runner.run(suite)
            results[label] = result.wasSuccessful()
        except Exception as e:
            progress_print(f"FAIL: Test crashed: {e}", enabled=progress)
            if verbose:
                import traceback

                traceback.print_exc()
            results[label] = False
    return results


def run_test_suite(
    verbose: bool = False, quick: bool = False, progress: bool = False
) -> Dict[str, Optional[bool]]:
    """
    Run complete Blender test suite.

    Args:
        verbose: Enable verbose output
        quick: Skip slow tests

    Returns:
        dict: Test results with pass/fail status
    """
    results = {}

    print("\n" + "=" * 70)
    print("BLENDER BLOCKING TOOL - TEST SUITE")
    print("=" * 70)

    # Test 1: Pure Python tests
    print("\n" + "-" * 70)
    print("[1/8] Pure Python Tests")
    print("-" * 70)
    results.update(
        run_unittest_modules(PURE_PYTHON_TESTS, verbose=verbose, progress=progress)
    )

    # Check Blender availability
    blender_ok, version = check_blender_available()
    if blender_ok:
        print(f"\nOK: Running in Blender {version}")
        print(f"OK: Python {sys.version.split()[0]}")
    else:
        print("\nWARN: Blender not available - Blender-only tests will be skipped")
        print("      Run: blender --background --python test_runner.py")

    # Test 2: Version compatibility
    print("\n" + "-" * 70)
    print("[2/8] Version Compatibility")
    print("-" * 70)
    if not blender_ok:
        print("SKIPPED - Blender required")
        results["version_compatibility"] = None
    else:
        try:
            from test_version_compatibility import (
                test_version_detection,
                test_boolean_solver_compatibility,
            )

            version_ok = test_version_detection()
            solver_ok = test_boolean_solver_compatibility()
            results["version_compatibility"] = version_ok and solver_ok
        except Exception as e:
            print(f"FAIL: Test crashed: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            results["version_compatibility"] = False

    # Test 3: Boolean solver enum
    print("\n" + "-" * 70)
    print("[3/8] Boolean Solver Enum")
    print("-" * 70)
    if not blender_ok:
        print("SKIPPED - Blender required")
        results["boolean_solver"] = None
    else:
        try:
            from test_blender_boolean import test_boolean_solver_enum

            results["boolean_solver"] = test_boolean_solver_enum()
        except Exception as e:
            print(f"FAIL: Test crashed: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            results["boolean_solver"] = False

    # Test 4: View layer update suppression
    print("\n" + "-" * 70)
    print("[4/8] View Layer Update Suppression")
    print("-" * 70)
    if not blender_ok:
        print("SKIPPED - Blender required")
        results["view_layer_updates"] = None
    else:
        try:
            from test_view_layer_fast_ops import test_view_layer_update_suppression

            results["view_layer_updates"] = test_view_layer_update_suppression()
        except Exception as e:
            print(f"FAIL: Test crashed: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            results["view_layer_updates"] = False

    # Test 5: MeshJoiner integration
    print("\n" + "-" * 70)
    print("[5/8] MeshJoiner Integration")
    print("-" * 70)
    if not blender_ok:
        print("SKIPPED - Blender required")
        results["mesh_joiner"] = None
    else:
        try:
            from test_blender_boolean import test_mesh_joiner

            results["mesh_joiner"] = test_mesh_joiner()
        except Exception as e:
            print(f"FAIL: Test crashed: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            results["mesh_joiner"] = False

    # Test 6: Slice shape matcher
    print("\n" + "-" * 70)
    print("[6/11] Slice Shape Matcher")
    print("-" * 70)
    if not blender_ok:
        print("SKIPPED - Blender required")
        results["slice_shape_matcher"] = None
    else:
        try:
            suite = unittest.defaultTestLoader.loadTestsFromName(
                "test_slice_shape_matcher"
            )
            runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
            result = runner.run(suite)
            results["slice_shape_matcher"] = result.wasSuccessful()
        except Exception as e:
            print(f"FAIL: Test crashed: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            results["slice_shape_matcher"] = False

    # Test 7: Mesh profile extractor
    print("\n" + "-" * 70)
    print("[7/11] Mesh Profile Extractor")
    print("-" * 70)
    if not blender_ok:
        print("SKIPPED - Blender required")
        results["mesh_profile_extractor"] = None
    else:
        try:
            suite = unittest.defaultTestLoader.loadTestsFromName(
                "test_mesh_profile_extractor"
            )
            runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
            result = runner.run(suite)
            results["mesh_profile_extractor"] = result.wasSuccessful()
        except Exception as e:
            print(f"FAIL: Test crashed: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            results["mesh_profile_extractor"] = False

    # Test 8: Silhouette render helpers
    print("\n" + "-" * 70)
    print("[8/11] Silhouette Rendering")
    print("-" * 70)
    if not blender_ok:
        print("SKIPPED - Blender required")
        results["silhouette_rendering"] = None
    else:
        try:
            suite = unittest.defaultTestLoader.loadTestsFromName(
                "test_silhouette_rendering"
            )
            runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
            result = runner.run(suite)
            results["silhouette_rendering"] = result.wasSuccessful()
        except Exception as e:
            print(f"FAIL: Test crashed: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            results["silhouette_rendering"] = False

    # Test 9: Full workflow test
    if not quick:
        print("\n" + "-" * 70)
        print("[9/11] Full Workflow (Procedural)")
        print("-" * 70)
        if not blender_ok:
            print("SKIPPED - Blender required")
            results["procedural_workflow"] = None
        else:
            try:
                from test_integration import test_procedural_generation

                results["procedural_workflow"] = test_procedural_generation()
            except Exception as e:
                print(f"FAIL: Test crashed: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()
                results["procedural_workflow"] = False
    else:
        print("\n" + "-" * 70)
        print("[9/11] Full Workflow (SKIPPED - quick mode)")
        print("-" * 70)
        results["procedural_workflow"] = None

    # Test 10: E2E Validation (Reference -> 3D -> Render -> Compare)
    if not quick:
        print("\n" + "-" * 70)
        print("[10/11] E2E Validation (Image -> Mesh -> Render -> IoU)")
        print("-" * 70)
        if not blender_ok:
            print("SKIPPED - Blender required")
            results["e2e_validation"] = None
        else:
            try:
                from test_e2e_validation import test_with_sample_images

                results["e2e_validation"] = test_with_sample_images()
            except Exception as e:
                print(f"FAIL: Test crashed: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()
                results["e2e_validation"] = False
    else:
        print("\n" + "-" * 70)
        print("[10/11] E2E Validation (SKIPPED - quick mode)")
        print("-" * 70)
        results["e2e_validation"] = None

    # Test 11: Dependency verification
    print("\n" + "-" * 70)
    print("[11/11] Dependency Check")
    print("-" * 70)
    try:
        # sys.path already configured at module level
        from verify_setup import verify_setup

        results["dependencies"] = verify_setup()
    except Exception as e:
        print(f"FAIL: Test crashed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        results["dependencies"] = False

    return results


def print_summary(results: Dict[str, Optional[bool]]) -> int:
    """Print test summary and return exit code."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = 0
    failed = 0
    skipped = 0

    for test_name, result in results.items():
        if result is True:
            status = "PASS"
            passed += 1
        elif result is False:
            status = "FAIL"
            failed += 1
        else:
            status = "- SKIP"
            skipped += 1

        print(f"  {test_name:.<50} {status}")

    print("=" * 70)
    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\nTESTS FAILED")
        print("=" * 70)
        return 1
    else:
        print("\nALL TESTS PASSED")
        print("=" * 70)
        return 0


def main() -> None:
    """Main entry point for test runner."""
    # Parse arguments (simple manual parsing since we're in Blender)
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    quick = "--quick" in sys.argv or "-q" in sys.argv
    progress = "--no-progress" not in sys.argv

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    # Run tests
    results = run_test_suite(verbose=verbose, quick=quick, progress=progress)

    # Print summary and exit with appropriate code
    exit_code = print_summary(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
