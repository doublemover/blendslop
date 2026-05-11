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
    python test_runner.py --phase pure
    python test_runner.py --phase bench --bench-case quality-smoke
    blender --background --python test_runner.py -- --phase quality-smoke

Exit codes:
    0: All required tests passed
    1: One or more tests failed
    2: Test runner error (unexpected crash)
"""

from __future__ import annotations

import argparse
import sys
import unittest
import site
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add blender_blocking directory to path for test module imports
BLENDER_BLOCKING_ROOT = Path(__file__).parent
REPO_ROOT = BLENDER_BLOCKING_ROOT.parent
sys.path.insert(0, str(BLENDER_BLOCKING_ROOT))
sys.path.insert(0, str(REPO_ROOT))

def _add_dependency_path(path: Path) -> None:
    """Expose optional dependency installs without shadowing bundled packages."""
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.append(path_str)


# Add user-installed dependency locations for Blender's bundled Python.
#
# Blender can launch with Python user-site disabled even when the same bundled
# python.exe sees it directly. Appending these paths keeps Blender's bundled
# packages first, but makes user installs of cv2/Pillow/scipy visible.
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

from utils.progress import iter_progress, progress_print

PURE_PYTHON_TESTS: List[Tuple[str, str]] = [
    ("pure_generation_context", "utils.test_generation_context"),
    ("pure_manifest_schema", "utils.test_manifest_schema"),
    ("pure_config_defaults", "test_config_defaults"),
    ("pure_config_validation", "test_config_validation"),
    ("pure_optional_deps", "test_optional_deps"),
    ("pure_profile_models", "test_profile_models"),
    ("pure_profile_band_distribution", "test_profile_band_distribution"),
    ("pure_primitive_placement_math", "placement.test_primitive_placement_math"),
    ("pure_proxy_distillation", "test_proxy_distillation"),
    ("pure_image_processor_rgba", "test_image_processor_rgba"),
    ("pure_silhouette_extraction", "test_silhouette_extraction"),
    ("pure_profile_sampling", "test_profile_sampling"),
    ("pure_elliptical_profile", "test_elliptical_profile"),
    ("pure_slice_sampling", "test_slice_sampling"),
    ("pure_silhouette_iou", "test_silhouette_iou"),
    ("pure_contour_analyzer", "test_contour_analyzer"),
    ("pure_shape_matcher", "test_shape_matcher"),
    ("pure_shape_program_grammar", "test_shape_program_grammar"),
    ("pure_profile_combination", "test_profile_combination"),
    ("pure_slice_shape_metrics", "test_slice_shape_metrics"),
    ("pure_topology_guard", "test_topology_guard"),
    ("pure_retopology_policy", "test_retopology_policy"),
    ("pure_resfitting_metrics", "test_resfitting_metrics"),
    ("pure_visual_hull", "integration.multi_view.test_visual_hull"),
    ("pure_silhouette_pipeline", "test_silhouette_pipeline"),
    ("pure_soft_silhouette", "test_soft_silhouette"),
    ("pure_differentiable_render", "test_differentiable_render"),
    ("pure_constraints_package", "test_constraints_package"),
    ("pure_volume", "test_volume"),
    ("pure_volume_chunk_cache", "test_volume_chunk_cache"),
    ("pure_sdf_projection", "test_sdf_projection"),
    ("pure_synthetic_factory", "test_synthetic_factory"),
    ("pure_quality_budget", "test_quality_budget"),
    ("pure_cost_model", "test_cost_model"),
    ("pure_evaluation_baselines", "test_evaluation_baselines"),
    ("pure_evaluation_calibration", "test_evaluation_calibration"),
    ("pure_evaluation_silhouette", "test_evaluation_silhouette"),
    ("pure_evaluation_uncertainty", "test_evaluation_uncertainty"),
    ("pure_evaluation_view_planning", "test_evaluation_view_planning"),
    ("pure_metrics_foundation", "test_metrics_foundation"),
    ("pure_metric_sensitivity", "test_metric_sensitivity"),
    ("pure_evaluation_bundle", "test_evaluation_bundle"),
    ("pure_reconstruction_contracts", "test_reconstruction_contracts"),
    ("pure_reconstruction_backends", "test_reconstruction_backends"),
    ("pure_reconstruction_pareto", "test_reconstruction_pareto"),
    ("pure_refinement_lab_contracts", "test_refinement_lab_contracts"),
    ("pure_refinement_lab_parameters", "test_refinement_lab_parameters"),
    ("pure_refinement_lab_presets", "test_refinement_lab_presets"),
    ("pure_refinement_lab_matrix", "test_refinement_lab_matrix"),
    ("pure_refinement_lab_parameter_search", "test_refinement_lab_parameter_search"),
    ("pure_refinement_lab_index", "test_refinement_lab_index"),
    ("pure_refinement_lab_autopsy", "test_refinement_lab_autopsy"),
    ("pure_refinement_lab_bounds_debug", "test_refinement_lab_bounds_debug"),
    ("pure_refinement_lab_report", "test_refinement_lab_report"),
    ("pure_refinement_lab_human_labels", "test_refinement_lab_human_labels"),
    ("pure_refinement_lab_editability_study", "test_refinement_lab_editability_study"),
    ("pure_refinement_lab_content_adaptive_patches", "test_refinement_lab_content_adaptive_patches"),
    ("pure_refinement_lab_adaptive_loop", "test_refinement_lab_adaptive_loop"),
    ("pure_refinement_lab_surrogate", "test_refinement_lab_surrogate"),
    ("pure_refinement_lab_runner", "test_refinement_lab_runner"),
    ("pure_refinement_lab_cli", "test_refinement_lab_cli"),
    ("pure_quality_refinement_smoke", "test_quality_refinement_smoke"),
    ("pure_e2e_novel_view_cli", "test_e2e_novel_view_cli"),
    ("pure_modularization_contracts", "test_modularization_contracts"),
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


RUNNER_PHASES = (
    "pure",
    "blender",
    "quick",
    "nightly",
    "bench",
    "quality-smoke",
    "quality-gated-smoke",
)


def _runner_argv() -> List[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def _parse_csv(value: str) -> Tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return items


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run blendslop validation phases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py --phase pure
  blender --background --python test_runner.py -- --phase quick
  blender --background --python test_runner.py -- --phase quality-smoke --budget-json ../configs/quality_perf_budget-smoke.json
  python test_runner.py --phase bench --bench-case quality-smoke --bench-json ../temp/benchmarks/quality-smoke.json
""",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--quick", "-q", action="store_true")
    parser.add_argument(
        "--phase",
        action="append",
        choices=RUNNER_PHASES,
        help="Validation phase to run. Can be repeated.",
    )
    parser.add_argument(
        "--phases",
        type=_parse_csv,
        default=None,
        help="Comma-separated validation phases.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "temp" / "runner",
        help="Root for JSON artifacts emitted by bench/quality phases.",
    )
    parser.add_argument("--bench-case", action="append", default=None)
    parser.add_argument("--bench-json", type=Path, default=None)
    parser.add_argument("--budget-json", type=Path, default=None)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--budget-report-json", type=Path, default=None)
    parser.add_argument("--synthetic-suite", default=None)
    parser.add_argument("--synthetic-modes", type=_parse_csv, default=None)
    parser.add_argument("--synthetic-count", type=int, default=None)
    parser.add_argument("--synthetic-seed", type=int, default=1234)
    parser.add_argument("--strict-skips", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        default=True,
        help="Disable progress output.",
    )
    return parser.parse_args(argv)


def _selected_phases(args: argparse.Namespace) -> List[str]:
    phases: List[str] = []
    if args.phase:
        phases.extend(args.phase)
    if args.phases:
        phases.extend(args.phases)
    if not phases:
        return []
    normalized = []
    for phase in phases:
        if phase not in RUNNER_PHASES:
            raise ValueError(f"unknown runner phase: {phase}")
        normalized.append(phase)
    return normalized


def run_pure_phase(verbose: bool = False, progress: bool = False) -> Dict[str, Optional[bool]]:
    print("\n" + "=" * 70)
    print("RUNNER PHASE: pure")
    print("=" * 70)
    return run_unittest_modules(PURE_PYTHON_TESTS, verbose=verbose, progress=progress)


def run_bench_phase(args: argparse.Namespace, *, case_default: str = "ci") -> Dict[str, Optional[bool]]:
    print("\n" + "=" * 70)
    print("RUNNER PHASE: bench")
    print("=" * 70)
    from benchmarks import benchmark_perf

    artifact_root = args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    bench_json = args.bench_json or artifact_root / f"bench-{case_default}.json"
    bench_args: List[str] = ["--json", str(bench_json), "--no-progress"]
    selected_cases = args.bench_case or [case_default]
    for case in selected_cases:
        bench_args.extend(["--case", case])
    if args.budget_json:
        bench_args.extend(["--budget-json", str(args.budget_json)])
    if args.baseline_json:
        bench_args.extend(["--baseline-json", str(args.baseline_json)])
    if args.budget_report_json:
        bench_args.extend(["--budget-report-json", str(args.budget_report_json)])
    bench_args.extend(["--fail-on-budget", "--fail-on-regression"])
    exit_code = benchmark_perf.main(bench_args)
    return {f"bench:{','.join(selected_cases)}": exit_code == 0}


def run_quality_smoke_phase(
    args: argparse.Namespace,
    *,
    progress: bool = False,
) -> Dict[str, Optional[bool]]:
    print("\n" + "=" * 70)
    print("RUNNER PHASE: quality-gated synthetic smoke")
    print("=" * 70)
    blender_ok, _ = check_blender_available()
    if not blender_ok:
        print("SKIPPED - Blender required")
        return {"quality_smoke_e2e": None}

    from scripts.quality_budget import evaluate_budget_files, write_report
    from test_e2e_validation import (
        DEFAULT_SYNTHETIC_MATRIX_MODES,
        run_synthetic_suite_matrix,
    )

    artifact_root = args.artifact_root / "quality-smoke"
    artifact_root.mkdir(parents=True, exist_ok=True)
    matrix_json = artifact_root / "e2e-matrix.json"
    bench_json = artifact_root / "benchmarks.json"
    budget_json = args.budget_json or REPO_ROOT / "configs" / "quality_perf_budget-smoke.json"
    e2e_report_json = artifact_root / "e2e-budget-report.json"
    bench_report_json = artifact_root / "bench-budget-report.json"
    modes = args.synthetic_modes or DEFAULT_SYNTHETIC_MATRIX_MODES

    e2e_ok = run_synthetic_suite_matrix(
        suite=args.synthetic_suite or "smoke",
        modes=modes,
        seed=args.synthetic_seed,
        count=args.synthetic_count,
        output_root=artifact_root / "synthetic",
        result_json=matrix_json,
        run_id=args.run_id,
        progress=progress,
        strict_skips=args.strict_skips,
    )
    if budget_json.exists():
        e2e_report = evaluate_budget_files(
            current_path=matrix_json,
            budget_path=budget_json,
            baseline_path=args.baseline_json,
        )
        write_report(e2e_report_json, e2e_report)
        e2e_ok = e2e_ok and bool(e2e_report.get("passed", False))
    else:
        print(f"WARN: budget file not found: {budget_json}")

    bench_args = argparse.Namespace(**vars(args))
    bench_args.artifact_root = artifact_root
    bench_args.bench_case = ["quality-smoke"]
    bench_args.bench_json = bench_json
    bench_args.budget_report_json = bench_report_json
    bench_args.budget_json = budget_json if budget_json.exists() else None
    bench_result = run_bench_phase(bench_args, case_default="quality-smoke")
    bench_ok = all(value is not False for value in bench_result.values())
    return {
        "quality_smoke_e2e": e2e_ok,
        "quality_smoke_bench": bench_ok,
    }


def run_nightly_phase(args: argparse.Namespace) -> Dict[str, Optional[bool]]:
    print("\n" + "=" * 70)
    print("RUNNER PHASE: nightly")
    print("=" * 70)
    results = run_test_suite(
        verbose=args.verbose,
        quick=False,
        progress=args.progress,
    )
    nightly_args = argparse.Namespace(**vars(args))
    nightly_args.synthetic_suite = args.synthetic_suite or "nightly-heavy"
    nightly_args.bench_case = args.bench_case or ["nightly"]
    bench_results = run_bench_phase(nightly_args, case_default="nightly")
    results.update(bench_results)
    return results


def run_named_phase(
    phase: str,
    args: argparse.Namespace,
) -> Dict[str, Optional[bool]]:
    if phase == "pure":
        return run_pure_phase(verbose=args.verbose, progress=args.progress)
    if phase == "quick":
        return run_test_suite(verbose=args.verbose, quick=True, progress=args.progress)
    if phase == "blender":
        return run_test_suite(verbose=args.verbose, quick=False, progress=args.progress)
    if phase == "bench":
        return run_bench_phase(args)
    if phase in {"quality-smoke", "quality-gated-smoke"}:
        return run_quality_smoke_phase(args, progress=args.progress)
    if phase == "nightly":
        return run_nightly_phase(args)
    raise ValueError(f"unknown runner phase: {phase}")


def main() -> None:
    """Main entry point for test runner."""
    args = _parse_args(_runner_argv())
    phases = _selected_phases(args)

    if phases:
        results: Dict[str, Optional[bool]] = {}
        for phase in phases:
            phase_results = run_named_phase(phase, args)
            for key, value in phase_results.items():
                results[f"{phase}:{key}"] = value
    else:
        results = run_test_suite(
            verbose=args.verbose,
            quick=args.quick,
            progress=args.progress,
        )

    # Print summary and exit with appropriate code
    exit_code = print_summary(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
