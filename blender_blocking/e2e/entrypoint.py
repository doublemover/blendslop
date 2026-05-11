# ruff: noqa: E402,F401,F403
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from blender_blocking.verify_setup import configure_dependency_paths
configure_dependency_paths()

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    BLENDER_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

from blender_blocking.config import BlockingConfig
from blender_blocking.config import CandidateConfig as ConfigCandidateConfig
from blender_blocking.config import RenderConfig
from blender_blocking.evaluation.cost_model import CostRecorder
from blender_blocking.evaluation.silhouette_eval import (
    SilhouetteGateConfig,
    evaluate_silhouette_pair,
    missing_silhouette_view,
    summarize_silhouette_views,
)
from blender_blocking.integration.blender_ops.render_utils import (
    parse_orbit_view_degrees,
    render_orthogonal_views,
)
from blender_blocking.integration.image_processing.image_loader import load_image
from blender_blocking.utils.generation_context import GenerationContext
from blender_blocking.utils.progress import progress_bar
from blender_blocking.validation.silhouette_iou import canonicalize_mask, mask_from_image_array
from blender_blocking.e2e.constants import *

from blender_blocking.e2e.cli_args import (
    _apply_cli_args,
    _apply_overrides,
    _derive_config_label,
    _parse_args,
    _resolve_novel_view_inputs,
)
from blender_blocking.e2e.console import _artifact_line, _console_print, _print_kv_table, _print_rule
from blender_blocking.e2e.matrix import run_synthetic_suite_matrix
from blender_blocking.e2e.refinement_bridge import _build_refinement_plan_from_args, _refinement_requested, _run_refinement_from_args
from blender_blocking.e2e.validator import E2EValidator, test_with_custom_images

def main(argv: Optional[list[str]] = None) -> int:
        if argv is not None:
            script_argv = list(argv)
        elif "--" in sys.argv:
            script_argv = sys.argv[sys.argv.index("--") + 1 :]
        elif not BLENDER_AVAILABLE:
            script_argv = sys.argv[1:]
        else:
            script_argv = []
        args = _parse_args(script_argv)

        if args.list_modes:
            _print_rule("RECONSTRUCTION MODES", width=72)
            for mode in ALL_RECONSTRUCTION_MODES:
                kind = "render-iou" if mode in RENDER_IOU_MODES else "backend-status"
                surface = "backend artifact" if mode in BACKEND_MODES else "Blender mesh"
                print(f"  {mode:<28} {kind:<16} {surface}")
            print(
                f"\nDefault ensemble candidates: {', '.join(DEFAULT_ENSEMBLE_CANDIDATES)}"
            )
            print(
                "Validation modes: auto, render-iou, backend-status, novel-view "
                "(PSNR/SSIM/optional LPIPS)"
            )
            return 0

        workflow_config = BlockingConfig()
        _apply_cli_args(workflow_config, args)

        overrides: Dict[str, Any] = {}
        if args.config_path:
            with open(args.config_path, "r", encoding="utf-8") as handle:
                overrides = json.load(handle)
        if args.config_json:
            inline = json.loads(args.config_json)
            if overrides:
                overrides.update(inline)
            else:
                overrides = inline
        _apply_overrides(workflow_config, overrides)
        workflow_config.validate()
        render_config = workflow_config.render_silhouette
        config_label = args.config_label or _derive_config_label(args)

        if args.print_config:
            _print_rule("RESOLVED CONFIG", width=72)
            print(json.dumps(workflow_config.to_dict(), indent=2, sort_keys=True))

        refinement_requested = _refinement_requested(args)

        custom_paths = [args.front, args.side, args.top]
        if any(custom_paths) and not all(custom_paths):
            print("ERROR: --front, --side, and --top must be provided together.")
            return 2
        custom_reference_paths = None
        if all(custom_paths):
            custom_reference_paths = {
                "front": Path(args.front),
                "side": Path(args.side),
                "top": Path(args.top),
            }
        try:
            (
                novel_reference_paths,
                novel_view_names,
                novel_compute_ssim,
                novel_compute_lpips,
                novel_psnr_threshold,
                novel_ssim_threshold,
                novel_lpips_threshold,
            ) = _resolve_novel_view_inputs(args)
        except (argparse.ArgumentTypeError, OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: {exc}")
            return 2

        if args.dry_run:
            if refinement_requested:
                plan, _track, summary = _build_refinement_plan_from_args(
                    args,
                    workflow_config,
                    custom_reference_paths=custom_reference_paths,
                )
                _print_rule("REFINEMENT PLAN", width=72)
                _print_kv_table(
                    (
                        ("suite", summary["suite"]),
                        ("track", summary["track"]),
                        ("search", summary["search"]),
                        ("objective", summary["objective"]),
                        ("run_root", plan.output_root),
                        ("cases", len(plan.cases)),
                        ("variants", len(plan.variants)),
                        ("plan_id", plan.plan_id),
                    )
                )
            print("\nDry run complete: config resolved and validated.")
            return 0

        if not BLENDER_AVAILABLE:
            can_delegate_refinement = refinement_requested and (
                args.refinement_subprocess
                or workflow_config.refinement_lab.allow_subprocess_blender
            )
            can_run_pure_synthetic_matrix = bool(
                args.synthetic_matrix
                and _synthetic_suite_is_pure_mask(
                    args.synthetic_suite or workflow_config.synthetic_factory.suite,
                    seed=(
                        args.synthetic_seed
                        if args.synthetic_seed is not None
                        else workflow_config.synthetic_factory.seed
                    ),
                    count=args.synthetic_count,
                )
                and args.validation_mode in {"auto", "backend-status"}
            )
            if not can_delegate_refinement and not can_run_pure_synthetic_matrix:
                print("ERROR: This validation CLI must be run inside Blender.")
                print(
                    "Run: blender --background --python blender_blocking/test_e2e_validation.py -- [options]"
                )
                print(
                    "For refinement labs from system Python, add "
                    "--refinement-subprocess --refinement-blender-exe <blender>."
                )
                return 1

        thresholds = {
            key: value
            for key, value in {
                "front": args.front_threshold,
                "side": args.side_threshold,
                "top": args.top_threshold,
            }.items()
            if value is not None
        }

        if refinement_requested:
            success = _run_refinement_from_args(
                args,
                workflow_config,
                custom_reference_paths=custom_reference_paths,
            )
        elif args.synthetic_matrix:
            matrix_root = Path(
                args.synthetic_output_root or workflow_config.synthetic_factory.output_root
            )
            matrix_json = args.result_json or matrix_root / "e2e_synthetic_matrix.json"
            success = run_synthetic_suite_matrix(
                suite=args.synthetic_suite or workflow_config.synthetic_factory.suite,
                modes=args.synthetic_modes or DEFAULT_SYNTHETIC_MATRIX_MODES,
                seed=(
                    args.synthetic_seed
                    if args.synthetic_seed is not None
                    else workflow_config.synthetic_factory.seed
                ),
                count=args.synthetic_count,
                output_root=matrix_root,
                base_config=workflow_config,
                iou_threshold=args.iou_threshold,
                view_thresholds=thresholds,
                boundary_iou_threshold=args.boundary_iou_threshold,
                signed_distance_loss_threshold=args.signed_distance_loss_threshold,
                validation_mode=args.validation_mode,
                config_label=config_label,
                result_json=matrix_json,
                run_id=args.run_id,
                novel_view_names=novel_view_names,
                novel_compute_ssim=novel_compute_ssim,
                novel_compute_lpips=novel_compute_lpips,
                novel_psnr_threshold=novel_psnr_threshold,
                novel_ssim_threshold=novel_ssim_threshold,
                novel_lpips_threshold=novel_lpips_threshold,
                cost_report_json=args.cost_report_json,
                cost_track_memory=args.cost_track_memory,
                cost_fail_max_wall_ms=args.cost_fail_max_wall_ms,
                cost_fail_max_backend_wall_ms=args.cost_fail_max_backend_wall_ms,
                progress=args.progress,
                strict_skips=args.synthetic_strict_skips,
                allow_failed_rows=args.synthetic_allow_failed_rows,
            )
            if args.quality_budget_json:
                from scripts.quality_budget import evaluate_budget_files, write_report

                report = evaluate_budget_files(
                    current_path=matrix_json,
                    budget_path=args.quality_budget_json,
                    baseline_path=args.quality_compare_baseline,
                )
                if args.quality_report_json:
                    write_report(args.quality_report_json, report)
                    _artifact_line("quality budget report", args.quality_report_json)
                threshold_passed = bool(report.get("threshold_passed", False))
                comparison_passed = bool(report.get("comparison_passed", True))
                _console_print(
                    "Quality budget: "
                    f"{'PASS' if report.get('passed') else 'FAIL'} "
                    f"(thresholds={'PASS' if threshold_passed else 'FAIL'}, "
                    f"comparisons={'PASS' if comparison_passed else 'FAIL'})",
                    color="green" if report.get("passed") else "red",
                )
                success = success and threshold_passed
                if workflow_config.quality_budget.fail_on_regression:
                    success = success and comparison_passed
        elif all(custom_paths):
            success = test_with_custom_images(
                args.front,
                args.side,
                args.top,
                num_slices=args.num_slices,
                iou_threshold=args.iou_threshold,
                view_thresholds=thresholds,
                boundary_iou_threshold=args.boundary_iou_threshold,
                signed_distance_loss_threshold=args.signed_distance_loss_threshold,
                render_config=render_config,
                workflow_config=workflow_config,
                config_label=config_label,
                validation_mode=args.validation_mode,
                render_output_dir=args.render_output_dir,
                artifact_root=args.artifact_output_root,
                result_json=args.result_json,
                run_id=args.run_id,
                novel_view_reference_paths=novel_reference_paths,
                novel_view_names=novel_view_names,
                novel_compute_ssim=novel_compute_ssim,
                novel_compute_lpips=novel_compute_lpips,
                novel_psnr_threshold=novel_psnr_threshold,
                novel_ssim_threshold=novel_ssim_threshold,
                novel_lpips_threshold=novel_lpips_threshold,
                cost_report_json=args.cost_report_json,
                cost_track_memory=args.cost_track_memory,
                cost_fail_max_wall_ms=args.cost_fail_max_wall_ms,
                cost_fail_max_backend_wall_ms=args.cost_fail_max_backend_wall_ms,
                progress=args.progress,
            )
        else:
            # Run test with sample images
            validator = E2EValidator(
                iou_threshold=args.iou_threshold,
                view_thresholds=thresholds,
                boundary_iou_threshold=args.boundary_iou_threshold,
                signed_distance_loss_threshold=args.signed_distance_loss_threshold,
                render_config=render_config,
                workflow_config=workflow_config,
                config_label=config_label,
                validation_mode=args.validation_mode,
                render_output_dir=args.render_output_dir,
                artifact_root=args.artifact_output_root,
                result_json=args.result_json,
                run_id=args.run_id,
                novel_view_reference_paths=novel_reference_paths,
                novel_view_names=novel_view_names,
                novel_compute_ssim=novel_compute_ssim,
                novel_compute_lpips=novel_compute_lpips,
                novel_psnr_threshold=novel_psnr_threshold,
                novel_ssim_threshold=novel_ssim_threshold,
                novel_lpips_threshold=novel_lpips_threshold,
                cost_report_json=args.cost_report_json,
                cost_track_memory=args.cost_track_memory,
                cost_fail_max_wall_ms=args.cost_fail_max_wall_ms,
                cost_fail_max_backend_wall_ms=args.cost_fail_max_backend_wall_ms,
                progress=args.progress,
            )
            base_dir = BLENDER_BLOCKING_ROOT
            test_images_dir = base_dir / "test_images"
            if not test_images_dir.exists():
                print("Creating test images...")
                import subprocess

                result = subprocess.run(
                    [sys.executable, str(base_dir / "create_test_images.py")],
                    capture_output=True,
                    text=True,
                    cwd=base_dir,
                )
                if result.returncode != 0:
                    print("ERROR: Failed to create test images")
                    print(result.stderr)
                    return 1
            success, _ = validator.validate_reconstruction(
                {
                    "front": str(test_images_dir / "vase_front.png"),
                    "side": str(test_images_dir / "vase_side.png"),
                    "top": str(test_images_dir / "vase_top.png"),
                },
                num_slices=args.num_slices,
            )
            if validator.results:
                validator.print_detailed_results()

        # Exit with appropriate code
        return 0 if success else 1


def _synthetic_suite_is_pure_mask(
    suite: str,
    *,
    seed: int,
    count: int | None,
) -> bool:
    try:
        from blender_blocking.synthetic.registry import get_definition, specs_for_suite
        from blender_blocking.e2e.matrix import _definition_name_from_spec
    except Exception:
        return False
    try:
        specs = specs_for_suite(suite, seed=seed, count=count)
        return bool(specs) and all(
            not get_definition(_definition_name_from_spec(spec)).blender_supported
            for spec in specs
        )
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
