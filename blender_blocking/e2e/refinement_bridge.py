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
from blender_blocking.e2e.console import _print_kv_table, _print_rule


def _refinement_requested(args: argparse.Namespace) -> bool:
    return any(
        (
            args.refinement_suite,
            args.refinement_track,
            args.refinement_search,
            args.refinement_objective,
            args.refinement_result_root,
            args.refinement_preset_json,
            args.refinement_variant_file,
        )
    )

def _build_refinement_plan_from_args(
    args: argparse.Namespace,
    workflow_config: BlockingConfig,
    *,
    custom_reference_paths: Optional[Mapping[str, Path]] = None,
):
    from blender_blocking.refinement_lab.matrix import (
        build_experiment_plan,
        load_variants_from_files,
    )
    from blender_blocking.refinement_lab.preset_catalog import get_track_preset

    refinement_cfg = workflow_config.refinement_lab
    preset_overrides: Dict[str, Any] = {}
    if args.refinement_preset_json:
        with open(args.refinement_preset_json, "r", encoding="utf-8") as handle:
            preset_overrides = json.load(handle)
    suite = (
        args.refinement_suite
        or preset_overrides.get("suite")
        or refinement_cfg.default_suite
    )
    track_name = (
        args.refinement_track
        or preset_overrides.get("track")
        or refinement_cfg.default_track
    )
    track = get_track_preset(track_name)
    search = (
        args.refinement_search
        or preset_overrides.get("search")
        or track.default_search
        or refinement_cfg.default_search
    )
    objective = (
        args.refinement_objective
        or preset_overrides.get("objective")
        or track.default_objective
        or refinement_cfg.default_objective
    )
    result_root = Path(
        args.refinement_result_root
        or preset_overrides.get("result_root", "")
        or refinement_cfg.default_output_root
    )
    seed = (
        args.refinement_seed
        if args.refinement_seed is not None
        else int(preset_overrides.get("seed", workflow_config.synthetic_factory.seed))
    )
    max_runs = (
        args.refinement_max_runs
        if args.refinement_max_runs is not None
        else preset_overrides.get("max_runs", refinement_cfg.max_runs)
    )
    top_k = (
        args.refinement_top_k
        if args.refinement_top_k is not None
        else int(preset_overrides.get("top_k", refinement_cfg.top_k))
    )
    variant_files = _refinement_variant_files(args, preset_overrides)
    variant_file_mode = (
        args.refinement_variant_file_mode
        or preset_overrides.get("variant_file_mode")
        or "append"
    )
    external_variants = load_variants_from_files(variant_files)
    plan = build_experiment_plan(
        suite=suite,
        track=track_name,
        search=search,
        objective=objective,
        output_root=result_root,
        seed=seed,
        max_runs=max_runs,
        top_k=top_k,
        custom_reference_paths=custom_reference_paths,
        external_variants=external_variants,
        external_variant_mode=variant_file_mode,
    )
    summary = {
        "suite": suite,
        "track": track_name,
        "search": search,
        "objective": objective,
        "seed": seed,
        "max_runs": max_runs,
        "top_k": top_k,
        "variant_files": [path.as_posix() for path in variant_files],
        "variant_file_mode": variant_file_mode,
        "external_variants": len(external_variants),
    }
    return plan, track, summary

def _refinement_variant_files(
    args: argparse.Namespace,
    preset_overrides: Mapping[str, Any],
) -> tuple[Path, ...]:
    if args.refinement_variant_file:
        return tuple(Path(path) for path in args.refinement_variant_file)
    value = preset_overrides.get("variant_files", preset_overrides.get("variant_file"))
    if value is None:
        return ()
    if isinstance(value, (str, Path)):
        return (Path(value),)
    if isinstance(value, Sequence):
        return tuple(Path(path) for path in value)
    raise ValueError("refinement variant_files must be a path or list of paths")

def _run_refinement_from_args(
    args: argparse.Namespace,
    workflow_config: BlockingConfig,
    *,
    custom_reference_paths: Optional[Mapping[str, Path]] = None,
) -> bool:
    from blender_blocking.refinement_lab.runner import RunOptions, runner_for_plan

    refinement_cfg = workflow_config.refinement_lab
    plan, track, summary = _build_refinement_plan_from_args(
        args,
        workflow_config,
        custom_reference_paths=custom_reference_paths,
    )
    options = RunOptions(
        html_report=refinement_cfg.html_report,
        write_overlays=refinement_cfg.write_overlays or track.force_overlays,
        write_bounds_debug=refinement_cfg.write_bounds_debug
        or track.force_bounds_debug,
        write_autopsy=refinement_cfg.write_autopsy or track.force_autopsy,
        copy_references=refinement_cfg.copy_references,
        stop_on_first_error=refinement_cfg.stop_on_first_error,
        fail_on_all_failed=refinement_cfg.fail_on_all_failed,
        append_global_index=refinement_cfg.append_leaderboard,
        report_failures=refinement_cfg.report_failures,
        subprocess_blender=bool(
            args.refinement_subprocess or refinement_cfg.allow_subprocess_blender
        ),
        blender_executable=args.refinement_blender_exe
        or refinement_cfg.blender_executable,
        progress=args.progress,
    )
    _print_rule("BLENDSLOP REFINEMENT LAB", width=72)
    _print_kv_table(
        (
            ("suite", summary["suite"]),
            ("track", summary["track"]),
            ("search", summary["search"]),
            ("objective", summary["objective"]),
            ("run_root", plan.output_root),
            ("cases", len(plan.cases)),
            ("variants", len(plan.variants)),
            ("external_variants", summary["external_variants"]),
            ("variant_file_mode", summary["variant_file_mode"]),
        )
    )
    ok, _results = runner_for_plan(
        plan,
        options=options,
        base_config=workflow_config,
    ).run()
    print(f"\nRefinement report: {plan.output_root / 'report.html'}")
    return ok
