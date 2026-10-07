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
from blender_blocking.e2e.payloads import _ordered_unique

from blender_blocking.e2e.novel_args import _parse_csv, _parse_rgba


def _coerce_override_value(current: Any, value: Any) -> Any:
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _candidate_configs(names: Sequence[str]) -> Tuple[ConfigCandidateConfig, ...]:
    return tuple(
        ConfigCandidateConfig(backend_name=name, candidate_id=f"{name}_{index:02d}")
        for index, name in enumerate(names)
    )


def _apply_dataclass_overrides(target: object, values: Mapping[str, Any]) -> None:
    valid_fields = {field.name for field in fields(target)}
    for key, value in values.items():
        if key not in valid_fields:
            raise ValueError(f"unknown config field {type(target).__name__}.{key}")
        current = getattr(target, key)
        if key == "candidates" and isinstance(value, (list, tuple)):
            candidates = []
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    candidates.append(ConfigCandidateConfig(**dict(item)))
                else:
                    candidates.append(
                        ConfigCandidateConfig(
                            backend_name=str(item),
                            candidate_id=f"{item}_{index:02d}",
                        )
                    )
            setattr(target, key, tuple(candidates))
        elif is_dataclass(current) and isinstance(value, Mapping):
            _apply_dataclass_overrides(current, value)
        else:
            setattr(target, key, _coerce_override_value(current, value))


def _apply_overrides(cfg: BlockingConfig, overrides: Dict[str, Any]) -> None:
    """Apply JSON/config-file overrides to any BlockingConfig group."""
    if not overrides:
        return
    for group_name, values in overrides.items():
        if not hasattr(cfg, group_name):
            raise ValueError(f"unknown config group {group_name!r}")
        target = getattr(cfg, group_name)
        if is_dataclass(target) and isinstance(values, Mapping):
            _apply_dataclass_overrides(target, values)
        else:
            setattr(cfg, group_name, values)


def _set_if_not_none(target: object, name: str, value: Any) -> None:
    if value is not None:
        setattr(target, name, value)


def _apply_silhouette_cli_args(cfg: BlockingConfig, args: argparse.Namespace) -> None:
    for prefix, target in (
        ("ref", cfg.silhouette_extract_ref),
        ("render", cfg.silhouette_extract_render),
    ):
        _set_if_not_none(
            target, "prefer_alpha", getattr(args, f"{prefix}_prefer_alpha")
        )
        _set_if_not_none(target, "polarity", getattr(args, f"{prefix}_polarity"))
        _set_if_not_none(
            target, "invert_policy", getattr(args, f"{prefix}_invert_policy")
        )
        _set_if_not_none(
            target, "alpha_threshold", getattr(args, f"{prefix}_alpha_threshold")
        )
        _set_if_not_none(
            target,
            "alpha_min_coverage",
            getattr(args, f"{prefix}_alpha_min_coverage"),
        )
        _set_if_not_none(
            target, "gray_threshold", getattr(args, f"{prefix}_gray_threshold")
        )
        _set_if_not_none(
            target, "morph_close_px", getattr(args, f"{prefix}_morph_close")
        )
        _set_if_not_none(target, "morph_open_px", getattr(args, f"{prefix}_morph_open"))
        _set_if_not_none(target, "fill_holes", getattr(args, f"{prefix}_fill_holes"))
        _set_if_not_none(
            target,
            "largest_component_only",
            getattr(args, f"{prefix}_largest_component"),
        )
        _set_if_not_none(
            target, "min_area_frac", getattr(args, f"{prefix}_min_area_frac")
        )
        _set_if_not_none(
            target, "max_area_frac", getattr(args, f"{prefix}_max_area_frac")
        )
        _set_if_not_none(
            target,
            "max_border_contact_frac",
            getattr(args, f"{prefix}_max_border_contact_frac"),
        )
        _set_if_not_none(
            target,
            "min_component_area_px",
            getattr(args, f"{prefix}_min_component_area_px"),
        )
        _set_if_not_none(
            target, "candidate_scoring", getattr(args, f"{prefix}_candidate_scoring")
        )
        _set_if_not_none(
            target, "emit_uncertainty", getattr(args, f"{prefix}_emit_uncertainty")
        )


def _apply_cli_args(cfg: BlockingConfig, args: argparse.Namespace) -> None:
    cfg.reconstruction.reconstruction_mode = args.reconstruction_mode
    cfg.reconstruction.num_slices = int(args.num_slices)
    _set_if_not_none(cfg.reconstruction, "unit_scale", args.unit_scale)

    cfg.render_silhouette.resolution = args.resolution
    cfg.render_silhouette.engine = args.engine
    cfg.render_silhouette.samples = int(args.samples)
    cfg.render_silhouette.margin_frac = float(args.margin)
    _set_if_not_none(cfg.render_silhouette, "color_mode", args.color_mode)
    _set_if_not_none(cfg.render_silhouette, "transparent_bg", args.transparent_bg)
    _set_if_not_none(cfg.render_silhouette, "force_material", args.force_material)
    _set_if_not_none(cfg.render_silhouette, "background_color", args.background_color)
    _set_if_not_none(cfg.render_silhouette, "silhouette_color", args.silhouette_color)
    _set_if_not_none(
        cfg.render_silhouette,
        "camera_distance_factor",
        args.camera_distance_factor,
    )
    _set_if_not_none(cfg.render_silhouette, "party_mode", args.party_mode)

    _set_if_not_none(cfg.profile_sampling, "num_samples", args.profile_samples)
    _set_if_not_none(cfg.profile_sampling, "sample_policy", args.profile_sample_policy)
    _set_if_not_none(cfg.profile_sampling, "fill_strategy", args.profile_fill_strategy)
    _set_if_not_none(
        cfg.profile_sampling, "smoothing_window", args.profile_smoothing_window
    )

    _set_if_not_none(cfg.mesh_from_profile, "surface_mode", args.mesh_surface_mode)
    _set_if_not_none(cfg.mesh_from_profile, "surface_subdivisions", args.mesh_surface_subdivisions)
    _set_if_not_none(
        cfg.mesh_from_profile, "radial_segments", args.mesh_radial_segments
    )
    _set_if_not_none(cfg.mesh_from_profile, "cap_mode", args.mesh_cap_mode)
    _set_if_not_none(cfg.mesh_from_profile, "min_radius_u", args.mesh_min_radius)
    _set_if_not_none(
        cfg.mesh_from_profile, "merge_threshold_u", args.mesh_merge_threshold
    )
    _set_if_not_none(cfg.mesh_from_profile, "recalc_normals", args.mesh_recalc_normals)
    _set_if_not_none(cfg.mesh_from_profile, "shade_smooth", args.mesh_shade_smooth)
    _set_if_not_none(
        cfg.mesh_from_profile,
        "weld_degenerate_rings",
        args.mesh_weld_degenerate_rings,
    )
    _set_if_not_none(
        cfg.mesh_from_profile,
        "adaptive_radial_segments",
        args.mesh_adaptive_radial_segments,
    )
    _set_if_not_none(
        cfg.mesh_from_profile,
        "min_adaptive_radial_segments",
        args.mesh_min_adaptive_radial_segments,
    )
    _set_if_not_none(
        cfg.mesh_from_profile,
        "max_adaptive_radial_segments",
        args.mesh_max_adaptive_radial_segments,
    )
    _set_if_not_none(
        cfg.mesh_from_profile, "topology_strict", args.mesh_topology_strict
    )
    _set_if_not_none(
        cfg.mesh_from_profile,
        "research_allow_low_radial_segments",
        args.mesh_research_allow_low_radial_segments,
    )

    _apply_silhouette_cli_args(cfg, args)

    _set_if_not_none(cfg.canonicalize, "output_size", args.canonical_output_size)
    _set_if_not_none(cfg.canonicalize, "padding_frac", args.canonical_padding_frac)
    _set_if_not_none(cfg.canonicalize, "anchor", args.canonical_anchor)
    _set_if_not_none(cfg.canonicalize, "interp", args.canonical_interp)
    _set_if_not_none(cfg.canonicalize, "use_cache", args.canonical_cache)
    _set_if_not_none(cfg.canonicalize, "digest_algorithm", args.canonical_digest)
    _set_if_not_none(cfg.canonicalize, "fill_holes", args.canonical_fill_holes)
    _set_if_not_none(
        cfg.canonicalize, "largest_component_only", args.canonical_largest_component
    )

    _set_if_not_none(cfg.mesh_join, "mode", args.mesh_join_mode)
    _set_if_not_none(cfg.mesh_join, "boolean_solver", args.boolean_solver)
    _set_if_not_none(
        cfg.mesh_join, "allow_degraded_simple_join", args.allow_degraded_simple_join
    )
    _set_if_not_none(cfg.mesh_join, "record_attempts", args.record_join_attempts)
    _set_if_not_none(cfg.mesh_join, "balanced_boolean_tree", args.balanced_boolean_tree)
    _set_if_not_none(
        cfg.silhouette_intersection,
        "extrude_distance",
        args.silhouette_extrude_distance,
    )
    _set_if_not_none(
        cfg.silhouette_intersection, "contour_mode", args.silhouette_contour_mode
    )
    _set_if_not_none(
        cfg.silhouette_intersection,
        "largest_component_only",
        args.silhouette_largest_component,
    )

    _set_if_not_none(cfg.visual_hull, "backend", args.vh_backend)
    _set_if_not_none(cfg.visual_hull, "resolution", args.vh_resolution)
    _set_if_not_none(cfg.visual_hull, "max_resolution", args.vh_max_resolution)
    _set_if_not_none(cfg.visual_hull, "chunk_size", args.vh_chunk_size)
    _set_if_not_none(cfg.visual_hull, "adaptive_max_depth", args.vh_adaptive_max_depth)
    _set_if_not_none(cfg.visual_hull, "boundary_refine", args.vh_boundary_refine)
    _set_if_not_none(cfg.visual_hull, "mesh_method", args.vh_mesh_method)
    _set_if_not_none(cfg.visual_hull, "postprocess", args.vh_postprocess)
    _set_if_not_none(cfg.visual_hull, "memory_budget_mb", args.vh_memory_budget_mb)
    _set_if_not_none(
        cfg.visual_hull, "occupancy_threshold", args.vh_occupancy_threshold
    )
    _set_if_not_none(
        cfg.visual_hull,
        "uncertainty_aggregation",
        args.vh_uncertainty_aggregation,
    )
    _set_if_not_none(cfg.visual_hull, "enable_cache", args.vh_cache)
    _set_if_not_none(cfg.visual_hull, "cache_directory", args.vh_cache_dir)
    _set_if_not_none(cfg.visual_hull, "cache_namespace", args.vh_cache_namespace)
    _set_if_not_none(cfg.visual_hull, "cache_read", args.vh_cache_read)
    _set_if_not_none(cfg.visual_hull, "cache_write", args.vh_cache_write)
    _set_if_not_none(cfg.volume, "backend", args.volume_backend)
    _set_if_not_none(cfg.volume, "sparse_chunk_size", args.volume_sparse_chunk_size)
    _set_if_not_none(cfg.volume, "serialization", args.volume_serialization)
    _set_if_not_none(cfg.volume, "export_openvdb", args.export_openvdb)

    _set_if_not_none(cfg.primitive_fit, "primitive_families", args.primitive_families)
    if args.primitive_loss_weights_json:
        cfg.primitive_fit.loss_weights = json.loads(args.primitive_loss_weights_json)
    _set_if_not_none(
        cfg.primitive_fit, "target_point_count", args.primitive_target_points
    )
    _set_if_not_none(cfg.primitive_fit, "min_primitives", args.primitive_min)
    _set_if_not_none(cfg.primitive_fit, "max_primitives", args.primitive_max)
    _set_if_not_none(cfg.primitive_fit, "optimization_steps", args.primitive_steps)
    _set_if_not_none(
        cfg.primitive_fit, "checkpoint_cadence", args.primitive_checkpoint_cadence
    )
    _set_if_not_none(
        cfg.primitive_fit, "fail_on_regression", args.primitive_fail_on_regression
    )
    _set_if_not_none(cfg.primitive_fit, "max_runtime_s", args.primitive_max_runtime_s)
    _set_if_not_none(
        cfg.primitive_fit,
        "max_objective_evaluations",
        args.primitive_max_objective_evaluations,
    )

    _set_if_not_none(cfg.gaussian_ellipsoid, "primitive_count", args.gaussian_count)
    _set_if_not_none(
        cfg.gaussian_ellipsoid, "initialization", args.gaussian_initialization
    )
    _set_if_not_none(cfg.gaussian_ellipsoid, "min_radius", args.gaussian_min_radius)
    _set_if_not_none(cfg.gaussian_ellipsoid, "max_radius", args.gaussian_max_radius)
    _set_if_not_none(cfg.gaussian_ellipsoid, "opacity_min", args.gaussian_opacity_min)
    _set_if_not_none(cfg.gaussian_ellipsoid, "opacity_max", args.gaussian_opacity_max)
    _set_if_not_none(cfg.gaussian_ellipsoid, "renderer", args.gaussian_renderer)
    _set_if_not_none(
        cfg.gaussian_ellipsoid,
        "export_mesh_proxy",
        args.gaussian_export_mesh_proxy,
    )

    _set_if_not_none(cfg.differentiable_render, "backend", args.diff_backend)
    _set_if_not_none(
        cfg.differentiable_render,
        "optional_dependency_policy",
        args.diff_optional_policy,
    )
    _set_if_not_none(
        cfg.differentiable_render, "gradient_mode", args.diff_gradient_mode
    )
    _set_if_not_none(
        cfg.differentiable_render, "finite_difference_epsilon", args.diff_epsilon
    )
    _set_if_not_none(
        cfg.differentiable_render, "primitive_count", args.diff_primitive_count
    )
    _set_if_not_none(
        cfg.differentiable_render, "target_point_count", args.diff_target_points
    )
    _set_if_not_none(
        cfg.differentiable_render,
        "visual_hull_resolution",
        args.diff_visual_hull_resolution,
    )
    _set_if_not_none(
        cfg.differentiable_render,
        "optimization_steps",
        args.diff_optimization_steps,
    )
    _set_if_not_none(
        cfg.differentiable_render,
        "optimization_initial_step",
        args.diff_initial_step,
    )
    _set_if_not_none(
        cfg.differentiable_render,
        "optimization_step_decay",
        args.diff_step_decay,
    )
    _set_if_not_none(
        cfg.differentiable_render,
        "optimization_min_step",
        args.diff_min_step,
    )
    _set_if_not_none(
        cfg.differentiable_render,
        "max_objective_evaluations",
        args.diff_max_objective_evaluations,
    )
    _set_if_not_none(
        cfg.differentiable_render, "max_runtime_s", args.diff_max_runtime
    )
    if args.diff_loss_weights_json:
        cfg.differentiable_render.loss_weights = json.loads(args.diff_loss_weights_json)

    _set_if_not_none(cfg.shape_program, "root_strategy", args.shape_root_strategy)
    _set_if_not_none(cfg.shape_program, "residual_policy", args.shape_residual_policy)
    _set_if_not_none(cfg.shape_program, "max_nodes", args.shape_max_nodes)
    _set_if_not_none(cfg.shape_program, "editability_bias", args.shape_editability_bias)
    _set_if_not_none(cfg.shape_program, "compile_blender", args.shape_compile_blender)
    _set_if_not_none(cfg.shape_program, "lathe_segments", args.shape_lathe_segments)
    _set_if_not_none(cfg.shape_program, "bevel_modifier", args.shape_bevel_modifier)
    _set_if_not_none(cfg.shape_program, "weighted_normals", args.shape_weighted_normals)
    _set_if_not_none(cfg.shape_program, "run_export_qa", args.shape_run_export_qa)
    _set_if_not_none(
        cfg.shape_program, "export_qa_targets", args.shape_export_qa_targets
    )
    _set_if_not_none(
        cfg.shape_program,
        "evaluate_texture_materials",
        args.evaluate_texture_materials,
    )
    _set_if_not_none(
        cfg.shape_program, "texture_reference_dir", args.texture_reference_dir
    )
    _set_if_not_none(cfg.shape_program, "uv_strict", args.uv_strict)
    _set_if_not_none(cfg.shape_program, "material_target", args.material_target)
    _set_if_not_none(
        cfg.shape_program,
        "max_texture_memory_mb",
        args.max_texture_memory_mb,
    )

    if args.ensemble_candidates:
        cfg.ensemble.candidates = _candidate_configs(args.ensemble_candidates)
    _set_if_not_none(cfg.ensemble, "selection_policy", args.ensemble_policy)
    _set_if_not_none(
        cfg.ensemble, "max_parallel_candidates", args.ensemble_max_parallel
    )
    _set_if_not_none(cfg.ensemble, "per_candidate_timeout_s", args.ensemble_timeout)
    _set_if_not_none(cfg.ensemble, "total_timeout_s", args.ensemble_total_timeout)
    _set_if_not_none(cfg.ensemble, "keep_all_artifacts", args.ensemble_keep_artifacts)
    _set_if_not_none(
        cfg.ensemble,
        "fail_if_no_candidate_passes_required_views",
        args.ensemble_fail_if_no_required_views,
    )

    if args.constraint_file:
        cfg.constraints.constraint_files = tuple(args.constraint_file)
    _set_if_not_none(
        cfg.constraints,
        "fail_on_unsatisfied_hard_constraints",
        args.fail_on_hard_constraints,
    )
    _set_if_not_none(
        cfg.constraints,
        "use_constraints_for_candidate_scoring",
        args.use_constraints_for_scoring,
    )
    _set_if_not_none(cfg.quality_budget, "budget_json", args.quality_budget_json)
    _set_if_not_none(
        cfg.quality_budget, "compare_baseline", args.quality_compare_baseline
    )
    _set_if_not_none(
        cfg.quality_budget, "fail_on_regression", args.quality_fail_on_regression
    )
    _set_if_not_none(
        cfg.quality_budget,
        "environment_compatibility",
        args.environment_compatibility,
    )

    _set_if_not_none(cfg.synthetic_factory, "suite", args.synthetic_suite)
    _set_if_not_none(cfg.synthetic_factory, "seed", args.synthetic_seed)
    _set_if_not_none(cfg.synthetic_factory, "output_root", args.synthetic_output_root)
    _set_if_not_none(
        cfg.synthetic_factory,
        "commit_small_fixtures_only",
        args.synthetic_commit_small_fixtures_only,
    )
    _set_if_not_none(
        cfg.synthetic_factory,
        "keep_heavy_artifacts",
        args.synthetic_keep_heavy_artifacts,
    )

    _set_if_not_none(cfg.refinement_lab, "default_suite", args.refinement_suite)
    _set_if_not_none(cfg.refinement_lab, "default_track", args.refinement_track)
    _set_if_not_none(cfg.refinement_lab, "default_search", args.refinement_search)
    _set_if_not_none(cfg.refinement_lab, "default_objective", args.refinement_objective)
    _set_if_not_none(
        cfg.refinement_lab,
        "default_output_root",
        str(args.refinement_result_root) if args.refinement_result_root else None,
    )
    _set_if_not_none(cfg.refinement_lab, "max_runs", args.refinement_max_runs)
    _set_if_not_none(cfg.refinement_lab, "top_k", args.refinement_top_k)
    _set_if_not_none(cfg.refinement_lab, "html_report", args.refinement_html_report)
    _set_if_not_none(
        cfg.refinement_lab, "write_overlays", args.refinement_write_overlays
    )
    _set_if_not_none(
        cfg.refinement_lab, "write_bounds_debug", args.refinement_bounds_debug
    )
    _set_if_not_none(cfg.refinement_lab, "write_autopsy", args.refinement_autopsy)
    _set_if_not_none(
        cfg.refinement_lab, "copy_references", args.refinement_copy_references
    )
    _set_if_not_none(
        cfg.refinement_lab, "fail_on_all_failed", args.refinement_fail_on_all_failed
    )
    _set_if_not_none(
        cfg.refinement_lab, "append_leaderboard", args.refinement_append_global_index
    )
    _set_if_not_none(
        cfg.refinement_lab, "report_failures", args.refinement_report_failures
    )
    _set_if_not_none(
        cfg.refinement_lab, "allow_subprocess_blender", args.refinement_subprocess
    )
    _set_if_not_none(
        cfg.refinement_lab, "blender_executable", args.refinement_blender_exe
    )
    if args.refinement_stop_on_first_error:
        cfg.refinement_lab.stop_on_first_error = True


def _derive_config_label(args: argparse.Namespace) -> str:
    if args.config_path:
        stem = Path(args.config_path).stem
        for mode in ALL_RECONSTRUCTION_MODES:
            prefix = f"{mode}-"
            if stem.startswith(prefix):
                return stem[len(prefix) :]
        parts = stem.split("-")
        return parts[-1] if len(parts) > 1 else stem
    if args.config_json:
        return "inline"
    return "default"
