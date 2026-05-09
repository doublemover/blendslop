"""Central refinement parameter to BlockingConfig path catalog."""

from __future__ import annotations

from typing import Mapping

CONFIG_PARAMETER_PATHS: Mapping[str, tuple[str, str]] = {
    "ref_polarity": ("silhouette_extract_ref", "polarity"),
    "ref_gray_threshold": ("silhouette_extract_ref", "gray_threshold"),
    "ref_morph_close": ("silhouette_extract_ref", "morph_close_px"),
    "ref_morph_open": ("silhouette_extract_ref", "morph_open_px"),
    "ref_min_component_area_px": ("silhouette_extract_ref", "min_component_area_px"),
    "ref_fill_holes": ("silhouette_extract_ref", "fill_holes"),
    "ref_largest_component": ("silhouette_extract_ref", "largest_component_only"),
    "canonical_padding_frac": ("canonicalize", "padding_frac"),
    "canonical_anchor": ("canonicalize", "anchor"),
    "profile_samples": ("profile_sampling", "num_samples"),
    "profile_sample_policy": ("profile_sampling", "sample_policy"),
    "profile_fill_strategy": ("profile_sampling", "fill_strategy"),
    "profile_smoothing_window": ("profile_sampling", "smoothing_window"),
    "mesh_radial_segments": ("mesh_from_profile", "radial_segments"),
    "mesh_min_radius": ("mesh_from_profile", "min_radius_u"),
    "mesh_merge_threshold": ("mesh_from_profile", "merge_threshold_u"),
    "mesh_cap_mode": ("mesh_from_profile", "cap_mode"),
    "mesh_adaptive_radial_segments": ("mesh_from_profile", "adaptive_radial_segments"),
    "mesh_shade_smooth": ("mesh_from_profile", "shade_smooth"),
    "vh_backend": ("visual_hull", "backend"),
    "vh_resolution": ("visual_hull", "resolution"),
    "vh_chunk_size": ("visual_hull", "chunk_size"),
    "vh_mesh_method": ("visual_hull", "mesh_method"),
    "vh_postprocess": ("visual_hull", "postprocess"),
    "vh_occupancy_threshold": ("visual_hull", "occupancy_threshold"),
    "vh_uncertainty_aggregation": ("visual_hull", "uncertainty_aggregation"),
    "primitive_families": ("primitive_fit", "primitive_families"),
    "primitive_target_points": ("primitive_fit", "target_point_count"),
    "primitive_min": ("primitive_fit", "min_primitives"),
    "primitive_max": ("primitive_fit", "max_primitives"),
    "primitive_steps": ("primitive_fit", "optimization_steps"),
    "primitive_fail_on_regression": ("primitive_fit", "fail_on_regression"),
    "primitive_loss_weights_json": ("primitive_fit", "loss_weights"),
    "primitive_max_runtime_s": ("primitive_fit", "max_runtime_s"),
    "primitive_max_objective_evaluations": (
        "primitive_fit",
        "max_objective_evaluations",
    ),
    "gaussian_count": ("gaussian_ellipsoid", "primitive_count"),
    "gaussian_initialization": ("gaussian_ellipsoid", "initialization"),
    "gaussian_min_radius": ("gaussian_ellipsoid", "min_radius"),
    "gaussian_opacity_min": ("gaussian_ellipsoid", "opacity_min"),
    "gaussian_opacity_max": ("gaussian_ellipsoid", "opacity_max"),
    "gaussian_export_mesh_proxy": ("gaussian_ellipsoid", "export_mesh_proxy"),
    "diff_backend": ("differentiable_render", "backend"),
    "diff_optional_policy": ("differentiable_render", "optional_dependency_policy"),
    "diff_gradient_mode": ("differentiable_render", "gradient_mode"),
    "diff_epsilon": ("differentiable_render", "finite_difference_epsilon"),
    "diff_loss_weights_json": ("differentiable_render", "loss_weights"),
    "ensemble_policy": ("ensemble", "selection_policy"),
    "shape_root_strategy": ("shape_program", "root_strategy"),
    "shape_residual_policy": ("shape_program", "residual_policy"),
    "shape_max_nodes": ("shape_program", "max_nodes"),
    "shape_editability_bias": ("shape_program", "editability_bias"),
    "shape_compile_blender": ("shape_program", "compile_blender"),
    "shape_lathe_segments": ("shape_program", "lathe_segments"),
    "shape_bevel_modifier": ("shape_program", "bevel_modifier"),
    "shape_weighted_normals": ("shape_program", "weighted_normals"),
    "shape_run_export_qa": ("shape_program", "run_export_qa"),
    "shape_export_qa_targets": ("shape_program", "export_qa_targets"),
    "shape_evaluate_texture_materials": (
        "shape_program",
        "evaluate_texture_materials",
    ),
    "shape_texture_reference_dir": ("shape_program", "texture_reference_dir"),
    "shape_uv_strict": ("shape_program", "uv_strict"),
    "shape_material_target": ("shape_program", "material_target"),
    "shape_max_texture_memory_mb": ("shape_program", "max_texture_memory_mb"),
}

SPECIAL_PARAMETER_NAMES = frozenset({"ensemble_candidates"})


def parameter_path(name: str) -> tuple[str, str] | None:
    """Return the BlockingConfig group/attribute path for a parameter."""
    return CONFIG_PARAMETER_PATHS.get(str(name))
