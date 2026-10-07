from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import math

from .dependencies import *


@dataclass
class VisualHullConfig:
    """Configuration for visual-hull reconstruction."""

    backend: str = "dense"
    resolution: int = 64
    max_resolution: int = 256
    chunk_size: Optional[int] = None
    adaptive_max_depth: int = 5
    adaptive_hull: bool = False
    preserve_thin_features: bool = False
    boundary_refine: bool = True
    mesh_method: str = "marching_cubes"
    postprocess: str = "none"
    postprocess_required: bool = False
    external_open3d_python: Optional[str] = None
    poisson_depth: int = 8
    poisson_density_quantile: Optional[float] = None
    poisson_timeout_s: float = 90.0
    poisson_crop_to_input_bounds: bool = True
    memory_budget_mb: Optional[int] = None
    occupancy_threshold: float = 0.5
    uncertainty_aggregation: str = "min"
    enable_cache: bool = False
    cache_directory: Optional[str] = None
    cache_namespace: str = "visual_hull"
    cache_read: bool = True
    cache_write: bool = True

    def validate(self) -> None:
        if not 1 <= self.poisson_depth <= 10 or not 0 < self.poisson_timeout_s <= 90:
            raise ValueError("Poisson depth must be 1..10 and helper timeout >0..90 seconds")
        if self.poisson_density_quantile is not None and not 0.0 <= self.poisson_density_quantile < 1.0:
            raise ValueError("poisson_density_quantile must be in [0, 1)")
        if self.backend not in _VALID_VOLUME_BACKENDS:
            raise ValueError(f"visual hull backend must be one of {_VALID_VOLUME_BACKENDS}")
        if self.resolution < 1:
            raise ValueError("visual hull resolution must be >= 1")
        if self.max_resolution < self.resolution:
            raise ValueError("max_resolution must be >= resolution")
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1 when provided")
        if self.adaptive_max_depth < 0:
            raise ValueError("adaptive_max_depth must be >= 0")
        if self.mesh_method not in _VALID_MESH_METHODS:
            raise ValueError(f"mesh_method must be one of {_VALID_MESH_METHODS}")
        if self.postprocess not in _VALID_POSTPROCESS:
            raise ValueError(f"postprocess must be one of {_VALID_POSTPROCESS}")
        if self.memory_budget_mb is not None and self.memory_budget_mb < 1:
            raise ValueError("memory_budget_mb must be >= 1 when provided")
        if not (0.0 <= self.occupancy_threshold <= 1.0):
            raise ValueError("occupancy_threshold must be in [0, 1]")
        if self.uncertainty_aggregation not in {"min", "product", "logit_sum"}:
            raise ValueError("uncertainty_aggregation must be min/product/logit_sum")
        if not self.cache_namespace:
            raise ValueError("cache_namespace must not be empty")
        if self.cache_directory is not None and not str(self.cache_directory).strip():
            raise ValueError("cache_directory must not be blank when provided")

    def to_dict(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "resolution": self.resolution,
            "max_resolution": self.max_resolution,
            "chunk_size": self.chunk_size,
            "adaptive_max_depth": self.adaptive_max_depth,
            "adaptive_hull": self.adaptive_hull,
            "preserve_thin_features": self.preserve_thin_features,
            "boundary_refine": self.boundary_refine,
            "mesh_method": self.mesh_method,
            "postprocess": self.postprocess,
            "postprocess_required": self.postprocess_required,
            "external_open3d_python": self.external_open3d_python,
            "poisson_depth": self.poisson_depth,
            "poisson_density_quantile": self.poisson_density_quantile,
            "poisson_timeout_s": self.poisson_timeout_s,
            "poisson_crop_to_input_bounds": self.poisson_crop_to_input_bounds,
            "memory_budget_mb": self.memory_budget_mb,
            "occupancy_threshold": self.occupancy_threshold,
            "uncertainty_aggregation": self.uncertainty_aggregation,
            "enable_cache": self.enable_cache,
            "cache_directory": self.cache_directory,
            "cache_namespace": self.cache_namespace,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
        }

@dataclass
class VolumeConfig:
    """Configuration for volume serialization and interchange."""

    backend: str = "dense"
    sparse_chunk_size: int = 32
    serialization: str = "npz"
    export_openvdb: bool = False

    def validate(self) -> None:
        if self.backend not in _VALID_VOLUME_BACKENDS:
            raise ValueError(f"volume backend must be one of {_VALID_VOLUME_BACKENDS}")
        if self.sparse_chunk_size < 1:
            raise ValueError("sparse_chunk_size must be >= 1")
        if self.serialization not in {"npz"}:
            raise ValueError("serialization must be npz")

    def to_dict(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "sparse_chunk_size": self.sparse_chunk_size,
            "serialization": self.serialization,
            "export_openvdb": self.export_openvdb,
        }

@dataclass
class CandidateConfig:
    """One backend candidate in an ensemble run."""

    backend_name: str
    enabled: bool = True
    candidate_id: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.backend_name:
            raise ValueError("candidate backend_name is required")

    def to_dict(self) -> Dict[str, object]:
        return {
            "backend_name": self.backend_name,
            "enabled": self.enabled,
            "candidate_id": self.candidate_id,
            "config": self.config,
        }

@dataclass
class EnsembleConfig:
    """Configuration for candidate ensemble reconstruction."""

    candidates: Tuple[CandidateConfig, ...] = field(default_factory=tuple)
    selection_policy: str = "best_score"
    evidence_routing: bool = False
    native_resident: bool = True
    projection_diagnostics: bool = False
    diagnostic_allocations: bool = False
    max_render_candidates: int = 3
    max_parallel_candidates: int = 2
    per_candidate_timeout_s: Optional[float] = None
    total_timeout_s: Optional[float] = None
    keep_all_artifacts: bool = True
    fail_if_no_candidate_passes_required_views: bool = True

    def validate(self) -> None:
        if self.max_render_candidates < 1:
            raise ValueError("measured routing requires at least one render candidate")
        if self.selection_policy not in _VALID_SELECTION_POLICIES:
            raise ValueError(
                f"selection_policy must be one of {_VALID_SELECTION_POLICIES}"
            )
        if self.max_parallel_candidates < 1:
            raise ValueError("max_parallel_candidates must be >= 1")
        if self.per_candidate_timeout_s is not None and self.per_candidate_timeout_s <= 0:
            raise ValueError("per_candidate_timeout_s must be > 0 when provided")
        if self.total_timeout_s is not None and self.total_timeout_s <= 0:
            raise ValueError("total_timeout_s must be > 0 when provided")
        for candidate in self.candidates:
            candidate.validate()

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selection_policy": self.selection_policy,
            "evidence_routing": self.evidence_routing,
            "native_resident": self.native_resident,
            "projection_diagnostics": self.projection_diagnostics,
            "diagnostic_allocations": self.diagnostic_allocations,
            "max_render_candidates": self.max_render_candidates,
            "max_parallel_candidates": self.max_parallel_candidates,
            "per_candidate_timeout_s": self.per_candidate_timeout_s,
            "total_timeout_s": self.total_timeout_s,
            "keep_all_artifacts": self.keep_all_artifacts,
            "fail_if_no_candidate_passes_required_views": self.fail_if_no_candidate_passes_required_views,
        }

@dataclass
class PrimitiveFitConfig:
    """Configuration for primitive fitting/refinement."""

    primitive_families: Tuple[str, ...] = ("superquadric", "superfrustum", "ellipsoid")
    silhouette_objective: str = "profile_rows"
    objective_mode: Optional[str] = None
    refinement_strategy: Optional[str] = None
    whole_support_search: Optional[bool] = None
    target_point_count: int = 2048
    min_primitives: int = 1
    max_primitives: int = 6
    optimization_steps: int = 12
    checkpoint_cadence: int = 10
    fail_on_regression: bool = True
    max_runtime_s: Optional[float] = 8.0
    max_objective_evaluations: Optional[int] = 256
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "surface_residual": 1.0,
            "visual_hull_occupancy": 0.08,
            "primitive_count": 0.01,
            "overlap_penalty": 0.04,
            "silhouette": 0.08,
            "topology_penalty": 0.03,
            "constraint_penalty": 0.05,
            "uncertainty_penalty": 0.02,
        }
    )

    kmeans_seed: str = "height"
    residual_rounds: int = 1
    residual_refinement_steps: int = 1
    max_residual_proposals: int = 3
    max_multistart_attempts: int = 2

    def validate(self) -> None:
        if self.objective_mode not in {None,"legacy_world_squared","normalized_area_v1"}:
            raise ValueError("unsupported primitive_fit objective_mode")
        if self.refinement_strategy not in {None,"coordinate","coupled_blocks"}:
            raise ValueError("unsupported primitive_fit refinement_strategy")
        if self.whole_support_search is not None and not isinstance(self.whole_support_search,bool):
            raise ValueError("whole_support_search must be bool when specified")
        if min(self.residual_rounds, self.residual_refinement_steps, self.max_residual_proposals) < 0:
            raise ValueError("residual search limits must be nonnegative")
        if not 1 <= self.max_multistart_attempts <= 16:
            raise ValueError("max_multistart_attempts must be in [1,16]")
        if self.kmeans_seed not in {"height", "farthest"}:
            raise ValueError("kmeans_seed must be height or farthest")
        if self.silhouette_objective not in {"profile_rows", "mesh_union"}:
            raise ValueError("silhouette_objective must be profile_rows or mesh_union")
        if not self.primitive_families:
            raise ValueError("primitive_families cannot be empty")
        if self.target_point_count < 1:
            raise ValueError("target_point_count must be >= 1")
        if self.min_primitives < 0 or self.max_primitives < self.min_primitives:
            raise ValueError("primitive count bounds are invalid")
        if self.optimization_steps < 0:
            raise ValueError("optimization_steps must be >= 0")
        if self.checkpoint_cadence < 1:
            raise ValueError("checkpoint_cadence must be >= 1")
        if self.max_runtime_s is not None and self.max_runtime_s <= 0:
            raise ValueError("max_runtime_s must be > 0 when provided")
        if (
            self.max_objective_evaluations is not None
            and self.max_objective_evaluations < 1
        ):
            raise ValueError("max_objective_evaluations must be >= 1 when provided")

    def to_dict(self) -> Dict[str, object]:
        return {
            "kmeans_seed": self.kmeans_seed,
            "residual_rounds": self.residual_rounds,
            "residual_refinement_steps": self.residual_refinement_steps,
            "max_residual_proposals": self.max_residual_proposals,
            "max_multistart_attempts": self.max_multistart_attempts,
            "primitive_families": list(self.primitive_families),
            "silhouette_objective": self.silhouette_objective,
            "objective_mode": self.objective_mode,
            "refinement_strategy": self.refinement_strategy,
            "whole_support_search": self.whole_support_search,
            "target_point_count": self.target_point_count,
            "min_primitives": self.min_primitives,
            "max_primitives": self.max_primitives,
            "optimization_steps": self.optimization_steps,
            "checkpoint_cadence": self.checkpoint_cadence,
            "fail_on_regression": self.fail_on_regression,
            "max_runtime_s": self.max_runtime_s,
            "max_objective_evaluations": self.max_objective_evaluations,
            "loss_weights": self.loss_weights,
        }

@dataclass
class GaussianEllipsoidConfig:
    """Configuration for Gaussian/ellipsoid proxy reconstruction."""

    primitive_count: int = 24
    cluster_sigma: float = 1.0
    negative_space_seed_guard: bool = False
    proxy_variant: Optional[str] = None
    proxy_fit_resolution: int = 48
    proxy_fit_evaluations: int = 192
    proxy_fit_iterations: int = 2
    initialization: str = "farthest_point"
    min_radius: float = 1e-4
    max_radius: Optional[float] = None
    opacity_min: float = 0.0
    opacity_max: float = 1.0
    renderer: str = "cpu_projected_ellipse"
    export_mesh_proxy: bool = True

    kmeans_seed: str = "height"

    def validate(self) -> None:
        if self.proxy_variant not in {None, 'initializer_only', 'fitted_opaque_union_v1'}:
            raise ValueError('unsupported proxy_variant')
        if not (1 <= self.proxy_fit_resolution <= 256 and 1 <= self.proxy_fit_evaluations <= 4096 and 1 <= self.proxy_fit_iterations <= 16):
            raise ValueError('fitted proxy budgets are out of bounds')
        if self.kmeans_seed not in {"height", "farthest"}:
            raise ValueError("kmeans_seed must be height or farthest")
        if self.primitive_count < 1:
            raise ValueError("primitive_count must be >= 1")
        if not (0.5 <= self.cluster_sigma <= 3.0):
            raise ValueError("cluster_sigma must be finite and in [0.5, 3.0]")
        if self.initialization not in {"farthest_point", "kmeans", "grid"}:
            raise ValueError("initialization must be farthest_point/kmeans/grid")
        if self.min_radius <= 0:
            raise ValueError("min_radius must be > 0")
        if self.max_radius is not None and self.max_radius < self.min_radius:
            raise ValueError("max_radius must be >= min_radius")
        if not (0.0 <= self.opacity_min <= self.opacity_max <= 1.0):
            raise ValueError("opacity bounds must be in [0, 1]")
        if self.renderer not in {"cpu_projected_ellipse", "gpu_splat"}:
            raise ValueError("renderer must be cpu_projected_ellipse/gpu_splat")

    def to_dict(self) -> Dict[str, object]:
        return {
            "kmeans_seed": self.kmeans_seed,
            "primitive_count": self.primitive_count,
            "cluster_sigma": self.cluster_sigma,
            "negative_space_seed_guard": self.negative_space_seed_guard,
            "proxy_variant": self.proxy_variant,
            "proxy_fit_resolution": self.proxy_fit_resolution,
            "proxy_fit_evaluations": self.proxy_fit_evaluations,
            "proxy_fit_iterations": self.proxy_fit_iterations,
            "initialization": self.initialization,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
            "opacity_min": self.opacity_min,
            "opacity_max": self.opacity_max,
            "renderer": self.renderer,
            "export_mesh_proxy": self.export_mesh_proxy,
        }

@dataclass
class DifferentiableRenderConfig:
    """Configuration for optional differentiable-rendering refinement."""

    backend: str = "cpu_soft_silhouette"
    optional_dependency_policy: str = "skip"
    gradient_mode: str = "finite_difference"
    finite_difference_epsilon: float = 1e-4
    softness: float = 72.0
    primitive_opacity_floor: float = 0.95
    mesh_proxy_scale: float = 1.0
    pixel_evidence_mode: Optional[str] = None
    optimization_resolution: int = 128
    finest_finishing: bool = True
    dvx_execution_approved: bool = False
    dvx_helper_python: Optional[str] = None
    dvx_resolution: int = 32
    dvx_steps: int = 12
    dvx_objective: Optional[str] = None
    dvx_parameterization: Optional[str] = None
    dvx_grid_levels: Optional[list[int]] = None
    dvx_seed_vertex_limit: int = 4096
    dvx_target_quadrature: int = 2
    dvx_ray_filter_mode: Optional[str] = None
    dvx_allow_subvoxel_ray_surrogate: bool = False
    dvx_differential_strength: float = 4.0
    dvx_warm_helper: Optional[bool] = None
    silhouette_bounds_padding: float = 0.92
    primitive_count: Optional[int] = 12
    target_point_count: int = 2048
    visual_hull_resolution: Optional[int] = None
    optimization_steps: int = 6
    optimization_initial_step: float = 0.05
    optimization_step_decay: float = 0.5
    optimization_min_step: float = 1.0e-4
    max_objective_evaluations: Optional[int] = 256
    max_runtime_s: Optional[float] = 20.0
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "silhouette_l2": 0.6,
            "soft_iou": 1.0,
            "area_iou": 0.45,
            "boundary_iou": 1.25,
            "signed_distance": 0.65,
            "depth_l2": 0.0,
        }
    )

    kmeans_seed: str = "farthest"

    def validate(self) -> None:
        if self.pixel_evidence_mode not in {None,'view_mean_legacy','pixel_reliability_v1'}:
            raise ValueError('unsupported pixel_evidence_mode')
        if self.kmeans_seed not in {"height", "farthest"}:
            raise ValueError("kmeans_seed must be height or farthest")
        if self.dvx_warm_helper is not None and not isinstance(self.dvx_warm_helper,bool):
            raise ValueError("dvx_warm_helper must be bool when specified")
        if self.dvx_objective not in {None,"legacy_occupancy","filtered_occupancy","observed_rays","observed_projected_rays"}:
            raise ValueError("unsupported dvx_objective")
        if self.dvx_ray_filter_mode not in {None,'pixel_cell_box_exact','subcell_quadrature'}:
            raise ValueError('unsupported dvx_ray_filter_mode')
        if self.dvx_parameterization not in {None,"vertices","cage","differential"}:
            raise ValueError("unsupported dvx_parameterization")
        if self.dvx_grid_levels is not None and (not self.dvx_grid_levels or
            sorted(set(self.dvx_grid_levels)) != list(self.dvx_grid_levels) or
            any(n not in {16,32,64} for n in self.dvx_grid_levels) or self.dvx_grid_levels[-1] != self.dvx_resolution):
            raise ValueError("dvx_grid_levels must increase to dvx_resolution")
        if self.dvx_seed_vertex_limit<4 or self.dvx_target_quadrature not in {1,2,3}:
            raise ValueError("invalid DVX seed/target preparation allowance")
        if self.dvx_differential_strength<0 or not math.isfinite(self.dvx_differential_strength):
            raise ValueError("dvx_differential_strength must be finite and nonnegative")
        if self.backend not in {"cpu_soft_silhouette", "blender_finite_difference", "nvdiffrast", "dvx"}:
            raise ValueError("invalid differentiable render backend")
        if self.optional_dependency_policy not in {"skip", "fail"}:
            raise ValueError("optional_dependency_policy must be skip/fail")
        if self.gradient_mode not in {"finite_difference", "backend"}:
            raise ValueError("gradient_mode must be finite_difference/backend")
        if self.finite_difference_epsilon <= 0:
            raise ValueError("finite_difference_epsilon must be > 0")
        if self.softness <= 0:
            raise ValueError("differentiable_render.softness must be > 0")
        if not (0.0 <= self.primitive_opacity_floor <= 1.0):
            raise ValueError("differentiable_render.primitive_opacity_floor must be in [0, 1]")
        if not (0.05 <= self.mesh_proxy_scale <= 10.0):
            raise ValueError("mesh_proxy_scale must be finite and in [0.05, 10.0]")
        if not (32 <= self.optimization_resolution <= 512):
            raise ValueError("optimization_resolution must be in [32, 512]")
        if self.silhouette_bounds_padding <= 0:
            raise ValueError("differentiable_render.silhouette_bounds_padding must be > 0")
        if self.primitive_count is not None and self.primitive_count < 1:
            raise ValueError("differentiable_render.primitive_count must be >= 1 when provided")
        if self.target_point_count < 1:
            raise ValueError("differentiable_render.target_point_count must be >= 1")
        if self.visual_hull_resolution is not None and self.visual_hull_resolution < 1:
            raise ValueError("differentiable_render.visual_hull_resolution must be >= 1 when provided")
        if self.optimization_steps < 0:
            raise ValueError("differentiable_render.optimization_steps must be >= 0")
        if self.optimization_initial_step <= 0.0:
            raise ValueError("differentiable_render.optimization_initial_step must be > 0")
        if not (0.0 < self.optimization_step_decay < 1.0):
            raise ValueError("differentiable_render.optimization_step_decay must be in (0, 1)")
        if self.optimization_min_step <= 0.0 or self.optimization_min_step >= self.optimization_initial_step:
            raise ValueError(
                "differentiable_render.optimization_min_step must be > 0 and "
                "< optimization_initial_step"
            )
        if (
            self.max_objective_evaluations is not None
            and self.max_objective_evaluations < 1
        ):
            raise ValueError("differentiable_render.max_objective_evaluations must be >= 1 when provided")
        if self.max_runtime_s is not None and self.max_runtime_s <= 0.0:
            raise ValueError("differentiable_render.max_runtime_s must be > 0 when provided")

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "backend": self.backend,
            "optional_dependency_policy": self.optional_dependency_policy,
            "gradient_mode": self.gradient_mode,
            "finite_difference_epsilon": self.finite_difference_epsilon,
            "softness": self.softness,
            "primitive_opacity_floor": self.primitive_opacity_floor,
            "mesh_proxy_scale": self.mesh_proxy_scale,
            "pixel_evidence_mode": self.pixel_evidence_mode,
            "optimization_resolution": self.optimization_resolution,
            "finest_finishing": self.finest_finishing,
            "dvx_execution_approved": self.dvx_execution_approved,
            "dvx_helper_python": self.dvx_helper_python,
            "dvx_resolution": self.dvx_resolution,
            "dvx_steps": self.dvx_steps,
            "dvx_objective": self.dvx_objective,
            "dvx_parameterization": self.dvx_parameterization,
            "dvx_grid_levels": self.dvx_grid_levels,
            "dvx_seed_vertex_limit": self.dvx_seed_vertex_limit,
            "dvx_target_quadrature": self.dvx_target_quadrature,
            "dvx_ray_filter_mode": self.dvx_ray_filter_mode,
            "dvx_allow_subvoxel_ray_surrogate": self.dvx_allow_subvoxel_ray_surrogate,
            "dvx_differential_strength": self.dvx_differential_strength,
            "dvx_warm_helper": self.dvx_warm_helper,
            "silhouette_bounds_padding": self.silhouette_bounds_padding,
            "target_point_count": self.target_point_count,
            "optimization_steps": self.optimization_steps,
            "optimization_initial_step": self.optimization_initial_step,
            "optimization_step_decay": self.optimization_step_decay,
            "optimization_min_step": self.optimization_min_step,
            "max_objective_evaluations": self.max_objective_evaluations,
            "max_runtime_s": self.max_runtime_s,
            "loss_weights": self.loss_weights,
        }
        if self.primitive_count is not None:
            payload["primitive_count"] = self.primitive_count
        if self.visual_hull_resolution is not None:
            payload["visual_hull_resolution"] = self.visual_hull_resolution
        return payload

@dataclass
class ShapeProgramConfig:
    """Configuration for editable shape-program research output."""

    program_search_candidates: int = 4
    program_timeout_s: float = 45.
    program_refinement_steps: int = 1
    program_refinement_trials: int = 6
    structural_search: bool = False
    subtractive_search: bool = False
    cuboid_search: bool = False
    convex_proxy_search: bool = False
    generalized_sweep_search: Optional[bool] = None
    root_strategy: str = "hybrid_profile_bounds"
    residual_policy: str = "suggest_patches"
    max_nodes: int = 64
    editability_bias: float = 1.0
    compile_blender: bool = True
    lathe_segments: int = 48
    bevel_modifier: bool = True
    weighted_normals: bool = True
    run_export_qa: bool = False
    export_qa_targets: Tuple[str, ...] = ("obj", "glb")
    evaluate_texture_materials: bool = False
    texture_reference_dir: Optional[str] = None
    uv_strict: bool = False
    material_target: str = "pbr"
    max_texture_memory_mb: Optional[float] = None

    def validate(self) -> None:
        if not 1 <= self.program_search_candidates <= 24 or not 0 <= self.program_refinement_steps <= 4 or not 1 <= self.program_refinement_trials <= 24 or not 0 < self.program_timeout_s <= 90:
            raise ValueError("shape program search must remain within bounded candidates/steps/time")
        if self.root_strategy not in {
            "profile_lathe",
            "bounds_box",
            "hybrid_profile_bounds",
        }:
            raise ValueError(
                "shape_program.root_strategy must be profile_lathe/"
                "bounds_box/hybrid_profile_bounds"
            )
        if self.residual_policy not in {"ignore", "report", "suggest_patches"}:
            raise ValueError(
                "shape_program.residual_policy must be ignore/report/suggest_patches"
            )
        if self.max_nodes < 1:
            raise ValueError("shape_program.max_nodes must be >= 1")
        if not (0.0 <= self.editability_bias <= 1.0):
            raise ValueError("shape_program.editability_bias must be in [0, 1]")
        if self.lathe_segments < 8:
            raise ValueError("shape_program.lathe_segments must be >= 8")
        if not isinstance(self.run_export_qa, bool):
            raise ValueError("shape_program.run_export_qa must be a boolean")
        invalid_targets = set(self.export_qa_targets) - {"obj", "glb", "gltf"}
        if invalid_targets:
            raise ValueError(
                "shape_program.export_qa_targets must contain only obj/glb/gltf"
            )
        if not isinstance(self.evaluate_texture_materials, bool):
            raise ValueError("shape_program.evaluate_texture_materials must be a boolean")
        if not isinstance(self.uv_strict, bool):
            raise ValueError("shape_program.uv_strict must be a boolean")
        if self.material_target not in {"pbr", "simple", "none"}:
            raise ValueError("shape_program.material_target must be pbr/simple/none")
        if self.texture_reference_dir is not None and not str(self.texture_reference_dir).strip():
            raise ValueError("shape_program.texture_reference_dir must not be blank")
        if self.max_texture_memory_mb is not None and self.max_texture_memory_mb <= 0.0:
            raise ValueError("shape_program.max_texture_memory_mb must be > 0 when provided")

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "root_strategy": self.root_strategy,
            "program_search_candidates": self.program_search_candidates,
            "program_refinement_steps": self.program_refinement_steps,
            "program_timeout_s": self.program_timeout_s,
            "program_refinement_trials": self.program_refinement_trials,
            "structural_search": self.structural_search,
            "subtractive_search": self.subtractive_search,
            "cuboid_search": self.cuboid_search,
            "convex_proxy_search": self.convex_proxy_search,
            "generalized_sweep_search": self.generalized_sweep_search,
            "residual_policy": self.residual_policy,
            "max_nodes": self.max_nodes,
            "editability_bias": self.editability_bias,
            "compile_blender": self.compile_blender,
            "lathe_segments": self.lathe_segments,
            "bevel_modifier": self.bevel_modifier,
            "weighted_normals": self.weighted_normals,
            "run_export_qa": self.run_export_qa,
            "export_qa_targets": list(self.export_qa_targets),
            "evaluate_texture_materials": self.evaluate_texture_materials,
            "uv_strict": self.uv_strict,
            "material_target": self.material_target,
        }
        if self.texture_reference_dir is not None:
            payload["texture_reference_dir"] = self.texture_reference_dir
        if self.max_texture_memory_mb is not None:
            payload["max_texture_memory_mb"] = self.max_texture_memory_mb
        return payload

@dataclass
class ConstraintConfig:
    """Configuration for human correction constraints."""

    constraint_files: Tuple[str, ...] = ()
    fail_on_unsatisfied_hard_constraints: bool = True
    use_constraints_for_candidate_scoring: bool = True

    def validate(self) -> None:
        if any(not path for path in self.constraint_files):
            raise ValueError("constraint_files cannot contain empty paths")

    def to_dict(self) -> Dict[str, object]:
        return {
            "constraint_files": list(self.constraint_files),
            "fail_on_unsatisfied_hard_constraints": self.fail_on_unsatisfied_hard_constraints,
            "use_constraints_for_candidate_scoring": self.use_constraints_for_candidate_scoring,
        }

@dataclass
class SilhouetteIntersectionConfig:
    """Configuration for silhouette intersection reconstruction."""

    extrude_distance: float = 1.0
    contour_mode: str = "external"
    largest_component_only: Optional[bool] = None
    silhouette_extract_override: Optional[Dict[str, object]] = None
    boolean_solver: str = "auto"

    def validate(self) -> None:
        """Validate configuration values."""
        if self.extrude_distance <= 0:
            raise ValueError("extrude_distance must be > 0")
        if self.contour_mode not in _VALID_CONTOUR_MODES:
            raise ValueError(f"contour_mode must be one of {_VALID_CONTOUR_MODES}")
        if self.boolean_solver not in _VALID_BOOLEAN_SOLVERS:
            raise ValueError(f"boolean_solver must be one of {_VALID_BOOLEAN_SOLVERS}")
        if self.silhouette_extract_override is not None:
            if not isinstance(self.silhouette_extract_override, dict):
                raise ValueError("silhouette_extract_override must be a dict or None")
            valid_keys = set(SilhouetteExtractConfig().to_dict().keys())
            invalid = set(self.silhouette_extract_override.keys()) - valid_keys
            if invalid:
                raise ValueError(
                    f"silhouette_extract_override has invalid keys: {sorted(invalid)}"
                )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "extrude_distance": self.extrude_distance,
            "contour_mode": self.contour_mode,
            "largest_component_only": self.largest_component_only,
            "silhouette_extract_override": self.silhouette_extract_override,
            "boolean_solver": self.boolean_solver,
        }

@dataclass
class ReconstructionConfig:
    """Top-level reconstruction settings."""

    quality_preset: str = "default"
    reconstruction_mode: str = "legacy"
    unit_scale: float = 0.01
    num_slices: int = 10
    view_calibration: Dict[str, Any] = field(default_factory=dict)
    valid_evidence_files: Dict[str, str] = field(default_factory=dict)
    view_crops: Dict[str, Any] = field(default_factory=dict)
    native_batch_queries: bool = False
    native_union_execution: bool = False
    native_union_solver: str = "EXACT"
    native_sdf_fallback: bool = False
    native_qualification_python: Optional[str] = None
    native_qualification_timeout_s: float = 15.0
    native_feature_thickness: Optional[float] = None

    def validate(self) -> None:
        """Validate configuration values."""
        from reconstruction.projection_contract import validate_view_calibration
        validate_view_calibration(self.view_calibration)
        if self.native_union_solver not in {"EXACT", "MANIFOLD"}:
            raise ValueError("native_union_solver must be EXACT or MANIFOLD")
        if self.native_qualification_timeout_s <= 0:
            raise ValueError("native qualification timeout must be positive")
        if self.native_feature_thickness is not None and self.native_feature_thickness <= 0:
            raise ValueError("native feature thickness must be positive when supplied")
        if set(self.valid_evidence_files) - {"front", "side", "top"} or set(self.view_crops) - {"front", "side", "top"}:
            raise ValueError("evidence masks/crops must use canonical view names")
        if self.quality_preset not in {"default", "quality"}:
            raise ValueError("quality_preset must be default or quality")
        if self.reconstruction_mode not in _VALID_RECON_MODES:
            raise ValueError(f"reconstruction_mode must be one of {_VALID_RECON_MODES}")
        if self.unit_scale < 0:
            raise ValueError("unit_scale must be >= 0")
        if self.num_slices < 1:
            raise ValueError("num_slices must be >= 1")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "reconstruction_mode": self.reconstruction_mode,
            "quality_preset": self.quality_preset,
            "unit_scale": self.unit_scale,
            "num_slices": self.num_slices,
            "view_calibration": self.view_calibration,
            "valid_evidence_files": dict(self.valid_evidence_files),
            "view_crops": dict(self.view_crops),
            "native_batch_queries": self.native_batch_queries,
            "native_union_execution": self.native_union_execution,
            "native_union_solver": self.native_union_solver,
            "native_sdf_fallback": self.native_sdf_fallback,
            "native_qualification_python": self.native_qualification_python,
            "native_qualification_timeout_s": self.native_qualification_timeout_s,
            "native_feature_thickness": self.native_feature_thickness,
        }
