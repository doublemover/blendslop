from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .dependencies import *


@dataclass
class VisualHullConfig:
    """Configuration for visual-hull reconstruction."""

    backend: str = "dense"
    resolution: int = 64
    max_resolution: int = 256
    chunk_size: Optional[int] = None
    adaptive_max_depth: int = 5
    boundary_refine: bool = True
    mesh_method: str = "marching_cubes"
    postprocess: str = "none"
    memory_budget_mb: Optional[int] = None
    occupancy_threshold: float = 0.5
    uncertainty_aggregation: str = "min"
    enable_cache: bool = False
    cache_directory: Optional[str] = None
    cache_namespace: str = "visual_hull"
    cache_read: bool = True
    cache_write: bool = True

    def validate(self) -> None:
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
            "boundary_refine": self.boundary_refine,
            "mesh_method": self.mesh_method,
            "postprocess": self.postprocess,
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
    max_parallel_candidates: int = 1
    per_candidate_timeout_s: Optional[float] = None
    total_timeout_s: Optional[float] = None
    keep_all_artifacts: bool = True
    fail_if_no_candidate_passes_required_views: bool = True

    def validate(self) -> None:
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
            "max_parallel_candidates": self.max_parallel_candidates,
            "per_candidate_timeout_s": self.per_candidate_timeout_s,
            "total_timeout_s": self.total_timeout_s,
            "keep_all_artifacts": self.keep_all_artifacts,
            "fail_if_no_candidate_passes_required_views": self.fail_if_no_candidate_passes_required_views,
        }

@dataclass
class PrimitiveFitConfig:
    """Configuration for primitive fitting/refinement."""

    primitive_families: Tuple[str, ...] = ("superfrustum", "ellipsoid")
    target_point_count: int = 4096
    min_primitives: int = 1
    max_primitives: int = 16
    optimization_steps: int = 50
    checkpoint_cadence: int = 10
    fail_on_regression: bool = True
    max_runtime_s: Optional[float] = 20.0
    max_objective_evaluations: Optional[int] = 768
    loss_weights: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
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
            "primitive_families": list(self.primitive_families),
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
    initialization: str = "farthest_point"
    min_radius: float = 1e-4
    max_radius: Optional[float] = None
    opacity_min: float = 0.0
    opacity_max: float = 1.0
    renderer: str = "cpu_projected_ellipse"
    export_mesh_proxy: bool = True

    def validate(self) -> None:
        if self.primitive_count < 1:
            raise ValueError("primitive_count must be >= 1")
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
            "primitive_count": self.primitive_count,
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
    primitive_count: Optional[int] = None
    target_point_count: int = 2048
    visual_hull_resolution: Optional[int] = None
    optimization_steps: int = 6
    optimization_initial_step: float = 0.05
    optimization_step_decay: float = 0.5
    optimization_min_step: float = 1.0e-4
    max_objective_evaluations: Optional[int] = 256
    max_runtime_s: Optional[float] = 20.0
    loss_weights: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.backend not in {"cpu_soft_silhouette", "blender_finite_difference", "nvdiffrast"}:
            raise ValueError("invalid differentiable render backend")
        if self.optional_dependency_policy not in {"skip", "fail"}:
            raise ValueError("optional_dependency_policy must be skip/fail")
        if self.gradient_mode not in {"finite_difference", "backend"}:
            raise ValueError("gradient_mode must be finite_difference/backend")
        if self.finite_difference_epsilon <= 0:
            raise ValueError("finite_difference_epsilon must be > 0")
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

    reconstruction_mode: str = "legacy"
    unit_scale: float = 0.01
    num_slices: int = 10

    def validate(self) -> None:
        """Validate configuration values."""
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
            "unit_scale": self.unit_scale,
            "num_slices": self.num_slices,
        }
