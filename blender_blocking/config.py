"""Configuration models for the Blender automated blocking tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


_VALID_RECON_MODES = {
    "legacy",
    "loft_profile",
    "profile_loft",
    "silhouette_intersection",
    "visual_hull_voxel",
    "hybrid_loft_hull",
    "primitive_fit_refine",
    "gaussian_ellipsoid_proxy",
    "differentiable_refine",
    "ensemble",
}
_VALID_JOIN_MODES = {"auto", "boolean", "voxel", "simple"}
_VALID_CAP_MODES = {"fan", "none", "ngon"}
_VALID_SAMPLE_POLICIES = {"endpoints", "cell_centers"}
_VALID_FILL_STRATEGIES = {"interp_linear", "interp_nearest", "constant"}
_VALID_CANON_ANCHORS = {"center", "bottom_center"}
_VALID_CANON_INTERP = {"nearest"}
_VALID_RENDER_ENGINES = {"BLENDER_EEVEE", "WORKBENCH"}
_VALID_COLOR_MODES = {"BW", "RGBA"}
_VALID_CONTOUR_MODES = {"external", "ccomp", "tree", "hierarchy"}
_VALID_BOOLEAN_SOLVERS = {"auto", "EXACT", "MANIFOLD", "FLOAT", "FAST"}
_VALID_INVERT_POLICIES = {"auto", "invert", "no_invert"}
_VALID_POLARITIES = {"auto", "dark_foreground", "light_foreground", "alpha_foreground"}
_VALID_MESH_METHODS = {"marching_cubes", "lewiner", "dual_contouring", "points"}
_VALID_VOLUME_BACKENDS = {"dense", "chunked", "sparse_hash", "openvdb"}
_VALID_SELECTION_POLICIES = {
    "best_score",
    "quality_first",
    "editability_first",
    "fast_preview",
    "pareto",
}
_VALID_POSTPROCESS = {"none", "poisson", "screened_poisson"}
_VALID_REFINEMENT_SEARCH = {"grid", "random", "coordinate", "successive_halving"}
_VALID_REFINEMENT_OBJECTIVES = {
    "quality_win",
    "min_view_iou",
    "mean_iou",
    "profile_editable",
    "visual_hull_alignment",
    "fast_preview",
    "human_adjusted",
}
_VALID_REFINEMENT_REPORT_FAILURES = {"top", "all", "none"}


@dataclass
class SilhouetteExtractConfig:
    """Configuration for silhouette extraction from images."""

    prefer_alpha: bool = True
    alpha_threshold: int = 127
    alpha_min_coverage: float = 0.001
    gray_threshold: Optional[int] = None
    invert_policy: str = "auto"
    polarity: str = "auto"
    min_area_frac: float = 0.0001
    max_area_frac: float = 0.98
    max_border_contact_frac: float = 0.95
    morph_close_px: int = 0
    morph_open_px: int = 0
    fill_holes: bool = True
    largest_component_only: bool = True
    min_component_area_px: int = 0
    candidate_scoring: bool = True
    emit_uncertainty: bool = True

    def validate(self) -> None:
        """Validate configuration values."""
        if not (0 <= self.alpha_threshold <= 255):
            raise ValueError("alpha_threshold must be in [0, 255]")
        if not (0.0 <= self.alpha_min_coverage <= 1.0):
            raise ValueError("alpha_min_coverage must be in [0, 1]")
        if self.gray_threshold is not None and not (0 <= self.gray_threshold <= 255):
            raise ValueError("gray_threshold must be in [0, 255] when provided")
        if self.invert_policy not in _VALID_INVERT_POLICIES:
            raise ValueError(f"invert_policy must be one of {_VALID_INVERT_POLICIES}")
        if self.polarity not in _VALID_POLARITIES:
            raise ValueError(f"polarity must be one of {_VALID_POLARITIES}")
        if not (0.0 <= self.min_area_frac <= 1.0):
            raise ValueError("min_area_frac must be in [0, 1]")
        if not (0.0 <= self.max_area_frac <= 1.0):
            raise ValueError("max_area_frac must be in [0, 1]")
        if self.min_area_frac > self.max_area_frac:
            raise ValueError("min_area_frac must be <= max_area_frac")
        if not (0.0 <= self.max_border_contact_frac <= 1.0):
            raise ValueError("max_border_contact_frac must be in [0, 1]")
        if self.morph_close_px < 0 or self.morph_open_px < 0:
            raise ValueError("morph_close_px and morph_open_px must be >= 0")
        if self.min_component_area_px < 0:
            raise ValueError("min_component_area_px must be >= 0")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "prefer_alpha": self.prefer_alpha,
            "alpha_threshold": self.alpha_threshold,
            "alpha_min_coverage": self.alpha_min_coverage,
            "gray_threshold": self.gray_threshold,
            "invert_policy": self.invert_policy,
            "polarity": self.polarity,
            "min_area_frac": self.min_area_frac,
            "max_area_frac": self.max_area_frac,
            "max_border_contact_frac": self.max_border_contact_frac,
            "morph_close_px": self.morph_close_px,
            "morph_open_px": self.morph_open_px,
            "fill_holes": self.fill_holes,
            "largest_component_only": self.largest_component_only,
            "min_component_area_px": self.min_component_area_px,
            "candidate_scoring": self.candidate_scoring,
            "emit_uncertainty": self.emit_uncertainty,
        }


@dataclass
class ProfileSamplingConfig:
    """Configuration for sampling silhouettes into profiles."""

    num_samples: int = 100
    sample_policy: str = "endpoints"
    fill_strategy: str = "interp_linear"
    smoothing_window: int = 3

    def validate(self) -> None:
        """Validate configuration values."""
        if self.num_samples < 2:
            raise ValueError("num_samples must be >= 2")
        if self.sample_policy not in _VALID_SAMPLE_POLICIES:
            raise ValueError(f"sample_policy must be one of {_VALID_SAMPLE_POLICIES}")
        if self.fill_strategy not in _VALID_FILL_STRATEGIES:
            raise ValueError(f"fill_strategy must be one of {_VALID_FILL_STRATEGIES}")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "num_samples": self.num_samples,
            "sample_policy": self.sample_policy,
            "fill_strategy": self.fill_strategy,
            "smoothing_window": self.smoothing_window,
        }


@dataclass
class LoftMeshOptions:
    """Configuration for loft mesh generation."""

    radial_segments: int = 24
    cap_mode: str = "fan"
    min_radius_u: float = 0.0
    merge_threshold_u: float = 0.0
    recalc_normals: bool = True
    shade_smooth: bool = True
    weld_degenerate_rings: bool = True
    adaptive_radial_segments: bool = False
    min_adaptive_radial_segments: int = 12
    max_adaptive_radial_segments: int = 96
    topology_strict: bool = True
    research_allow_low_radial_segments: bool = False

    def validate(self) -> None:
        """Validate configuration values."""
        min_segments = 3 if self.research_allow_low_radial_segments else 12
        if self.radial_segments < min_segments:
            raise ValueError(f"radial_segments must be >= {min_segments}")
        if self.cap_mode not in _VALID_CAP_MODES:
            raise ValueError(f"cap_mode must be one of {_VALID_CAP_MODES}")
        if self.min_radius_u < 0 or self.merge_threshold_u < 0:
            raise ValueError("min_radius_u and merge_threshold_u must be >= 0")
        if self.min_adaptive_radial_segments < 3:
            raise ValueError("min_adaptive_radial_segments must be >= 3")
        if self.max_adaptive_radial_segments < self.min_adaptive_radial_segments:
            raise ValueError(
                "max_adaptive_radial_segments must be >= min_adaptive_radial_segments"
            )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "radial_segments": self.radial_segments,
            "cap_mode": self.cap_mode,
            "min_radius_u": self.min_radius_u,
            "merge_threshold_u": self.merge_threshold_u,
            "recalc_normals": self.recalc_normals,
            "shade_smooth": self.shade_smooth,
            "weld_degenerate_rings": self.weld_degenerate_rings,
            "adaptive_radial_segments": self.adaptive_radial_segments,
            "min_adaptive_radial_segments": self.min_adaptive_radial_segments,
            "max_adaptive_radial_segments": self.max_adaptive_radial_segments,
            "topology_strict": self.topology_strict,
            "research_allow_low_radial_segments": self.research_allow_low_radial_segments,
        }


@dataclass
class RenderConfig:
    """Configuration for rendering orthographic silhouettes."""

    resolution: Tuple[int, int] = (512, 512)
    engine: str = "BLENDER_EEVEE"
    transparent_bg: bool = True
    samples: int = 1
    margin_frac: float = 0.08
    color_mode: str = "RGBA"
    force_material: bool = False
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    silhouette_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    camera_distance_factor: float = 2.0
    party_mode: bool = False

    def validate(self) -> None:
        """Validate configuration values."""
        if len(self.resolution) != 2:
            raise ValueError("resolution must be a (width, height) tuple")
        if any(val < 32 for val in self.resolution):
            raise ValueError("resolution values must be >= 32")
        if self.engine not in _VALID_RENDER_ENGINES:
            raise ValueError(f"engine must be one of {_VALID_RENDER_ENGINES}")
        if self.samples < 1:
            raise ValueError("samples must be >= 1")
        if not (0.0 <= self.margin_frac <= 1.0):
            raise ValueError("margin_frac must be in [0, 1]")
        if self.color_mode not in _VALID_COLOR_MODES:
            raise ValueError(f"color_mode must be one of {_VALID_COLOR_MODES}")
        if len(self.background_color) != 4:
            raise ValueError("background_color must be RGBA with 4 values")
        if len(self.silhouette_color) != 4:
            raise ValueError("silhouette_color must be RGBA with 4 values")
        if any(not (0.0 <= val <= 1.0) for val in self.background_color):
            raise ValueError("background_color values must be in [0, 1]")
        if any(not (0.0 <= val <= 1.0) for val in self.silhouette_color):
            raise ValueError("silhouette_color values must be in [0, 1]")
        if self.camera_distance_factor <= 0:
            raise ValueError("camera_distance_factor must be > 0")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "resolution": list(self.resolution),
            "engine": self.engine,
            "transparent_bg": self.transparent_bg,
            "samples": self.samples,
            "margin_frac": self.margin_frac,
            "color_mode": self.color_mode,
            "force_material": self.force_material,
            "background_color": list(self.background_color),
            "silhouette_color": list(self.silhouette_color),
            "camera_distance_factor": self.camera_distance_factor,
            "party_mode": self.party_mode,
        }


@dataclass
class CanonicalizeConfig:
    """Configuration for canonicalizing silhouette masks."""

    output_size: int = 256
    padding_frac: float = 0.1
    anchor: str = "bottom_center"
    interp: str = "nearest"
    use_cache: bool = True
    digest_algorithm: str = "sha256"
    fill_holes: Optional[bool] = None
    largest_component_only: Optional[bool] = None

    def validate(self) -> None:
        """Validate configuration values."""
        if self.output_size < 32:
            raise ValueError("output_size must be >= 32")
        if not (0.0 <= self.padding_frac <= 1.0):
            raise ValueError("padding_frac must be in [0, 1]")
        if self.anchor not in _VALID_CANON_ANCHORS:
            raise ValueError(f"anchor must be one of {_VALID_CANON_ANCHORS}")
        if self.interp not in _VALID_CANON_INTERP:
            raise ValueError(f"interp must be one of {_VALID_CANON_INTERP}")
        if self.digest_algorithm not in {"sha256"}:
            raise ValueError("digest_algorithm must be sha256")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "output_size": self.output_size,
            "padding_frac": self.padding_frac,
            "anchor": self.anchor,
            "interp": self.interp,
            "use_cache": self.use_cache,
            "digest_algorithm": self.digest_algorithm,
            "fill_holes": self.fill_holes,
            "largest_component_only": self.largest_component_only,
        }


@dataclass
class MeshJoinConfig:
    """Configuration for mesh join behavior."""

    mode: str = "boolean"
    boolean_solver: str = "auto"
    allow_degraded_simple_join: bool = True
    record_attempts: bool = True
    balanced_boolean_tree: bool = True

    def validate(self) -> None:
        """Validate configuration values."""
        if self.mode not in _VALID_JOIN_MODES:
            raise ValueError(f"mode must be one of {_VALID_JOIN_MODES}")
        if self.boolean_solver not in _VALID_BOOLEAN_SOLVERS:
            raise ValueError(f"boolean_solver must be one of {_VALID_BOOLEAN_SOLVERS}")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "mode": self.mode,
            "boolean_solver": self.boolean_solver,
            "allow_degraded_simple_join": self.allow_degraded_simple_join,
            "record_attempts": self.record_attempts,
            "balanced_boolean_tree": self.balanced_boolean_tree,
        }


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

    def to_dict(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "optional_dependency_policy": self.optional_dependency_policy,
            "gradient_mode": self.gradient_mode,
            "finite_difference_epsilon": self.finite_difference_epsilon,
            "loss_weights": self.loss_weights,
        }


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
class SyntheticFactoryConfig:
    """Configuration for synthetic fixture generation."""

    suite: str = "smoke"
    seed: int = 1234
    output_root: str = "test_output/synthetic"
    commit_small_fixtures_only: bool = True
    keep_heavy_artifacts: bool = False

    def validate(self) -> None:
        if not self.suite:
            raise ValueError("synthetic suite is required")
        if not self.output_root:
            raise ValueError("synthetic output_root is required")

    def to_dict(self) -> Dict[str, object]:
        return {
            "suite": self.suite,
            "seed": self.seed,
            "output_root": self.output_root,
            "commit_small_fixtures_only": self.commit_small_fixtures_only,
            "keep_heavy_artifacts": self.keep_heavy_artifacts,
        }


@dataclass
class QualityBudgetConfig:
    """Configuration for quality/performance budget checks."""

    budget_json: Optional[str] = None
    compare_baseline: Optional[str] = None
    fail_on_regression: bool = False
    environment_compatibility: str = "warn"

    def validate(self) -> None:
        if self.environment_compatibility not in {"warn", "strict", "ignore"}:
            raise ValueError("environment_compatibility must be warn/strict/ignore")

    def to_dict(self) -> Dict[str, object]:
        return {
            "budget_json": self.budget_json,
            "compare_baseline": self.compare_baseline,
            "fail_on_regression": self.fail_on_regression,
            "environment_compatibility": self.environment_compatibility,
        }


@dataclass
class RefinementLabConfig:
    """Configuration for local reconstruction refinement experiments."""

    default_output_root: str = "temp/refinement-runs"
    default_suite: str = "default-vase"
    default_track: str = "profile-loft-refinement"
    default_search: str = "grid"
    default_objective: str = "quality_win"
    max_runs: Optional[int] = None
    top_k: int = 10
    html_report: bool = True
    write_overlays: bool = True
    write_bounds_debug: bool = True
    write_autopsy: bool = True
    append_leaderboard: bool = True
    fail_on_all_failed: bool = True
    allow_subprocess_blender: bool = False
    blender_executable: Optional[str] = None
    copy_references: bool = False
    report_failures: str = "top"
    stop_on_first_error: bool = False

    def validate(self) -> None:
        if not self.default_output_root:
            raise ValueError("refinement_lab.default_output_root is required")
        if not self.default_suite:
            raise ValueError("refinement_lab.default_suite is required")
        if not self.default_track:
            raise ValueError("refinement_lab.default_track is required")
        if self.default_search not in _VALID_REFINEMENT_SEARCH:
            raise ValueError(
                f"refinement default_search must be one of {_VALID_REFINEMENT_SEARCH}"
            )
        if self.default_objective not in _VALID_REFINEMENT_OBJECTIVES:
            raise ValueError(
                f"refinement default_objective must be one of {_VALID_REFINEMENT_OBJECTIVES}"
            )
        if self.max_runs is not None and self.max_runs < 1:
            raise ValueError("refinement max_runs must be >= 1 when provided")
        if self.top_k < 1:
            raise ValueError("refinement top_k must be >= 1")
        if self.report_failures not in _VALID_REFINEMENT_REPORT_FAILURES:
            raise ValueError(
                "refinement report_failures must be one of "
                f"{_VALID_REFINEMENT_REPORT_FAILURES}"
            )

    def to_dict(self) -> Dict[str, object]:
        return {
            "default_output_root": self.default_output_root,
            "default_suite": self.default_suite,
            "default_track": self.default_track,
            "default_search": self.default_search,
            "default_objective": self.default_objective,
            "max_runs": self.max_runs,
            "top_k": self.top_k,
            "html_report": self.html_report,
            "write_overlays": self.write_overlays,
            "write_bounds_debug": self.write_bounds_debug,
            "write_autopsy": self.write_autopsy,
            "append_leaderboard": self.append_leaderboard,
            "fail_on_all_failed": self.fail_on_all_failed,
            "allow_subprocess_blender": self.allow_subprocess_blender,
            "blender_executable": self.blender_executable,
            "copy_references": self.copy_references,
            "report_failures": self.report_failures,
            "stop_on_first_error": self.stop_on_first_error,
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


@dataclass
class BlockingConfig:
    """Root configuration for the blocking workflow."""

    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    mesh_join: MeshJoinConfig = field(default_factory=MeshJoinConfig)
    silhouette_extract_ref: SilhouetteExtractConfig = field(
        default_factory=SilhouetteExtractConfig
    )
    silhouette_extract_render: SilhouetteExtractConfig = field(
        default_factory=SilhouetteExtractConfig
    )
    silhouette_intersection: SilhouetteIntersectionConfig = field(
        default_factory=SilhouetteIntersectionConfig
    )
    profile_sampling: ProfileSamplingConfig = field(
        default_factory=ProfileSamplingConfig
    )
    mesh_from_profile: LoftMeshOptions = field(default_factory=LoftMeshOptions)
    render_silhouette: RenderConfig = field(default_factory=RenderConfig)
    canonicalize: CanonicalizeConfig = field(default_factory=CanonicalizeConfig)
    visual_hull: VisualHullConfig = field(default_factory=VisualHullConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    primitive_fit: PrimitiveFitConfig = field(default_factory=PrimitiveFitConfig)
    gaussian_ellipsoid: GaussianEllipsoidConfig = field(
        default_factory=GaussianEllipsoidConfig
    )
    differentiable_render: DifferentiableRenderConfig = field(
        default_factory=DifferentiableRenderConfig
    )
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    synthetic_factory: SyntheticFactoryConfig = field(default_factory=SyntheticFactoryConfig)
    quality_budget: QualityBudgetConfig = field(default_factory=QualityBudgetConfig)
    refinement_lab: RefinementLabConfig = field(default_factory=RefinementLabConfig)

    def validate(self) -> None:
        """Validate configuration values across groups."""
        self.reconstruction.validate()
        self.mesh_join.validate()
        self.silhouette_extract_ref.validate()
        self.silhouette_extract_render.validate()
        self.silhouette_intersection.validate()
        self.profile_sampling.validate()
        self.mesh_from_profile.validate()
        self.render_silhouette.validate()
        self.canonicalize.validate()
        self.visual_hull.validate()
        self.volume.validate()
        self.ensemble.validate()
        self.primitive_fit.validate()
        self.gaussian_ellipsoid.validate()
        self.differentiable_render.validate()
        self.constraints.validate()
        self.synthetic_factory.validate()
        self.quality_budget.validate()
        self.refinement_lab.validate()

        # Placeholder for mutually exclusive scale policies if added later.

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict matching the canonical schema."""
        return {
            "reconstruction": self.reconstruction.to_dict(),
            "mesh_join": self.mesh_join.to_dict(),
            "silhouette_extract_ref": self.silhouette_extract_ref.to_dict(),
            "silhouette_extract_render": self.silhouette_extract_render.to_dict(),
            "silhouette_intersection": self.silhouette_intersection.to_dict(),
            "profile_sampling": self.profile_sampling.to_dict(),
            "mesh_from_profile": self.mesh_from_profile.to_dict(),
            "canonicalize": self.canonicalize.to_dict(),
            "render_silhouette": self.render_silhouette.to_dict(),
            "visual_hull": self.visual_hull.to_dict(),
            "volume": self.volume.to_dict(),
            "ensemble": self.ensemble.to_dict(),
            "primitive_fit": self.primitive_fit.to_dict(),
            "gaussian_ellipsoid": self.gaussian_ellipsoid.to_dict(),
            "differentiable_render": self.differentiable_render.to_dict(),
            "constraints": self.constraints.to_dict(),
            "synthetic_factory": self.synthetic_factory.to_dict(),
            "quality_budget": self.quality_budget.to_dict(),
            "refinement_lab": self.refinement_lab.to_dict(),
        }
