"""Preset suites and parameter tracks for the refinement lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import ParameterSpec


BLENDER_BLOCKING_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SuitePreset:
    name: str
    source: str
    description: str
    synthetic_suites: tuple[str, ...] = ()
    reference_paths: Mapping[str, Path] = field(default_factory=dict)
    default_seed: int = 1234
    default_count: int | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "description": self.description,
            "synthetic_suites": list(self.synthetic_suites),
            "reference_paths": {
                key: path.as_posix() for key, path in self.reference_paths.items()
            },
            "default_seed": self.default_seed,
            "default_count": self.default_count,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class TrackPreset:
    name: str
    description: str
    modes: tuple[str, ...]
    parameters: tuple[ParameterSpec, ...]
    default_validation_mode: str = "render-iou"
    default_search: str = "grid"
    default_objective: str = "quality_win"
    max_runs_hint: int | None = None
    force_bounds_debug: bool = False
    force_autopsy: bool = True
    force_overlays: bool = True
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "modes": list(self.modes),
            "default_validation_mode": self.default_validation_mode,
            "default_search": self.default_search,
            "default_objective": self.default_objective,
            "max_runs_hint": self.max_runs_hint,
            "force_bounds_debug": self.force_bounds_debug,
            "force_autopsy": self.force_autopsy,
            "force_overlays": self.force_overlays,
            "tags": list(self.tags),
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }


def _p(
    name: str,
    cli_arg: str,
    values: Sequence[object],
    *,
    value_type: str = "string",
    config_path: str | None = None,
    group: str = "",
    description: str = "",
) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        cli_arg=cli_arg,
        config_path=config_path,
        value_type=value_type,
        values=tuple(values),
        group=group,
        description=description,
    )


def _csv_p(
    name: str,
    cli_arg: str,
    values: Sequence[object],
    *,
    config_path: str | None = None,
    group: str = "",
    description: str = "",
) -> ParameterSpec:
    return _p(
        name,
        cli_arg,
        values,
        value_type="csv",
        config_path=config_path,
        group=group,
        description=description,
    )


def _json_p(
    name: str,
    cli_arg: str,
    values: Sequence[object],
    *,
    config_path: str | None = None,
    group: str = "",
    description: str = "",
) -> ParameterSpec:
    return _p(
        name,
        cli_arg,
        values,
        value_type="json",
        config_path=config_path,
        group=group,
        description=description,
    )


_MASK_PARAMETERS = (
    _p(
        "ref_polarity",
        "--ref-polarity",
        ("auto", "dark_foreground", "light_foreground"),
        group="silhouette",
    ),
    _p(
        "ref_gray_threshold",
        "--ref-gray-threshold",
        (96, 112, 128, 144, 160),
        value_type="int",
        group="silhouette",
    ),
    _p(
        "ref_morph_close",
        "--ref-morph-close",
        (0, 3, 5, 7),
        value_type="int",
        group="silhouette",
    ),
    _p(
        "ref_morph_open",
        "--ref-morph-open",
        (0, 1, 3),
        value_type="int",
        group="silhouette",
    ),
    _p(
        "ref_min_component_area_px",
        "--ref-min-component-area-px",
        (0, 16, 64, 256),
        value_type="int",
        group="silhouette",
    ),
    _p(
        "ref_fill_holes",
        "--ref-fill-holes",
        (True, False),
        value_type="bool",
        group="silhouette",
    ),
    _p(
        "ref_largest_component",
        "--ref-largest-component",
        (True, False),
        value_type="bool",
        group="silhouette",
    ),
    _p(
        "canonical_padding_frac",
        "--canonical-padding-frac",
        (0.06, 0.08, 0.10, 0.12),
        value_type="float",
        group="canonical",
    ),
    _p(
        "canonical_anchor",
        "--canonical-anchor",
        ("bottom_center", "center"),
        group="canonical",
    ),
)


SUITES: dict[str, SuitePreset] = {
    "default-vase": SuitePreset(
        name="default-vase",
        source="builtin_sample",
        description="Built-in orthogonal vase references used for fast real-image regression.",
        reference_paths={
            "front": BLENDER_BLOCKING_ROOT / "test_images" / "vase_front.png",
            "side": BLENDER_BLOCKING_ROOT / "test_images" / "vase_side.png",
            "top": BLENDER_BLOCKING_ROOT / "test_images" / "vase_top.png",
        },
        tags=("quick", "real-image", "vase"),
    ),
    "synthetic-smoke": SuitePreset(
        name="synthetic-smoke",
        source="synthetic",
        description="Small synthetic sanity suite.",
        synthetic_suites=("smoke",),
        tags=("quick", "synthetic"),
    ),
    "synthetic-visual-hull": SuitePreset(
        name="synthetic-visual-hull",
        source="synthetic",
        description="Synthetic shapes that stress visual hull volume and meshing behavior.",
        synthetic_suites=("visual-hull",),
        tags=("synthetic", "visual-hull"),
    ),
    "synthetic-profile-band": SuitePreset(
        name="synthetic-profile-band",
        source="synthetic",
        description="Synthetic profile/lathe shapes for profile-band and loft tuning.",
        synthetic_suites=("profile-band",),
        tags=("synthetic", "profile"),
    ),
    "synthetic-primitive-fit": SuitePreset(
        name="synthetic-primitive-fit",
        source="synthetic",
        description="Synthetic primitive fixtures for primitive, Gaussian, and differentiable fitting.",
        synthetic_suites=("primitive-fit",),
        tags=("synthetic", "primitive"),
    ),
    "synthetic-adversarial": SuitePreset(
        name="synthetic-adversarial",
        source="synthetic",
        description="Combined synthetic edge cases for mask extraction and robustness.",
        synthetic_suites=("silhouette-edge-cases", "degradation-stress"),
        tags=("synthetic", "adversarial"),
    ),
    "synthetic-nightly": SuitePreset(
        name="synthetic-nightly",
        source="synthetic",
        description="Full heavy synthetic suite for broad regression checks.",
        synthetic_suites=("nightly-heavy",),
        tags=("synthetic", "nightly"),
    ),
}


TRACKS: dict[str, TrackPreset] = {
    "mask-refinement": TrackPreset(
        name="mask-refinement",
        description="Sweep silhouette extraction and canonicalization knobs across modes.",
        modes=("profile_loft", "visual_hull_voxel", "ensemble"),
        parameters=_MASK_PARAMETERS,
        default_objective="min_view_iou",
        max_runs_hint=96,
        tags=("mask", "canonicalization"),
    ),
    "profile-loft-refinement": TrackPreset(
        name="profile-loft-refinement",
        description="Tune profile sampling and loft mesh settings for render IoU and editability.",
        modes=("profile_loft",),
        parameters=(
            _p(
                "profile_samples",
                "--profile-samples",
                (80, 120, 160, 240),
                value_type="int",
                group="profile",
            ),
            _p(
                "profile_sample_policy",
                "--profile-sample-policy",
                ("endpoints", "cell_centers"),
                group="profile",
            ),
            _p(
                "profile_fill_strategy",
                "--profile-fill-strategy",
                ("interp_linear", "interp_nearest"),
                group="profile",
            ),
            _p(
                "profile_smoothing_window",
                "--profile-smoothing-window",
                (1, 3, 5, 7),
                value_type="int",
                group="profile",
            ),
            _p(
                "mesh_radial_segments",
                "--mesh-radial-segments",
                (24, 32, 48, 64, 96),
                value_type="int",
                group="mesh",
            ),
            _p(
                "mesh_min_radius",
                "--mesh-min-radius",
                (0.0, 0.0001, 0.001),
                value_type="float",
                group="mesh",
            ),
            _p(
                "mesh_merge_threshold",
                "--mesh-merge-threshold",
                (0.0, 0.0001, 0.001),
                value_type="float",
                group="mesh",
            ),
            _p("mesh_cap_mode", "--mesh-cap-mode", ("fan", "none"), group="mesh"),
            _p(
                "mesh_adaptive_radial_segments",
                "--mesh-adaptive-radial-segments",
                (True, False),
                value_type="bool",
                group="mesh",
            ),
            _p(
                "mesh_shade_smooth",
                "--mesh-shade-smooth",
                (True, False),
                value_type="bool",
                group="mesh",
            ),
        ),
        default_objective="profile_editable",
        max_runs_hint=128,
        tags=("profile", "loft"),
    ),
    "visual-hull-transform": TrackPreset(
        name="visual-hull-transform",
        description="Diagnose visual hull bounds, projection, axis order, scale, and render framing.",
        modes=("visual_hull_voxel",),
        parameters=(
            _p(
                "vh_backend",
                "--vh-backend",
                ("dense", "chunked", "sparse_hash"),
                group="visual_hull",
            ),
            _p(
                "vh_resolution",
                "--vh-resolution",
                (32, 48, 64, 96),
                value_type="int",
                group="visual_hull",
            ),
            _p(
                "vh_chunk_size",
                "--vh-chunk-size",
                (8, 16, 32),
                value_type="int",
                group="visual_hull",
            ),
            _p(
                "vh_mesh_method",
                "--vh-mesh-method",
                ("marching_cubes",),
                group="visual_hull",
            ),
            _p(
                "vh_occupancy_threshold",
                "--vh-occupancy-threshold",
                (0.25, 0.5, 0.75),
                value_type="float",
                group="visual_hull",
            ),
            _p(
                "ref_gray_threshold",
                "--ref-gray-threshold",
                (128,),
                value_type="int",
                group="silhouette",
            ),
            _p(
                "ref_morph_close",
                "--ref-morph-close",
                (5,),
                value_type="int",
                group="silhouette",
            ),
            _p(
                "ref_morph_open",
                "--ref-morph-open",
                (3,),
                value_type="int",
                group="silhouette",
            ),
            _p(
                "ref_fill_holes",
                "--ref-fill-holes",
                (True,),
                value_type="bool",
                group="silhouette",
            ),
            _p(
                "ref_largest_component",
                "--ref-largest-component",
                (True,),
                value_type="bool",
                group="silhouette",
            ),
        ),
        default_search="coordinate",
        default_objective="visual_hull_alignment",
        max_runs_hint=48,
        force_bounds_debug=True,
        tags=("visual-hull", "bounds", "transform"),
    ),
    "visual-hull-quality": TrackPreset(
        name="visual-hull-quality",
        description="Tune visual hull resolution, sparse backend, meshing, postprocess, and uncertainty aggregation.",
        modes=("visual_hull_voxel", "hybrid_loft_hull"),
        parameters=(
            _p(
                "vh_backend",
                "--vh-backend",
                ("chunked", "sparse_hash", "openvdb"),
                group="visual_hull",
            ),
            _p(
                "vh_resolution",
                "--vh-resolution",
                (64, 96, 128, 192),
                value_type="int",
                group="visual_hull",
            ),
            _p(
                "vh_chunk_size",
                "--vh-chunk-size",
                (16, 32),
                value_type="int",
                group="visual_hull",
            ),
            _p(
                "vh_mesh_method",
                "--vh-mesh-method",
                ("marching_cubes", "lewiner"),
                group="visual_hull",
            ),
            _p(
                "vh_postprocess",
                "--vh-postprocess",
                ("none", "poisson", "screened_poisson"),
                group="visual_hull",
            ),
            _p(
                "vh_uncertainty_aggregation",
                "--vh-uncertainty-aggregation",
                ("min", "product", "logit_sum"),
                group="visual_hull",
            ),
        ),
        max_runs_hint=96,
        force_bounds_debug=True,
        tags=("visual-hull", "quality"),
    ),
    "primitive-fit": TrackPreset(
        name="primitive-fit",
        description="Tune primitive families, target points, primitive counts, and objective weights.",
        modes=("primitive_fit_refine",),
        parameters=(
            _csv_p(
                "primitive_families",
                "--primitive-families",
                (
                    ("superfrustum",),
                    ("ellipsoid",),
                    ("superfrustum", "ellipsoid"),
                    ("superquadric",),
                ),
            ),
            _p(
                "primitive_target_points",
                "--primitive-target-points",
                (256, 512, 1024, 2048),
                value_type="int",
            ),
            _p("primitive_min", "--primitive-min", (1, 2, 4), value_type="int"),
            _p("primitive_max", "--primitive-max", (4, 6, 8, 12, 16), value_type="int"),
            _p(
                "primitive_steps",
                "--primitive-steps",
                (0, 5, 10, 25, 50),
                value_type="int",
            ),
            _p(
                "primitive_max_runtime_s",
                "--primitive-max-runtime-s",
                (5.0, 10.0, 20.0),
                value_type="float",
            ),
            _p(
                "primitive_max_objective_evaluations",
                "--primitive-max-objective-evaluations",
                (128, 384, 768, 1536),
                value_type="int",
            ),
            _p(
                "primitive_fail_on_regression",
                "--primitive-fail-on-regression",
                (True, False),
                value_type="bool",
            ),
            _json_p(
                "primitive_loss_weights_json",
                "--primitive-loss-weights-json",
                (
                    {"silhouette": 1.0, "surface": 0.5},
                    {"silhouette": 1.0, "surface": 1.0, "topology": 0.25},
                ),
            ),
        ),
        max_runs_hint=96,
        tags=("primitive",),
    ),
    "gaussian-proxy": TrackPreset(
        name="gaussian-proxy",
        description="Tune Gaussian/ellipsoid proxy count, initialization, radius, opacity, and mesh export.",
        modes=("gaussian_ellipsoid_proxy",),
        parameters=(
            _p(
                "gaussian_count",
                "--gaussian-count",
                (8, 12, 16, 24, 32, 48),
                value_type="int",
            ),
            _p(
                "gaussian_initialization",
                "--gaussian-initialization",
                ("farthest_point", "kmeans", "grid"),
            ),
            _p(
                "gaussian_min_radius",
                "--gaussian-min-radius",
                (0.0001, 0.001, 0.01),
                value_type="float",
            ),
            _p(
                "gaussian_opacity_min",
                "--gaussian-opacity-min",
                (0.0, 0.1, 0.25),
                value_type="float",
            ),
            _p(
                "gaussian_opacity_max",
                "--gaussian-opacity-max",
                (0.75, 1.0),
                value_type="float",
            ),
            _p(
                "gaussian_export_mesh_proxy",
                "--gaussian-export-mesh-proxy",
                (True, False),
                value_type="bool",
            ),
        ),
        tags=("gaussian", "metric-only-risk"),
    ),
    "differentiable-refine": TrackPreset(
        name="differentiable-refine",
        description="Exercise CPU/GPU differentiable-style refinement and optional dependency policies.",
        modes=("differentiable_refine",),
        parameters=(
            _p(
                "diff_backend",
                "--diff-backend",
                ("cpu_soft_silhouette", "blender_finite_difference", "nvdiffrast"),
            ),
            _p("diff_optional_policy", "--diff-optional-policy", ("skip", "fail")),
            _p(
                "diff_gradient_mode",
                "--diff-gradient-mode",
                ("finite_difference", "backend"),
            ),
            _p(
                "diff_epsilon", "--diff-epsilon", (1e-3, 1e-4, 1e-5), value_type="float"
            ),
            _json_p(
                "diff_loss_weights_json",
                "--diff-loss-weights-json",
                ({"silhouette": 1.0}, {"silhouette": 1.0, "sdf": 0.25}),
            ),
        ),
        tags=("differentiable", "research"),
    ),
    "ensemble-selection": TrackPreset(
        name="ensemble-selection",
        description="Tune candidate sets and policies so selected ensemble candidates are evidence-backed.",
        modes=("ensemble",),
        parameters=(
            _csv_p(
                "ensemble_candidates",
                "--ensemble-candidates",
                (
                    ("profile_loft", "visual_hull_voxel"),
                    ("profile_loft", "visual_hull_voxel", "gaussian_ellipsoid_proxy"),
                    (
                        "profile_loft",
                        "visual_hull_voxel",
                        "primitive_fit_refine",
                        "gaussian_ellipsoid_proxy",
                        "differentiable_refine",
                    ),
                    ("visual_hull_voxel", "primitive_fit_refine", "shape_program"),
                    (
                        "visual_hull_voxel",
                        "primitive_fit_refine",
                        "gaussian_ellipsoid_proxy",
                        "differentiable_refine",
                        "shape_program",
                    ),
                ),
            ),
            _p(
                "ensemble_policy",
                "--ensemble-policy",
                (
                    "best_score",
                    "balanced",
                    "fidelity",
                    "quality_first",
                    "editable",
                    "editability_first",
                    "printable",
                    "fast_preview",
                    "pareto",
                    "research_fidelity",
                ),
            ),
            _p("primitive_max", "--primitive-max", (6, 12), value_type="int"),
            _p("gaussian_count", "--gaussian-count", (16, 24), value_type="int"),
        ),
        tags=("ensemble",),
    ),
    "shape-program-editability": TrackPreset(
        name="shape-program-editability",
        description="Emit editable primitive programs and residual patch hints for Blender-first reconstruction.",
        modes=("shape_program", "ensemble"),
        parameters=(
            _p(
                "shape_root_strategy",
                "--shape-root-strategy",
                ("profile_lathe", "bounds_box", "hybrid_profile_bounds"),
                group="shape_program",
            ),
            _p(
                "shape_residual_policy",
                "--shape-residual-policy",
                ("ignore", "report", "suggest_patches"),
                group="shape_program",
            ),
            _p(
                "shape_max_nodes",
                "--shape-max-nodes",
                (24, 48, 64, 96, 128),
                value_type="int",
                group="shape_program",
            ),
            _p(
                "shape_editability_bias",
                "--shape-editability-bias",
                (0.7, 0.85, 1.0),
                value_type="float",
                group="shape_program",
            ),
            _p(
                "shape_compile_blender",
                "--shape-compile-blender",
                (True, False),
                value_type="bool",
                group="shape_program",
            ),
            _p(
                "shape_lathe_segments",
                "--shape-lathe-segments",
                (24, 48, 64, 96),
                value_type="int",
                group="shape_program",
            ),
            _p(
                "shape_bevel_modifier",
                "--shape-bevel-modifier",
                (True, False),
                value_type="bool",
                group="shape_program",
            ),
            _p(
                "shape_weighted_normals",
                "--shape-weighted-normals",
                (True, False),
                value_type="bool",
                group="shape_program",
            ),
            _csv_p(
                "ensemble_candidates",
                "--ensemble-candidates",
                (
                    ("shape_program",),
                    ("visual_hull_voxel", "shape_program"),
                    ("primitive_fit_refine", "shape_program"),
                    ("visual_hull_voxel", "primitive_fit_refine", "shape_program"),
                ),
                group="ensemble",
            ),
            _p(
                "ensemble_policy",
                "--ensemble-policy",
                ("editable", "printable", "research_fidelity"),
                group="ensemble",
            ),
        ),
        default_validation_mode="backend-status",
        default_search="coordinate",
        default_objective="profile_editable",
        max_runs_hint=72,
        force_autopsy=True,
        force_overlays=False,
        tags=("shape-program", "editable", "research"),
    ),
    "content-adaptive-patches": TrackPreset(
        name="content-adaptive-patches",
        description="Drive global-plus-local residual refinement using boundary, SDF, uncertainty, and surface-coverage failures.",
        modes=("ensemble",),
        parameters=(
            _csv_p(
                "ensemble_candidates",
                "--ensemble-candidates",
                (
                    ("visual_hull_voxel", "shape_program"),
                    ("visual_hull_voxel", "primitive_fit_refine", "shape_program"),
                    (
                        "visual_hull_voxel",
                        "primitive_fit_refine",
                        "shape_program",
                        "differentiable_refine",
                    ),
                    (
                        "visual_hull_voxel",
                        "primitive_fit_refine",
                        "gaussian_ellipsoid_proxy",
                        "shape_program",
                        "differentiable_refine",
                    ),
                ),
                group="ensemble",
            ),
            _p(
                "ensemble_policy",
                "--ensemble-policy",
                ("fidelity", "editable", "research_fidelity", "pareto"),
                group="ensemble",
            ),
            _p(
                "shape_residual_policy",
                "--shape-residual-policy",
                ("suggest_patches", "report"),
                group="shape_program",
            ),
            _p(
                "shape_max_nodes",
                "--shape-max-nodes",
                (64, 96, 128),
                value_type="int",
                group="shape_program",
            ),
            _p(
                "shape_editability_bias",
                "--shape-editability-bias",
                (0.85, 1.0),
                value_type="float",
                group="shape_program",
            ),
            _p(
                "vh_resolution",
                "--vh-resolution",
                (96, 128, 192),
                value_type="int",
                group="visual_hull",
            ),
            _p(
                "vh_mesh_method",
                "--vh-mesh-method",
                ("lewiner", "marching_cubes"),
                group="visual_hull",
            ),
            _p(
                "vh_uncertainty_aggregation",
                "--vh-uncertainty-aggregation",
                ("product", "logit_sum"),
                group="visual_hull",
            ),
            _p(
                "ref_morph_close",
                "--ref-morph-close",
                (3, 5, 7),
                value_type="int",
                group="silhouette",
            ),
            _p(
                "ref_fill_holes",
                "--ref-fill-holes",
                (True,),
                value_type="bool",
                group="silhouette",
            ),
            _json_p(
                "primitive_loss_weights_json",
                "--primitive-loss-weights-json",
                (
                    {"silhouette": 1.0, "boundary": 0.35, "sdf": 0.25, "surface": 0.5},
                    {
                        "silhouette": 1.0,
                        "boundary": 0.5,
                        "sdf": 0.35,
                        "surface": 0.75,
                        "topology": 0.25,
                    },
                ),
                group="primitive",
            ),
            _json_p(
                "diff_loss_weights_json",
                "--diff-loss-weights-json",
                (
                    {"silhouette": 1.0, "boundary": 0.35, "sdf": 0.35},
                    {"silhouette": 1.0, "boundary": 0.5, "sdf": 0.5},
                ),
                group="differentiable",
            ),
        ),
        default_validation_mode="backend-status",
        default_search="successive_halving",
        default_objective="quality_win",
        max_runs_hint=160,
        force_bounds_debug=True,
        force_autopsy=True,
        force_overlays=True,
        tags=("content-adaptive-patches", "boundary", "sdf", "detail", "research"),
    ),
    "sota-metric-bundle": TrackPreset(
        name="sota-metric-bundle",
        description="Exercise full SOTA-style evidence bundles: per-view IoU, boundary IoU, SDF loss, topology, editability, and cost.",
        modes=("ensemble",),
        parameters=(
            _csv_p(
                "ensemble_candidates",
                "--ensemble-candidates",
                (
                    ("visual_hull_voxel", "primitive_fit_refine"),
                    (
                        "visual_hull_voxel",
                        "primitive_fit_refine",
                        "gaussian_ellipsoid_proxy",
                    ),
                    (
                        "visual_hull_voxel",
                        "primitive_fit_refine",
                        "gaussian_ellipsoid_proxy",
                        "differentiable_refine",
                        "shape_program",
                    ),
                ),
                group="ensemble",
            ),
            _p(
                "ensemble_policy",
                "--ensemble-policy",
                ("balanced", "fidelity", "editable", "printable", "research_fidelity"),
                group="ensemble",
            ),
            _p(
                "vh_mesh_method",
                "--vh-mesh-method",
                ("marching_cubes", "lewiner", "dual_contouring"),
                group="visual_hull",
            ),
            _p(
                "vh_uncertainty_aggregation",
                "--vh-uncertainty-aggregation",
                ("min", "product", "logit_sum"),
                group="visual_hull",
            ),
            _json_p(
                "primitive_loss_weights_json",
                "--primitive-loss-weights-json",
                (
                    {"silhouette": 1.0, "boundary": 0.25, "topology": 0.15},
                    {
                        "silhouette": 1.0,
                        "sdf": 0.35,
                        "topology": 0.25,
                        "constraint": 0.2,
                    },
                ),
                group="primitive",
            ),
            _json_p(
                "diff_loss_weights_json",
                "--diff-loss-weights-json",
                (
                    {"silhouette": 1.0, "sdf": 0.25},
                    {"silhouette": 1.0, "boundary": 0.35, "sdf": 0.35},
                ),
                group="differentiable",
            ),
        ),
        default_validation_mode="backend-status",
        default_search="successive_halving",
        default_objective="quality_win",
        max_runs_hint=144,
        force_bounds_debug=True,
        force_autopsy=True,
        force_overlays=True,
        tags=("sota-metrics", "ensemble", "quality-gate"),
    ),
}


def list_suites() -> tuple[str, ...]:
    return tuple(sorted(SUITES))


def list_tracks() -> tuple[str, ...]:
    return tuple(sorted(TRACKS))


def get_suite_preset(name: str) -> SuitePreset:
    try:
        return SUITES[name]
    except KeyError as exc:
        raise KeyError(f"unknown refinement suite {name!r}") from exc


def get_track_preset(name: str) -> TrackPreset:
    try:
        return TRACKS[name]
    except KeyError as exc:
        raise KeyError(f"unknown refinement track {name!r}") from exc
