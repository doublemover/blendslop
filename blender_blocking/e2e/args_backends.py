from __future__ import annotations

import argparse
from pathlib import Path

from blender_blocking.e2e.novel_args import _parse_csv

def add_visual_hull_args(parser: argparse.ArgumentParser) -> None:
    hull = parser.add_argument_group("visual hull and volume")
    hull.add_argument(
        "--vh-backend",
        choices=("dense", "chunked", "sparse_hash", "openvdb"),
        default=None,
    )
    hull.add_argument("--vh-resolution", type=int, default=None)
    hull.add_argument("--vh-max-resolution", type=int, default=None)
    hull.add_argument("--vh-chunk-size", type=int, default=None)
    hull.add_argument("--vh-adaptive-max-depth", type=int, default=None)
    hull.add_argument(
        "--vh-boundary-refine", action=argparse.BooleanOptionalAction, default=None
    )
    hull.add_argument(
        "--vh-mesh-method",
        choices=("marching_cubes", "lewiner", "dual_contouring", "points"),
        default=None,
    )
    hull.add_argument(
        "--vh-postprocess",
        choices=(
            "none",
            "poisson",
            "screened_poisson",
            "smooth_guarded",
            "topology_repair",
        ),
        default=None,
    )
    hull.add_argument("--vh-memory-budget-mb", type=int, default=None)
    hull.add_argument("--vh-occupancy-threshold", type=float, default=None)
    hull.add_argument(
        "--vh-uncertainty-aggregation",
        choices=("min", "product", "logit_sum"),
        default=None,
    )
    hull.add_argument("--vh-cache", action=argparse.BooleanOptionalAction, default=None)
    hull.add_argument("--vh-cache-dir", type=str, default=None)
    hull.add_argument("--vh-cache-namespace", type=str, default=None)
    hull.add_argument("--vh-cache-read", action=argparse.BooleanOptionalAction, default=None)
    hull.add_argument("--vh-cache-write", action=argparse.BooleanOptionalAction, default=None)
    hull.add_argument(
        "--volume-backend",
        choices=("dense", "chunked", "sparse_hash", "openvdb"),
        default=None,
    )
    hull.add_argument("--volume-sparse-chunk-size", type=int, default=None)
    hull.add_argument("--volume-serialization", choices=("npz",), default=None)
    hull.add_argument(
        "--export-openvdb", action=argparse.BooleanOptionalAction, default=None
    )


def add_primitive_fit_args(parser: argparse.ArgumentParser) -> None:
    primitive = parser.add_argument_group("primitive fitting")
    primitive.add_argument("--primitive-families", type=_parse_csv, default=None)
    primitive.add_argument("--primitive-loss-weights-json", type=str, default=None)
    primitive.add_argument("--primitive-target-points", type=int, default=None)
    primitive.add_argument("--primitive-min", type=int, default=None)
    primitive.add_argument("--primitive-max", type=int, default=None)
    primitive.add_argument("--primitive-steps", type=int, default=None)
    primitive.add_argument("--primitive-checkpoint-cadence", type=int, default=None)
    primitive.add_argument(
        "--primitive-fail-on-regression",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    primitive.add_argument("--primitive-max-runtime-s", type=float, default=None)
    primitive.add_argument(
        "--primitive-max-objective-evaluations", type=int, default=None
    )


def add_gaussian_proxy_args(parser: argparse.ArgumentParser) -> None:
    gaussian = parser.add_argument_group("gaussian and ellipsoid proxy")
    gaussian.add_argument("--gaussian-count", type=int, default=None)
    gaussian.add_argument(
        "--gaussian-initialization",
        choices=("farthest_point", "kmeans", "grid"),
        default=None,
    )
    gaussian.add_argument("--gaussian-min-radius", type=float, default=None)
    gaussian.add_argument("--gaussian-max-radius", type=float, default=None)
    gaussian.add_argument("--gaussian-opacity-min", type=float, default=None)
    gaussian.add_argument("--gaussian-opacity-max", type=float, default=None)
    gaussian.add_argument(
        "--gaussian-renderer",
        choices=("cpu_projected_ellipse", "gpu_splat"),
        default=None,
    )
    gaussian.add_argument(
        "--gaussian-export-mesh-proxy",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


def add_differentiable_refine_args(parser: argparse.ArgumentParser) -> None:
    diff = parser.add_argument_group("differentiable refinement")
    diff.add_argument(
        "--diff-backend",
        choices=("cpu_soft_silhouette", "blender_finite_difference", "nvdiffrast"),
        default=None,
    )
    diff.add_argument("--diff-optional-policy", choices=("skip", "fail"), default=None)
    diff.add_argument(
        "--diff-gradient-mode", choices=("finite_difference", "backend"), default=None
    )
    diff.add_argument("--diff-epsilon", type=float, default=None)
    diff.add_argument("--diff-primitive-count", type=int, default=None)
    diff.add_argument("--diff-target-points", type=int, default=None)
    diff.add_argument("--diff-visual-hull-resolution", type=int, default=None)
    diff.add_argument("--diff-optimization-steps", type=int, default=None)
    diff.add_argument("--diff-initial-step", type=float, default=None)
    diff.add_argument("--diff-step-decay", type=float, default=None)
    diff.add_argument("--diff-min-step", type=float, default=None)
    diff.add_argument("--diff-max-objective-evaluations", type=int, default=None)
    diff.add_argument("--diff-max-runtime", type=float, default=None)
    diff.add_argument("--diff-loss-weights-json", type=str, default=None)


def add_shape_program_args(parser: argparse.ArgumentParser) -> None:
    shape_program = parser.add_argument_group("editable shape program")
    shape_program.add_argument(
        "--shape-root-strategy",
        choices=("profile_lathe", "bounds_box", "hybrid_profile_bounds"),
        default=None,
    )
    shape_program.add_argument(
        "--shape-residual-policy",
        choices=("ignore", "report", "suggest_patches"),
        default=None,
    )
    shape_program.add_argument("--shape-max-nodes", type=int, default=None)
    shape_program.add_argument("--shape-editability-bias", type=float, default=None)
    shape_program.add_argument(
        "--shape-compile-blender",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    shape_program.add_argument("--shape-lathe-segments", type=int, default=None)
    shape_program.add_argument(
        "--shape-bevel-modifier",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    shape_program.add_argument(
        "--shape-weighted-normals",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    shape_program.add_argument(
        "--shape-run-export-qa",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    shape_program.add_argument(
        "--shape-export-qa-targets",
        type=_parse_csv,
        default=None,
        help="Comma-separated editable export QA targets, e.g. obj,glb.",
    )
    shape_program.add_argument(
        "--evaluate-texture-materials",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Evaluate UV, texture, material, and appearance-attribution evidence for editable outputs.",
    )
    shape_program.add_argument(
        "--texture-reference-dir",
        type=str,
        default=None,
        help="Optional directory containing held-out texture/material reference images.",
    )
    shape_program.add_argument(
        "--uv-strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Treat missing or invalid UV evidence as a hard appearance failure.",
    )
    shape_program.add_argument(
        "--material-target",
        choices=("pbr", "simple", "none"),
        default=None,
        help="Expected editable material target for appearance evaluation.",
    )
    shape_program.add_argument("--max-texture-memory-mb", type=float, default=None)


def add_ensemble_args(parser: argparse.ArgumentParser) -> None:
    ensemble = parser.add_argument_group("ensemble")
    ensemble.add_argument(
        "--ensemble-candidates",
        type=_parse_csv,
        default=None,
        help="Comma-separated backend list.",
    )
    ensemble.add_argument(
        "--ensemble-policy",
        choices=(
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
        default=None,
    )
    ensemble.add_argument("--ensemble-max-parallel", type=int, default=None)
    ensemble.add_argument("--ensemble-timeout", type=float, default=None)
    ensemble.add_argument("--ensemble-total-timeout", type=float, default=None)
    ensemble.add_argument(
        "--ensemble-keep-artifacts", action=argparse.BooleanOptionalAction, default=None
    )
    ensemble.add_argument(
        "--ensemble-fail-if-no-required-views",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
