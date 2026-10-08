from __future__ import annotations

import argparse
from pathlib import Path

from blender_blocking.e2e.novel_args import _parse_resolution, _parse_rgba

def add_render_args(parser: argparse.ArgumentParser) -> None:
    render = parser.add_argument_group("render")
    render.add_argument(
        "--resolution",
        type=_parse_resolution,
        default=(512, 512),
        help="Render resolution (N or WxH, e.g., 512 or 1024x1024)",
    )
    render.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Render samples (EEVEE only)",
    )
    render.add_argument(
        "--engine",
        choices=("BLENDER_EEVEE", "WORKBENCH"),
        default="BLENDER_EEVEE",
        help="Render engine",
    )
    render.add_argument(
        "--margin",
        type=float,
        default=0.08,
        help="Camera framing margin as fraction of bounds",
    )
    render.add_argument("--color-mode", choices=("BW", "RGBA"), default=None)
    render.add_argument(
        "--transparent-bg", action=argparse.BooleanOptionalAction, default=None
    )
    render.add_argument(
        "--force-material", action=argparse.BooleanOptionalAction, default=None
    )
    render.add_argument(
        "--background-color", type=_parse_rgba, default=None, help="RGBA as r,g,b,a"
    )
    render.add_argument(
        "--silhouette-color", type=_parse_rgba, default=None, help="RGBA as r,g,b,a"
    )
    render.add_argument("--camera-distance-factor", type=float, default=None)
    render.add_argument(
        "--party-mode", action=argparse.BooleanOptionalAction, default=None
    )
    render.add_argument(
        "--render-output-dir",
        type=Path,
        default=None,
        help="Directory for rendered validation views.",
    )


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    profile = parser.add_argument_group("profile and loft")
    profile.add_argument("--legacy-profile-geometry", choices=("auto", "connected", "stacked"), default=None)
    profile.add_argument("--mesh-regularization-window", type=int, default=None,
                         help="Odd bounded Savitzky-Golay radius window; zero disables")
    profile.add_argument("--mesh-regularization-max-deviation", type=float, default=None,
                         help="Maximum source-section radius displacement in unchanged world units")
    profile.add_argument("--mesh-surface-mode", choices=("smooth", "stepped", "sharp"), default=None,
                         help="Connected loft geometry: smooth curve, stepped shoulders, or sharp section corners")
    profile.add_argument("--mesh-surface-subdivisions", type=int, default=None,
                         help="Smooth loft sections per input interval, 1..16")
    profile.add_argument(
        "--profile-samples",
        type=int,
        default=None,
        help="Profile sample count (profile_sampling.num_samples)",
    )
    profile.add_argument(
        "--profile-sample-policy",
        choices=("endpoints", "cell_centers"),
        default=None,
        help="Profile sampling policy",
    )
    profile.add_argument(
        "--profile-fill-strategy",
        choices=("interp_linear", "interp_nearest", "constant"),
        default=None,
        help="Profile fill strategy",
    )
    profile.add_argument(
        "--profile-smoothing-window",
        type=int,
        default=None,
        help="Median filter window for profile smoothing",
    )
    profile.add_argument(
        "--mesh-radial-segments",
        type=int,
        default=None,
        help="Loft mesh radial segments",
    )
    profile.add_argument(
        "--mesh-cap-mode",
        choices=("fan", "none", "ngon"),
        default=None,
        help="Loft mesh cap mode",
    )
    profile.add_argument(
        "--mesh-min-radius",
        type=float,
        default=None,
        help="Minimum radius (world units) for loft mesh slices",
    )
    profile.add_argument(
        "--mesh-merge-threshold",
        type=float,
        default=None,
        help="Merge threshold for loft mesh (world units)",
    )
    profile.add_argument(
        "--mesh-recalc-normals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recalculate loft mesh normals",
    )
    profile.add_argument(
        "--mesh-shade-smooth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable smooth shading on loft mesh",
    )
    profile.add_argument(
        "--mesh-weld-degenerate-rings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Weld degenerate rings in loft mesh",
    )
    profile.add_argument(
        "--mesh-adaptive-radial-segments",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    profile.add_argument("--mesh-min-adaptive-radial-segments", type=int, default=None)
    profile.add_argument("--mesh-max-adaptive-radial-segments", type=int, default=None)
    profile.add_argument(
        "--mesh-topology-strict", action=argparse.BooleanOptionalAction, default=None
    )
    profile.add_argument(
        "--mesh-research-allow-low-radial-segments",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


def add_silhouette_args(parser: argparse.ArgumentParser) -> None:
    silhouette = parser.add_argument_group("silhouette extraction")
    silhouette.add_argument(
        "--ref-prefer-alpha", action=argparse.BooleanOptionalAction, default=None
    )
    silhouette.add_argument(
        "--render-prefer-alpha", action=argparse.BooleanOptionalAction, default=None
    )
    silhouette.add_argument(
        "--ref-polarity",
        choices=("auto", "dark_foreground", "light_foreground", "alpha_foreground"),
        default=None,
    )
    silhouette.add_argument(
        "--render-polarity",
        choices=("auto", "dark_foreground", "light_foreground", "alpha_foreground"),
        default=None,
    )
    silhouette.add_argument(
        "--ref-invert-policy", choices=("auto", "invert", "no_invert"), default=None
    )
    silhouette.add_argument(
        "--render-invert-policy", choices=("auto", "invert", "no_invert"), default=None
    )
    silhouette.add_argument("--ref-alpha-threshold", type=int, default=None)
    silhouette.add_argument("--render-alpha-threshold", type=int, default=None)
    silhouette.add_argument("--ref-alpha-min-coverage", type=float, default=None)
    silhouette.add_argument("--render-alpha-min-coverage", type=float, default=None)
    silhouette.add_argument("--ref-gray-threshold", type=int, default=None)
    silhouette.add_argument("--render-gray-threshold", type=int, default=None)
    silhouette.add_argument("--ref-morph-close", type=int, default=None)
    silhouette.add_argument("--render-morph-close", type=int, default=None)
    silhouette.add_argument("--ref-morph-open", type=int, default=None)
    silhouette.add_argument("--render-morph-open", type=int, default=None)
    silhouette.add_argument(
        "--ref-fill-holes", action=argparse.BooleanOptionalAction, default=None
    )
    silhouette.add_argument(
        "--render-fill-holes", action=argparse.BooleanOptionalAction, default=None
    )
    silhouette.add_argument(
        "--ref-largest-component", action=argparse.BooleanOptionalAction, default=None
    )
    silhouette.add_argument(
        "--render-largest-component",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    silhouette.add_argument("--ref-min-area-frac", type=float, default=None)
    silhouette.add_argument("--render-min-area-frac", type=float, default=None)
    silhouette.add_argument("--ref-max-area-frac", type=float, default=None)
    silhouette.add_argument("--render-max-area-frac", type=float, default=None)
    silhouette.add_argument("--ref-max-border-contact-frac", type=float, default=None)
    silhouette.add_argument(
        "--render-max-border-contact-frac", type=float, default=None
    )
    silhouette.add_argument("--ref-min-component-area-px", type=int, default=None)
    silhouette.add_argument("--render-min-component-area-px", type=int, default=None)
    silhouette.add_argument(
        "--ref-candidate-scoring", action=argparse.BooleanOptionalAction, default=None
    )
    silhouette.add_argument(
        "--render-candidate-scoring",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    silhouette.add_argument(
        "--ref-emit-uncertainty", action=argparse.BooleanOptionalAction, default=None
    )
    silhouette.add_argument(
        "--render-emit-uncertainty", action=argparse.BooleanOptionalAction, default=None
    )


def add_canonicalization_args(parser: argparse.ArgumentParser) -> None:
    canonical = parser.add_argument_group("canonicalization and IoU")
    canonical.add_argument("--canonical-output-size", type=int, default=None)
    canonical.add_argument("--canonical-padding-frac", type=float, default=None)
    canonical.add_argument(
        "--canonical-anchor", choices=("center", "bottom_center"), default=None
    )
    canonical.add_argument("--canonical-interp", choices=("nearest",), default=None)
    canonical.add_argument(
        "--canonical-cache", action=argparse.BooleanOptionalAction, default=None
    )
    canonical.add_argument("--canonical-digest", choices=("sha256",), default=None)
    canonical.add_argument(
        "--canonical-fill-holes", action=argparse.BooleanOptionalAction, default=None
    )
    canonical.add_argument(
        "--canonical-largest-component",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


def add_mesh_join_args(parser: argparse.ArgumentParser) -> None:
    join = parser.add_argument_group("mesh join and silhouette intersection")
    join.add_argument(
        "--mesh-join-mode", choices=("auto", "boolean", "voxel", "simple"), default=None
    )
    join.add_argument(
        "--boolean-solver",
        choices=("auto", "EXACT", "MANIFOLD", "FLOAT", "FAST"),
        default=None,
    )
    join.add_argument(
        "--allow-degraded-simple-join",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    join.add_argument(
        "--record-join-attempts", action=argparse.BooleanOptionalAction, default=None
    )
    join.add_argument(
        "--balanced-boolean-tree", action=argparse.BooleanOptionalAction, default=None
    )
    join.add_argument("--silhouette-extrude-distance", type=float, default=None)
    join.add_argument(
        "--silhouette-contour-mode",
        choices=("external", "ccomp", "tree", "hierarchy"),
        default=None,
    )
    join.add_argument(
        "--silhouette-largest-component",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
