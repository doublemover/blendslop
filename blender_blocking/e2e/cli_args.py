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


def _parse_resolution(value: str) -> Tuple[int, int]:
    if "x" in value:
        parts = value.lower().split("x", 1)
    elif "," in value:
        parts = value.split(",", 1)
    else:
        parts = [value]
    try:
        if len(parts) == 1:
            size = int(parts[0])
            return (size, size)
        width = int(parts[0])
        height = int(parts[1])
        return (width, height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "resolution must be N or WxH (e.g., 512 or 1024x1024)"
        ) from exc

def _parse_csv(value: str) -> Tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return items

def _parse_rgba(value: str) -> Tuple[float, float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("RGBA must be r,g,b,a")
    try:
        rgba = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RGBA values must be numbers") from exc
    if any(channel < 0.0 or channel > 1.0 for channel in rgba):
        raise argparse.ArgumentTypeError("RGBA values must be in [0, 1]")
    return rgba  # type: ignore[return-value]

def _parse_view_reference_entries(entries: Iterable[str]) -> Dict[str, Path]:
    references: Dict[str, Path] = {}
    for entry in entries or ():
        if "=" not in str(entry):
            raise argparse.ArgumentTypeError(
                "--novel-view-reference must use VIEW=PATH"
            )
        view, path = str(entry).split("=", 1)
        view = _validate_novel_view_name(view.strip())
        references[view] = Path(path.strip())
    return references

def _novel_view_names_from_args(args: argparse.Namespace) -> Tuple[str, ...]:
    angle_views = []
    for angle_text in args.novel_view_angles or ():
        try:
            angle = float(angle_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid --novel-view-angles value: {angle_text!r}"
            ) from exc
        angle_views.append(f"orbit_{int(round(angle % 360.0)):03d}")
    references = _parse_view_reference_entries(args.novel_view_reference)
    return _ordered_unique(
        tuple(args.novel_view or ())
        + tuple(angle_views)
        + tuple(references.keys())
    )

def _resolve_novel_view_inputs(
    args: argparse.Namespace,
) -> tuple[
    Dict[str, Path],
    Tuple[str, ...],
    bool,
    bool,
    Optional[float],
    Optional[float],
    Optional[float],
]:
    manifest = _load_novel_view_manifest(getattr(args, "novel_view_manifest", None))
    manual_references = _parse_view_reference_entries(args.novel_view_reference)
    references = {
        **manifest["references"],
        **manual_references,
    }
    view_names = _ordered_unique(
        tuple(manifest["view_names"])
        + tuple(_novel_view_names_from_args(args))
        + tuple(references.keys())
    )
    options = manifest["options"]
    compute_ssim = bool(
        args.novel_compute_ssim
        if args.novel_compute_ssim is not None
        else options.get("compute_ssim", True)
    )
    compute_lpips = bool(
        args.novel_compute_lpips
        if args.novel_compute_lpips is not None
        else options.get("compute_lpips", False)
    )
    psnr_threshold = _novel_threshold(
        args.novel_psnr_threshold
        if args.novel_psnr_threshold is not None
        else options.get("psnr_threshold", 20.0)
    )
    ssim_threshold = _novel_threshold(
        args.novel_ssim_threshold
        if args.novel_ssim_threshold is not None
        else options.get("ssim_threshold", 0.65)
    )
    lpips_threshold = _novel_threshold(
        args.novel_lpips_threshold
        if args.novel_lpips_threshold is not None
        else options.get("lpips_threshold")
    )
    return (
        references,
        view_names,
        compute_ssim,
        compute_lpips,
        psnr_threshold,
        ssim_threshold,
        lpips_threshold,
    )

def _load_novel_view_manifest(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {"references": {}, "view_names": (), "options": {}}
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise argparse.ArgumentTypeError("--novel-view-manifest must be a JSON object")
    base_dir = manifest_path.parent
    references: Dict[str, Path] = {}
    view_names: list[str] = []

    references_payload = payload.get("references", {})
    if isinstance(references_payload, Mapping):
        for view, raw_path in references_payload.items():
            view_name = _validate_novel_view_name(str(view))
            references[view_name] = _manifest_relative_path(base_dir, raw_path)
            view_names.append(view_name)

    views_payload = payload.get("views", ())
    if isinstance(views_payload, Sequence) and not isinstance(
        views_payload, (str, bytes)
    ):
        for item in views_payload:
            if isinstance(item, Mapping):
                view_name = _validate_novel_view_name(str(item.get("view", "")))
                view_names.append(view_name)
                if item.get("reference") is not None:
                    references[view_name] = _manifest_relative_path(
                        base_dir,
                        item.get("reference"),
                    )
            elif item:
                view_names.append(_validate_novel_view_name(str(item)))

    for angle in _manifest_angles(payload.get("angles", ())):
        view_names.append(f"orbit_{int(round(angle % 360.0)):03d}")

    metric_options = payload.get("metrics", {})
    options: Dict[str, Any] = {}
    if isinstance(metric_options, Mapping):
        options.update(metric_options)
    for source_key, target_key in (
        ("compute_ssim", "compute_ssim"),
        ("compute_lpips", "compute_lpips"),
        ("psnr_threshold", "psnr_threshold"),
        ("ssim_threshold", "ssim_threshold"),
        ("lpips_threshold", "lpips_threshold"),
    ):
        if source_key in payload:
            options[target_key] = payload[source_key]
    return {
        "references": references,
        "view_names": _ordered_unique(view_names),
        "options": options,
    }

def _validate_novel_view_name(value: str) -> str:
    view = value.strip()
    if not view:
        raise argparse.ArgumentTypeError("novel view name cannot be empty")
    if view not in {"front", "side", "top"} and parse_orbit_view_degrees(view) is None:
        raise argparse.ArgumentTypeError(
            "novel view names must be front, side, top, or orbit/azimuth names"
        )
    return view

def _manifest_relative_path(base_dir: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path

def _manifest_angles(value: object) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Iterable[object] = value.split(",")
    elif isinstance(value, Sequence):
        values = value
    else:
        values = (value,)
    angles = []
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        try:
            angles.append(float(text))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid novel-view manifest angle: {text!r}"
            ) from exc
    return tuple(angles)

def _novel_threshold(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return None if value <= 0.0 else value

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate blendslop reconstruction modes from reference silhouettes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  blender --background --python blender_blocking/test_e2e_validation.py -- --reconstruction-mode legacy --no-progress
  blender --background --python blender_blocking/test_e2e_validation.py -- --reconstruction-mode loft_profile --mesh-radial-segments 32 --profile-samples 120 --no-progress
  blender --background --python blender_blocking/test_e2e_validation.py -- --reconstruction-mode visual_hull_voxel --validation-mode backend-status --vh-resolution 32 --vh-backend chunked --vh-chunk-size 16 --vh-mesh-method points --no-progress
  blender --background --python blender_blocking/test_e2e_validation.py -- --reconstruction-mode legacy --validation-mode novel-view --novel-view-angles 45,135 --no-progress
  blender --background --python blender_blocking/test_e2e_validation.py -- --reconstruction-mode ensemble --validation-mode backend-status --ensemble-candidates visual_hull_voxel,primitive_fit_refine,gaussian_ellipsoid_proxy --primitive-max 6 --gaussian-count 8 --no-progress
  python blender_blocking/test_e2e_validation.py --reconstruction-mode ensemble --print-config --dry-run

Modes:
  {", ".join(ALL_RECONSTRUCTION_MODES)}

Default ensemble:
  {", ".join(DEFAULT_ENSEMBLE_CANDIDATES)}
""",
    )

    core = parser.add_argument_group("core")
    core.add_argument(
        "--list-modes",
        action="store_true",
        help="Print available reconstruction modes and exit.",
    )
    core.add_argument(
        "--config-json",
        type=str,
        default=None,
        help='Inline JSON overrides for BlockingConfig (e.g. \'{"reconstruction": {"num_slices": 160}}\')',
    )
    core.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to JSON file with BlockingConfig overrides",
    )
    core.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config before running.",
    )
    core.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate CLI/config overrides, print config if requested, then exit.",
    )
    core.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Write validation/backend result JSON to this path.",
    )
    core.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Deterministic run id used for manifests and backend artifacts.",
    )
    core.add_argument(
        "--config-label",
        type=str,
        default=None,
        help="Label included in render filenames. Defaults to config path/inline/default.",
    )
    core.add_argument(
        "--front",
        type=str,
        default=None,
        help="Path to front view image (PNG/JPG)",
    )
    core.add_argument(
        "--side",
        type=str,
        default=None,
        help="Path to side view image (PNG/JPG)",
    )
    core.add_argument(
        "--top",
        type=str,
        default=None,
        help="Path to top view image (PNG/JPG)",
    )
    core.add_argument(
        "--validation-mode",
        choices=VALIDATION_MODES,
        default="auto",
        help="auto renders mesh modes, checks backend status for artifact-only modes, or run novel-view image metrics explicitly.",
    )
    core.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="Per-required-view IoU threshold for render-iou validation.",
    )
    core.add_argument("--front-threshold", type=float, default=None)
    core.add_argument("--side-threshold", type=float, default=None)
    core.add_argument("--top-threshold", type=float, default=None)
    core.add_argument(
        "--boundary-iou-threshold",
        type=float,
        default=None,
        help="Optional per-view Boundary IoU threshold for render-iou validation.",
    )
    core.add_argument(
        "--signed-distance-loss-threshold",
        type=float,
        default=None,
        help="Optional maximum per-view signed-distance silhouette loss.",
    )
    core.add_argument(
        "--novel-view-reference",
        action="append",
        default=[],
        metavar="VIEW=PATH",
        help="Reference image for image metrics. Repeat for views like orbit_045=path/to/ref.png.",
    )
    core.add_argument(
        "--novel-view-manifest",
        type=Path,
        default=None,
        help="JSON manifest containing held-out novel-view references, extra views, angles, and metric options.",
    )
    core.add_argument(
        "--novel-view",
        action="append",
        default=[],
        metavar="VIEW",
        help="Additional rendered view name for novel-view validation, e.g. orbit_045.",
    )
    core.add_argument(
        "--novel-view-angles",
        type=_parse_csv,
        default=(),
        help="Comma-separated orbit degrees to render as orbit_XXX views for novel-view validation.",
    )
    core.add_argument(
        "--novel-compute-ssim",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute SSIM for novel-view/image validation.",
    )
    core.add_argument(
        "--novel-compute-lpips",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute LPIPS for novel-view/image validation when torch/lpips are available.",
    )
    core.add_argument(
        "--novel-psnr-threshold",
        type=float,
        default=None,
        help="Minimum aggregate and per-view PSNR for novel-view validation; use 0 to disable.",
    )
    core.add_argument(
        "--novel-ssim-threshold",
        type=float,
        default=None,
        help="Minimum aggregate and per-view SSIM for novel-view validation; use 0 to disable.",
    )
    core.add_argument(
        "--novel-lpips-threshold",
        type=float,
        default=None,
        help="Optional maximum aggregate and per-view LPIPS for novel-view validation.",
    )
    core.add_argument(
        "--reconstruction-mode",
        choices=ALL_RECONSTRUCTION_MODES,
        default="legacy",
        help="Reconstruction mode to validate.",
    )
    core.add_argument(
        "--num-slices",
        type=int,
        default=120,
        help="Number of slices used for legacy/loft/profile workflows.",
    )
    core.add_argument(
        "--unit-scale",
        type=float,
        default=None,
        help="World units per pixel.",
    )

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

    profile = parser.add_argument_group("profile and loft")
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

    misc = parser.add_argument_group("constraints and quality")
    misc.add_argument("--constraint-file", action="append", default=None)
    misc.add_argument(
        "--fail-on-hard-constraints",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    misc.add_argument(
        "--use-constraints-for-scoring",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    misc.add_argument("--quality-budget-json", type=str, default=None)
    misc.add_argument("--quality-compare-baseline", type=str, default=None)
    misc.add_argument(
        "--quality-fail-on-regression",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    misc.add_argument(
        "--cost-report-json",
        type=Path,
        default=None,
        help="Write a top-level E2E cost report with validation stages and nested backend cost.",
    )
    misc.add_argument(
        "--cost-track-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Track peak Python allocation memory for cost stages.",
    )
    misc.add_argument(
        "--cost-fail-max-wall-ms",
        type=float,
        default=None,
        help="Fail the run if combined validation plus backend wall time exceeds this many milliseconds.",
    )
    misc.add_argument(
        "--cost-fail-max-backend-wall-ms",
        type=float,
        default=None,
        help="Fail the run if nested backend reconstruction wall time exceeds this many milliseconds.",
    )
    misc.add_argument(
        "--environment-compatibility",
        choices=("warn", "strict", "ignore"),
        default=None,
    )
    misc.add_argument("--synthetic-suite", type=str, default=None)
    misc.add_argument("--synthetic-seed", type=int, default=None)
    misc.add_argument("--synthetic-output-root", type=str, default=None)
    misc.add_argument(
        "--synthetic-matrix",
        action="store_true",
        help="Run the synthetic suite x reconstruction mode matrix instead of a single sample/custom validation.",
    )
    misc.add_argument(
        "--synthetic-modes",
        type=_parse_csv,
        default=None,
        help="Comma-separated reconstruction modes for --synthetic-matrix.",
    )
    misc.add_argument(
        "--synthetic-count",
        type=int,
        default=None,
        help="Optional synthetic spec count for --synthetic-matrix.",
    )
    misc.add_argument(
        "--synthetic-strict-skips",
        action="store_true",
        help="Treat skipped synthetic matrix rows as failures.",
    )
    misc.add_argument(
        "--quality-report-json",
        type=Path,
        default=None,
        help="Write quality budget report JSON after --synthetic-matrix.",
    )
    misc.add_argument(
        "--synthetic-commit-small-fixtures-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    misc.add_argument(
        "--synthetic-keep-heavy-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    misc.add_argument(
        "--artifact-output-root",
        type=Path,
        default=None,
        help="Write reconstruction backend artifacts under this root.",
    )

    refinement = parser.add_argument_group("refinement lab")
    refinement.add_argument("--refinement-suite", type=str, default=None)
    refinement.add_argument("--refinement-track", type=str, default=None)
    refinement.add_argument(
        "--refinement-search",
        choices=("grid", "random", "coordinate", "successive_halving"),
        default=None,
    )
    refinement.add_argument(
        "--refinement-objective",
        choices=(
            "quality_win",
            "min_view_iou",
            "mean_iou",
            "profile_editable",
            "visual_hull_alignment",
            "fast_preview",
            "human_adjusted",
        ),
        default=None,
    )
    refinement.add_argument("--refinement-result-root", type=Path, default=None)
    refinement.add_argument("--refinement-preset-json", type=Path, default=None)
    refinement.add_argument("--refinement-max-runs", type=int, default=None)
    refinement.add_argument("--refinement-top-k", type=int, default=None)
    refinement.add_argument("--refinement-seed", type=int, default=None)
    refinement.add_argument(
        "--refinement-variant-file",
        type=Path,
        action="append",
        default=None,
        help="Adaptive variant JSON file to append or replace generated variants.",
    )
    refinement.add_argument(
        "--refinement-variant-file-mode",
        choices=("append", "replace"),
        default=None,
    )
    refinement.add_argument(
        "--refinement-html-report", action=argparse.BooleanOptionalAction, default=None
    )
    refinement.add_argument(
        "--refinement-write-overlays",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    refinement.add_argument(
        "--refinement-bounds-debug", action=argparse.BooleanOptionalAction, default=None
    )
    refinement.add_argument(
        "--refinement-autopsy", action=argparse.BooleanOptionalAction, default=None
    )
    refinement.add_argument(
        "--refinement-copy-references",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    refinement.add_argument("--refinement-stop-on-first-error", action="store_true")
    refinement.add_argument(
        "--refinement-fail-on-all-failed",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    refinement.add_argument(
        "--refinement-append-global-index",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    refinement.add_argument(
        "--refinement-report-failures", choices=("top", "all", "none"), default=None
    )
    refinement.add_argument("--refinement-subprocess", action="store_true")
    refinement.add_argument("--refinement-blender-exe", type=str, default=None)

    misc.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        default=True,
        help="Disable progress bars",
    )
    return parser.parse_args(argv)

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
