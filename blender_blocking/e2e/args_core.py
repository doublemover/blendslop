from __future__ import annotations

import argparse
from pathlib import Path

from blender_blocking.e2e.constants import ALL_RECONSTRUCTION_MODES, VALIDATION_MODES
from blender_blocking.e2e.novel_args import _parse_csv

def add_core_args(parser: argparse.ArgumentParser) -> None:
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
