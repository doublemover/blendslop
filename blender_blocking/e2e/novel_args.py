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
    return rgba


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
