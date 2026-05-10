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
from blender_blocking.e2e.payloads import _mean_optional, _optional_float


def _load_novel_pair(reference_path: str, rendered_path: str) -> tuple[Any, Any, tuple[str, ...]]:
    reference = load_image(reference_path)
    rendered = load_image(rendered_path)
    warnings: list[str] = []
    if np.asarray(reference).shape != np.asarray(rendered).shape:
        if not PIL_AVAILABLE:
            raise ValueError(
                "novel-view reference/render image shapes differ and Pillow is unavailable"
            )
        rendered = _resize_image_like(rendered, reference)
        warnings.append("rendered image resized to match reference for image metrics")
    return reference, rendered, tuple(warnings)

def _resize_image_like(image: Any, reference: Any) -> np.ndarray:
    reference_shape = np.asarray(reference).shape
    if len(reference_shape) < 2:
        raise ValueError("reference image must have at least two dimensions")
    height, width = int(reference_shape[0]), int(reference_shape[1])
    array = np.asarray(image)
    mode = None
    if array.ndim == 2:
        mode = "L"
    pil_image = Image.fromarray(array.astype(np.uint8), mode=mode)
    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    resized = pil_image.resize((width, height), resample=resample)
    return np.asarray(resized)

def _novel_view_gate(
    report: Mapping[str, Any],
    *,
    psnr_threshold: Optional[float],
    ssim_threshold: Optional[float],
    lpips_threshold: Optional[float],
) -> Dict[str, Any]:
    failures: list[str] = []
    _threshold_min(
        report,
        "psnr",
        psnr_threshold,
        failures,
        label="PSNR",
    )
    _threshold_min(
        report,
        "ssim",
        ssim_threshold,
        failures,
        label="SSIM",
    )
    _threshold_max(
        report,
        "lpips",
        lpips_threshold,
        failures,
        label="LPIPS",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "thresholds": {
            "psnr_min": psnr_threshold,
            "ssim_min": ssim_threshold,
            "lpips_max": lpips_threshold,
        },
    }

def _threshold_min(
    report: Mapping[str, Any],
    key: str,
    threshold: Optional[float],
    failures: list[str],
    *,
    label: str,
) -> None:
    if threshold is None:
        return
    value = _optional_float(report.get(key))
    if value is None:
        failures.append(f"{label} missing")
    elif value < float(threshold):
        failures.append(f"{label} {value:.4g} below {float(threshold):.4g}")

def _threshold_max(
    report: Mapping[str, Any],
    key: str,
    threshold: Optional[float],
    failures: list[str],
    *,
    label: str,
) -> None:
    if threshold is None:
        return
    value = _optional_float(report.get(key))
    if value is None:
        failures.append(f"{label} missing")
    elif value > float(threshold):
        failures.append(f"{label} {value:.4g} above {float(threshold):.4g}")

def _aggregate_novel_reports(
    pair_reports: Mapping[str, Mapping[str, Any]],
    *,
    missing_views: Sequence[str],
    psnr_threshold: Optional[float],
    ssim_threshold: Optional[float],
    lpips_threshold: Optional[float],
) -> Dict[str, Any]:
    metrics = {
        key: _mean_optional(
            report.get(key)
            for report in pair_reports.values()
            if isinstance(report, Mapping)
        )
        for key in ("psnr", "ssim", "lpips", "mse")
    }
    gate = _novel_view_gate(
        metrics,
        psnr_threshold=psnr_threshold,
        ssim_threshold=ssim_threshold,
        lpips_threshold=lpips_threshold,
    )
    failed_views = [
        view
        for view, report in pair_reports.items()
        if not bool(report.get("gate", {}).get("passed", False))
    ]
    warnings = [
        str(warning)
        for report in pair_reports.values()
        for warning in report.get("warnings", [])
    ]
    dependency_state = {
        view: dict(report.get("dependency_state", {}))
        for view, report in pair_reports.items()
        if isinstance(report.get("dependency_state"), Mapping)
        and report.get("dependency_state")
    }
    return {
        **metrics,
        "image_count": len(pair_reports),
        "missing_views": list(missing_views),
        "failed_views": failed_views,
        "passed": bool(pair_reports) and not missing_views and not failed_views and gate["passed"],
        "gate": gate,
        "thresholds": gate["thresholds"],
        "warnings": warnings,
        "dependency_state": dependency_state,
    }
