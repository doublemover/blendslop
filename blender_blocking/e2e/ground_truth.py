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


def _synthetic_ground_truth_row(
    spec: object,
    result_payload: Mapping[str, Any],
    *,
    reference_paths: Optional[Mapping[str, str]] = None,
    config: Optional[BlockingConfig] = None,
) -> Dict[str, Any]:
    try:
        from blender_blocking.synthetic.ground_truth import (
            build_pure_artifacts,
            geometry_payload_from_candidate,
            recoverable_envelope_from_views,
        )
    except Exception:
        return {"available": False, "reason": "synthetic ground truth module unavailable"}

    try:
        reference = build_pure_artifacts(spec, volume_resolution=48)  # type: ignore[arg-type]
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    metadata = dict(reference.get("metadata", {}) or {})
    quality_targets = reference.get("quality_targets", {})
    row: Dict[str, Any] = {
        "available": True,
        "shape_id": getattr(spec, "shape_id", ""),
        "ground_truth_level": metadata.get("ground_truth_level", ""),
        "metadata": metadata,
        "quality_targets": quality_targets,
        "metrics": {},
    }
    if not reference.get("sdf_samples"):
        row["reason"] = "no analytic SDF geometry for this synthetic fixture"
        return row

    mesh_path = _mesh_path_from_payload(result_payload)
    if mesh_path is None:
        row["reason"] = "candidate mesh artifact not found"
        return row
    candidate_points = _obj_vertices(mesh_path, max_points=8192)
    if candidate_points is None or len(candidate_points) == 0:
        row["reason"] = f"candidate mesh had no readable vertices: {mesh_path}"
        return row
    recoverable_surface_points = None
    recoverable_occupancy = None
    if reference_paths:
        try:
            reference_views = {
                view: load_image(path)
                for view, path in reference_paths.items()
                if path and Path(path).exists()
            }
            if reference_views:
                recoverable = recoverable_envelope_from_views(
                    reference_views,
                    config=config,
                    resolution=32,
                    max_surface_points=8192,
                    profile_samples=64,
                )
                recoverable_surface_points = recoverable.get("surface_points")
                recoverable_occupancy = recoverable.get("occupancy")
                row["recoverable_envelope"] = recoverable.get("metadata", {})
        except Exception as exc:
            row["recoverable_envelope"] = {
                "available": False,
                "reason": str(exc),
            }

    try:
        payload = geometry_payload_from_candidate(
            reference,
            candidate_surface_points=candidate_points,
            recoverable_surface_points=recoverable_surface_points,
            recoverable_occupancy=recoverable_occupancy,
            tolerance=0.03,
        )
    except Exception as exc:
        row["reason"] = str(exc)
        return row

    row["geometry_payload"] = payload
    row["mesh_path"] = mesh_path.as_posix()
    row["metrics"] = _flatten_geometry_payload(payload)
    return row

def _mesh_path_from_payload(payload: Mapping[str, Any]) -> Optional[Path]:
    candidates: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"mesh_path", "mesh_obj"} and item:
                    candidates.append(str(item))
                else:
                    visit(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                visit(item)

    visit(payload)
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None

def _obj_vertices(path: Path, *, max_points: int) -> Optional[Any]:
    try:
        import numpy as np

        points: list[tuple[float, float, float]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    points.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError:
                    continue
        if not points:
            return None
        array = np.asarray(points, dtype=np.float32)
        if len(array) > max_points:
            indices = np.linspace(0, len(array) - 1, int(max_points)).round().astype(int)
            array = array[indices]
        return array
    except Exception:
        return None

def _flatten_geometry_payload(payload: Mapping[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                next_prefix = f"{prefix}_{key}" if prefix else str(key)
                visit(next_prefix, item)
            return
        if isinstance(value, bool):
            metrics[prefix] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            metrics[prefix] = float(value)

    visit("synthetic", payload)
    return metrics
