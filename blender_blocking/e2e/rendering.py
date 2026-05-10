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
from blender_blocking.e2e.payloads import _is_renderable_mesh


def _import_obj_for_render(path: Path) -> Optional[object]:
    before = {obj.name for obj in bpy.context.scene.objects}
    try:
        if hasattr(bpy.ops.wm, "obj_import"):
            # Backend OBJ writers emit Blender Z-up coordinates directly.  The
            # Blender 5 importer defaults to Y-up OBJ conversion, which rotates
            # visual-hull meshes into a top-like front render and collapses
            # side/top IoU.  Preserve the file coordinates during validation.
            bpy.ops.wm.obj_import(
                filepath=str(path),
                forward_axis="Y",
                up_axis="Z",
            )
        else:
            bpy.ops.import_scene.obj(
                filepath=str(path),
                axis_forward="Y",
                axis_up="Z",
            )
    except Exception as exc:
        print(f"Warning: failed to import OBJ for render validation: {exc}")
        return None
    imported = [
        obj
        for obj in bpy.context.scene.objects
        if obj.name not in before and getattr(obj, "type", None) == "MESH"
    ]
    if imported:
        return imported[0]
    active = bpy.context.active_object
    return active if _is_renderable_mesh(active) else None
