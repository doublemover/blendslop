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


def _ordered_unique(values: Iterable[str]) -> Tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return tuple(ordered)

def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _mean_optional(values: Iterable[object]) -> Optional[float]:
    filtered = [
        float(value)
        for value in values
        if value is not None and _optional_float(value) is not None
    ]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))

def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

def _bounded_slug(value: object, *, max_len: int = 32) -> str:
    text = str(value or "default").strip()
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-._")
    if not slug:
        slug = "default"
    if len(slug) <= max_len:
        return slug
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:8]
    head_len = max(1, max_len - len(digest) - 1)
    return f"{slug[:head_len]}-{digest}"

def _render_filename_prefix(
    base_name: Optional[str],
    technique: str,
    config_label: str,
) -> str:
    """Build a readable but path-safe prefix for refinement render outputs."""
    parts = []
    if base_name:
        parts.append(_bounded_slug(base_name, max_len=10))
    parts.append(_bounded_slug(technique or "legacy", max_len=14))
    parts.append(_bounded_slug(config_label or "default", max_len=18))
    return "_".join(parts) + "_"

def _is_renderable_mesh(value: object) -> bool:
    return getattr(value, "type", None) == "MESH"

def _find_renderable_mesh(value: object) -> Optional[object]:
    if _is_renderable_mesh(value):
        return value
    if any(_is_renderable_mesh(child) for child in getattr(value, 'children_recursive', ())):
        # Compiled shape programs use an Empty root. Keep the entire part tree.
        return value
    if isinstance(value, Mapping):
        for key in ("profile_payload", "mesh", "object", "payload"):
            if key in value:
                found = _find_renderable_mesh(value[key])
                if found is not None:
                    return found
        for nested in value.values():
            found = _find_renderable_mesh(nested)
            if found is not None:
                return found
    if hasattr(value, "selected"):
        selected = getattr(value, "selected", None)
        found = _find_renderable_mesh(selected)
        if found is not None:
            return found
    if hasattr(value, "payload"):
        return _find_renderable_mesh(getattr(value, "payload"))
    return None

def _selected_result_dict(result: object) -> Dict[str, Any]:
    if result is None:
        return {}
    if hasattr(result, "selected"):
        selected = getattr(result, "selected", None)
        return selected.to_dict() if hasattr(selected, "to_dict") else {}
    return result.to_dict() if hasattr(result, "to_dict") else {}

def _mesh_path_from_backend_result(result: object) -> Optional[Path]:
    data = _selected_result_dict(result)
    mesh_path = data.get("mesh_path") or data.get("artifacts", {}).get("mesh_obj")
    if not mesh_path:
        return None
    path = Path(str(mesh_path))
    return path if path.exists() else None

def _evaluation_payload_from_result(result: object) -> Dict[str, Any]:
    """Expose EvaluationBundle data in e2e JSON, even for single candidates."""
    if result is None:
        return {}
    if hasattr(result, "evaluation_bundles"):
        bundles = [
            _to_jsonable(bundle)
            for bundle in (getattr(result, "evaluation_bundles", ()) or ())
        ]
        payload: Dict[str, Any] = {}
        if bundles:
            payload["evaluation_bundles"] = bundles
        autopsies = [
            _to_jsonable(pack) for pack in (getattr(result, "autopsy_packs", ()) or ())
        ]
        if autopsies:
            payload["autopsy_packs"] = autopsies
        return payload
    if hasattr(result, "to_evaluation_bundle"):
        try:
            bundle = result.to_evaluation_bundle(
                suite=str(getattr(result, "backend_name", "")),
                run_id=str(getattr(result, "candidate_id", "")),
            )
            payload = {"evaluation_bundle": _to_jsonable(bundle)}
            try:
                from blender_blocking.evaluation.autopsy import (
                    autopsy_pack_from_bundle,
                )

                payload["autopsy_pack"] = _to_jsonable(autopsy_pack_from_bundle(bundle))
            except Exception as exc:
                payload["autopsy_pack_error"] = str(exc)
            return payload
        except Exception as exc:
            return {"evaluation_bundle_error": str(exc)}
    return {}

def _e2e_payload_with_evaluation(
    payload: Mapping[str, Any],
    result: object,
) -> Dict[str, Any]:
    merged = dict(payload)
    evaluation = _evaluation_payload_from_result(result)
    if evaluation:
        merged.update(evaluation)
    return merged

def _evaluation_outputs_from_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    if not isinstance(payload, Mapping):
        return outputs
    bundles = payload.get("evaluation_bundles")
    if isinstance(bundles, Sequence) and not isinstance(bundles, (str, bytes)):
        outputs["evaluation_bundles"] = list(bundles)
    bundle = payload.get("evaluation_bundle")
    if isinstance(bundle, Mapping):
        outputs["evaluation_bundle"] = dict(bundle)
    for key in ("autopsy_packs", "autopsy_pack"):
        value = payload.get(key)
        if value:
            outputs[key] = value
    backend = payload.get("backend_result")
    if isinstance(backend, Mapping):
        nested = _evaluation_outputs_from_payload(backend)
        for key, value in nested.items():
            outputs.setdefault(key, value)
    return outputs

def _to_jsonable(value: object) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
