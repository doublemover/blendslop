from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from blender_blocking.utils.optional_deps import dependency_report, probe_dependency
except Exception:  # pragma: no cover - script-style imports
    from utils.optional_deps import dependency_report, probe_dependency


def _visual_hull_cache_directory(config: Mapping[str, Any]) -> Path | None:
    raw = (
        config.get("cache_directory")
        or config.get("volume_cache_directory")
        or config.get("vh_cache_directory")
    )
    if raw:
        return Path(str(raw))
    if bool(config.get("enable_cache") or config.get("cache_chunks")):
        return Path("temp") / "volume-chunk-cache"
    return None

def _sdf_projection_enabled(config: Mapping[str, Any]) -> bool:
    return bool(
        config.get("sdf_projection")
        or config.get("project_sdf")
        or config.get("signed_distance_projection")
        or _mesh_uses_sdf(config)
    )

def _mesh_uses_sdf(config: Mapping[str, Any]) -> bool:
    mesh_source = str(config.get("mesh_source", "")).strip().lower()
    return bool(
        config.get("mesh_from_sdf")
        or mesh_source in {"sdf", "signed_distance", "signed_distance_field"}
    )

def _sdf_required(config: Mapping[str, Any]) -> bool:
    return bool(
        config.get("require_sdf")
        or config.get("sdf_required")
        or config.get("fail_on_sdf_skip")
        or _mesh_uses_sdf(config)
    )
