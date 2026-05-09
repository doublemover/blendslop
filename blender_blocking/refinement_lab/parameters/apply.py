"""Apply refinement variant parameters to BlockingConfig."""

from __future__ import annotations

import json
from typing import Any

from .catalog import CONFIG_PARAMETER_PATHS, SPECIAL_PARAMETER_NAMES


def apply_variant_parameter_to_config(cfg: Any, key: str, value: Any) -> bool:
    """Apply one variant parameter to a BlockingConfig-like object.

    Returns True when the parameter is known and applied, False when the
    parameter is intentionally unknown to this catalog and should be ignored by
    callers that support lab-only metadata.
    """
    key = str(key)
    if key == "ensemble_candidates":
        _apply_ensemble_candidates(cfg, value)
        return True
    target_path = CONFIG_PARAMETER_PATHS.get(key)
    if target_path is None:
        return False
    group_name, attr_name = target_path
    target = getattr(cfg, group_name)
    current = getattr(target, attr_name)
    setattr(target, attr_name, _coerce_parameter_value(key, attr_name, value, current))
    return True


def _apply_ensemble_candidates(cfg: Any, value: Any) -> None:
    try:
        from blender_blocking.config import CandidateConfig
    except ImportError:  # pragma: no cover - script-style imports
        from config import CandidateConfig

    cfg.ensemble.candidates = tuple(
        CandidateConfig(
            backend_name=str(name),
            candidate_id=f"{name}_{idx:02d}",
        )
        for idx, name in enumerate(value)
    )


def _coerce_parameter_value(
    key: str,
    attr_name: str,
    value: Any,
    current: Any,
) -> Any:
    if attr_name == "loss_weights" and isinstance(value, str):
        return json.loads(value)
    if attr_name == "primitive_families":
        return (
            tuple(value)
            if isinstance(value, (list, tuple))
            else tuple(str(value).split(","))
        )
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    if key.endswith("_targets") and isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return value


def known_parameter_name(key: str) -> bool:
    return str(key) in CONFIG_PARAMETER_PATHS or str(key) in SPECIAL_PARAMETER_NAMES
