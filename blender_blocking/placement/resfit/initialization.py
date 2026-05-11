from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..resfit_initialization import (
    PrimitiveInitializationConfig,
    initialize_ellipsoids_from_points,
    initialize_from_profile_bands,
    initialize_gaussians_from_points,
    initialize_superfrusta_from_points,
    initialize_superquadrics_from_points,
)
from ..resfit_objective import (
    PenaltyHook,
    ResFitLossWeights,
    ResFitObjectiveResult,
    SilhouetteHook,
    evaluate_resfit_objective,
)
from ..resfit_optimizer import (
    CoordinateDescentConfig,
    OptimizationRecord,
    coordinate_descent_optimize,
)

from .config import ResFitPipelineConfig, ResFitPipelineResult

InitializerFn = Callable[[np.ndarray, PrimitiveInitializationConfig], Sequence[object]]


def get_initializer(family: str) -> InitializerFn:
    """Return the initializer for a primitive family."""
    normalized = family.lower().strip()
    if normalized in ("superfrustum", "superfrusta", "frustum"):
        return initialize_superfrusta_from_points
    if normalized in ("ellipsoid", "ellipsoids"):
        return initialize_ellipsoids_from_points
    if normalized in ("superquadric", "superquadrics", "boxy_superquadric"):
        return initialize_superquadrics_from_points
    if normalized in ("gaussian", "gaussians", "anisotropic_gaussian"):
        return initialize_gaussians_from_points
    raise ValueError(f"unknown primitive family: {family}")


def _family_supports_profile_init(primitive_family: str) -> bool:
    return str(primitive_family).lower().strip() in {"superfrustum", "superfrusta", "frustum"}


def _first_family(config: Mapping[str, object]) -> str:
    family = config.get("primitive_family")
    if family:
        return str(family)
    families = config.get("primitive_families")
    if isinstance(families, (list, tuple)) and families:
        return str(families[0])
    return "superfrustum"


def _candidate_families(config: Mapping[str, object]) -> tuple[str, ...]:
    family = config.get("primitive_family")
    if family:
        return (str(family),)
    families = config.get("primitive_families")
    if isinstance(families, str):
        raw = tuple(item.strip() for item in families.split(","))
    elif isinstance(families, Sequence):
        raw = tuple(str(item).strip() for item in families)
    else:
        raw = ("superfrustum",)
    deduped: list[str] = []
    for item in raw:
        if item and item not in deduped:
            deduped.append(item)
    return tuple(deduped or ("superfrustum",))
