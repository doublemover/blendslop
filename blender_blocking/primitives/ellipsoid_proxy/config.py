from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from metrics.topology import mesh_topology_report
from placement.resfit_initialization import (
    PrimitiveInitializationConfig,
    initialize_ellipsoids_from_points,
    initialize_gaussians_from_points,
)
from reconstruction.artifacts import write_json
from reconstruction.mesh_io import (
    combine_primitive_meshes,
    mesh_arrays_from_object,
    write_obj,
    write_primitive_set,
)
from reconstruction.point_cloud import target_surface_points
from reconstruction.types import CandidateMetrics, CandidateResult

try:
    from primitives.shape_program import ShapeNode, ShapeProgram, validate_shape_program
    from primitives.proxy_distillation import distill_proxy_field, write_proxy_field_npz
except ImportError:  # pragma: no cover - package import path
    from ..shape_program import ShapeNode, ShapeProgram, validate_shape_program
    from ..proxy_distillation import distill_proxy_field, write_proxy_field_npz


_ALLOWED_FAMILIES = {"gaussian", "gaussians", "ellipsoid", "ellipsoids"}
_ALLOWED_INITIALIZERS = {"farthest_point", "kmeans", "grid"}
_ALLOWED_RENDERERS = {"cpu_projected_ellipse", "gpu_splat"}
_DEFAULT_OBJECTIVE_WEIGHTS = {
    "coverage": 0.55,
    "complexity": 0.20,
    "topology": 0.10,
    "constraint": 0.08,
    "uncertainty": 0.07,
}


def validate_gaussian_ellipsoid_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate gaussian-ellipsoid config and return human-readable errors."""
    _, errors, _ = _normalize_gaussian_ellipsoid_config(config)
    return tuple(errors)


def _normalize_gaussian_ellipsoid_config(
    config: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    if not isinstance(config, Mapping):
        return {}, [f"config must be a mapping, got {type(config)!r}"], []
    errors: list[str] = []
    warnings: list[str] = []
    normalized: dict[str, Any] = {}

    normalized["family"] = str(
        config.get(
            "family", config.get("proxy_family", config.get("primitive_family", "gaussian"))
        )
    ).strip().lower()
    if normalized["family"] not in _ALLOWED_FAMILIES:
        errors.append(
            f"family must be one of {sorted(_ALLOWED_FAMILIES)}, got {normalized['family']!r}"
        )

    primitive_count = _coerce_int(
        config.get("primitive_count", config.get("max_primitives", 24)),
        "primitive_count",
        min_value=1,
        max_value=1024,
        errors=errors,
        default=24,
    )
    target_point_count = _coerce_int(
        config.get("target_point_count", 4096),
        "target_point_count",
        min_value=1,
        max_value=65535,
        errors=errors,
        default=4096,
    )
    visual_hull_resolution = _coerce_int(
        config.get("visual_hull_resolution", config.get("resolution", 48)),
        "visual_hull_resolution",
        min_value=16,
        max_value=512,
        errors=errors,
        default=48,
    )
    min_radius = _coerce_float(
        config.get("min_radius", 1e-4),
        "min_radius",
        min_value=1e-9,
        errors=errors,
        default=1e-4,
    )
    max_radius = _coerce_float_or_none(config.get("max_radius"), "max_radius", errors=errors)
    if max_radius is not None and max_radius < min_radius:
        errors.append("max_radius must be >= min_radius")
    covariance_floor = _coerce_float(
        config.get("covariance_floor", 1e-4),
        "covariance_floor",
        min_value=0.0,
        errors=errors,
        default=1e-4,
    )
    kmeans_iterations = _coerce_int(
        config.get("kmeans_iterations", 8),
        "kmeans_iterations",
        min_value=1,
        max_value=64,
        errors=errors,
        default=8,
    )
    initialization = str(config.get("initialization", "kmeans")).strip().lower()
    if initialization not in _ALLOWED_INITIALIZERS:
        errors.append(
            f"initialization must be one of {sorted(_ALLOWED_INITIALIZERS)}, got {initialization!r}"
        )

    renderer = str(config.get("renderer", "cpu_projected_ellipse")).strip().lower()
    if renderer not in _ALLOWED_RENDERERS:
        errors.append(
            f"renderer must be one of {sorted(_ALLOWED_RENDERERS)}, got {renderer!r}"
        )

    chunk_size = _coerce_optional_int(
        config.get("chunk_size", None),
        "chunk_size",
        errors=errors,
        default=None,
    )
    if chunk_size is not None:
        normalized["chunk_size"] = int(chunk_size)
    else:
        normalized["chunk_size"] = None

    opacity_min = _coerce_float(
        config.get("opacity_min", 0.0),
        "opacity_min",
        min_value=0.0,
        max_value=1.0,
        errors=errors,
        default=0.0,
    )
    opacity_max = _coerce_float(
        config.get("opacity_max", 1.0),
        "opacity_max",
        min_value=0.0,
        max_value=1.0,
        errors=errors,
        default=1.0,
    )
    if opacity_max < opacity_min:
        errors.append("opacity_max must be >= opacity_min")

    objective_weights, objective_weight_warnings = _coerce_objective_weights(
        config.get("objective_weights"),
        errors=errors,
    )
    warnings.extend(objective_weight_warnings)

    normalized["primitive_count"] = primitive_count
    normalized["target_point_count"] = target_point_count
    normalized["visual_hull_resolution"] = visual_hull_resolution
    normalized["distillation_resolution"] = _coerce_int(
        config.get("distillation_resolution", 24),
        "distillation_resolution",
        min_value=4,
        max_value=128,
        errors=errors,
        default=24,
    )
    normalized["distillation_padding"] = _coerce_float(
        config.get("distillation_padding", 0.08),
        "distillation_padding",
        min_value=0.0,
        max_value=1.0,
        errors=errors,
        default=0.08,
    )
    normalized["min_radius"] = min_radius
    normalized["max_radius"] = max_radius
    normalized["covariance_floor"] = covariance_floor
    normalized["kmeans_iterations"] = kmeans_iterations
    normalized["initialization"] = initialization
    normalized["renderer"] = renderer
    normalized["opacity_min"] = opacity_min
    normalized["opacity_max"] = opacity_max
    normalized["objective_weights"] = objective_weights
    normalized["export_mesh_proxy"] = bool(config.get("export_mesh_proxy", True))
    normalized["editable_proxy_sigma"] = _coerce_float(
        config.get("editable_proxy_sigma", 1.0),
        "editable_proxy_sigma",
        min_value=0.05,
        max_value=4.0,
        errors=errors,
        default=1.0,
    )
    return normalized, errors, warnings


def _coerce_int(
    value: Any,
    name: str,
    *,
    min_value: int,
    max_value: int,
    errors: list[str],
    default: int,
) -> int:
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        errors.append(f"{name} must be an integer, got {type(value)!r}")
        return default
    if parsed < min_value:
        errors.append(f"{name} must be >= {min_value}, got {parsed}")
    if parsed > max_value:
        errors.append(f"{name} must be <= {max_value}, got {parsed}")
    return parsed


def _coerce_optional_int(
    value: Any,
    name: str,
    *,
    errors: list[str],
    default: int | None = None,
) -> int | None:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        errors.append(f"{name} must be an integer, got {type(value)!r}")
        return default
    if parsed < 1:
        errors.append(f"{name} must be >= 1, got {parsed}")
        return default
    return parsed


def _coerce_float(
    value: Any,
    name: str,
    *,
    min_value: float,
    max_value: float | None = None,
    errors: list[str],
    default: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a float, got {type(value)!r}")
        return default
    if not np.isfinite(parsed):
        errors.append(f"{name} must be finite, got {parsed!r}")
        return default
    if parsed < min_value:
        errors.append(f"{name} must be >= {min_value}, got {parsed}")
    if max_value is not None and parsed > max_value:
        errors.append(f"{name} must be <= {max_value}, got {parsed}")
    return parsed


def _coerce_float_or_none(
    value: Any,
    name: str,
    errors: list[str],
) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a float or null, got {type(value)!r}")
        return None
    if not np.isfinite(parsed):
        errors.append(f"{name} must be finite or null, got {parsed!r}")
        return None
    return parsed


def _coerce_objective_weights(
    value: Any,
    errors: list[str],
) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    weights = dict(_DEFAULT_OBJECTIVE_WEIGHTS)
    if value is None:
        return weights, warnings
    if not isinstance(value, Mapping):
        errors.append(f"objective_weights must be a mapping, got {type(value)!r}")
        return weights, warnings
    for key in _DEFAULT_OBJECTIVE_WEIGHTS:
        raw = value.get(key)
        if raw is None:
            continue
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            errors.append(
                f"objective_weights[{key}] must be numeric, got {type(raw)!r}"
            )
            continue
        if not np.isfinite(parsed):
            errors.append(f"objective_weights[{key}] must be finite, got {parsed!r}")
            continue
        if parsed < 0.0:
            warnings.append(
                f"objective_weights[{key}] is negative ({parsed}); clamping to 0"
            )
            parsed = 0.0
        weights[key] = parsed
    return weights, warnings
