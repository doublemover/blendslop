from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence

import numpy as np

try:
    from utils.optional_deps import optional_policy_decision, probe_dependency
    from config_models.coercion import (
        coerce_float,
        coerce_int,
        coerce_optional_float,
        coerce_optional_int,
    )
except Exception:  # pragma: no cover
    from ...utils.optional_deps import optional_policy_decision, probe_dependency
    from ...config_models.coercion import (
        coerce_float,
        coerce_int,
        coerce_optional_float,
        coerce_optional_int,
    )

try:
    from primitives.analytic_primitives import (
        AnisotropicGaussianPrimitive,
        EllipsoidPrimitive,
        SuperquadricPrimitive,
    )
    from primitives.primitive_protocol import MeshData
    from primitives.soft_silhouette import (
        OrthographicCamera,
        render_projected_soft_silhouette,
        soft_mask_metrics,
    )
    from primitives.superfrustum import SuperFrustum
    from reconstruction.artifacts import write_json
except ImportError:  # pragma: no cover - package import path.
    from ...primitives.analytic_primitives import (
        AnisotropicGaussianPrimitive,
        EllipsoidPrimitive,
        SuperquadricPrimitive,
    )
    from ...primitives.primitive_protocol import MeshData
    from ...primitives.soft_silhouette import (
        OrthographicCamera,
        render_projected_soft_silhouette,
        soft_mask_metrics,
    )
    from ...primitives.superfrustum import SuperFrustum
    from ...reconstruction.artifacts import write_json

from .contracts import LossWeights

_SUPPORTED_BACKENDS = {"cpu_soft_silhouette", "blender_finite_difference", "nvdiffrast", "dvx"}
_DIFF_OPTIONAL_POLICIES = {"skip", "fail"}
_DIFF_GRADIENT_MODES = {"finite_difference", "backend"}
_DIFF_WEIGHT_DEFAULTS = {
    "silhouette_l2": 1.0,
    "soft_iou": 1.0,
    "area_iou": 0.25,
    "boundary_iou": 0.35,
    "signed_distance": 0.25,
    "depth_l2": 0.0,
}
_DIFF_WEIGHT_ALIASES = {
    "silhouette": "silhouette_l2",
    "l2": "silhouette_l2",
    "area": "area_iou",
    "mask_iou": "area_iou",
    "boundary": "boundary_iou",
    "boundary_iou_loss": "boundary_iou",
    "sdf": "signed_distance",
    "silhouette_sdf": "signed_distance",
    "signed_distance_loss": "signed_distance",
}
_DIFF_WEIGHT_LEGACY_ALIASES = {
    "silhouette_l2_weight": "silhouette_l2",
    "silhouette_weight": "silhouette_l2",
    "soft_iou_weight": "soft_iou",
    "area_iou_weight": "area_iou",
    "area_weight": "area_iou",
    "boundary_iou_weight": "boundary_iou",
    "boundary_weight": "boundary_iou",
    "sdf_weight": "signed_distance",
    "silhouette_sdf_weight": "signed_distance",
    "signed_distance_weight": "signed_distance",
    "signed_distance_loss_weight": "signed_distance",
    "depth_l2_weight": "depth_l2",
}

def _coerce_float(
    value: object,
    name: str,
    errors: list[str],
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    return coerce_float(
        value,
        name,
        errors,
        min_value=min_value,
        max_value=max_value,
        default_on_bounds=True,
        real_number_message=True,
    )

def _coerce_int(
    value: object,
    name: str,
    errors: list[str],
    *,
    min_value: int = 0,
) -> int | None:
    return coerce_int(
        value,
        name,
        errors,
        min_value=min_value,
        default_on_bounds=True,
    )

def _coerce_optional_int(
    value: object,
    name: str,
    errors: list[str],
    *,
    min_value: int = 1,
) -> int | None:
    return coerce_optional_int(
        value,
        name,
        errors,
        min_value=min_value,
        default_on_bounds=True,
    )

def _coerce_optional_float(
    value: object,
    name: str,
    errors: list[str],
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    return coerce_optional_float(
        value,
        name,
        errors,
        min_value=min_value,
        max_value=max_value,
        default_on_bounds=True,
        real_number_message=True,
    )

def _minimum_positive_float(*values: object) -> float | None:
    positives: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            casted = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(casted) and casted > 0.0:
            positives.append(casted)
    return min(positives) if positives else None

def _coerce_loss_weights(config: Mapping[str, object], errors: list[str], warnings: list[str]) -> LossWeights:
    raw_weights = config.get("loss_weights", {})
    if raw_weights is None:
        raw_weights = {}
    if not isinstance(raw_weights, Mapping):
        errors.append("loss_weights must be a mapping")
        raw_weights = {}
    parsed = dict(_DIFF_WEIGHT_DEFAULTS)
    unknown: set[str] = set()
    for key, value in dict(raw_weights).items():
        weight_key = str(key)
        canonical = _DIFF_WEIGHT_ALIASES.get(weight_key, weight_key)
        if canonical not in parsed:
            unknown.add(weight_key)
            continue
        casted = _coerce_float(value, f"loss_weights.{weight_key}", errors, min_value=0.0)
        if casted is not None:
            parsed[canonical] = casted
    for legacy_name, canonical in _DIFF_WEIGHT_LEGACY_ALIASES.items():
        if canonical in raw_weights:
            continue
        if legacy_name not in config:
            continue
        casted = _coerce_float(config[legacy_name], f"{legacy_name}", errors, min_value=0.0)
        if casted is not None:
            parsed[canonical] = casted
    if unknown:
        warnings.append(
            "ignored unknown loss weight keys: "
            + ", ".join(sorted(unknown))
        )
    return LossWeights(**parsed)

def _normalize_differentiable_config(config: Mapping[str, object]) -> tuple[
    dict[str, object],
    tuple[str, ...],
    tuple[str, ...],
]:
    if not isinstance(config, Mapping):
        return (
            {},
            (f"config must be a mapping, got {type(config)!r}",),
            (),
        )

    errors: list[str] = []
    warnings: list[str] = []
    normalized: dict[str, object] = {}

    backend = str(config.get("backend", "cpu_soft_silhouette"))
    if backend not in _SUPPORTED_BACKENDS:
        errors.append(
            "backend must be one of "
            f"{sorted(_SUPPORTED_BACKENDS)}"
        )
    normalized["backend"] = backend

    optional_dependency_policy = str(config.get("optional_dependency_policy", "skip"))
    if optional_dependency_policy not in _DIFF_OPTIONAL_POLICIES:
        errors.append(
            "optional_dependency_policy must be one of "
            f"{sorted(_DIFF_OPTIONAL_POLICIES)}"
        )
    normalized["optional_dependency_policy"] = optional_dependency_policy

    gradient_mode = str(config.get("gradient_mode", "finite_difference"))
    if gradient_mode not in _DIFF_GRADIENT_MODES:
        errors.append(
            "gradient_mode must be one of "
            f"{sorted(_DIFF_GRADIENT_MODES)}"
        )
    normalized["gradient_mode"] = gradient_mode

    finite_difference_epsilon = _coerce_float(
        config.get("finite_difference_epsilon", 1e-4),
        "finite_difference_epsilon",
        errors,
        min_value=1.0e-12,
    )
    if finite_difference_epsilon is None:
        finite_difference_epsilon = 1.0e-4
    normalized["finite_difference_epsilon"] = float(finite_difference_epsilon)

    softness = _coerce_float(config.get("softness", 24.0), "softness", errors, min_value=1e-6)
    if softness is None:
        softness = 24.0
    normalized["softness"] = float(softness)

    min_variance = _coerce_float(
        config.get("min_variance", 1.0e-6),
        "min_variance",
        errors,
        min_value=1.0e-12,
    )
    if min_variance is None:
        min_variance = 1.0e-6
    normalized["min_variance"] = float(min_variance)

    primitive_opacity_floor = _coerce_float(
        config.get(
            "primitive_opacity_floor",
            config.get("render_opacity_floor", config.get("initial_opacity_floor", 0.0)),
        ),
        "primitive_opacity_floor",
        errors,
        min_value=0.0,
        max_value=1.0,
    )
    if primitive_opacity_floor is None:
        primitive_opacity_floor = 0.0
    normalized["primitive_opacity_floor"] = float(primitive_opacity_floor)

    silhouette_bounds_padding = _coerce_float(
        config.get("silhouette_bounds_padding", 1.0),
        "silhouette_bounds_padding",
        errors,
        min_value=0.05,
        max_value=10.0,
    )
    if silhouette_bounds_padding is None:
        silhouette_bounds_padding = 1.0
    normalized["silhouette_bounds_padding"] = float(silhouette_bounds_padding)
    normalized["pixel_evidence_mode"]=str(config.get('pixel_evidence_mode') or 'view_mean_legacy')
    if normalized['pixel_evidence_mode'] not in {'view_mean_legacy','pixel_reliability_v1'}:
        errors.append('pixel_evidence_mode must be view_mean_legacy or pixel_reliability_v1')
    normalized["include_bounds_proxy"] = bool(config.get("include_bounds_proxy", True))
    normalized["calibrate_silhouette_bounds"] = bool(
        config.get("calibrate_silhouette_bounds", False)
    )
    normalized["optimize_boundary_sdf_first"] = bool(
        config.get("optimize_boundary_sdf_first", True)
    )
    mesh_proxy_scale = _coerce_float(
        config.get("mesh_proxy_scale", 1.0),
        "mesh_proxy_scale",
        errors,
        min_value=0.05,
        max_value=10.0,
    )
    if mesh_proxy_scale is None:
        mesh_proxy_scale = 1.0
    normalized["mesh_proxy_scale"] = float(mesh_proxy_scale)

    visual_hull_resolution = _coerce_int(
        config.get("visual_hull_resolution", config.get("resolution", 40)),
        "visual_hull_resolution",
        errors,
        min_value=1,
    )
    if visual_hull_resolution is None:
        visual_hull_resolution = 40
    normalized["visual_hull_resolution"] = int(visual_hull_resolution)

    primitive_count = _coerce_int(
        config.get("primitive_count", 12),
        "primitive_count",
        errors,
        min_value=1,
    )
    if primitive_count is None:
        primitive_count = 12
    normalized["primitive_count"] = int(primitive_count)

    target_point_count = _coerce_int(
        config.get("target_point_count", 2048),
        "target_point_count",
        errors,
        min_value=1,
    )
    if target_point_count is None:
        target_point_count = 2048
    normalized["target_point_count"] = int(target_point_count)

    min_radius = _coerce_float(
        config.get("min_radius", 0.04),
        "min_radius",
        errors,
        min_value=1.0e-12,
    )
    if min_radius is None:
        min_radius = 0.04
    normalized["min_radius"] = float(min_radius)

    covariance_floor = _coerce_float(
        config.get("covariance_floor", 1.0e-4),
        "covariance_floor",
        errors,
        min_value=0.0,
    )
    if covariance_floor is None:
        covariance_floor = 1.0e-4
    normalized["covariance_floor"] = float(covariance_floor)

    kmeans_iterations = _coerce_int(
        config.get("kmeans_iterations", 8),
        "kmeans_iterations",
        errors,
        min_value=1,
    )
    if kmeans_iterations is None:
        kmeans_iterations = 8
    normalized["kmeans_iterations"] = int(kmeans_iterations)

    chunk_size = _coerce_optional_int(
        config.get("chunk_size"),
        "chunk_size",
        errors,
        min_value=1,
    )
    if chunk_size is not None:
        normalized["chunk_size"] = chunk_size
    else:
        normalized["chunk_size"] = None
    optimization_steps = _coerce_int(
        config.get("optimization_steps", config.get("refinement_steps", 6)),
        "optimization_steps",
        errors,
        min_value=0,
    )
    if optimization_steps is None:
        optimization_steps = 6
    normalized["optimization_steps"] = int(optimization_steps)

    optimization_initial_step = _coerce_float(
        config.get("optimization_initial_step", config.get("initial_step", 0.05)),
        "optimization_initial_step",
        errors,
        min_value=1.0e-9,
        max_value=1.0,
    )
    if optimization_initial_step is None:
        optimization_initial_step = 0.05
    normalized["optimization_initial_step"] = float(optimization_initial_step)

    optimization_step_decay = _coerce_float(
        config.get("optimization_step_decay", config.get("step_decay", 0.5)),
        "optimization_step_decay",
        errors,
        min_value=1.0e-9,
        max_value=0.999999,
    )
    if optimization_step_decay is None:
        optimization_step_decay = 0.5
    normalized["optimization_step_decay"] = float(optimization_step_decay)

    optimization_min_step = _coerce_float(
        config.get("optimization_min_step", config.get("min_step", 1.0e-4)),
        "optimization_min_step",
        errors,
        min_value=1.0e-12,
        max_value=1.0,
    )
    if optimization_min_step is None:
        optimization_min_step = 1.0e-4
    normalized["optimization_min_step"] = float(optimization_min_step)

    max_objective_evaluations = _coerce_optional_int(
        config.get("max_objective_evaluations", 256),
        "max_objective_evaluations",
        errors,
        min_value=1,
    )
    normalized["max_objective_evaluations"] = max_objective_evaluations

    max_runtime_s = _coerce_optional_float(
        config.get("max_runtime_s", 20.0),
        "max_runtime_s",
        errors,
        min_value=1.0e-6,
    )
    normalized["max_runtime_s"] = max_runtime_s

    normalized["loss_weights"] = _coerce_loss_weights(config, errors, warnings)
    normalized["loss_weights_dict"] = {
        "silhouette_l2": normalized["loss_weights"].silhouette_l2,
        "soft_iou": normalized["loss_weights"].soft_iou,
        "area_iou": normalized["loss_weights"].area_iou,
        "boundary_iou": normalized["loss_weights"].boundary_iou,
        "signed_distance": normalized["loss_weights"].signed_distance,
        "depth_l2": normalized["loss_weights"].depth_l2,
    }
    normalized["warnings"] = tuple(warnings)

    return normalized, tuple(errors), tuple(warnings)
