"""
Optional differentiable rendering interface and CPU fallback backend.

This module is intentionally Blender-free by default. GPU renderers can plug in
behind lazy imports, while the CPU backend gives primitive fitting a deterministic
soft-silhouette objective today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence

import numpy as np

try:
    from utils.optional_deps import optional_policy_decision, probe_dependency
except Exception:  # pragma: no cover
    from ..utils.optional_deps import optional_policy_decision, probe_dependency

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
    from ..primitives.analytic_primitives import (
        AnisotropicGaussianPrimitive,
        EllipsoidPrimitive,
        SuperquadricPrimitive,
    )
    from ..primitives.primitive_protocol import MeshData
    from ..primitives.soft_silhouette import (
        OrthographicCamera,
        render_projected_soft_silhouette,
        soft_mask_metrics,
    )
    from ..primitives.superfrustum import SuperFrustum
    from ..reconstruction.artifacts import write_json


@dataclass(frozen=True)
class CameraSpec:
    name: str = "front"
    axes: tuple[int, int] = (0, 2)
    image_size: tuple[int, int] = (128, 128)
    world_bounds: tuple[float, float, float, float] = (-1.5, 1.5, -1.5, 1.5)

    def to_orthographic_camera(self) -> OrthographicCamera:
        return OrthographicCamera(
            name=self.name,
            axes=self.axes,
            image_size=self.image_size,
            world_bounds=self.world_bounds,
        )


@dataclass(frozen=True)
class RenderablePrimitive:
    primitive_type: str
    parameters: Mapping[str, object]
    mesh_proxy: MeshData | None = None


@dataclass(frozen=True)
class RenderableScene:
    primitives: tuple[RenderablePrimitive, ...] = ()
    mesh: MeshData | None = None
    transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))


@dataclass(frozen=True)
class RenderBatch:
    silhouettes: Mapping[str, np.ndarray]
    depths: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconstructionTarget:
    silhouettes: Mapping[str, np.ndarray] = field(default_factory=dict)
    depths: Mapping[str, np.ndarray] = field(default_factory=dict)
    surface_points: np.ndarray | None = None


@dataclass(frozen=True)
class LossWeights:
    silhouette_l2: float = 1.0
    soft_iou: float = 1.0
    area_iou: float = 0.25
    boundary_iou: float = 0.35
    signed_distance: float = 0.25
    depth_l2: float = 0.0


@dataclass(frozen=True)
class LossResult:
    total: float
    terms: Mapping[str, float]
    per_view: Mapping[str, Mapping[str, float]]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class GradientBatch:
    gradients: Mapping[str, np.ndarray | float]
    epsilon: float
    warnings: tuple[str, ...] = ()


class DifferentiableRenderBackend(Protocol):
    name: str
    supports_gradients: bool
    supports_silhouette: bool
    supports_depth: bool

    def render(self, scene: RenderableScene, cameras: Sequence[CameraSpec]) -> RenderBatch:
        ...

    def loss(
        self,
        render_batch: RenderBatch,
        target: ReconstructionTarget,
        weights: LossWeights,
        view_weights: Mapping[str, float] | None = None,
    ) -> LossResult:
        ...

    def backward(self, loss: LossResult) -> GradientBatch:
        ...


_SUPPORTED_BACKENDS = {"cpu_soft_silhouette", "blender_finite_difference", "nvdiffrast"}
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
    try:
        casted = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a real number, got {value!r}")
        return None
    if not math.isfinite(casted):
        errors.append(f"{name} must be finite, got {casted}")
        return None
    if min_value is not None and casted < min_value:
        errors.append(f"{name} must be >= {min_value}, got {casted}")
        return None
    if max_value is not None and casted > max_value:
        errors.append(f"{name} must be <= {max_value}, got {casted}")
        return None
    return casted


def _coerce_int(
    value: object,
    name: str,
    errors: list[str],
    *,
    min_value: int = 0,
) -> int | None:
    if isinstance(value, bool):
        errors.append(f"{name} must be an integer, got {value!r}")
        return None
    try:
        casted = int(value)
    except (OverflowError, TypeError, ValueError):
        errors.append(f"{name} must be an integer, got {value!r}")
        return None
    if casted < min_value:
        errors.append(f"{name} must be >= {min_value}, got {casted}")
        return None
    return casted


def _coerce_optional_int(
    value: object,
    name: str,
    errors: list[str],
    *,
    min_value: int = 1,
) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, name, errors, min_value=min_value)


def _coerce_optional_float(
    value: object,
    name: str,
    errors: list[str],
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _coerce_float(
        value,
        name,
        errors,
        min_value=min_value,
        max_value=max_value,
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


def _collect_target_view_signal_weights(
    target: object,
) -> tuple[dict[str, float], dict[str, dict[str, object]], tuple[str, ...]]:
    """Collect per-view optional weights from available target signals."""
    constraints = tuple(getattr(target, "constraints", ()))
    extras = getattr(target, "extras", {})
    if not isinstance(extras, Mapping):
        extras = {}
    weights: dict[str, float] = {}
    details: dict[str, dict[str, object]] = {}
    warnings: list[str] = []

    extra_view_diagnostics = extras.get("view_diagnostics")
    if not isinstance(extra_view_diagnostics, Mapping):
        extra_view_diagnostics = {}

    for constraint in constraints:
        view = str(getattr(constraint, "view", ""))
        if not view:
            continue
        values: list[float] = []
        source_count = 0
        source_values: list[float] = []
        source_details: dict[str, object] = {
            "sources": [],
            "available": False,
        }

        uncertainty = getattr(constraint, "uncertainty", None)
        if uncertainty is not None:
            raw_confidence = np.asarray(getattr(uncertainty, "confidence", ()))
            if raw_confidence.size:
                mean_conf = float(np.mean(raw_confidence))
                if math.isfinite(mean_conf):
                    values.append(float(np.clip(mean_conf, 0.0, 1.0)))
                    source_count += 1
                    source_values.append(float(np.clip(mean_conf, 0.0, 1.0)))
                    source_details["sources"].append("constraint_uncertainty")
                    source_details["constraint_confidence_mean"] = source_values[-1]

        view_diag = extra_view_diagnostics.get(view)
        if isinstance(view_diag, Mapping):
            uncertainty_diag = view_diag.get("uncertainty")
            if isinstance(uncertainty_diag, Mapping):
                conf = uncertainty_diag.get("confidence_mean")
                if conf is not None:
                    try:
                        parsed = float(conf)
                    except (TypeError, ValueError):
                        warnings.append(
                            f"invalid confidence_mean signal for view {view!r}: {conf!r}"
                        )
                    else:
                        if math.isfinite(parsed):
                            parsed = float(np.clip(parsed, 0.0, 1.0))
                            values.append(parsed)
                            source_count += 1
                            source_values.append(parsed)
                            source_details["sources"].append(
                                "extras.view_diagnostics.uncertainty.confidence_mean"
                            )
                            source_details["view_diagnostics_confidence_mean"] = parsed

        explicit = extras.get("view_signals")
        if isinstance(explicit, Mapping):
            explicit_value = explicit.get(view)
            if explicit_value is not None:
                try:
                    parsed = float(explicit_value)
                except (TypeError, ValueError):
                    warnings.append(
                        f"invalid explicit view signal for {view!r}: {explicit_value!r}"
                    )
                else:
                    if math.isfinite(parsed):
                        parsed = float(np.clip(parsed, 0.0, 1.0))
                        values.append(parsed)
                        source_count += 1
                        source_values.append(parsed)
                        source_details["sources"].append("extras.view_signals")
                        source_details["view_signal_value"] = parsed

        weight = 1.0
        if values:
            weight = float(np.mean(values))
            source_details["available"] = True
            source_details["source_count"] = source_count
            source_details["source_values"] = tuple(source_values)
        source_details["count"] = source_count
        details[view] = source_details
        if 0.0 <= weight <= 1.0:
            weights[view] = weight

    return weights, details, tuple(warnings)


def _coerce_view_weight(
    value: object,
    *,
    view_name: str,
    warnings: list[str],
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        warnings.append(f"invalid signal weight for view {view_name!r}: {value!r}")
        return 1.0
    if not math.isfinite(parsed):
        warnings.append(f"non-finite signal weight for view {view_name!r}: {parsed}")
        return 1.0
    return float(np.clip(parsed, 0.0, 1.0))


def _weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for raw, weight in zip(values, weights):
        numerator += float(raw) * float(weight)
        denominator += float(weight)
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _silhouette_view_history(
    name: str,
    predicted: np.ndarray,
    target: np.ndarray,
    metrics: Mapping[str, float],
) -> dict[str, object]:
    predicted_array = np.asarray(predicted, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    intersection = float(np.minimum(predicted_array, target_array).sum())
    union = float(np.maximum(predicted_array, target_array).sum())
    return {
        "view": name,
        "shape": tuple(int(value) for value in predicted_array.shape),
        "pred_area_ratio": float(predicted_array.mean()),
        "target_area_ratio": float(target_array.mean()),
        "intersection_area": intersection,
        "union_area": union,
        "soft_l1": float(metrics.get("soft_l1", 0.0)),
        "soft_l2": float(metrics.get("soft_l2", 0.0)),
        "soft_iou_loss": float(metrics.get("soft_iou_loss", 0.0)),
        "area_iou_loss": float(metrics.get("area_iou_loss", 0.0)),
        "boundary_iou": float(metrics.get("boundary_iou", 0.0)),
        "boundary_iou_loss": float(metrics.get("boundary_iou_loss", 1.0)),
        "signed_distance_loss": float(metrics.get("signed_distance_loss", 0.0)),
    }


def _candidate_per_view_metrics(
    per_view: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, object]]:
    """Convert soft-renderer losses into the shared CandidateMetrics contract."""
    converted: dict[str, dict[str, object]] = {}
    for view, values in per_view.items():
        area_loss = _metric_float(values, "area_iou_loss", 1.0)
        soft_loss = _metric_float(values, "soft_iou_loss", area_loss)
        signed_distance = max(
            0.0,
            _metric_float(
                values,
                "signed_distance_loss",
                _metric_float(values, "soft_l2", soft_loss),
            ),
        )
        area_iou = _clamp01(1.0 - area_loss)
        soft_iou = _clamp01(1.0 - soft_loss)
        boundary_iou = _clamp01(
            _metric_float(values, "boundary_iou", soft_iou)
        )
        passed = bool(
            area_iou >= 0.5
            and boundary_iou > 0.0
            and math.isfinite(signed_distance)
        )
        reason = "" if passed else "soft silhouette did not satisfy required view gate"
        converted[str(view)] = {
            **dict(values),
            "area_iou": area_iou,
            "soft_iou": soft_iou,
            "boundary_iou": boundary_iou,
            "signed_distance_loss": signed_distance,
            "required": True,
            "passed": passed,
            "pass": passed,
            "reason": reason,
        }
    return converted


def _mean_candidate_metric(
    per_view: Mapping[str, Mapping[str, object]],
    name: str,
    fallback: float,
) -> float:
    values = []
    for payload in per_view.values():
        value = payload.get(name)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return fallback
    return float(np.mean(values))


def _min_candidate_metric(
    per_view: Mapping[str, Mapping[str, object]],
    name: str,
    fallback: float,
) -> float:
    values = []
    for payload in per_view.values():
        value = payload.get(name)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return fallback
    return float(min(values))


def _metric_float(
    values: Mapping[str, float],
    name: str,
    default: float,
) -> float:
    try:
        value = float(values.get(name, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def primitive_from_renderable(renderable: RenderablePrimitive) -> object:
    """Create a known primitive object from a renderable primitive record."""
    ptype = renderable.primitive_type.lower()
    params = dict(renderable.parameters)
    if ptype == "ellipsoid":
        return EllipsoidPrimitive.from_dict(params)
    if ptype in ("gaussian", "anisotropic_gaussian"):
        return AnisotropicGaussianPrimitive.from_dict(params)
    if ptype == "superquadric":
        return SuperquadricPrimitive.from_dict(params)
    if ptype == "superfrustum":
        return SuperFrustum.from_dict(params)
    raise ValueError(f"unsupported renderable primitive type: {renderable.primitive_type}")


def renderable_from_primitive(primitive: object) -> RenderablePrimitive:
    """Serialize a known primitive into the renderable scene contract."""
    if not hasattr(primitive, "to_dict"):
        raise TypeError(f"primitive lacks to_dict: {type(primitive)!r}")
    params = primitive.to_dict()
    ptype = str(params.get("type", type(primitive).__name__.lower()))
    mesh = primitive.to_mesh_data(32) if hasattr(primitive, "to_mesh_data") else None
    return RenderablePrimitive(primitive_type=ptype, parameters=params, mesh_proxy=mesh)


def evaluate_render_loss(
    render_batch: RenderBatch,
    target: ReconstructionTarget,
    weights: LossWeights = LossWeights(),
    view_weights: Mapping[str, float] | None = None,
) -> LossResult:
    """Compute soft silhouette/depth losses for a rendered batch."""
    terms: Dict[str, float] = {}
    per_view: Dict[str, Mapping[str, float]] = {}
    warnings: list[str] = []
    configured_view_weights = dict(view_weights or {})

    silhouette_weight_sum = 0.0
    depth_terms_sum: float = 0.0
    depth_weight_sum: float = 0.0

    for name, predicted in render_batch.silhouettes.items():
        target_mask = target.silhouettes.get(name)
        if target_mask is None:
            warnings.append(f"missing target silhouette for view {name}")
            continue
        metrics = dict(soft_mask_metrics(predicted, target_mask))
        weight = _coerce_view_weight(
            configured_view_weights.get(name, 1.0),
            view_name=name,
            warnings=warnings,
        )
        per_view[name] = {
            **metrics,
            "weight": weight,
        }
        silhouette_weight_sum += weight
        for key, value in metrics.items():
            terms[f"{name}_{key}"] = float(value)

    silhouette_l2 = (
        _weighted_average(
            [m["soft_l2"] for m in per_view.values()],
            [m["weight"] for m in per_view.values()],
        )
        if per_view
        else 0.0
    )
    if silhouette_weight_sum <= 0.0 and per_view:
        warnings.append(
            "all silhouette view weights were zero; fell back to unweighted aggregation"
        )
        silhouette_l2 = float(np.mean([m["soft_l2"] for m in per_view.values()]))
        soft_iou = float(np.mean([m["soft_iou_loss"] for m in per_view.values()]))
        area_iou = float(np.mean([m["area_iou_loss"] for m in per_view.values()]))
    else:
        soft_iou = float(
            _weighted_average(
                [m["soft_iou_loss"] for m in per_view.values()],
                [m["weight"] for m in per_view.values()],
            )
            if per_view
            else 0.0
        )
        area_iou = float(
            _weighted_average(
                [m["area_iou_loss"] for m in per_view.values()],
                [m["weight"] for m in per_view.values()],
            )
            if per_view
            else 0.0
        )
        silhouette_l2 = float(silhouette_l2)

    if silhouette_weight_sum <= 0.0 and per_view:
        boundary_iou = float(
            np.mean([m.get("boundary_iou_loss", 1.0) for m in per_view.values()])
        )
        signed_distance = float(
            np.mean([m.get("signed_distance_loss", 0.0) for m in per_view.values()])
        )
    else:
        boundary_iou = float(
            _weighted_average(
                [m.get("boundary_iou_loss", 1.0) for m in per_view.values()],
                [m["weight"] for m in per_view.values()],
            )
            if per_view
            else 0.0
        )
        signed_distance = float(
            _weighted_average(
                [m.get("signed_distance_loss", 0.0) for m in per_view.values()],
                [m["weight"] for m in per_view.values()],
            )
            if per_view
            else 0.0
        )

    terms["silhouette_l2"] = silhouette_l2
    terms["soft_iou"] = soft_iou
    terms["area_iou"] = area_iou
    terms["boundary_iou"] = boundary_iou
    terms["boundary_iou_loss"] = boundary_iou
    terms["boundary_iou_score"] = float(1.0 - boundary_iou)
    terms["signed_distance"] = signed_distance
    terms["signed_distance_loss"] = signed_distance
    terms["silhouette_weight_sum"] = float(silhouette_weight_sum)

    depth_losses = []
    for name, predicted_depth in render_batch.depths.items():
        target_depth = target.depths.get(name)
        if target_depth is None:
            continue
        pred = np.asarray(predicted_depth, dtype=np.float64)
        tgt = np.asarray(target_depth, dtype=np.float64)
        if pred.shape != tgt.shape:
            warnings.append(f"depth shape mismatch for view {name}")
            continue
        weight = _coerce_view_weight(
            configured_view_weights.get(name, 1.0),
            view_name=name,
            warnings=warnings,
        )
        depth_value = float(np.mean((pred - tgt) ** 2))
        depth_losses.append(depth_value)
        if weight > 0.0:
            depth_terms_sum += depth_value * weight
            depth_weight_sum += weight

    if depth_weight_sum > 0.0:
        depth_l2 = float(depth_terms_sum / depth_weight_sum)
    elif depth_losses:
        warnings.append(
            "all depth view weights were zero; fell back to unweighted depth aggregation"
        )
        depth_l2 = float(np.mean(depth_losses))
    else:
        depth_l2 = 0.0
    terms["depth_l2"] = depth_l2
    terms["depth_weight_sum"] = float(depth_weight_sum)

    total = (
        weights.silhouette_l2 * silhouette_l2
        + weights.soft_iou * soft_iou
        + weights.area_iou * area_iou
        + weights.boundary_iou * boundary_iou
        + weights.signed_distance * signed_distance
        + weights.depth_l2 * depth_l2
    )
    return LossResult(
        total=float(total),
        terms=terms,
        per_view=per_view,
        warnings=tuple(warnings),
    )


def finite_difference_scalar(
    fn: Callable[[float], float],
    value: float,
    epsilon: float = 1e-5,
    central: bool = True,
) -> float:
    """Finite-difference derivative for a scalar value."""
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if central:
        return float((fn(value + epsilon) - fn(value - epsilon)) / (2.0 * epsilon))
    return float((fn(value + epsilon) - fn(value)) / epsilon)


def finite_difference_array(
    fn: Callable[[np.ndarray], float],
    values: np.ndarray,
    epsilon: float = 1e-5,
    central: bool = True,
) -> np.ndarray:
    """Finite-difference gradient for an ndarray input."""
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    values = np.asarray(values, dtype=np.float64)
    gradient = np.zeros_like(values, dtype=np.float64)
    for index in np.ndindex(values.shape):
        plus = values.copy()
        plus[index] += epsilon
        if central:
            minus = values.copy()
            minus[index] -= epsilon
            gradient[index] = (fn(plus) - fn(minus)) / (2.0 * epsilon)
        else:
            gradient[index] = (fn(plus) - fn(values)) / epsilon
    return gradient


class CpuSoftSilhouetteBackend:
    """Pure Python soft silhouette backend for ellipsoid/Gaussian proxies."""

    name = "cpu_soft_silhouette"
    supports_gradients = False
    supports_silhouette = True
    supports_depth = False

    def __init__(self, softness: float = 24.0, min_variance: float = 1.0e-6) -> None:
        self.softness = float(softness)
        self.min_variance = float(min_variance)
        if not math.isfinite(self.softness) or self.softness <= 0.0:
            raise ValueError("softness must be a finite positive number")
        if not math.isfinite(self.min_variance) or self.min_variance <= 0.0:
            raise ValueError("min_variance must be a finite positive number")
        self._last_scene: RenderableScene | None = None
        self._last_cameras: tuple[CameraSpec, ...] = ()
        self._last_target: ReconstructionTarget | None = None
        self._last_weights: LossWeights = LossWeights()
        self._last_metadata: dict[str, object] = {}

    def validate_config(self, config: Mapping[str, object]) -> list[str]:
        _, errors, _ = _normalize_differentiable_config(config)
        return list(errors)

    def render(self, scene: RenderableScene, cameras: Sequence[CameraSpec]) -> RenderBatch:
        primitives = [primitive_from_renderable(primitive) for primitive in scene.primitives]
        view_stats: list[dict[str, object]] = []
        silhouettes = {}
        for camera in cameras:
            camera_name = camera.name
            silhouettes[camera.name] = render_projected_soft_silhouette(
                primitives,
                camera.to_orthographic_camera(),
                softness=self.softness,
                min_variance=self.min_variance,
            )
            mask = silhouettes[camera.name]
            if mask.ndim != 2:
                raise ValueError(f"rendered silhouette for {camera_name} must be 2D")
            view_stats.append(
                {
                    "view": camera_name,
                    "image_size": tuple(int(size) for size in camera.image_size),
                    "coverage": float(mask.mean()),
                    "area": float(mask.sum()),
                }
            )
        self._last_scene = scene
        self._last_cameras = tuple(cameras)
        self._last_metadata = {
            "backend": self.name,
            "softness": self.softness,
            "min_variance": self.min_variance,
            "primitive_count": len(primitives),
            "camera_count": len(view_stats),
            "view_stats": view_stats,
        }
        return RenderBatch(
            silhouettes=silhouettes,
            metadata=self._last_metadata,
        )

    def loss(
        self,
        render_batch: RenderBatch,
        target: ReconstructionTarget,
        weights: LossWeights = LossWeights(),
        view_weights: Mapping[str, float] | None = None,
    ) -> LossResult:
        self._last_target = target
        self._last_weights = weights
        return evaluate_render_loss(render_batch, target, weights, view_weights=view_weights)

    def backward(self, loss: LossResult) -> GradientBatch:
        return GradientBatch(
            gradients={},
            epsilon=0.0,
            warnings=("CPU soft silhouette backend uses finite-difference helpers externally",),
        )


class BlenderFiniteDifferenceBackend:
    """
    Slow finite-difference adapter around injected render/loss callbacks.

    The class is available without Blender, but real rendering requires callers
    to inject callbacks that know how to build and render Blender scene state.
    """

    name = "blender_finite_difference"
    supports_gradients = False
    supports_silhouette = True
    supports_depth = True

    def __init__(
        self,
        render_callback: Callable[[RenderableScene, Sequence[CameraSpec]], RenderBatch] | None = None,
        loss_callback: Callable[[RenderBatch, ReconstructionTarget, LossWeights], LossResult] | None = None,
        epsilon: float = 1e-4,
    ) -> None:
        self.render_callback = render_callback
        self.loss_callback = loss_callback or evaluate_render_loss
        self.epsilon = float(epsilon)

    def render(self, scene: RenderableScene, cameras: Sequence[CameraSpec]) -> RenderBatch:
        if self.render_callback is None:
            raise RuntimeError("BlenderFiniteDifferenceBackend requires render_callback")
        return self.render_callback(scene, cameras)

    def loss(
        self,
        render_batch: RenderBatch,
        target: ReconstructionTarget,
        weights: LossWeights = LossWeights(),
        view_weights: Mapping[str, float] | None = None,
    ) -> LossResult:
        return self.loss_callback(render_batch, target, weights, view_weights=view_weights)

    def backward(self, loss: LossResult) -> GradientBatch:
        return GradientBatch(
            gradients={},
            epsilon=self.epsilon,
            warnings=("Blender backend has no direct gradients; perturb parameters with finite_difference_array",),
        )


class NvdiffrastBackend:
    """Lazy optional nvdiffrast adapter with explicit availability reporting."""

    name = "nvdiffrast"
    supports_gradients = True
    supports_silhouette = True
    supports_depth = True

    def __init__(self) -> None:
        self._module = None
        self._dr = None
        self._nvdiffrast = probe_dependency("nvdiffrast")
        self._torch = probe_dependency("torch")
        self._nvdiffrast_torch = None
        self._gpu_runtime_status: dict[str, object] = {
            "available": False,
            "status": "not_checked",
            "message": "GPU runtime has not been checked",
        }
        self._unmet_dependencies: list[str] = []
        if not self._nvdiffrast.available:
            self._unmet_dependencies.append(self._nvdiffrast.skip_reason)
        if not self._torch.available:
            self._unmet_dependencies.append(self._torch.skip_reason)
        elif self._torch.module is not None:
            self._gpu_runtime_status = self._detect_torch_gpu_runtime(self._torch.module)
            if not bool(self._gpu_runtime_status.get("available")):
                self._unmet_dependencies.append(
                    "torch GPU runtime: "
                    f"{self._gpu_runtime_status.get('message', 'unavailable')}"
                )
        self._module = self._nvdiffrast.module if self._nvdiffrast.available else None
        if self._module is not None:
            self._nvdiffrast_torch = probe_dependency("nvdiffrast.torch")
            if self._nvdiffrast_torch.available:
                self._dr = self._nvdiffrast_torch.module
            else:
                self._unmet_dependencies.append(self._nvdiffrast_torch.skip_reason)
        self.unavailable_reason: str | None = None
        if self._unmet_dependencies:
            self.unavailable_reason = "; ".join(self._unmet_dependencies)
        self._contexts: dict[str, object] = {}
        self._last_render_batch: RenderBatch | None = None
        self._last_tensors: dict[str, object] = {}

    @property
    def available(self) -> bool:
        return (
            self._module is not None
            and self._dr is not None
            and self._torch.available
            and bool(self._gpu_runtime_status.get("available"))
        )

    @property
    def dependency_report(self) -> str:
        if self.available:
            return "dependencies satisfied"
        return "; ".join(self._unmet_dependencies)

    @property
    def dependency_state(self) -> dict[str, object]:
        state: dict[str, object] = {
            "nvdiffrast": self._nvdiffrast.to_dict(),
            "torch": self._torch.to_dict(),
            "gpu_runtime": dict(self._gpu_runtime_status),
        }
        if self._nvdiffrast_torch is not None:
            state["nvdiffrast.torch"] = self._nvdiffrast_torch.to_dict()
        return state

    def _require_available(self) -> None:
        if not self.available:
            raise RuntimeError(
                "nvdiffrast is unavailable; use CpuSoftSilhouetteBackend or "
                f"BlenderFiniteDifferenceBackend instead ({self.dependency_report})"
            )

    def render(self, scene: RenderableScene, cameras: Sequence[CameraSpec]) -> RenderBatch:
        self._require_available()
        torch = self._torch.require()
        dr = self._dr
        vertices, faces = _scene_mesh_arrays(scene)
        if len(vertices) == 0 or len(faces) == 0:
            raise RuntimeError("nvdiffrast render requires a non-empty triangle mesh")
        device_name = "cuda"
        device = torch.device(device_name)
        faces_tensor = torch.as_tensor(faces, dtype=torch.int32, device=device)
        ctx = self._context_for_device(dr, torch, device_name)

        silhouettes: dict[str, np.ndarray] = {}
        depths: dict[str, np.ndarray] = {}
        view_stats: list[dict[str, object]] = []
        tensor_records: dict[str, object] = {}
        for camera in cameras:
            width, height = (int(camera.image_size[0]), int(camera.image_size[1]))
            clip_vertices = _project_vertices_to_clip(vertices, camera)
            pos = torch.as_tensor(
                clip_vertices,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            rast, _ = dr.rasterize(
                ctx,
                pos,
                faces_tensor,
                resolution=[height, width],
            )
            hard_mask = (rast[..., 3:4] > 0).to(torch.float32)
            try:
                mask_tensor = dr.antialias(hard_mask, rast, pos, faces_tensor)[0, ..., 0]
            except Exception:
                mask_tensor = hard_mask[0, ..., 0]
            depth_tensor = torch.where(
                hard_mask[0, ..., 0] > 0.0,
                rast[0, ..., 2],
                torch.zeros_like(rast[0, ..., 2]),
            )
            mask_np = mask_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
            depth_np = depth_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
            silhouettes[camera.name] = mask_np
            depths[camera.name] = depth_np
            tensor_records[camera.name] = {
                "position": pos,
                "raster": rast,
                "mask": mask_tensor,
                "depth": depth_tensor,
            }
            view_stats.append(
                {
                    "view": camera.name,
                    "image_size": (width, height),
                    "coverage": float(mask_np.mean()),
                    "area": float(mask_np.sum()),
                    "device": device_name,
                }
            )

        metadata = {
            "backend": self.name,
            "device": device_name,
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "camera_count": int(len(cameras)),
            "view_stats": view_stats,
            "gradient_path": "nvdiffrast.torch",
        }
        batch = RenderBatch(
            silhouettes=silhouettes,
            depths=depths,
            metadata=metadata,
        )
        self._last_render_batch = batch
        self._last_tensors = tensor_records
        return batch

    def loss(
        self,
        render_batch: RenderBatch,
        target: ReconstructionTarget,
        weights: LossWeights = LossWeights(),
        view_weights: Mapping[str, float] | None = None,
    ) -> LossResult:
        self._require_available()
        return evaluate_render_loss(render_batch, target, weights, view_weights=view_weights)

    def backward(self, loss: LossResult) -> GradientBatch:
        self._require_available()
        _ = loss
        return GradientBatch(
            gradients={},
            epsilon=0.0,
            warnings=(
                "nvdiffrast raster tensors were retained for external torch "
                "optimization; scalar LossResult gradients are not materialized by "
                "this candidate wrapper",
            ),
        )

    def _context_for_device(self, dr: Any, torch: Any, device_name: str) -> object:
        cached = self._contexts.get(device_name)
        if cached is not None:
            return cached
        errors: list[str] = []
        if device_name == "cuda" and hasattr(dr, "RasterizeCudaContext"):
            try:
                context = dr.RasterizeCudaContext(device=torch.device(device_name))
                self._contexts[device_name] = context
                return context
            except Exception as exc:
                errors.append(f"RasterizeCudaContext: {exc}")
        if hasattr(dr, "RasterizeGLContext"):
            try:
                context = dr.RasterizeGLContext()
                self._contexts[device_name] = context
                return context
            except Exception as exc:
                errors.append(f"RasterizeGLContext: {exc}")
        raise RuntimeError("no usable nvdiffrast raster context: " + "; ".join(errors))

    @staticmethod
    def _detect_torch_gpu_runtime(torch: Any) -> dict[str, object]:
        version = getattr(torch, "version", None)
        cuda_version = getattr(version, "cuda", None)
        hip_version = getattr(version, "hip", None)
        cuda_api = getattr(torch, "cuda", None)
        cuda_available = False
        device_count = 0
        try:
            cuda_available = (
                bool(cuda_api.is_available()) if cuda_api is not None else False
            )
        except Exception as exc:
            return {
                "available": False,
                "status": "unusable",
                "message": f"torch.cuda availability check failed: {exc}",
                "error_type": type(exc).__name__,
                "cuda_version": cuda_version,
                "hip_version": hip_version,
            }
        try:
            device_count = (
                int(cuda_api.device_count())
                if cuda_api is not None and cuda_available
                else 0
            )
        except Exception:
            device_count = 0
        if hip_version and not cuda_version:
            return {
                "available": False,
                "status": "rocm_unsupported",
                "message": (
                    "ROCm/HIP PyTorch runtime detected; nvdiffrast has no official "
                    "ROCm backend and must not silently fall back to CPU"
                ),
                "cuda_version": cuda_version,
                "hip_version": hip_version,
                "device_count": device_count,
                "supports_rocm": False,
            }
        if not cuda_available:
            return {
                "available": False,
                "status": "cuda_unavailable",
                "message": (
                    "torch.cuda is unavailable; nvdiffrast requires an NVIDIA CUDA "
                    "runtime for this backend"
                ),
                "cuda_version": cuda_version,
                "hip_version": hip_version,
                "device_count": device_count,
                "supports_rocm": False,
            }
        return {
            "available": True,
            "status": "cuda_available",
            "message": "torch CUDA runtime is available for nvdiffrast",
            "cuda_version": cuda_version,
            "hip_version": hip_version,
            "device_count": device_count,
            "supports_rocm": False,
        }


def run_refinement_candidate(request: object) -> object:
    """Run a CPU differentiable-rendering-inspired candidate.

    This path creates an editable ellipsoid proxy, renders soft silhouettes, and
    records loss terms through the same backend protocol that GPU
    renderers can implement.
    """
    import time

    from placement.resfit_objective import ResFitObjectiveResult
    from placement.resfit_optimizer import (
        CoordinateDescentConfig,
        coordinate_descent_optimize,
    )
    from placement.resfit_initialization import (
        PrimitiveInitializationConfig,
        initialize_ellipsoids_from_points,
    )
    from reconstruction.mesh_io import (
        combine_primitive_meshes,
        write_obj,
        write_primitive_set,
    )
    from reconstruction.point_cloud import target_surface_points
    from reconstruction.types import CandidateMetrics, CandidateResult
    from metrics.topology import mesh_topology_report

    start = time.perf_counter()
    config = dict(getattr(request, "config", {}) or {})
    parsed_config, config_errors, config_warnings = _normalize_differentiable_config(config)
    if config_errors:
        return CandidateResult(
            candidate_id=getattr(request, "candidate_id"),
            backend_name=getattr(request, "backend_name", "differentiable_refine"),
            status="failed",
            errors=config_errors,
            warnings=tuple(config_warnings),
        )
    backend_choice = str(parsed_config["backend"])
    candidate_id = getattr(request, "candidate_id")
    backend_name = getattr(request, "backend_name", "differentiable_refine")
    target = getattr(request, "target")
    request_budget = getattr(request, "budget", None)
    request_timeout_s = getattr(request_budget, "timeout_s", None)
    optional_dependency_policy = str(parsed_config["optional_dependency_policy"])
    (
        target_view_weights,
        view_signal_details,
        target_signal_warnings,
    ) = _collect_target_view_signal_weights(target)

    nvd_renderer = None
    if backend_choice == "nvdiffrast":
        nvd_renderer = NvdiffrastBackend()
        if not nvd_renderer.available:
            nvd_dependency = probe_dependency("nvdiffrast")
            decision = optional_policy_decision(
                nvd_dependency,
                policy=optional_dependency_policy,
                feature="differentiable_refine.nvdiffrast",
            )
            status = str(decision["result_status"])
            reason = nvd_renderer.unavailable_reason or str(decision["message"])
            metrics = CandidateMetrics(
                extras={
                    "optional_dependencies": getattr(nvd_renderer, "dependency_state", {}),
                    "optional_dependency_policy": decision,
                    "backend": backend_choice,
                }
            )
            return CandidateResult(
                candidate_id=candidate_id,
                backend_name=backend_name,
                status=status,
                metric_result=metrics,
                warnings=(
                    reason,
                    *config_warnings,
                    *tuple(target_signal_warnings),
                ),
                errors=(reason,) if status == "failed" else (),
            )

    try:
        points, point_meta = target_surface_points(
            target,
            resolution=int(parsed_config["visual_hull_resolution"]),
            max_points=int(parsed_config["target_point_count"]),
            chunk_size=parsed_config["chunk_size"],
        )
        primitives = tuple(
            initialize_ellipsoids_from_points(
                points,
                PrimitiveInitializationConfig(
                    primitive_count=int(parsed_config["primitive_count"]),
                    target_point_count=int(parsed_config["target_point_count"]),
                    min_radius=float(parsed_config["min_radius"]),
                    covariance_floor=float(parsed_config["covariance_floor"]),
                    kmeans_iterations=int(parsed_config["kmeans_iterations"]),
                ),
            )
        )
        renderables = tuple(renderable_from_primitive(primitive) for primitive in primitives)
        scene = RenderableScene(primitives=renderables)
        cameras, silhouettes = _target_cameras_and_masks(target)
        if backend_choice == "nvdiffrast":
            renderer = nvd_renderer or NvdiffrastBackend()
        else:
            renderer = CpuSoftSilhouetteBackend(
                softness=float(parsed_config["softness"]),
                min_variance=float(parsed_config["min_variance"]),
            )
        target_record = ReconstructionTarget(
            silhouettes=silhouettes,
            surface_points=points,
            depths=getattr(target, "depths", {}),
        )
        baseline_render_batch = RenderBatch(
            silhouettes={
                name: np.zeros_like(mask, dtype=np.float64)
                for name, mask in silhouettes.items()
            },
            metadata={"stage": "baseline"},
        )
        baseline_loss = renderer.loss(
            baseline_render_batch,
            target_record,
            parsed_config["loss_weights"],
            view_weights=target_view_weights,
        )
        initial_render_batch = renderer.render(scene, cameras)
        initial_loss = renderer.loss(
            initial_render_batch,
            target_record,
            parsed_config["loss_weights"],
            view_weights=target_view_weights,
        )
        optimized_primitives = primitives
        optimization_history: list[dict[str, object]] = []
        optimization_summary: dict[str, object] = {
            "enabled": False,
            "reason": "not_run",
            "objective_evaluations": 0,
            "elapsed_s": 0.0,
            "initial_total": float(initial_loss.total),
            "final_total": float(initial_loss.total),
        }
        if (
            backend_choice == "cpu_soft_silhouette"
            and int(parsed_config["optimization_steps"]) > 0
            and primitives
            and cameras
        ):
            max_elapsed_s = _minimum_positive_float(
                parsed_config.get("max_runtime_s"),
                request_timeout_s,
            )
            optimizer_config = CoordinateDescentConfig(
                iterations=int(parsed_config["optimization_steps"]),
                initial_step=float(parsed_config["optimization_initial_step"]),
                step_decay=float(parsed_config["optimization_step_decay"]),
                min_step=float(parsed_config["optimization_min_step"]),
                max_objective_evaluations=parsed_config["max_objective_evaluations"],
                max_elapsed_s=max_elapsed_s,
            )

            def objective(primitives_to_score: Sequence[object]) -> ResFitObjectiveResult:
                trial_scene = RenderableScene(
                    primitives=tuple(
                        renderable_from_primitive(primitive)
                        for primitive in primitives_to_score
                    )
                )
                trial_batch = renderer.render(trial_scene, cameras)
                trial_loss = renderer.loss(
                    trial_batch,
                    target_record,
                    parsed_config["loss_weights"],
                    view_weights=target_view_weights,
                )
                return ResFitObjectiveResult(
                    total=float(trial_loss.total),
                    terms=dict(trial_loss.terms),
                    warnings=tuple(trial_loss.warnings),
                )

            optimization = coordinate_descent_optimize(
                primitives,
                objective,
                optimizer_config,
            )
            optimized_primitives = optimization.primitives
            optimization_history = [
                {
                    "iteration": record.iteration,
                    "total": record.total,
                    "terms": dict(record.terms),
                    "accepted_moves": record.accepted_moves,
                    "rejected_moves": getattr(record, "rejected_moves", 0),
                    "step_size": record.step_size,
                    "reason": getattr(record, "reason", ""),
                }
                for record in optimization.history
            ]
            optimization_summary = {
                "enabled": True,
                "reason": optimization.termination_reason,
                "objective_evaluations": optimization.objective_evaluations,
                "elapsed_s": optimization.elapsed_s,
                "initial_total": float(initial_loss.total),
                "best_total": float(optimization.best_loss),
                "accepted_moves": sum(
                    int(record.accepted_moves) for record in optimization.history
                ),
                "history_length": len(optimization_history),
                "config": {
                    "iterations": int(parsed_config["optimization_steps"]),
                    "initial_step": float(parsed_config["optimization_initial_step"]),
                    "step_decay": float(parsed_config["optimization_step_decay"]),
                    "min_step": float(parsed_config["optimization_min_step"]),
                    "max_objective_evaluations": parsed_config["max_objective_evaluations"],
                    "max_elapsed_s": max_elapsed_s,
                },
            }
        elif int(parsed_config["optimization_steps"]) <= 0:
            optimization_summary["reason"] = "optimization_steps_zero"
        elif backend_choice != "cpu_soft_silhouette":
            optimization_summary["reason"] = f"optimizer disabled for backend {backend_choice}"
        elif not cameras:
            optimization_summary["reason"] = "no target cameras"
        elif not primitives:
            optimization_summary["reason"] = "no primitives"

        if optimization_summary.get("enabled"):
            renderables = tuple(
                renderable_from_primitive(primitive) for primitive in optimized_primitives
            )
            scene = RenderableScene(primitives=renderables)
            render_batch = renderer.render(scene, cameras)
            loss = renderer.loss(
                render_batch,
                target_record,
                parsed_config["loss_weights"],
                view_weights=target_view_weights,
            )
        else:
            render_batch = initial_render_batch
            loss = initial_loss
        optimization_summary["final_total"] = float(loss.total)
    except Exception as exc:
        if backend_choice == "nvdiffrast":
            dependency_state = (
                nvd_renderer.dependency_state if nvd_renderer is not None else {}
            )
            status = "failed" if optional_dependency_policy == "fail" else "skipped"
            message = f"nvdiffrast render path unavailable: {exc}"
            metrics = CandidateMetrics(
                extras={
                    "optional_dependencies": dependency_state,
                    "optional_dependency_policy": {
                        "policy": optional_dependency_policy,
                        "feature": "differentiable_refine.nvdiffrast",
                        "status": status,
                        "result_status": status,
                        "message": message,
                    },
                    "backend": backend_choice,
                }
            )
            return CandidateResult(
                candidate_id=candidate_id,
                backend_name=backend_name,
                status=status,
                metric_result=metrics,
                warnings=(
                    message,
                    *config_warnings,
                    *tuple(target_signal_warnings),
                ),
                errors=(message,) if status == "failed" else (),
            )
        return CandidateResult(
            candidate_id=candidate_id,
            backend_name=backend_name,
            status="failed",
            errors=(str(exc),),
        )

    root = request.candidate_artifact_root()
    primitive_path = None
    mesh_path = None
    artifacts = {}
    mesh_proxy = combine_primitive_meshes(optimized_primitives, resolution=20)
    try:
        topology_report = mesh_topology_report(mesh_proxy.vertices, mesh_proxy.faces)
        topology_payload = topology_report.to_dict()
        topology_score = float(topology_payload.get("topology_score", 0.0))
        topology_penalty = float(topology_payload.get("penalty", 0.0))
    except Exception as exc:
        topology_payload = {
            "required": True,
            "passed": False,
            "pass": False,
            "reason": f"mesh topology report failed: {exc}",
            "topology_score": 0.0,
            "penalty": 1.0,
        }
        topology_score = 0.0
        topology_penalty = 1.0
    refinement_history: list[dict[str, object]] = []
    for name, target_mask in silhouettes.items():
        view_metrics = dict(loss.per_view.get(name, {}))
        if name in render_batch.silhouettes:
            refinement_history.append(
                _silhouette_view_history(
                    name=name,
                    predicted=np.asarray(render_batch.silhouettes[name], dtype=np.float64),
                    target=np.asarray(target_mask, dtype=np.float64),
                    metrics=view_metrics,
                )
            )
    baseline_refinement_history = [
        _silhouette_view_history(
            name=name,
            predicted=np.zeros_like(target_mask, dtype=np.float64),
            target=np.asarray(target_mask, dtype=np.float64),
            metrics=dict(baseline_loss.per_view.get(name, {})),
        )
        for name, target_mask in silhouettes.items()
    ]
    objective_improvement = float(initial_loss.total - loss.total)
    zero_baseline_improvement = float(baseline_loss.total - loss.total)
    objective_history = [
        {
            "stage": "baseline_zero",
            "loss_total": float(baseline_loss.total),
            "loss_terms": dict(baseline_loss.terms),
            "loss_warnings": tuple(baseline_loss.warnings),
            "loss_view_weights": dict(target_view_weights),
            "views": baseline_refinement_history,
        },
        {
            "stage": "initial_render_and_score",
            "loss_total": float(initial_loss.total),
            "loss_terms": dict(initial_loss.terms),
            "loss_warnings": tuple(initial_loss.warnings),
            "loss_view_weights": dict(target_view_weights),
            "views": [
                _silhouette_view_history(
                    name=name,
                    predicted=np.asarray(initial_render_batch.silhouettes[name], dtype=np.float64),
                    target=np.asarray(target_mask, dtype=np.float64),
                    metrics=dict(initial_loss.per_view.get(name, {})),
                )
                for name, target_mask in silhouettes.items()
                if name in initial_render_batch.silhouettes
            ],
        },
        {
            "stage": "optimized_render_and_score",
            "loss_total": float(loss.total),
            "loss_terms": dict(loss.terms),
            "loss_warnings": tuple(loss.warnings),
            "loss_view_weights": dict(target_view_weights),
            "views": refinement_history,
            "optimization": optimization_summary,
        },
    ]
    objective_improvement_record = {
        "baseline_total": float(baseline_loss.total),
        "initial_candidate_total": float(initial_loss.total),
        "objective_total": float(loss.total),
        "objective_improvement": float(objective_improvement),
        "zero_baseline_improvement": float(zero_baseline_improvement),
        "weight_sum": {
            "silhouette_weight_sum": float(sum(weight for weight in target_view_weights.values())),
            "depth_weight_sum": float(sum(weight for weight in target_view_weights.values())),
        },
        "history": objective_history,
        "view_signal_weights": dict(target_view_weights),
        "view_signal_details": dict(view_signal_details),
        "optimization": optimization_summary,
    }
    history_payload = [
        {
            "step": "baseline_zero",
            "backend": backend_choice,
            "softness": float(parsed_config["softness"]),
            "min_variance": float(parsed_config["min_variance"]),
            "loss_total": float(baseline_loss.total),
            "loss_terms": dict(baseline_loss.terms),
            "loss_warnings": tuple(baseline_loss.warnings),
            "loss_view_weights": dict(target_view_weights),
            "views": baseline_refinement_history,
            "render_metadata": dict(getattr(baseline_render_batch, "metadata", {})),
        },
        {
            "step": "initial_render_and_score",
            "backend": backend_choice,
            "softness": float(parsed_config["softness"]),
            "min_variance": float(parsed_config["min_variance"]),
            "loss_total": float(initial_loss.total),
            "loss_terms": dict(initial_loss.terms),
            "loss_warnings": tuple(initial_loss.warnings),
            "loss_view_weights": dict(target_view_weights),
            "view_signal_details": dict(view_signal_details),
            "views": [
                _silhouette_view_history(
                    name=name,
                    predicted=np.asarray(initial_render_batch.silhouettes[name], dtype=np.float64),
                    target=np.asarray(target_mask, dtype=np.float64),
                    metrics=dict(initial_loss.per_view.get(name, {})),
                )
                for name, target_mask in silhouettes.items()
                if name in initial_render_batch.silhouettes
            ],
            "render_metadata": dict(getattr(initial_render_batch, "metadata", {})),
        },
        {
            "step": "optimized_render_and_score",
            "backend": backend_choice,
            "softness": float(parsed_config["softness"]),
            "min_variance": float(parsed_config["min_variance"]),
            "loss_total": float(loss.total),
            "loss_terms": dict(loss.terms),
            "loss_warnings": tuple(loss.warnings),
            "loss_view_weights": dict(target_view_weights),
            "view_signal_details": dict(view_signal_details),
            "views": refinement_history,
            "render_metadata": dict(getattr(render_batch, "metadata", {})),
            "optimization": optimization_summary,
            "optimization_history": optimization_history,
        }
    ]
    if root is not None:
        primitive_path = write_primitive_set(
            root / "primitives" / "differentiable-refine.json",
            optimized_primitives,
            metadata={
                "backend": backend_choice,
                "loss": loss.terms,
                "per_view": loss.per_view,
                "surface_points": point_meta,
                "topology": topology_payload,
                "optimization": optimization_summary,
            },
        )
        artifacts["primitive_json"] = primitive_path
        mesh_path = write_obj(
            root / "mesh" / "differentiable-refine.obj",
            mesh_proxy,
            header=(f"candidate {candidate_id}", backend_name),
        )
        artifacts["mesh_obj"] = mesh_path
        objective_path = root / "artifacts" / "differentiable-refine-objective-history.json"
        write_json(
            objective_path,
            {
                "candidate_id": candidate_id,
                "backend": backend_choice,
                "objective_improvement": objective_improvement_record,
                "history": objective_history,
                "config": {
                    "backend": backend_choice,
                    "optional_dependency_policy": optional_dependency_policy,
                    "gradient_mode": str(parsed_config["gradient_mode"]),
                    "finite_difference_epsilon": float(parsed_config["finite_difference_epsilon"]),
                    "softness": float(parsed_config["softness"]),
                    "min_variance": float(parsed_config["min_variance"]),
                    "visual_hull_resolution": int(parsed_config["visual_hull_resolution"]),
                    "primitive_count": int(parsed_config["primitive_count"]),
                    "target_point_count": int(parsed_config["target_point_count"]),
                    "min_radius": float(parsed_config["min_radius"]),
                    "covariance_floor": float(parsed_config["covariance_floor"]),
                    "kmeans_iterations": int(parsed_config["kmeans_iterations"]),
                    "chunk_size": parsed_config["chunk_size"],
                    "optimization_steps": int(parsed_config["optimization_steps"]),
                    "optimization_initial_step": float(parsed_config["optimization_initial_step"]),
                    "optimization_step_decay": float(parsed_config["optimization_step_decay"]),
                    "optimization_min_step": float(parsed_config["optimization_min_step"]),
                    "max_objective_evaluations": parsed_config["max_objective_evaluations"],
                    "max_runtime_s": parsed_config["max_runtime_s"],
                    "loss_weights": dict(parsed_config["loss_weights_dict"]),
                    "warnings": tuple(config_warnings),
                },
                "loss": {
                    "baseline_total": float(baseline_loss.total),
                    "initial_total": float(initial_loss.total),
                    "final_total": float(loss.total),
                    "baseline_terms": dict(baseline_loss.terms),
                    "initial_terms": dict(initial_loss.terms),
                    "final_terms": dict(loss.terms),
                },
                "optimization": optimization_summary,
                "optimization_history": optimization_history,
                "view_signal_weights": dict(target_view_weights),
                "view_signal_details": dict(view_signal_details),
                "view_signal_warnings": tuple(target_signal_warnings),
            },
        )
        artifacts["objective_history"] = objective_path
        history_path = root / "artifacts" / "differentiable-refine-history.json"
        write_json(
            history_path,
            {
                "candidate_id": candidate_id,
                "backend": backend_choice,
                "config": {
                    "backend": backend_choice,
                    "optional_dependency_policy": optional_dependency_policy,
                    "gradient_mode": str(parsed_config["gradient_mode"]),
                    "finite_difference_epsilon": float(parsed_config["finite_difference_epsilon"]),
                    "softness": float(parsed_config["softness"]),
                    "min_variance": float(parsed_config["min_variance"]),
                    "visual_hull_resolution": int(parsed_config["visual_hull_resolution"]),
                    "primitive_count": int(parsed_config["primitive_count"]),
                    "target_point_count": int(parsed_config["target_point_count"]),
                    "min_radius": float(parsed_config["min_radius"]),
                    "covariance_floor": float(parsed_config["covariance_floor"]),
                    "kmeans_iterations": int(parsed_config["kmeans_iterations"]),
                    "chunk_size": parsed_config["chunk_size"],
                    "optimization_steps": int(parsed_config["optimization_steps"]),
                    "optimization_initial_step": float(parsed_config["optimization_initial_step"]),
                    "optimization_step_decay": float(parsed_config["optimization_step_decay"]),
                    "optimization_min_step": float(parsed_config["optimization_min_step"]),
                    "max_objective_evaluations": parsed_config["max_objective_evaluations"],
                    "max_runtime_s": parsed_config["max_runtime_s"],
                    "loss_weights": dict(parsed_config["loss_weights_dict"]),
                    "warnings": tuple(config_warnings),
                },
                "history": history_payload,
                "objective_improvement": objective_improvement_record,
                "objective_history": objective_history,
                "optimization": optimization_summary,
                "optimization_history": optimization_history,
                "loss": {
                    "total": float(loss.total),
                    "terms": dict(loss.terms),
                    "warnings": tuple(loss.warnings),
                },
                "per_view": {name: dict(values) for name, values in loss.per_view.items()},
            },
        )
        artifacts["refinement_history"] = history_path

    elapsed = time.perf_counter() - start
    area_iou_mean = 1.0 - float(loss.terms.get("area_iou", 1.0))
    soft_iou_mean = 1.0 - float(loss.terms.get("soft_iou", 1.0))
    candidate_per_view = _candidate_per_view_metrics(loss.per_view)
    area_iou_mean = _mean_candidate_metric(
        candidate_per_view,
        "area_iou",
        fallback=max(0.0, area_iou_mean),
    )
    area_iou_min = _min_candidate_metric(
        candidate_per_view,
        "area_iou",
        fallback=max(0.0, min(area_iou_mean, soft_iou_mean)),
    )
    boundary_iou_mean = _mean_candidate_metric(
        candidate_per_view,
        "boundary_iou",
        fallback=max(0.0, soft_iou_mean),
    )
    failed_required_views = sum(
        1
        for payload in candidate_per_view.values()
        if bool(payload.get("required", True)) and not bool(payload.get("passed", False))
    )
    metrics = CandidateMetrics(
        area_iou_min=max(0.0, area_iou_min),
        area_iou_mean=max(0.0, area_iou_mean),
        boundary_iou_mean=max(0.0, boundary_iou_mean),
        topology_score=topology_score,
        topology_penalty=topology_penalty,
        editability_score=0.65,
        complexity_penalty=min(1.0, len(optimized_primitives) / 96.0),
        elapsed_s=elapsed,
        per_view=candidate_per_view,
        extras={
            "backend": backend_choice,
            "loss_total": loss.total,
            "loss_terms": loss.terms,
            "loss_warnings": loss.warnings,
            "primitive_count": len(optimized_primitives),
            "surface_points": point_meta,
            "render_metadata": dict(getattr(render_batch, "metadata", {})),
            "topology": topology_payload,
            "baseline_loss": dict(baseline_loss.terms),
            "baseline_total": float(baseline_loss.total),
            "baseline_warnings": tuple(baseline_loss.warnings),
            "initial_loss": dict(initial_loss.terms),
            "initial_total": float(initial_loss.total),
            "initial_warnings": tuple(initial_loss.warnings),
            "objective_total": float(loss.total),
            "objective_improvement": float(objective_improvement),
            "zero_baseline_improvement": float(zero_baseline_improvement),
            "objective_improvement_record": objective_improvement_record,
            "objective_history": objective_history,
            "optimization": optimization_summary,
            "optimization_history": optimization_history,
            "view_signal_weights": dict(target_view_weights),
            "view_signal_details": dict(view_signal_details),
            "view_signal_warnings": tuple(target_signal_warnings),
            "history": history_payload,
            "objective_history_artifact": str(artifacts.get("objective_history", "")),
            "history_file": str(artifacts.get("refinement_history", "")),
            "validation_warnings": tuple(config_warnings),
        },
    )
    warnings = (
        tuple(loss.warnings)
        + tuple(initial_loss.warnings)
        + tuple(baseline_loss.warnings)
        + tuple(config_warnings)
        + tuple(target_signal_warnings)
    )
    errors: tuple[str, ...] = ()
    status = "success" if optimized_primitives else "skipped"
    degraded = False
    require_improvement = bool(
        config.get("fail_on_objective_regression")
        or config.get("require_objective_improvement")
    )
    if objective_improvement < 0.0 or (
        require_improvement and objective_improvement <= 0.0
    ):
        regression_message = (
            "objective worsened relative to initial differentiable candidate"
            if objective_improvement < 0.0
            else "objective did not improve relative to initial differentiable candidate"
        )
        warnings = warnings + (regression_message,)
        if require_improvement:
            status = "failed"
            errors = (regression_message,)
        elif optimized_primitives:
            status = "degraded"
            degraded = True
    if optimized_primitives and failed_required_views and status == "success":
        status = "degraded"
        degraded = True
        warnings = warnings + (
            f"{failed_required_views} required soft-silhouette view(s) failed metric gates",
        )
    return CandidateResult(
        candidate_id=candidate_id,
        backend_name=backend_name,
        status=status,
        primitive_path=primitive_path,
        mesh_path=mesh_path,
        metric_result=metrics,
        artifacts=artifacts,
        warnings=warnings,
        errors=errors,
        degraded=degraded,
        payload={
            "primitives": optimized_primitives,
            "loss": loss,
            "baseline_loss": baseline_loss,
            "history": history_payload,
            "objective_history": objective_history,
            "objective_improvement": objective_improvement_record,
            "render_batch": render_batch,
            "view_signal_weights": dict(target_view_weights),
        },
    )


def _target_cameras_and_masks(target: object) -> tuple[tuple[CameraSpec, ...], dict[str, np.ndarray]]:
    from reconstruction.point_cloud import target_bounds

    bounds = target_bounds(target)
    cameras = []
    silhouettes = {}
    for constraint in getattr(target, "constraints", ()):
        mask = np.asarray(getattr(constraint.mask, "mask", constraint.mask), dtype=np.float32)
        if mask.ndim != 2:
            continue
        height, width = mask.shape
        if constraint.view == "side":
            axes = (1, 2)
            world_bounds = (bounds.min_y, bounds.max_y, bounds.min_z, bounds.max_z)
        elif constraint.view == "top":
            axes = (0, 1)
            world_bounds = (bounds.min_x, bounds.max_x, bounds.min_y, bounds.max_y)
        else:
            axes = (0, 2)
            world_bounds = (bounds.min_x, bounds.max_x, bounds.min_z, bounds.max_z)
        cameras.append(
            CameraSpec(
                name=constraint.view,
                axes=axes,
                image_size=(int(width), int(height)),
                world_bounds=world_bounds,
            )
        )
        silhouettes[constraint.view] = mask
    return tuple(cameras), silhouettes


def _scene_mesh_arrays(scene: RenderableScene) -> tuple[np.ndarray, np.ndarray]:
    vertices_blocks: list[np.ndarray] = []
    face_blocks: list[np.ndarray] = []
    offset = 0
    if scene.mesh is not None:
        vertices, faces = _mesh_arrays(scene.mesh)
        vertices_blocks.append(vertices)
        face_blocks.append(_triangulated_mesh_faces(faces))
        offset += len(vertices)
    for renderable in scene.primitives:
        if renderable.mesh_proxy is None:
            continue
        vertices, faces = _mesh_arrays(renderable.mesh_proxy)
        triangles = _triangulated_mesh_faces(faces)
        vertices_blocks.append(vertices)
        if len(triangles):
            face_blocks.append(triangles + offset)
        offset += len(vertices)
    if not vertices_blocks:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    vertices_all = np.vstack(vertices_blocks).astype(np.float32, copy=False)
    faces_all = (
        np.vstack(face_blocks).astype(np.int32, copy=False)
        if face_blocks
        else np.empty((0, 3), dtype=np.int32)
    )
    return vertices_all, faces_all


def _mesh_arrays(mesh: Any) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    vertices = np.asarray(getattr(mesh, "vertices", ()), dtype=np.float32)
    faces = tuple(tuple(int(index) for index in face) for face in getattr(mesh, "faces", ()))
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError(f"mesh vertices must have shape (N, 3), got {vertices.shape}")
    return vertices, faces


def _triangulated_mesh_faces(faces: Sequence[Sequence[int]]) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    for face in faces:
        if len(face) < 3:
            continue
        first = int(face[0])
        for index in range(1, len(face) - 1):
            tri = (first, int(face[index]), int(face[index + 1]))
            if len(set(tri)) == 3:
                triangles.append(tri)
    if not triangles:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(triangles, dtype=np.int32)


def _project_vertices_to_clip(vertices: np.ndarray, camera: CameraSpec) -> np.ndarray:
    vertex_array = np.asarray(vertices, dtype=np.float32)
    axes = tuple(int(axis) for axis in camera.axes)
    if len(axes) != 2 or axes[0] == axes[1] or any(axis not in (0, 1, 2) for axis in axes):
        raise RuntimeError(f"invalid camera axes for nvdiffrast: {camera.axes!r}")
    depth_axis = next(axis for axis in (0, 1, 2) if axis not in axes)
    xmin, xmax, ymin, ymax = (float(value) for value in camera.world_bounds)
    xden = max(abs(xmax - xmin), 1.0e-12)
    yden = max(abs(ymax - ymin), 1.0e-12)
    x_clip = ((vertex_array[:, axes[0]] - xmin) / xden) * 2.0 - 1.0
    y_clip = ((vertex_array[:, axes[1]] - ymin) / yden) * 2.0 - 1.0
    depth_values = vertex_array[:, depth_axis]
    depth_min = float(np.min(depth_values)) if len(depth_values) else 0.0
    depth_max = float(np.max(depth_values)) if len(depth_values) else 1.0
    depth_den = max(abs(depth_max - depth_min), 1.0e-12)
    z_clip = ((depth_values - depth_min) / depth_den) * 2.0 - 1.0
    w_clip = np.ones_like(z_clip, dtype=np.float32)
    return np.column_stack((x_clip, y_clip, z_clip, w_clip)).astype(
        np.float32,
        copy=False,
    )
