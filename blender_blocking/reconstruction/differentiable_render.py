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
    from utils.optional_deps import probe_dependency
except Exception:  # pragma: no cover
    from ..utils.optional_deps import probe_dependency

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
    "depth_l2": 0.0,
}
_DIFF_WEIGHT_LEGACY_ALIASES = {
    "silhouette_l2_weight": "silhouette_l2",
    "soft_iou_weight": "soft_iou",
    "area_iou_weight": "area_iou",
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
        if key not in parsed:
            unknown.add(str(key))
            continue
        casted = _coerce_float(value, f"loss_weights.{key}", errors, min_value=0.0)
        if casted is not None:
            parsed[key] = casted
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

    normalized["loss_weights"] = _coerce_loss_weights(config, errors, warnings)
    normalized["loss_weights_dict"] = {
        "silhouette_l2": normalized["loss_weights"].silhouette_l2,
        "soft_iou": normalized["loss_weights"].soft_iou,
        "area_iou": normalized["loss_weights"].area_iou,
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
    }


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

    terms["silhouette_l2"] = silhouette_l2
    terms["soft_iou"] = soft_iou
    terms["area_iou"] = area_iou
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
        self._nvdiffrast = probe_dependency("nvdiffrast")
        self._torch = probe_dependency("torch")
        self._unmet_dependencies: list[str] = []
        if not self._nvdiffrast.available:
            self._unmet_dependencies.append(f"nvdiffrast: {self._nvdiffrast.error}")
        if not self._torch.available:
            self._unmet_dependencies.append(f"torch: {self._torch.error}")
        self._module = self._nvdiffrast.module if self._nvdiffrast.available else None
        self.unavailable_reason: str | None = None
        if self._unmet_dependencies:
            self.unavailable_reason = "; ".join(self._unmet_dependencies)

    @property
    def available(self) -> bool:
        return self._module is not None and self._torch.available

    @property
    def dependency_report(self) -> str:
        if self.available:
            return "dependencies satisfied"
        return "; ".join(self._unmet_dependencies)

    def _require_available(self) -> None:
        if not self.available:
            raise RuntimeError(
                "nvdiffrast is unavailable; use CpuSoftSilhouetteBackend or "
                f"BlenderFiniteDifferenceBackend instead ({self.dependency_report})"
            )

    def render(self, scene: RenderableScene, cameras: Sequence[CameraSpec]) -> RenderBatch:
        self._require_available()
        raise RuntimeError(
            "nvdiffrast is installed, but mesh scene translation is unavailable; "
            "select cpu_soft_silhouette for the current pure-Python backend"
        )

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
        raise RuntimeError(
            "nvdiffrast is installed, but gradient extraction is unavailable; "
            "select finite_difference gradient mode for the current backend"
        )


def run_refinement_candidate(request: object) -> object:
    """Run a CPU differentiable-rendering-inspired candidate.

    This path creates an editable ellipsoid proxy, renders soft silhouettes, and
    records loss terms through the same backend protocol that GPU
    renderers can implement.
    """
    import time

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
    optional_dependency_policy = str(parsed_config["optional_dependency_policy"])
    (
        target_view_weights,
        view_signal_details,
        target_signal_warnings,
    ) = _collect_target_view_signal_weights(target)

    if backend_choice == "nvdiffrast":
        nvd = NvdiffrastBackend()
        if not nvd.available:
            status = "failed" if optional_dependency_policy == "fail" else "skipped"
            reason = (
                nvd.unavailable_reason
                or nvd.dependency_report
                or "optional GPU dependency unavailable"
            )
            return CandidateResult(
                candidate_id=candidate_id,
                backend_name=backend_name,
                status=status,
                warnings=(
                    f"optional dependency unavailable: {reason}",
                    *config_warnings,
                    *tuple(target_signal_warnings),
                ),
            )
        status = "failed" if optional_dependency_policy == "fail" else "skipped"
        return CandidateResult(
            candidate_id=candidate_id,
            backend_name=backend_name,
            status=status,
            warnings=(
                "nvdiffrast was detected, but this adapter needs explicit mesh "
                "translation and gradient extraction before it can run",
                *config_warnings,
                *tuple(target_signal_warnings),
            ),
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
        render_batch = renderer.render(scene, cameras)
        loss = renderer.loss(
            render_batch,
            target_record,
            parsed_config["loss_weights"],
            view_weights=target_view_weights,
        )
    except Exception as exc:
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
    mesh_proxy = combine_primitive_meshes(primitives, resolution=20)
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
    objective_improvement = float(baseline_loss.total - loss.total)
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
            "stage": "render_and_score",
            "loss_total": float(loss.total),
            "loss_terms": dict(loss.terms),
            "loss_warnings": tuple(loss.warnings),
            "loss_view_weights": dict(target_view_weights),
            "views": refinement_history,
        },
    ]
    objective_improvement_record = {
        "baseline_total": float(baseline_loss.total),
        "objective_total": float(loss.total),
        "objective_improvement": float(objective_improvement),
        "weight_sum": {
            "silhouette_weight_sum": float(sum(weight for weight in target_view_weights.values())),
            "depth_weight_sum": float(sum(weight for weight in target_view_weights.values())),
        },
        "history": objective_history,
        "view_signal_weights": dict(target_view_weights),
        "view_signal_details": dict(view_signal_details),
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
            "step": "render_and_score",
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
        }
    ]
    if root is not None:
        primitive_path = write_primitive_set(
            root / "primitives" / "differentiable-refine.json",
            primitives,
            metadata={
                "backend": backend_choice,
                "loss": loss.terms,
                "per_view": loss.per_view,
                "surface_points": point_meta,
                "topology": topology_payload,
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
                    "loss_weights": dict(parsed_config["loss_weights_dict"]),
                    "warnings": tuple(config_warnings),
                },
                "loss": {
                    "baseline_total": float(baseline_loss.total),
                    "final_total": float(loss.total),
                    "baseline_terms": dict(baseline_loss.terms),
                    "final_terms": dict(loss.terms),
                },
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
                    "loss_weights": dict(parsed_config["loss_weights_dict"]),
                    "warnings": tuple(config_warnings),
                },
                "history": history_payload,
                "objective_improvement": objective_improvement_record,
                "objective_history": objective_history,
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
    metrics = CandidateMetrics(
        area_iou_min=max(0.0, min(area_iou_mean, soft_iou_mean)),
        area_iou_mean=max(0.0, area_iou_mean),
        boundary_iou_mean=max(0.0, soft_iou_mean),
        topology_score=topology_score,
        topology_penalty=topology_penalty,
        editability_score=0.65,
        complexity_penalty=min(1.0, len(primitives) / 96.0),
        elapsed_s=elapsed,
        per_view={
            view: {
                "passed": values.get("area_iou_loss", 1.0) <= 0.5,
                "required": True,
                **dict(values),
            }
            for view, values in loss.per_view.items()
        },
        extras={
            "backend": backend_choice,
            "loss_total": loss.total,
            "loss_terms": loss.terms,
            "loss_warnings": loss.warnings,
            "primitive_count": len(primitives),
            "surface_points": point_meta,
            "render_metadata": dict(getattr(render_batch, "metadata", {})),
            "topology": topology_payload,
            "baseline_loss": dict(baseline_loss.terms),
            "baseline_total": float(baseline_loss.total),
            "baseline_warnings": tuple(baseline_loss.warnings),
            "objective_total": float(loss.total),
            "objective_improvement": float(objective_improvement),
            "objective_improvement_record": objective_improvement_record,
            "objective_history": objective_history,
            "view_signal_weights": dict(target_view_weights),
            "view_signal_details": dict(view_signal_details),
            "view_signal_warnings": tuple(target_signal_warnings),
            "history": history_payload,
            "objective_history_artifact": str(artifacts.get("objective_history", "")),
            "history_file": str(artifacts.get("refinement_history", "")),
            "validation_warnings": tuple(config_warnings),
        },
    )
    warnings = tuple(loss.warnings) + tuple(baseline_loss.warnings) + tuple(config_warnings) + tuple(target_signal_warnings)
    if objective_improvement < 0.0:
        warnings = warnings + ("objective worsened relative to baseline zero silhouette",)
    return CandidateResult(
        candidate_id=candidate_id,
        backend_name=backend_name,
        status="success" if primitives else "skipped",
        primitive_path=primitive_path,
        mesh_path=mesh_path,
        metric_result=metrics,
        artifacts=artifacts,
        warnings=warnings,
        payload={
            "primitives": primitives,
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
