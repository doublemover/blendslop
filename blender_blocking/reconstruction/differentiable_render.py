"""
Optional differentiable rendering interface and CPU fallback backend.

This module is intentionally Blender-free by default. GPU renderers can plug in
behind lazy imports, while the CPU backend gives primitive fitting a deterministic
soft-silhouette objective today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Protocol, Sequence

import numpy as np

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
    ) -> LossResult:
        ...

    def backward(self, loss: LossResult) -> GradientBatch:
        ...


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
) -> LossResult:
    """Compute soft silhouette/depth losses for a rendered batch."""
    terms: Dict[str, float] = {}
    per_view: Dict[str, Mapping[str, float]] = {}
    warnings = []

    for name, predicted in render_batch.silhouettes.items():
        target_mask = target.silhouettes.get(name)
        if target_mask is None:
            warnings.append(f"missing target silhouette for view {name}")
            continue
        metrics = dict(soft_mask_metrics(predicted, target_mask))
        per_view[name] = metrics
        for key, value in metrics.items():
            terms[f"{name}_{key}"] = float(value)

    silhouette_l2 = float(
        np.mean([m["soft_l2"] for m in per_view.values()])
        if per_view
        else 0.0
    )
    soft_iou = float(
        np.mean([m["soft_iou_loss"] for m in per_view.values()])
        if per_view
        else 0.0
    )
    area_iou = float(
        np.mean([m["area_iou_loss"] for m in per_view.values()])
        if per_view
        else 0.0
    )
    terms["silhouette_l2"] = silhouette_l2
    terms["soft_iou"] = soft_iou
    terms["area_iou"] = area_iou

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
        depth_losses.append(float(np.mean((pred - tgt) ** 2)))
    terms["depth_l2"] = float(np.mean(depth_losses)) if depth_losses else 0.0

    total = (
        weights.silhouette_l2 * silhouette_l2
        + weights.soft_iou * soft_iou
        + weights.area_iou * area_iou
        + weights.depth_l2 * terms["depth_l2"]
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

    def __init__(self, softness: float = 24.0) -> None:
        self.softness = float(softness)
        self._last_scene: RenderableScene | None = None
        self._last_cameras: tuple[CameraSpec, ...] = ()
        self._last_target: ReconstructionTarget | None = None
        self._last_weights: LossWeights = LossWeights()

    def render(self, scene: RenderableScene, cameras: Sequence[CameraSpec]) -> RenderBatch:
        primitives = [primitive_from_renderable(primitive) for primitive in scene.primitives]
        silhouettes = {}
        for camera in cameras:
            silhouettes[camera.name] = render_projected_soft_silhouette(
                primitives,
                camera.to_orthographic_camera(),
                softness=self.softness,
            )
        self._last_scene = scene
        self._last_cameras = tuple(cameras)
        return RenderBatch(
            silhouettes=silhouettes,
            metadata={"backend": self.name, "softness": self.softness},
        )

    def loss(
        self,
        render_batch: RenderBatch,
        target: ReconstructionTarget,
        weights: LossWeights = LossWeights(),
    ) -> LossResult:
        self._last_target = target
        self._last_weights = weights
        return evaluate_render_loss(render_batch, target, weights)

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
    ) -> LossResult:
        return self.loss_callback(render_batch, target, weights)

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
        self.unavailable_reason: str | None = None
        try:
            import nvdiffrast  # type: ignore

            self._module = nvdiffrast
        except Exception as exc:  # pragma: no cover - dependency optional.
            self.unavailable_reason = str(exc)

    @property
    def available(self) -> bool:
        return self._module is not None

    def _require_available(self) -> None:
        if not self.available:
            raise RuntimeError(
                "nvdiffrast is unavailable; use CpuSoftSilhouetteBackend or "
                f"BlenderFiniteDifferenceBackend instead ({self.unavailable_reason})"
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
    ) -> LossResult:
        self._require_available()
        return evaluate_render_loss(render_batch, target, weights)

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
    from reconstruction.point_cloud import target_bounds, target_surface_points
    from reconstruction.types import CandidateMetrics, CandidateResult

    start = time.perf_counter()
    config = dict(getattr(request, "config", {}) or {})
    backend_choice = str(config.get("backend", "cpu_soft_silhouette"))
    candidate_id = getattr(request, "candidate_id")
    backend_name = getattr(request, "backend_name", "differentiable_refine")
    target = getattr(request, "target")

    if backend_choice == "nvdiffrast":
        nvd = NvdiffrastBackend()
        if not nvd.available:
            policy = str(config.get("optional_dependency_policy", "skip"))
            status = "failed" if policy == "fail" else "skipped"
            return CandidateResult(
                candidate_id=candidate_id,
                backend_name=backend_name,
                status=status,
                warnings=(nvd.unavailable_reason or "nvdiffrast unavailable",),
            )
        policy = str(config.get("optional_dependency_policy", "skip"))
        status = "failed" if policy == "fail" else "skipped"
        return CandidateResult(
            candidate_id=candidate_id,
            backend_name=backend_name,
            status=status,
            warnings=(
                "nvdiffrast was detected, but this adapter needs explicit mesh "
                "translation and gradient extraction before it can run",
            ),
        )

    try:
        points, point_meta = target_surface_points(
            target,
            resolution=int(config.get("visual_hull_resolution", config.get("resolution", 40))),
            max_points=int(config.get("target_point_count", 2048)),
            chunk_size=config.get("chunk_size"),
        )
        primitives = tuple(
            initialize_ellipsoids_from_points(
                points,
                PrimitiveInitializationConfig(
                    primitive_count=int(config.get("primitive_count", 12)),
                    target_point_count=int(config.get("target_point_count", 2048)),
                    min_radius=float(config.get("min_radius", 0.04)),
                    covariance_floor=float(config.get("covariance_floor", 1e-4)),
                    kmeans_iterations=int(config.get("kmeans_iterations", 8)),
                ),
            )
        )
        renderables = tuple(renderable_from_primitive(primitive) for primitive in primitives)
        scene = RenderableScene(primitives=renderables)
        cameras, silhouettes = _target_cameras_and_masks(target)
        renderer = CpuSoftSilhouetteBackend(softness=float(config.get("softness", 24.0)))
        render_batch = renderer.render(scene, cameras)
        loss = renderer.loss(
            render_batch,
            ReconstructionTarget(silhouettes=silhouettes, surface_points=points),
            LossWeights(
                silhouette_l2=float(config.get("silhouette_l2_weight", 1.0)),
                soft_iou=float(config.get("soft_iou_weight", 1.0)),
                area_iou=float(config.get("area_iou_weight", 0.25)),
                depth_l2=float(config.get("depth_l2_weight", 0.0)),
            ),
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
    if root is not None:
        primitive_path = write_primitive_set(
            root / "primitives" / "differentiable-refine.json",
            primitives,
            metadata={
                "backend": backend_choice,
                "loss": loss.terms,
                "per_view": loss.per_view,
                "surface_points": point_meta,
            },
        )
        artifacts["primitive_json"] = primitive_path
        mesh_proxy = combine_primitive_meshes(primitives, resolution=20)
        mesh_path = write_obj(
            root / "mesh" / "differentiable-refine.obj",
            mesh_proxy,
            header=(f"candidate {candidate_id}", backend_name),
        )
        artifacts["mesh_obj"] = mesh_path

    elapsed = time.perf_counter() - start
    area_iou_mean = 1.0 - float(loss.terms.get("area_iou", 1.0))
    soft_iou_mean = 1.0 - float(loss.terms.get("soft_iou", 1.0))
    metrics = CandidateMetrics(
        area_iou_min=max(0.0, min(area_iou_mean, soft_iou_mean)),
        area_iou_mean=max(0.0, area_iou_mean),
        topology_score=0.65,
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
        },
    )
    return CandidateResult(
        candidate_id=candidate_id,
        backend_name=backend_name,
        status="success" if primitives else "skipped",
        primitive_path=primitive_path,
        mesh_path=mesh_path,
        metric_result=metrics,
        artifacts=artifacts,
        warnings=tuple(loss.warnings),
        payload={"primitives": primitives, "loss": loss, "render_batch": render_batch},
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
