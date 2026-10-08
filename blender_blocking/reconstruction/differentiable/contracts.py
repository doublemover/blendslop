from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence

import numpy as np

try:
    from utils.optional_deps import optional_policy_decision, probe_dependency
except Exception:  # pragma: no cover
    from ...utils.optional_deps import optional_policy_decision, probe_dependency

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
    valid_masks: Mapping[str, np.ndarray] = field(default_factory=dict)
    probability_masks: Mapping[str, np.ndarray] = field(default_factory=dict)
    pixel_weights: Mapping[str, np.ndarray] = field(default_factory=dict)
    proposal_valid_masks: Mapping[str,np.ndarray] = field(default_factory=dict)

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
