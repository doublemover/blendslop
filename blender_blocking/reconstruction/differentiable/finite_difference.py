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
from .contracts import CameraSpec, GradientBatch, LossResult, LossWeights, ReconstructionTarget, RenderBatch, RenderableScene
from .losses import evaluate_render_loss


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
