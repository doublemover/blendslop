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
from .config import _normalize_differentiable_config
from .contracts import CameraSpec, GradientBatch, LossResult, LossWeights, ReconstructionTarget, RenderBatch, RenderableScene
from .losses import evaluate_render_loss
from .target_adapter import primitive_from_renderable


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
        from blender_blocking.primitives.shape_aware_silhouette import (
            is_ellipse, ShapeAwareSilhouetteCache,
        )
        contour_reports = []
        if any(not is_ellipse(part) for part in primitives):
            cache = ShapeAwareSilhouetteCache(
                [camera.to_orthographic_camera() for camera in cameras],
                self.softness, self.min_variance)
            silhouettes = cache.render(primitives)
            contour_reports = [{"view": key[0], "part_index": key[1], **report}
                               for key, report in cache.approximation_reports.items()]
        for camera in cameras:
            camera_name = camera.name
            if camera.name not in silhouettes:
                silhouettes[camera.name] = render_projected_soft_silhouette(
                    primitives, camera.to_orthographic_camera(), softness=self.softness,
                    min_variance=self.min_variance)
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
            "contour_approximation_reports": contour_reports,
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
