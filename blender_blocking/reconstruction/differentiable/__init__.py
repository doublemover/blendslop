from __future__ import annotations

from . import candidate_adapter
from .candidate_adapter import run_refinement_candidate
from .config import _normalize_differentiable_config
from .contracts import (
    CameraSpec,
    DifferentiableRenderBackend,
    GradientBatch,
    LossResult,
    LossWeights,
    ReconstructionTarget,
    RenderBatch,
    RenderablePrimitive,
    RenderableScene,
)
from .cpu_soft import CpuSoftSilhouetteBackend
from .finite_difference import BlenderFiniteDifferenceBackend, finite_difference_array, finite_difference_scalar
from .losses import evaluate_render_loss
from .nvdiffrast_adapter import NvdiffrastBackend
from .target_adapter import _candidate_per_view_metrics, primitive_from_renderable, renderable_from_primitive

__all__ = [
    "BlenderFiniteDifferenceBackend",
    "CameraSpec",
    "CpuSoftSilhouetteBackend",
    "DifferentiableRenderBackend",
    "GradientBatch",
    "LossResult",
    "LossWeights",
    "NvdiffrastBackend",
    "ReconstructionTarget",
    "RenderBatch",
    "RenderablePrimitive",
    "RenderableScene",
    "_candidate_per_view_metrics",
    "_normalize_differentiable_config",
    "candidate_adapter",
    "evaluate_render_loss",
    "finite_difference_array",
    "finite_difference_scalar",
    "primitive_from_renderable",
    "renderable_from_primitive",
    "run_refinement_candidate",
]
