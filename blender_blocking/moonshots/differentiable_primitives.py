"""Differentiable editable primitive fitting moonshot."""

from __future__ import annotations

from .contracts import MoonshotExperiment, MoonshotRequest, MoonshotResult, research_candidate_result
from .papers import NEURAL_MESH_RENDERER, SOFT_RASTERIZER, SUPERQUADRICS


EXPERIMENT = MoonshotExperiment(
    experiment_id="differentiable_primitives",
    title="Differentiable silhouette fitting for editable primitive programs",
    subsystem="differentiable",
    hypothesis=(
        "A CPU-compatible soft silhouette objective can refine primitive transforms "
        "and dimensions without requiring CUDA-only rasterization."
    ),
    expected_wins={
        "quality": "lower boundary loss after primitive initialization",
        "editability": "directly optimized primitives instead of post-hoc mesh cleanup",
    },
    required_inputs=("primitive_program", "silhouette_targets", "profile_bands"),
    optional_dependencies=("torch", "rocm-capable-array-backend"),
    validation_metrics=("boundary_iou", "signed_distance_loss", "objective_delta"),
    papers=(NEURAL_MESH_RENDERER, SOFT_RASTERIZER, SUPERQUADRICS),
)


def run(request: MoonshotRequest) -> MoonshotResult:
    return research_candidate_result(
        request,
        next_steps=(
            "generalize CPU soft silhouette kernels from point primitives to superfrusta and CSG programs",
            "add finite-difference and analytic gradient crosschecks on synthetic fixtures",
            "gate acceptance on objective improvement and per-view metric improvement, not average IoU alone",
        ),
    )
