"""Implicit SDF proxy moonshot for post-hull detail recovery."""

from __future__ import annotations

from .contracts import MoonshotExperiment, MoonshotRequest, MoonshotResult, research_candidate_result
from .papers import POISSON, SCREENED_POISSON, SPACE_CARVING


EXPERIMENT = MoonshotExperiment(
    experiment_id="implicit_sdf_proxy",
    title="Hybrid visual-hull SDF proxy and editable extraction",
    subsystem="volume",
    hypothesis=(
        "A calibrated SDF proxy can smooth voxel artifacts and preserve silhouette constraints "
        "before editable mesh or primitive extraction."
    ),
    expected_wins={
        "quality": "smoother surfaces, better normal consistency, fewer staircase artifacts",
        "performance": "sparse SDF chunks reuse the existing visual-hull chunk cache",
    },
    required_inputs=("visual_hull_volume", "silhouette_signed_distance_fields"),
    validation_metrics=("chamfer_l2", "normal_consistency", "volumetric_iou", "watertightness"),
    papers=(SPACE_CARVING, POISSON, SCREENED_POISSON),
)


def run(request: MoonshotRequest) -> MoonshotResult:
    return research_candidate_result(
        request,
        next_steps=(
            "project per-view signed distance into sparse volume chunks with uncertainty weights",
            "extract candidate meshes with screened-Poisson and marching-cubes baselines",
            "compare against synthetic SDF ground truth and editable retopology gates",
        ),
    )
