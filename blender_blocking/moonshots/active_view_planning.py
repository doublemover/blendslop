"""Active-view planning moonshot for ambiguity reduction."""

from __future__ import annotations

from .contracts import MoonshotExperiment, MoonshotRequest, MoonshotResult, research_candidate_result
from .papers import ACTIVE_VISION, VISUAL_HULL


EXPERIMENT = MoonshotExperiment(
    experiment_id="active_view_planning",
    title="Next-best-view ambiguity planner",
    subsystem="validation",
    hypothesis=(
        "Candidate disagreement, visual-hull uncertainty, and per-view boundary loss "
        "can select the next camera angle that resolves the largest silhouette ambiguity."
    ),
    expected_wins={
        "quality": "higher min_view_iou and lower signed-distance loss from one extra view",
        "throughput": "avoid brute-force dense multi-view capture when ambiguity is local",
    },
    required_inputs=("candidate_set", "uncertainty_volume", "camera_constraints"),
    validation_metrics=("uncertainty_reduction", "min_view_iou_delta", "capture_cost"),
    papers=(VISUAL_HULL, ACTIVE_VISION),
)


def run(request: MoonshotRequest) -> MoonshotResult:
    return research_candidate_result(
        request,
        next_steps=(
            "render disagreement heatmaps for candidate ensembles across candidate camera azimuths",
            "estimate silhouette information gain from uncertainty-volume projections",
            "emit a ranked next-view manifest with expected metric deltas and capture cost",
        ),
    )
