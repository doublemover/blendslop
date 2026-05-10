"""Program-search moonshot for editable shape grammars."""

from __future__ import annotations

from .contracts import MoonshotExperiment, MoonshotRequest, MoonshotResult, research_candidate_result
from .papers import SUPERQUADRICS


EXPERIMENT = MoonshotExperiment(
    experiment_id="shape_grammar_search",
    title="Shape grammar search over editable primitives",
    subsystem="shape_program",
    hypothesis=(
        "Search over constructive shape programs can recover cleaner editable "
        "structure than direct mesh extraction for blocky, furniture, and vehicle inputs."
    ),
    expected_wins={
        "editability": "clean named primitives, lower boolean/mesh cleanup load",
        "quality": "better thin-support and repeated-part reconstruction",
    },
    required_inputs=("multi_view_silhouettes", "profile_bands", "constraints"),
    validation_metrics=("min_view_iou", "boundary_iou", "editability_index", "component_sanity"),
    papers=(SUPERQUADRICS,),
)


def run(request: MoonshotRequest) -> MoonshotResult:
    return research_candidate_result(
        request,
        next_steps=(
            "add grammar production inventory with typed dimensions and constraints",
            "build beam search seeded by profile bands and synthetic ground truth labels",
            "score programs by silhouette, topology, primitive count, and editability metrics",
        ),
    )
