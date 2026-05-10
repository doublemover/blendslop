"""Human-in-the-loop constraint learning moonshot."""

from __future__ import annotations

from .contracts import MoonshotExperiment, MoonshotRequest, MoonshotResult, research_candidate_result
from .papers import VISUAL_HULL


EXPERIMENT = MoonshotExperiment(
    experiment_id="human_constraint_learning",
    title="Human constraint learning from edit and rating feedback",
    subsystem="constraints",
    hypothesis=(
        "Small human labels about intended symmetry, thin structures, and editable parts "
        "can be converted into reusable constraints that improve future reconstructions."
    ),
    expected_wins={
        "quality": "recover intended semantics that silhouettes alone cannot disambiguate",
        "throughput": "reduce repeated manual tuning by turning edits into priors",
    },
    required_inputs=("candidate_rankings", "human_labels", "constraint_schema"),
    validation_metrics=("label_satisfaction", "candidate_rank_delta", "editability_index"),
    papers=(VISUAL_HULL,),
)


def run(request: MoonshotRequest) -> MoonshotResult:
    return research_candidate_result(
        request,
        next_steps=(
            "normalize human labels into explicit constraint-schema records",
            "train or fit lightweight priors for symmetry, support thickness, and part salience",
            "report constraint satisfaction and disagreement in candidate manifests",
        ),
    )
