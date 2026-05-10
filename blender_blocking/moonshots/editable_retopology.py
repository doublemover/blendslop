"""Editable retopology moonshot for Blender output quality."""

from __future__ import annotations

from .contracts import MoonshotExperiment, MoonshotRequest, MoonshotResult, research_candidate_result
from .papers import LEWINER_MC, MARCHING_CUBES, SCREENED_POISSON


EXPERIMENT = MoonshotExperiment(
    experiment_id="editable_retopology",
    title="Topology-aware editable retopology pass",
    subsystem="mesh_generation",
    hypothesis=(
        "Mesh extraction should optimize for Blender editability, not only silhouette agreement; "
        "quad-friendly topology and component sanity can be scored as first-class outputs."
    ),
    expected_wins={
        "editability": "cleaner components, fewer non-manifold artifacts, better sculpt/readiness",
        "reliability": "explicit failure when visual quality is mesh-hostile",
    },
    required_inputs=("mesh_candidate", "topology_report", "shape_program_or_primitive_proxy"),
    validation_metrics=("manifoldness", "watertightness", "component_sanity", "editable_index"),
    papers=(MARCHING_CUBES, LEWINER_MC, SCREENED_POISSON),
)


def run(request: MoonshotRequest) -> MoonshotResult:
    return research_candidate_result(
        request,
        next_steps=(
            "derive editability defects from mesh topology reports and Blender mesh QA",
            "prototype remesh policies tied to defect classes instead of unconditional voxel remesh",
            "record before/after topology metrics and reject passes that harm silhouette or manifoldness",
        ),
    )
