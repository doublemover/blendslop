"""Experiment tooling for iterative reconstruction refinement."""

from __future__ import annotations

from .adaptive_planner import (
    RefinementProposal,
    proposals_from_bundle,
    proposals_from_result_payload,
    variants_from_bundle,
)
from .content_adaptive_patches import (
    AdaptivePatch,
    PatchBox,
    PatchFusionResult,
    fuse_patch_predictions,
    score_map_from_signals,
    select_adaptive_patches,
)
from .contracts import (
    ExperimentCase,
    ExperimentPlan,
    ExperimentResult,
    ExperimentVariant,
    ParameterSpec,
    RefinementRunManifest,
)
from .editability_study import (
    EditabilityCriterion,
    EditabilityStudyItem,
    EditabilityStudyPack,
    build_editability_study_pack,
    score_review_row,
    summarize_review_rows,
    write_editability_study_pack,
)

__all__ = [
    "ExperimentCase",
    "ExperimentPlan",
    "ExperimentResult",
    "ExperimentVariant",
    "EditabilityCriterion",
    "EditabilityStudyItem",
    "EditabilityStudyPack",
    "AdaptivePatch",
    "ParameterSpec",
    "PatchBox",
    "PatchFusionResult",
    "RefinementProposal",
    "RefinementRunManifest",
    "build_editability_study_pack",
    "fuse_patch_predictions",
    "proposals_from_bundle",
    "proposals_from_result_payload",
    "score_map_from_signals",
    "score_review_row",
    "select_adaptive_patches",
    "summarize_review_rows",
    "variants_from_bundle",
    "write_editability_study_pack",
]
