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
    ContentAdaptiveRefinementResult,
    PatchBox,
    PatchFusionResult,
    fuse_patch_predictions,
    run_content_adaptive_patch_refinement,
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
from .parameters import (
    CONFIG_PARAMETER_PATHS,
    ParameterCatalogIssue,
    apply_variant_parameter_to_config,
    validate_parameter_catalog,
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
    "CONFIG_PARAMETER_PATHS",
    "ContentAdaptiveRefinementResult",
    "ParameterCatalogIssue",
    "ParameterSpec",
    "PatchBox",
    "PatchFusionResult",
    "RefinementProposal",
    "RefinementRunManifest",
    "apply_variant_parameter_to_config",
    "build_editability_study_pack",
    "fuse_patch_predictions",
    "proposals_from_bundle",
    "proposals_from_result_payload",
    "run_content_adaptive_patch_refinement",
    "score_map_from_signals",
    "score_review_row",
    "select_adaptive_patches",
    "summarize_review_rows",
    "validate_parameter_catalog",
    "variants_from_bundle",
    "write_editability_study_pack",
]
