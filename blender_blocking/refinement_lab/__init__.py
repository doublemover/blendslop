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

__all__ = [
    "ExperimentCase",
    "ExperimentPlan",
    "ExperimentResult",
    "ExperimentVariant",
    "AdaptivePatch",
    "ParameterSpec",
    "PatchBox",
    "PatchFusionResult",
    "RefinementProposal",
    "RefinementRunManifest",
    "fuse_patch_predictions",
    "proposals_from_bundle",
    "proposals_from_result_payload",
    "score_map_from_signals",
    "select_adaptive_patches",
    "variants_from_bundle",
]
