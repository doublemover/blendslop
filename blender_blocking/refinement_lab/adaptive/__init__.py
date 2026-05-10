"""Adaptive refinement proposal package."""

from __future__ import annotations

from .planner import (
    RefinementProposal,
    ambiguity_signal,
    merge_proposals,
    proposals_from_bundle,
    proposals_from_result_payload,
    variants_from_bundle,
)

__all__ = [
    "RefinementProposal",
    "ambiguity_signal",
    "merge_proposals",
    "proposals_from_bundle",
    "proposals_from_result_payload",
    "variants_from_bundle",
]
