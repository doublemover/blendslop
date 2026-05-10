"""Adaptive refinement proposal package."""

from __future__ import annotations

from .contracts import RefinementProposal
from .selection import (
    merge_proposals,
    proposals_from_bundle,
    proposals_from_result_payload,
    variants_from_bundle,
)
from .signals import ambiguity_signal

__all__ = [
    "RefinementProposal",
    "ambiguity_signal",
    "merge_proposals",
    "proposals_from_bundle",
    "proposals_from_result_payload",
    "variants_from_bundle",
]
