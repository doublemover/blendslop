"""Synthetic shape factory package."""

from __future__ import annotations

from .registry import get_definition, list_definitions, list_suites
from .specs import (
    GENERATOR_VERSION,
    SyntheticArtifactSet,
    SyntheticShapeSpec,
    SyntheticViewSpec,
)

__all__ = [
    "GENERATOR_VERSION",
    "SyntheticArtifactSet",
    "SyntheticShapeSpec",
    "SyntheticViewSpec",
    "get_definition",
    "list_definitions",
    "list_suites",
]
