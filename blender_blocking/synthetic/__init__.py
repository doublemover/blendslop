"""Synthetic shape factory package."""

from __future__ import annotations

from .artifact_writer import (
    build_manifest,
    normalize_generation_policy,
    should_keep_generated_artifacts,
    validate_manifest,
    validate_manifest_tree,
    write_artifact_set,
)
from .registry import get_definition, list_definitions, list_suites
from .specs import (
    GENERATOR_VERSION,
    SyntheticArtifactSet,
    SyntheticShapeSpec,
    SyntheticViewSpec,
)

__all__ = [
    "GENERATOR_VERSION",
    "build_manifest",
    "normalize_generation_policy",
    "should_keep_generated_artifacts",
    "SyntheticArtifactSet",
    "SyntheticShapeSpec",
    "SyntheticViewSpec",
    "validate_manifest",
    "validate_manifest_tree",
    "write_artifact_set",
    "get_definition",
    "list_definitions",
    "list_suites",
]
