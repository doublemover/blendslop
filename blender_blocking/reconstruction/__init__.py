"""Shared reconstruction contracts, backend registry, and orchestration."""

from .backend import BackendBudget, BackendCapabilities, ReconstructionBackend
from .registry import (
    get_backend,
    list_backends,
    register_backend,
    register_builtin_backends,
)
from .ensemble import EnsembleRunner
from .target_builder import TargetBuildResult, build_target_from_images
from .types import (
    Bounds2D,
    Bounds3D,
    CandidateMetrics,
    CandidateRequest,
    CandidateResult,
    CandidateScore,
    CandidateScoreTerm,
    MeshQualityReport,
    ReconstructionTarget,
)

__all__ = [
    "BackendBudget",
    "BackendCapabilities",
    "Bounds2D",
    "Bounds3D",
    "CandidateMetrics",
    "CandidateRequest",
    "CandidateResult",
    "CandidateScore",
    "CandidateScoreTerm",
    "EnsembleRunner",
    "MeshQualityReport",
    "ReconstructionBackend",
    "ReconstructionTarget",
    "TargetBuildResult",
    "build_target_from_images",
    "get_backend",
    "list_backends",
    "register_backend",
    "register_builtin_backends",
]
