"""Shared reconstruction contracts, backend registry, and orchestration."""

from .backend import BackendBudget, BackendCapabilities, ReconstructionBackend
from .registry import (
    get_backend,
    list_backends,
    register_backend,
    register_builtin_backends,
)
from .ensemble import EnsembleRunner
from .pareto import (
    DEFAULT_PARETO_OBJECTIVES,
    ParetoCandidate,
    ParetoObjective,
    ParetoReport,
    pareto_report_from_candidates,
)
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
    "DEFAULT_PARETO_OBJECTIVES",
    "MeshQualityReport",
    "ParetoCandidate",
    "ParetoObjective",
    "ParetoReport",
    "ReconstructionBackend",
    "ReconstructionTarget",
    "TargetBuildResult",
    "build_target_from_images",
    "get_backend",
    "list_backends",
    "pareto_report_from_candidates",
    "register_backend",
    "register_builtin_backends",
]
