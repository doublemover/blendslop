"""Backend protocol and capability contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .types import CandidateRequest, CandidateResult


@dataclass(frozen=True)
class BackendCapabilities:
    requires_blender: bool = False
    supports_pure_python: bool = True
    supports_multi_view: bool = False
    supports_top_view: bool = False
    supports_uncertainty: bool = False
    supports_constraints: bool = False
    outputs_mesh: bool = False
    outputs_volume: bool = False
    outputs_primitive_set: bool = False
    supports_gradients: bool = False
    editability_score: float = 0.0
    optional_dependencies: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requires_blender": self.requires_blender,
            "supports_pure_python": self.supports_pure_python,
            "supports_multi_view": self.supports_multi_view,
            "supports_top_view": self.supports_top_view,
            "supports_uncertainty": self.supports_uncertainty,
            "supports_constraints": self.supports_constraints,
            "outputs_mesh": self.outputs_mesh,
            "outputs_volume": self.outputs_volume,
            "outputs_primitive_set": self.outputs_primitive_set,
            "supports_gradients": self.supports_gradients,
            "editability_score": self.editability_score,
            "optional_dependencies": list(self.optional_dependencies),
        }


@dataclass(frozen=True)
class BackendBudget:
    estimated_seconds: float = 0.0
    estimated_memory_mb: float = 0.0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_seconds": self.estimated_seconds,
            "estimated_memory_mb": self.estimated_memory_mb,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class BackendInfo:
    name: str
    version: str
    capabilities: BackendCapabilities

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": self.capabilities.to_dict(),
        }


class ReconstructionBackend(Protocol):
    """Protocol every reconstruction backend must implement."""

    name: str
    version: str
    capabilities: BackendCapabilities

    def validate_config(self, config: Mapping[str, Any]) -> list[str]:
        """Return validation error messages. Empty means valid."""

    def estimate_budget(
        self, target: Any, config: Mapping[str, Any]
    ) -> BackendBudget:
        """Estimate runtime/memory before scheduling a candidate."""

    def reconstruct(self, request: CandidateRequest) -> CandidateResult:
        """Run reconstruction and return a structured result."""

    def benchmark_cases(self) -> Sequence[Mapping[str, Any]]:
        """Return backend-specific benchmark case descriptors."""


@dataclass
class BaseBackend:
    """Small helper base for simple built-in backends."""

    name: str
    version: str = "1"
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)

    def validate_config(self, config: Mapping[str, Any]) -> list[str]:
        return []

    def estimate_budget(
        self, target: Any, config: Mapping[str, Any]
    ) -> BackendBudget:
        return BackendBudget(notes=(f"{self.name} has no calibrated budget yet",))

    def benchmark_cases(self) -> Sequence[Mapping[str, Any]]:
        return ()

    def unavailable(self, request: CandidateRequest, reason: str) -> CandidateResult:
        return CandidateResult(
            candidate_id=request.candidate_id,
            backend_name=self.name,
            status="skipped",
            warnings=(reason,),
        )
