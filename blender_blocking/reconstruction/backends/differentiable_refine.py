"""Differentiable-rendering-inspired refinement backend."""

from __future__ import annotations

from typing import Mapping

from ..backend import BackendCapabilities, BaseBackend
from ..types import CandidateRequest, CandidateResult


class DifferentiableRefinementBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            name="differentiable_refine",
            capabilities=BackendCapabilities(
                requires_blender=False,
                supports_pure_python=True,
                supports_multi_view=True,
                supports_uncertainty=True,
                supports_constraints=True,
                outputs_mesh=True,
                supports_gradients=True,
                optional_dependencies=("nvdiffrast", "torch"),
                editability_score=0.65,
            ),
        )

    def validate_config(self, config: Mapping[str, object]) -> list[str]:
        try:
            from reconstruction.differentiable import _normalize_differentiable_config
        except Exception:
            return ["differentiable render module is unavailable"]

        _, errors, _ = _normalize_differentiable_config(config)
        return list(errors)

    def reconstruct(self, request: CandidateRequest) -> CandidateResult:
        config_errors = self.validate_config(getattr(request, "config", {}))
        if config_errors:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="failed",
                errors=tuple(config_errors),
            )
        try:
            from reconstruction.differentiable import run_refinement_candidate
        except Exception as exc:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="failed",
                errors=(f"differentiable refinement unavailable: {exc}",),
            )
        return run_refinement_candidate(request)
