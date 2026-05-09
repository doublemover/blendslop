"""Differentiable-rendering-inspired refinement backend."""

from __future__ import annotations

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

    def reconstruct(self, request: CandidateRequest) -> CandidateResult:
        try:
            from reconstruction.differentiable_render import run_refinement_candidate
        except Exception as exc:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="skipped",
                warnings=(f"differentiable refinement unavailable: {exc}",),
            )
        return run_refinement_candidate(request)
