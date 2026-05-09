"""Primitive fitting backend."""

from __future__ import annotations

from ..backend import BackendCapabilities, BaseBackend
from ..types import CandidateRequest, CandidateResult


class PrimitiveFitBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            name="primitive_fit_refine",
            capabilities=BackendCapabilities(
                requires_blender=False,
                supports_pure_python=True,
                supports_multi_view=True,
                supports_uncertainty=True,
                supports_constraints=True,
                outputs_primitive_set=True,
                outputs_mesh=True,
                editability_score=0.9,
            ),
        )

    def reconstruct(self, request: CandidateRequest) -> CandidateResult:
        try:
            from placement.resfit_pipeline import run_primitive_fit_pipeline
        except Exception as exc:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="skipped",
                warnings=(f"primitive fit pipeline unavailable: {exc}",),
            )
        return run_primitive_fit_pipeline(request)
