"""Profile-loft backend wrapper."""

from __future__ import annotations

from typing import Any, Mapping

from ..backend import BackendCapabilities, BaseBackend, BackendBudget
from ..types import CandidateMetrics, CandidateRequest, CandidateResult


class ProfileLoftBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            name="profile_loft",
            capabilities=BackendCapabilities(
                requires_blender=True,
                supports_pure_python=False,
                supports_multi_view=True,
                supports_top_view=True,
                supports_uncertainty=True,
                outputs_mesh=True,
                editability_score=0.55,
            ),
        )

    def estimate_budget(
        self, target: Any, config: Mapping[str, Any]
    ) -> BackendBudget:
        return BackendBudget(
            estimated_seconds=2.0,
            notes=("profile loft avoids legacy sequential boolean unions",),
        )

    def reconstruct(self, request: CandidateRequest) -> CandidateResult:
        workflow = getattr(request.context, "workflow", None)
        if workflow is None:
            return self.unavailable(
                request,
                "ProfileLoftBackend requires a live BlockingWorkflow context.",
            )
        try:
            obj = workflow.create_3d_blockout_loft()
        except Exception as exc:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="failed",
                errors=(str(exc),),
            )
        return CandidateResult(
            candidate_id=request.candidate_id,
            backend_name=self.name,
            status="success" if obj is not None else "failed",
            metric_result=CandidateMetrics(editability_score=0.55),
            payload=obj,
            warnings=() if obj is not None else ("profile loft returned no object",),
        )
