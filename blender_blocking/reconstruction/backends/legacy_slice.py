"""Legacy slice backend wrapper."""

from __future__ import annotations

from typing import Any, Mapping

from ..backend import BackendCapabilities, BaseBackend, BackendBudget
from ..types import CandidateMetrics, CandidateRequest, CandidateResult


class LegacySliceBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            name="legacy",
            capabilities=BackendCapabilities(
                requires_blender=True,
                supports_pure_python=False,
                supports_multi_view=False,
                outputs_mesh=True,
                editability_score=0.25,
            ),
        )

    def estimate_budget(
        self, target: Any, config: Mapping[str, Any]
    ) -> BackendBudget:
        num_slices = int(config.get("num_slices", 10))
        return BackendBudget(
            estimated_seconds=max(1.0, num_slices * 0.2),
            notes=("legacy sequential primitives/joins scale with slice count",),
        )

    def reconstruct(self, request: CandidateRequest) -> CandidateResult:
        workflow = getattr(request.context, "workflow", None)
        if workflow is None:
            return self.unavailable(
                request,
                "LegacySliceBackend requires a live BlockingWorkflow context.",
            )
        try:
            obj = workflow.create_3d_blockout()
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
            metric_result=CandidateMetrics(editability_score=0.25),
            payload=obj,
            warnings=() if obj is not None else ("legacy workflow returned no object",),
        )
