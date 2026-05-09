"""Hybrid loft/hull backend."""

from __future__ import annotations

from ..backend import BackendCapabilities, BaseBackend
from ..types import CandidateMetrics, CandidateRequest, CandidateResult


class HybridLoftHullBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            name="hybrid_loft_hull",
            capabilities=BackendCapabilities(
                requires_blender=True,
                supports_pure_python=False,
                supports_multi_view=True,
                supports_top_view=True,
                supports_uncertainty=True,
                outputs_mesh=True,
                outputs_volume=True,
                editability_score=0.45,
            ),
        )

    def reconstruct(self, request: CandidateRequest) -> CandidateResult:
        warnings = []
        profile_payload = None
        workflow = getattr(request.context, "workflow", None)
        if workflow is not None:
            try:
                profile_payload = workflow.create_3d_blockout_loft()
            except Exception as exc:
                warnings.append(f"profile loft branch failed: {exc}")

        try:
            from .visual_hull import VisualHullBackend

            hull_result = VisualHullBackend().reconstruct(request)
        except Exception as exc:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="failed",
                errors=(f"visual hull branch failed: {exc}",),
                warnings=tuple(warnings),
            )

        status = hull_result.status
        if profile_payload is not None and status == "success":
            status = "success"
        elif profile_payload is not None:
            status = "degraded"
        elif hull_result.status == "success":
            status = "degraded"

        metrics = CandidateMetrics(
            area_iou_min=hull_result.metric_result.area_iou_min,
            area_iou_mean=hull_result.metric_result.area_iou_mean,
            topology_score=max(0.0, hull_result.metric_result.topology_score),
            editability_score=0.45,
            complexity_penalty=hull_result.metric_result.complexity_penalty,
            elapsed_s=hull_result.metric_result.elapsed_s,
            extras={
                **dict(hull_result.metric_result.extras),
                "profile_loft_payload": getattr(profile_payload, "name", None),
                "hybrid_strategy": "profile_loft_plus_visual_hull_diagnostics",
            },
        )
        return CandidateResult(
            candidate_id=request.candidate_id,
            backend_name=self.name,
            status=status,
            mesh_path=hull_result.mesh_path,
            volume_path=hull_result.volume_path,
            render_paths=hull_result.render_paths,
            metric_result=metrics,
            artifacts=hull_result.artifacts,
            warnings=tuple(warnings + list(hull_result.warnings)),
            errors=hull_result.errors,
            degraded=status == "degraded",
            payload={
                "profile_payload": profile_payload,
                "hull_payload": hull_result.payload,
            },
        )
