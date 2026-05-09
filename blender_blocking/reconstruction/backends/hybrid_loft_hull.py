"""Hybrid profile-loft plus visual-hull backend."""

from __future__ import annotations

from ..backend import BackendCapabilities, BaseBackend
from ..types import CandidateMetrics, CandidateRequest, CandidateResult


class HybridLoftHullBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            name="hybrid_loft_hull",
            capabilities=BackendCapabilities(
                requires_blender=False,
                supports_pure_python=True,
                supports_multi_view=True,
                supports_top_view=True,
                supports_uncertainty=True,
                outputs_mesh=True,
                outputs_volume=True,
                editability_score=0.45,
            ),
        )

    def reconstruct(self, request: CandidateRequest) -> CandidateResult:
        warnings: list[str] = []
        artifacts = {}
        profile_result = None

        try:
            from .visual_hull import VisualHullBackend

            hull_request = CandidateRequest(
                candidate_id=f"{request.candidate_id}-visual_hull",
                backend_name="visual_hull_voxel",
                target=request.target,
                config=request.config,
                budget=request.budget,
                artifact_root=request.artifact_root,
                context=request.context,
            )
            hull_result = VisualHullBackend().reconstruct(hull_request)
        except Exception as exc:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="failed",
                errors=(f"visual hull branch failed: {exc}",),
            )

        profile_config = _profile_config_from_hybrid(request.config)
        try:
            from .profile_loft import ProfileLoftBackend

            profile_request = CandidateRequest(
                candidate_id=f"{request.candidate_id}-profile_loft",
                backend_name="profile_loft",
                target=request.target,
                config=profile_config,
                budget=request.budget,
                artifact_root=request.artifact_root,
                context=request.context,
            )
            profile_backend = ProfileLoftBackend()
            profile_errors = profile_backend.validate_config(profile_config)
            if profile_errors:
                profile_result = CandidateResult(
                    candidate_id=profile_request.candidate_id,
                    backend_name=profile_backend.name,
                    status="failed",
                    errors=tuple(profile_errors),
                )
            else:
                profile_result = profile_backend.reconstruct(profile_request)
        except Exception as exc:
            profile_result = CandidateResult(
                candidate_id=f"{request.candidate_id}-profile_loft",
                backend_name="profile_loft",
                status="failed",
                errors=(f"profile loft branch failed: {exc}",),
            )

        for key, value in hull_result.artifacts.items():
            artifacts[f"hull_{key}"] = value
        if profile_result is not None:
            for key, value in profile_result.artifacts.items():
                artifacts[f"profile_{key}"] = value

        hull_ok = hull_result.succeeded
        profile_ok = bool(profile_result and profile_result.succeeded)
        if hull_ok and profile_ok:
            status = "success"
        elif hull_ok or profile_ok:
            status = "degraded"
        elif hull_result.status == "skipped" and profile_result is not None:
            status = profile_result.status
        else:
            status = hull_result.status

        warnings.extend(hull_result.warnings)
        if profile_result is not None:
            warnings.extend(profile_result.warnings)
        errors = list(hull_result.errors)
        if profile_result is not None:
            errors.extend(profile_result.errors)
        if status in {"success", "degraded"}:
            errors = []

        profile_metrics = (
            profile_result.metric_result if profile_result is not None else CandidateMetrics()
        )
        metrics = CandidateMetrics(
            area_iou_min=hull_result.metric_result.area_iou_min,
            area_iou_mean=hull_result.metric_result.area_iou_mean,
            boundary_iou_mean=hull_result.metric_result.boundary_iou_mean,
            topology_score=max(
                hull_result.metric_result.topology_score,
                profile_metrics.topology_score,
            ),
            topology_penalty=min(
                _positive_or_one(hull_result.metric_result.topology_penalty),
                _positive_or_one(profile_metrics.topology_penalty),
            ),
            constraint_score=max(
                hull_result.metric_result.constraint_score,
                profile_metrics.constraint_score,
            ),
            constraint_penalty=min(
                _positive_or_one(hull_result.metric_result.constraint_penalty),
                _positive_or_one(profile_metrics.constraint_penalty),
            ),
            editability_score=0.45,
            complexity_penalty=max(
                hull_result.metric_result.complexity_penalty,
                profile_metrics.complexity_penalty,
            ),
            elapsed_s=(
                hull_result.metric_result.elapsed_s + profile_metrics.elapsed_s
            ),
            extras={
                "hybrid_strategy": "profile_loft_plus_visual_hull_backends",
                "hull_status": hull_result.status,
                "profile_status": profile_result.status if profile_result else "not_run",
                "hull_metrics": hull_result.metric_result.to_dict(),
                "profile_metrics": (
                    profile_result.metric_result.to_dict()
                    if profile_result is not None
                    else {}
                ),
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
            artifacts=artifacts,
            warnings=tuple(warnings),
            errors=tuple(errors),
            degraded=status == "degraded",
            payload={
                "profile_payload": profile_result.payload if profile_result else None,
                "hull_payload": hull_result.payload,
                "profile_result": profile_result.to_dict() if profile_result else None,
                "hull_result": hull_result.to_dict(),
            },
        )


def _profile_config_from_hybrid(config: object) -> dict[str, object]:
    if not isinstance(config, dict):
        config = dict(config or {})
    profile_sampling = dict(config.get("profile_sampling") or {})
    mesh_from_profile = dict(config.get("mesh_from_profile") or {})
    return {
        "unit_scale": config.get("unit_scale", 0.01),
        "num_slices": config.get("num_slices", 10),
        **profile_sampling,
        **mesh_from_profile,
    }


def _positive_or_one(value: float) -> float:
    value = float(value)
    return value if value > 0.0 else 1.0
