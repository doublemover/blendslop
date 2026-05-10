"""Gaussian/ellipsoid proxy backend."""

from __future__ import annotations

from collections.abc import Mapping

from ..backend import BackendCapabilities, BaseBackend
from ..types import CandidateRequest, CandidateResult


class GaussianEllipsoidBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            name="gaussian_ellipsoid_proxy",
            capabilities=BackendCapabilities(
                requires_blender=False,
                supports_pure_python=True,
                supports_multi_view=True,
                supports_top_view=True,
                supports_uncertainty=True,
                supports_constraints=True,
                outputs_primitive_set=True,
                outputs_mesh=True,
                editability_score=0.75,
            ),
        )

    def validate_config(self, config: Mapping[str, object]) -> list[str]:
        from primitives.ellipsoid_proxy import validate_gaussian_ellipsoid_config

        return list(validate_gaussian_ellipsoid_config(config))

    def reconstruct(self, request: CandidateRequest) -> CandidateResult:
        try:
            from primitives.ellipsoid_proxy import run_gaussian_ellipsoid_proxy
        except Exception as exc:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="skipped",
                warnings=(f"gaussian/ellipsoid proxy unavailable: {exc}",),
            )
        return run_gaussian_ellipsoid_proxy(request)
