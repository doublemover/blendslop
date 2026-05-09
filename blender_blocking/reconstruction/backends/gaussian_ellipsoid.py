"""Gaussian/ellipsoid proxy backend."""

from __future__ import annotations

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
                supports_uncertainty=True,
                outputs_primitive_set=True,
                editability_score=0.75,
            ),
        )

    def reconstruct(self, request: CandidateRequest) -> CandidateResult:
        try:
            from primitives.gaussian_ellipsoid import run_gaussian_ellipsoid_proxy
        except Exception as exc:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="skipped",
                warnings=(f"gaussian/ellipsoid proxy unavailable: {exc}",),
            )
        return run_gaussian_ellipsoid_proxy(request)
