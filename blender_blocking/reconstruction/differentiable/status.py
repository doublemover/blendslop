from __future__ import annotations

from typing import Mapping


def differentiable_candidate_status(
    *,
    config: Mapping[str, object],
    optimized_primitives_present: bool,
    objective_improvement: float,
    boundary_or_sdf_improved: bool,
    failed_required_views: int,
    warnings: tuple[str, ...],
) -> tuple[str, bool, tuple[str, ...], tuple[str, ...]]:
    errors: tuple[str, ...] = ()
    status = "success" if optimized_primitives_present else "skipped"
    degraded = False
    require_improvement = bool(
        config.get("fail_on_objective_regression")
        or config.get("require_objective_improvement")
    )
    if objective_improvement < 0.0 or (
        require_improvement and objective_improvement <= 0.0
    ):
        regression_message = (
            "objective worsened relative to initial differentiable candidate"
            if objective_improvement < 0.0
            else "objective did not improve relative to initial differentiable candidate"
        )
        warnings = warnings + (regression_message,)
        if require_improvement:
            status = "failed"
            errors = (regression_message,)
        elif optimized_primitives_present:
            status = "degraded"
            degraded = True
    if (
        optimized_primitives_present
        and objective_improvement > 0.0
        and not boundary_or_sdf_improved
        and status == "success"
    ):
        status = "degraded"
        degraded = True
        warnings = warnings + (
            "differentiable refinement did not improve boundary or signed-distance losses",
        )
    if optimized_primitives_present and failed_required_views and status == "success":
        status = "degraded"
        degraded = True
        warnings = warnings + (
            f"{failed_required_views} required soft-silhouette view(s) failed metric gates",
        )
    return status, degraded, errors, warnings
