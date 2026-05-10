from __future__ import annotations

from typing import Any, Mapping


def resfit_candidate_status(
    *,
    config: Mapping[str, object],
    result: Any,
    improved: float,
    profile_rows_present: bool,
    profile_init_warning: str,
) -> tuple[str, bool, tuple[str, ...], tuple[str, ...]]:
    warnings = list(result.warnings)
    if profile_init_warning:
        warnings.append(profile_init_warning)
    if improved <= 0.0:
        warnings.append("objective did not improve during refinement")
    if improved >= 0.0 and not profile_rows_present:
        warnings.append("using geometric seeding; no profile constraints were available")
    if result.optimization_termination_reason in {
        "elapsed_time_budget",
        "objective_evaluation_budget",
    }:
        warnings.append(
            f"optimization stopped by {result.optimization_termination_reason}"
        )
    status = "success" if result.primitives else "skipped"
    degraded = False
    errors_out: tuple[str, ...] = ()
    if result.primitives and improved <= 0.0:
        no_improvement_message = "primitive fit objective did not improve"
        if bool(
            config.get("require_objective_improvement")
            or config.get("fail_on_no_improvement")
        ):
            status = "failed"
            errors_out = (no_improvement_message,)
        else:
            status = "degraded"
            degraded = True
    if (
        result.primitives
        and result.optimization_termination_reason
        in {"elapsed_time_budget", "objective_evaluation_budget"}
        and status == "success"
    ):
        status = "degraded"
        degraded = True
    return status, degraded, errors_out, tuple(warnings)
