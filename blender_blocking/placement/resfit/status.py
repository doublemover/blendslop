from __future__ import annotations

import math
from typing import Any, Mapping


_BUDGET_TERMINATIONS = {
    "elapsed_time_budget",
    "objective_evaluation_budget",
}


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
    termination_reason = str(getattr(result, "optimization_termination_reason", "") or "")
    budget_limited = termination_reason in _BUDGET_TERMINATIONS
    if budget_limited:
        warnings.append(
            f"optimization stopped by {termination_reason}"
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
    if result.primitives and budget_limited and status == "success":
        if _budget_limit_requires_degraded(config) or not _budget_limited_solution_valid(
            result,
            improved,
        ):
            status = "degraded"
            degraded = True
        else:
            warnings.append(
                "budget-limited primitive fit accepted: objective improved and "
                "valid primitives were emitted"
            )
    return status, degraded, errors_out, tuple(warnings)


def _budget_limit_requires_degraded(config: Mapping[str, object]) -> bool:
    return bool(
        config.get("degrade_on_budget_exhaustion")
        or config.get("fail_on_budget_exhaustion")
        or config.get("require_optimizer_completion")
    )


def _budget_limited_solution_valid(result: Any, improved: float) -> bool:
    if not getattr(result, "primitives", ()):
        return False
    if not math.isfinite(float(improved)) or float(improved) <= 0.0:
        return False
    final_loss = getattr(result, "final_loss", None)
    final_total = getattr(final_loss, "total", None)
    try:
        parsed_total = float(final_total)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed_total)
