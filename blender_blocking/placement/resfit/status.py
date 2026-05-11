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


def apply_resfit_quality_floors(
    *,
    config: Mapping[str, object],
    status: str,
    degraded: bool,
    errors: tuple[str, ...],
    warnings: tuple[str, ...],
    metric: Any,
) -> tuple[str, bool, tuple[str, ...], tuple[str, ...]]:
    """Apply backend-status quality floors to primitive-fit candidates."""
    if status not in {"success", "degraded"}:
        return status, degraded, errors, warnings
    warnings_out = list(warnings)
    errors_out = list(errors)
    fail_reasons: list[str] = []
    degrade_reasons: list[str] = []

    extras = getattr(metric, "extras", {}) or {}
    min_iou = _float_attr(metric, "area_iou_min")
    if min_iou is not None:
        if min_iou < 0.35:
            message = _backend_iou_floor_message(
                min_iou,
                floor=0.35,
                metric=metric,
                extras=extras,
            )
            if _strict_backend_iou_floor_failure(config):
                fail_reasons.append(message)
            else:
                degrade_reasons.append(message)
        elif min_iou < 0.55:
            degrade_reasons.append(
                _backend_iou_floor_message(
                    min_iou,
                    floor=0.55,
                    metric=metric,
                    extras=extras,
                )
            )

    topology_score = _float_attr(metric, "topology_score")
    if topology_score is not None and topology_score < 0.75:
        degrade_reasons.append(
            f"topology score {topology_score:.3f} below 0.750"
        )

    topology = extras.get("topology") if isinstance(extras, Mapping) else None
    if isinstance(topology, Mapping):
        if topology.get("watertight") is False:
            degrade_reasons.append("primitive mesh proxy is not watertight")
        if int(topology.get("boundary_edges", 0) or 0) > 0:
            degrade_reasons.append("primitive mesh proxy has boundary edges")
        if int(topology.get("non_manifold_edges", 0) or 0) > 0:
            degrade_reasons.append("primitive mesh proxy has non-manifold edges")

    objective = extras.get("objective") if isinstance(extras, Mapping) else None
    if isinstance(objective, Mapping) and objective.get("improved") is False:
        message = "primitive objective did not improve"
        if bool(
            config.get("require_objective_improvement")
            or config.get("fail_on_no_improvement")
        ):
            fail_reasons.append(message)
        else:
            degrade_reasons.append(message)
    if isinstance(objective, Mapping) and objective.get("all_attempts_noop") is True:
        message = "primitive optimizer made no accepted objective-improving moves"
        if bool(
            config.get("fail_on_noop_optimization")
            or config.get("fail_on_no_improvement")
        ):
            fail_reasons.append(message)
        else:
            degrade_reasons.append(message)

    if fail_reasons:
        status = "failed"
        degraded = False
        warnings_out = [
            warning
            for warning in warnings_out
            if "budget-limited primitive fit accepted" not in warning
        ]
        errors_out.extend(fail_reasons)
    elif degrade_reasons and status == "success":
        status = "degraded"
        degraded = True

    warnings_out.extend(degrade_reasons)
    return (
        status,
        degraded,
        tuple(dict.fromkeys(errors_out)),
        tuple(dict.fromkeys(warnings_out)),
    )


def _budget_limit_requires_degraded(config: Mapping[str, object]) -> bool:
    return bool(
        config.get("degrade_on_budget_exhaustion")
        or config.get("fail_on_budget_exhaustion")
        or config.get("require_optimizer_completion")
    )


def _strict_backend_iou_floor_failure(config: Mapping[str, object]) -> bool:
    """Return whether weak primitive proxy/backend IoU should be fatal."""
    return bool(
        config.get("fail_on_quality_floor")
        or config.get("fail_on_backend_quality_floor")
        or config.get("fail_on_backend_iou_floor")
        or config.get("fail_on_proxy_iou_floor")
        or config.get("fail_on_internal_proxy_floor")
        or config.get("require_backend_quality_floor")
    )


def _backend_iou_floor_message(
    min_iou: float,
    *,
    floor: float,
    metric: Any,
    extras: Mapping[str, Any],
) -> str:
    source = "internal/proxy"
    if isinstance(extras, Mapping):
        raw_source = extras.get("backend_quality_source")
        if raw_source:
            source = str(raw_source).replace("_", " ")
    primitive_count = None
    if isinstance(extras, Mapping):
        primitive_count = extras.get("primitive_count")
    suffix = ""
    if primitive_count is not None:
        suffix = f" after emitting {primitive_count} primitive artifact(s)"
    return (
        f"primitive backend emitted renderable artifacts{suffix}, but {source} "
        f"min IoU {min_iou:.3f} is below {floor:.3f}"
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


def _float_attr(value: Any, name: str) -> float | None:
    try:
        raw = getattr(value, name)
    except Exception:
        return None
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
