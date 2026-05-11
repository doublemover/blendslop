"""Active-view planning moonshot for ambiguity reduction."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import (
    MoonshotExperiment,
    MoonshotRequest,
    MoonshotResult,
    bundle_result,
    error_result,
    skipped_result,
)
from .papers import ACTIVE_VISION, VISUAL_HULL
from .support import (
    bounded,
    candidate_rows,
    per_view_area_iou,
    per_view_boundary_iou,
    per_view_signed_distance_loss,
    row_metric,
    target_signals,
)


EXPERIMENT = MoonshotExperiment(
    experiment_id="active_view_planning",
    title="Next-best-view ambiguity planner",
    subsystem="validation",
    hypothesis=(
        "Candidate disagreement, visual-hull uncertainty, and per-view boundary loss "
        "can select the next camera angle that resolves the largest silhouette ambiguity."
    ),
    expected_wins={
        "quality": "higher min_view_iou and lower signed-distance loss from one extra view",
        "throughput": "avoid brute-force dense multi-view capture when ambiguity is local",
    },
    required_inputs=("candidate_set", "uncertainty_volume", "camera_constraints"),
    validation_metrics=("uncertainty_reduction", "min_view_iou_delta", "capture_cost"),
    papers=(VISUAL_HULL, ACTIVE_VISION),
)


def run(request: MoonshotRequest) -> MoonshotResult:
    try:
        rows = candidate_rows(request.candidate)
        signals = target_signals(request)
        has_target = any(bool(group.get("available")) for group in signals.values())
        if not rows and not has_target:
            return skipped_result(
                request,
                reason="active view planning needs candidate rows or target uncertainty signals",
                next_steps=("run after at least one candidate or target signal extraction",),
            )
        return _plan_views(request, rows=rows, signals=signals)
    except Exception as exc:
        return error_result(request, error=f"{type(exc).__name__}: {exc}")


def _plan_views(
    request: MoonshotRequest,
    *,
    rows: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
) -> MoonshotResult:
    config = dict(request.config or {})
    max_views = max(1, int(config.get("max_view_suggestions", 5) or 5))
    existing_views = _existing_views(request, rows, signals)
    view_stats = _view_stats(rows, signals)
    candidates = _candidate_camera_views(config)
    ranked = []
    for candidate in candidates:
        related = tuple(candidate.get("related_views", ()) or ())
        if not related:
            related = ("front", "side", "top")
        weakness = max(
            (view_stats.get(view, {}).get("quality_gap", 0.0) for view in related),
            default=0.0,
        )
        disagreement = max((view_stats.get(view, {}).get("disagreement", 0.0) for view in related), default=0.0)
        uncertainty = max((view_stats.get(view, {}).get("uncertainty", 0.0) for view in related), default=0.0)
        boundary_pressure = max(
            (view_stats.get(view, {}).get("boundary_pressure", 0.0) for view in related),
            default=0.0,
        )
        novelty = _novelty(candidate, existing_views)
        is_new = 0.0 if str(candidate["view_id"]) in existing_views else 1.0
        capture_cost = float(candidate.get("capture_cost", 1.0) or 1.0)
        expected_delta = bounded(
            0.025
            + 0.16 * disagreement
            + 0.12 * uncertainty
            + 0.08 * weakness
            + 0.07 * boundary_pressure
            + 0.035 * is_new,
            0.0,
            0.35,
        )
        score = (expected_delta + 0.025 * novelty) / max(0.1, capture_cost)
        ranked.append(
            {
                **candidate,
                "score": score,
                "expected_metric_delta": expected_delta,
                "expected_uncertainty_reduction": bounded(0.08 + 0.4 * uncertainty),
                "candidate_disagreement": disagreement,
                "weak_view_pressure": weakness,
                "boundary_pressure": boundary_pressure,
                "novelty_score": novelty,
                "already_captured": str(candidate["view_id"]) in existing_views,
                "decision_factors": {
                    "weakness": weakness,
                    "disagreement": disagreement,
                    "uncertainty": uncertainty,
                    "boundary_pressure": boundary_pressure,
                    "novelty": novelty,
                    "capture_cost": capture_cost,
                },
            }
        )
    ranked.sort(key=lambda item: (float(item["score"]), str(item["view_id"])), reverse=True)
    suggestions = ranked[:max_views]
    sequence = _sequence_plan(ranked, max_views=max_views)
    ambiguity = _ambiguity_index(view_stats)
    evidence = {
        "existing_views": sorted(existing_views),
        "view_stats": view_stats,
        "requests": suggestions,
        "sequence_plan": sequence,
        "view_pair_pressure": _view_pair_pressure(view_stats),
        "stop_rule": {
            "minimum_expected_metric_delta": float(config.get("min_expected_view_delta", 0.03) or 0.03),
            "capture_until": "expected_metric_delta drops below threshold or measured delta fails guard",
        },
        "candidate_count": len(candidates),
        "ambiguity_index": ambiguity,
    }
    metrics = {
        "ran": 1.0,
        "suggestion_count": float(len(suggestions)),
        "best_expected_metric_delta": float(suggestions[0]["expected_metric_delta"]) if suggestions else 0.0,
        "best_expected_uncertainty_reduction": float(suggestions[0]["expected_uncertainty_reduction"]) if suggestions else 0.0,
        "cumulative_expected_metric_delta": sum(
            float(item.get("marginal_expected_metric_delta", 0.0) or 0.0)
            for item in sequence
        ),
        "ambiguity_index": ambiguity,
        "weak_view_count": float(
            sum(1 for item in view_stats.values() if item.get("quality_gap", 0.0) > 0.15)
        ),
    }
    return bundle_result(
        request,
        status="ran",
        metrics=metrics,
        evidence=evidence,
        artifact_name="active-view-plan.json",
        next_steps=(
            "capture the top suggested view before accepting ambiguous candidates",
            "compare measured delta against expected_metric_delta in the next smoke run",
        ),
    )


def _existing_views(
    request: MoonshotRequest,
    rows: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    views: set[str] = set()
    target = request.target
    if target is not None and hasattr(target, "views"):
        try:
            views.update(str(view) for view in target.views())
        except Exception:
            pass
    constraint_views = signals.get("constraints", {}).get("constraint_views")
    if isinstance(constraint_views, Mapping):
        views.update(str(view) for view in constraint_views)
    for row in rows:
        views.update(per_view_area_iou(row))
    return views or {"front", "side", "top"}


def _view_stats(
    rows: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    details = signals.get("uncertainty", {}).get("view_details")
    details = details if isinstance(details, Mapping) else {}
    output: dict[str, dict[str, float]] = {}
    for view in ("front", "side", "top"):
        values = [per_view_area_iou(row).get(view) for row in rows]
        present = [float(value) for value in values if value is not None]
        boundary_values = [per_view_boundary_iou(row).get(view) for row in rows]
        boundary_present = [float(value) for value in boundary_values if value is not None]
        sdf_values = [per_view_signed_distance_loss(row).get(view) for row in rows]
        sdf_present = [float(value) for value in sdf_values if value is not None]
        if present:
            mean_iou = sum(present) / len(present)
            disagreement = max(present) - min(present) if len(present) > 1 else 1.0 - mean_iou
        else:
            mean_iou = 0.0
            disagreement = 0.25
        mean_boundary = sum(boundary_present) / len(boundary_present) if boundary_present else mean_iou
        mean_sdf = sum(sdf_present) / len(sdf_present) if sdf_present else 0.0
        uncertainty_payload = details.get(view, {}) if isinstance(details, Mapping) else {}
        uncertainty = (
            float(uncertainty_payload.get("boundary_uncertainty_mean", 0.0) or 0.0)
            if isinstance(uncertainty_payload, Mapping)
            else 0.0
        )
        if uncertainty <= 0.0:
            uncertainty = float(signals.get("uncertainty", {}).get("overall_boundary_uncertainty_mean", 0.0) or 0.0)
        output[view] = {
            "mean_iou": bounded(mean_iou),
            "disagreement": bounded(disagreement),
            "uncertainty": bounded(uncertainty),
            "mean_boundary_iou": bounded(mean_boundary),
            "mean_signed_distance_loss": max(0.0, mean_sdf),
            "boundary_pressure": bounded((1.0 - mean_boundary) * 0.65 + min(1.0, mean_sdf * 18.0) * 0.35),
            "quality_gap": bounded(max(1.0 - mean_iou, 1.0 - mean_boundary)),
            "sample_count": float(len(present)),
            "failed": 1.0 if present and min(present) < 0.7 else 0.0,
        }
    return output


def _candidate_camera_views(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    configured = config.get("candidate_views")
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        rows = tuple(item for item in configured if isinstance(item, Mapping))
        if rows:
            return tuple(dict(item) for item in rows)
    return (
        {
            "view_id": "front_orthographic_refresh",
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "related_views": ("front",),
            "capture_cost": 0.85,
        },
        {
            "view_id": "side_orthographic_refresh",
            "azimuth_deg": 90.0,
            "elevation_deg": 0.0,
            "related_views": ("side",),
            "capture_cost": 0.85,
        },
        {
            "view_id": "top_orthographic_refresh",
            "azimuth_deg": 0.0,
            "elevation_deg": 90.0,
            "related_views": ("top",),
            "capture_cost": 0.9,
        },
        {
            "view_id": "front_side_diagonal_045",
            "azimuth_deg": 45.0,
            "elevation_deg": 0.0,
            "related_views": ("front", "side"),
            "capture_cost": 1.0,
        },
        {
            "view_id": "front_side_diagonal_135",
            "azimuth_deg": 135.0,
            "elevation_deg": 0.0,
            "related_views": ("front", "side"),
            "capture_cost": 1.0,
        },
        {
            "view_id": "rear_side_diagonal_225",
            "azimuth_deg": 225.0,
            "elevation_deg": 0.0,
            "related_views": ("front", "side"),
            "capture_cost": 1.15,
        },
        {
            "view_id": "top_oblique_060",
            "azimuth_deg": 60.0,
            "elevation_deg": 45.0,
            "related_views": ("top", "front"),
            "capture_cost": 1.25,
        },
        {
            "view_id": "top_oblique_300",
            "azimuth_deg": 300.0,
            "elevation_deg": 45.0,
            "related_views": ("top", "side"),
            "capture_cost": 1.25,
        },
        {
            "view_id": "low_oblique_120",
            "azimuth_deg": 120.0,
            "elevation_deg": -25.0,
            "related_views": ("front", "side"),
            "capture_cost": 1.35,
        },
        {
            "view_id": "low_oblique_240",
            "azimuth_deg": 240.0,
            "elevation_deg": -25.0,
            "related_views": ("front", "side"),
            "capture_cost": 1.35,
        },
    )


def _novelty(candidate: Mapping[str, Any], existing_views: set[str]) -> float:
    view_id = str(candidate.get("view_id", ""))
    if view_id in existing_views:
        return 0.0
    related = set(str(view) for view in candidate.get("related_views", ()) or ())
    if related and related.issubset(existing_views):
        return 0.55
    return 1.0


def _sequence_plan(
    ranked: Sequence[Mapping[str, Any]],
    *,
    max_views: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    covered: set[str] = set()
    for item in ranked:
        if len(selected) >= max_views:
            break
        related = set(str(view) for view in item.get("related_views", ()) or ())
        overlap = len(covered.intersection(related))
        diversity_discount = 1.0 - min(0.45, overlap * 0.18)
        marginal = float(item.get("expected_metric_delta", 0.0) or 0.0) * diversity_discount
        selected.append(
            {
                "order": len(selected) + 1,
                "view_id": str(item.get("view_id", "")),
                "related_views": sorted(related),
                "marginal_expected_metric_delta": bounded(marginal, 0.0, 0.35),
                "guard": "rerank after capture before taking the next planned view",
            }
        )
        covered.update(related)
    return selected


def _view_pair_pressure(view_stats: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    pairs = (("front", "side"), ("front", "top"), ("side", "top"))
    output = []
    for left, right in pairs:
        left_stats = view_stats.get(left, {})
        right_stats = view_stats.get(right, {})
        pressure = max(
            float(left_stats.get("quality_gap", 0.0) or 0.0),
            float(right_stats.get("quality_gap", 0.0) or 0.0),
            float(left_stats.get("boundary_pressure", 0.0) or 0.0),
            float(right_stats.get("boundary_pressure", 0.0) or 0.0),
        )
        output.append({"views": [left, right], "pressure": bounded(pressure)})
    return sorted(output, key=lambda item: float(item["pressure"]), reverse=True)


def _ambiguity_index(view_stats: Mapping[str, Mapping[str, float]]) -> float:
    if not view_stats:
        return 0.0
    values = []
    for stats in view_stats.values():
        values.append(
            max(
                float(stats.get("quality_gap", 0.0) or 0.0),
                float(stats.get("boundary_pressure", 0.0) or 0.0),
                float(stats.get("disagreement", 0.0) or 0.0),
                float(stats.get("uncertainty", 0.0) or 0.0),
            )
        )
    return bounded(sum(values) / max(1, len(values)))
