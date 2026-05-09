"""Active view planning for silhouette-only reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ViewRequest:
    view_id: str
    label: str
    azimuth_deg: float
    elevation_deg: float = 0.0
    roll_deg: float = 0.0
    priority: int = 50
    expected_information_gain: float = 0.0
    reason: str = ""
    target_failures: tuple[str, ...] = ()
    capture_notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "view_id": self.view_id,
            "label": self.label,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "roll_deg": self.roll_deg,
            "priority": self.priority,
            "expected_information_gain": self.expected_information_gain,
            "reason": self.reason,
            "target_failures": list(self.target_failures),
            "capture_notes": list(self.capture_notes),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ViewDisagreementSignal:
    candidate_count: int
    view_scores: Mapping[str, float] = field(default_factory=dict)
    view_stddev: Mapping[str, float] = field(default_factory=dict)
    view_mean_iou: Mapping[str, float] = field(default_factory=dict)
    recommended_view_scores: Mapping[str, float] = field(default_factory=dict)
    max_disagreement: float = 0.0
    entropy: float = 0.0
    source_views: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "view_scores": dict(self.view_scores),
            "view_stddev": dict(self.view_stddev),
            "view_mean_iou": dict(self.view_mean_iou),
            "recommended_view_scores": dict(self.recommended_view_scores),
            "max_disagreement": self.max_disagreement,
            "entropy": self.entropy,
            "source_views": list(self.source_views),
        }


DEFAULT_ACTIVE_VIEWS = (
    ViewRequest(
        "front_side_45",
        "45 degree front-side silhouette",
        45.0,
        priority=10,
        expected_information_gain=0.8,
        reason="Breaks front/side visual-hull ambiguity and reveals diagonal protrusions.",
    ),
    ViewRequest(
        "front_side_135",
        "135 degree opposite diagonal silhouette",
        135.0,
        priority=20,
        expected_information_gain=0.75,
        reason="Cross-checks asymmetry hidden by front/side silhouettes.",
    ),
    ViewRequest(
        "top_oblique_45",
        "Top oblique silhouette",
        45.0,
        elevation_deg=35.0,
        priority=30,
        expected_information_gain=0.65,
        reason="Separates height/depth ambiguity and catches thin supports.",
    ),
    ViewRequest(
        "rear",
        "Rear silhouette",
        180.0,
        priority=40,
        expected_information_gain=0.5,
        reason="Detects back-side asymmetry and missing rear protrusions.",
    ),
)


def suggest_next_views(
    bundle: Any,
    *,
    existing_views: Sequence[str] = (),
    candidate_bundles: Sequence[Any] = (),
    max_views: int = 3,
) -> tuple[ViewRequest, ...]:
    metrics = _metric_index(bundle)
    failures = _failure_codes(bundle)
    existing = {str(view).lower() for view in existing_views}
    disagreement = ensemble_disagreement_signal(
        candidate_bundles or _candidate_bundles_from_payload(bundle)
    )
    scored: list[ViewRequest] = []

    min_iou = _metric(metrics, "silhouette.min_view_iou")
    boundary = _metric(metrics, "silhouette.mean_boundary_iou")
    ambiguity = _metric(metrics, "geometry.ambiguity_gap")
    fscore = _metric(metrics, "geometry.fscore_tau", default=1.0)
    status = _status(bundle)

    for base in DEFAULT_ACTIVE_VIEWS:
        if base.view_id.lower() in existing or base.label.lower() in existing:
            continue
        gain = base.expected_information_gain
        priority = base.priority
        reasons = [base.reason]
        target_failures = list(failures)
        if min_iou < 0.7:
            gain += 0.2
            priority -= 10
            reasons.append("At least one required view has low silhouette IoU.")
        if boundary < 0.55:
            gain += 0.1
            reasons.append("Boundary IoU suggests contour detail is underconstrained.")
        if ambiguity > 0.1:
            gain += 0.25
            priority -= 10
            reasons.append("Geometry ambiguity gap suggests silhouette-only underconstraint.")
        if fscore < 0.6:
            gain += 0.15
            reasons.append("Surface F-score indicates missing recoverable geometry.")
        if status in {"research_only", "degraded"}:
            gain += 0.1
            reasons.append(f"Candidate status is {status}; another view can confirm whether the artifact should be promoted.")
        if "top" not in existing and base.view_id == "top_oblique_45":
            priority -= 15
            gain += 0.2
            reasons.append("No top view was listed; oblique top capture is high leverage.")
        view_signal = float(disagreement.recommended_view_scores.get(base.view_id, 0.0))
        if view_signal > 0.0:
            gain += min(0.3, view_signal * 0.35)
            priority -= int(round(view_signal * 20.0))
            reasons.append(
                "Ensemble disagreement indicates this view should reduce candidate uncertainty."
            )
        scored.append(
            ViewRequest(
                view_id=base.view_id,
                label=base.label,
                azimuth_deg=base.azimuth_deg,
                elevation_deg=base.elevation_deg,
                roll_deg=base.roll_deg,
                priority=max(1, priority),
                expected_information_gain=min(1.0, gain),
                reason=" ".join(reasons),
                target_failures=tuple(target_failures),
                capture_notes=_capture_notes(base),
                metadata={
                    "base_priority": base.priority,
                    "bundle_status": status,
                    "ensemble_disagreement": disagreement.to_dict(),
                    "view_signal": view_signal,
                },
            )
        )
    return tuple(
        sorted(scored, key=lambda item: (item.priority, -item.expected_information_gain))[
            : max(1, int(max_views))
        ]
    )


def active_view_plan_payload(
    bundle: Any,
    *,
    existing_views: Sequence[str] = (),
    candidate_bundles: Sequence[Any] = (),
    max_views: int = 3,
) -> dict[str, object]:
    requests = suggest_next_views(
        bundle,
        existing_views=existing_views,
        candidate_bundles=candidate_bundles,
        max_views=max_views,
    )
    disagreement = ensemble_disagreement_signal(
        candidate_bundles or _candidate_bundles_from_payload(bundle)
    )
    return {
        "schema_version": "active-view-plan-v1",
        "candidate_id": str(getattr(bundle, "candidate_id", "")) if not isinstance(bundle, Mapping) else str(bundle.get("candidate_id", "")),
        "ensemble_disagreement": disagreement.to_dict(),
        "requests": [request.to_dict() for request in requests],
    }


def ensemble_disagreement_signal(
    candidate_bundles: Sequence[Any],
) -> ViewDisagreementSignal:
    per_candidate = [_per_view_iou(item) for item in candidate_bundles]
    per_candidate = [item for item in per_candidate if item]
    by_view: dict[str, list[float]] = {}
    for metrics in per_candidate:
        for view, value in metrics.items():
            by_view.setdefault(view, []).append(value)

    view_stddev: dict[str, float] = {}
    view_mean: dict[str, float] = {}
    view_scores: dict[str, float] = {}
    for view, values in by_view.items():
        if not values:
            continue
        mean_value = _clamp01(sum(values) / len(values))
        stddev = statistics.pstdev(values) if len(values) > 1 else 0.0
        disagreement = _clamp01(stddev + max(0.0, 0.75 - mean_value) * 0.35)
        view_mean[view] = mean_value
        view_stddev[view] = stddev
        view_scores[view] = disagreement

    recommended = _recommended_scores_from_views(view_scores, view_mean)
    max_disagreement = max(view_scores.values(), default=0.0)
    entropy = _normalized_entropy(tuple(recommended.values()))
    return ViewDisagreementSignal(
        candidate_count=len(per_candidate),
        view_scores=view_scores,
        view_stddev=view_stddev,
        view_mean_iou=view_mean,
        recommended_view_scores=recommended,
        max_disagreement=max_disagreement,
        entropy=entropy,
        source_views=tuple(sorted(view_scores)),
    )


def _candidate_bundles_from_payload(bundle: Any) -> tuple[Any, ...]:
    if not isinstance(bundle, Mapping):
        return ()
    for key in (
        "candidate_bundles",
        "evaluation_bundles",
        "bundles",
        "candidates",
        "ranked_candidates",
    ):
        value = bundle.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return tuple(item for item in value if item)
    backend = bundle.get("backend_result")
    if isinstance(backend, Mapping):
        return _candidate_bundles_from_payload(backend)
    return ()


def _per_view_iou(bundle: Any) -> Mapping[str, float]:
    metrics = _metric_index(bundle)
    values: dict[str, float] = {}
    for key, value in metrics.items():
        parts = key.split(".")
        if len(parts) >= 4 and parts[:2] == ["silhouette", "per_view"]:
            view = parts[2]
            metric = parts[3]
            if metric in {"area_iou", "iou"}:
                values[view] = _clamp01(value)
        elif key.endswith("_iou") and key[:-4] in {"front", "side", "top"}:
            values[key[:-4]] = _clamp01(value)
    if values:
        return values

    source = bundle if isinstance(bundle, Mapping) else {}
    backend = source.get("backend_result") if isinstance(source, Mapping) else None
    if isinstance(backend, Mapping):
        nested = _per_view_iou(backend)
        if nested:
            return nested
    metric_result = source.get("metric_result") if isinstance(source, Mapping) else None
    if isinstance(metric_result, Mapping):
        per_view = metric_result.get("per_view")
        if isinstance(per_view, Mapping):
            for view, payload in per_view.items():
                if isinstance(payload, Mapping):
                    value = payload.get("area_iou", payload.get("iou"))
                    if value is not None:
                        values[str(view)] = _clamp01(_float(value))
    return values


def _recommended_scores_from_views(
    view_scores: Mapping[str, float],
    view_mean: Mapping[str, float],
) -> Mapping[str, float]:
    front = float(view_scores.get("front", 0.0))
    side = float(view_scores.get("side", 0.0))
    top = float(view_scores.get("top", 0.0))
    front_mean = float(view_mean.get("front", 1.0))
    side_mean = float(view_mean.get("side", 1.0))
    top_mean = float(view_mean.get("top", 1.0))
    diagonal = _clamp01(max(front, side) + abs(front_mean - side_mean) * 0.35)
    top_oblique = _clamp01(max(top, (1.0 - top_mean) * 0.45))
    rear = _clamp01(max(0.0, (front + side + top) / 3.0 - 0.05))
    return {
        "front_side_45": diagonal,
        "front_side_135": _clamp01(diagonal * 0.9 + rear * 0.1),
        "top_oblique_45": top_oblique,
        "rear": rear,
    }


def _normalized_entropy(values: Sequence[float]) -> float:
    positive = [max(0.0, float(value)) for value in values if value > 0.0]
    if len(positive) <= 1:
        return 0.0
    total = sum(positive)
    if total <= 0.0:
        return 0.0
    probs = [value / total for value in positive]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    return _clamp01(entropy / math.log(len(probs)))


def _capture_notes(request: ViewRequest) -> tuple[str, ...]:
    notes = [
        "Use the same silhouette extraction polarity and background as existing reference views.",
        "Keep the object scale and framing consistent with the current orthographic views.",
    ]
    if request.elevation_deg:
        notes.append("Preserve visible footprint; avoid perspective distortion if possible.")
    return tuple(notes)


def _metric_index(bundle: Any) -> Mapping[str, float]:
    if hasattr(bundle, "metric_index"):
        return {
            key: _float(getattr(metric, "value", None))
            for key, metric in bundle.metric_index().items()
        }
    if isinstance(bundle, Mapping):
        metrics: dict[str, float] = {}
        direct = bundle.get("metrics")
        if isinstance(direct, Mapping):
            for key, value in direct.items():
                if isinstance(value, Mapping):
                    continue
                metrics[str(key)] = _float(value)
        for group in bundle.get("metric_groups", ()) or ():
            if not isinstance(group, Mapping):
                continue
            for metric in group.get("metrics", ()) or ():
                if isinstance(metric, Mapping):
                    metrics[str(metric.get("name", ""))] = _float(metric.get("value"))
        return metrics
    return {}


def _failure_codes(bundle: Any) -> tuple[str, ...]:
    failures = getattr(bundle, "failures", None)
    if failures is None and isinstance(bundle, Mapping):
        failures = bundle.get("failures", ())
    codes = []
    for failure in failures or ():
        if hasattr(failure, "code"):
            codes.append(str(failure.code))
        elif isinstance(failure, Mapping):
            codes.append(str(failure.get("code", "")))
    return tuple(code for code in codes if code)


def _status(bundle: Any) -> str:
    if hasattr(bundle, "status"):
        return str(bundle.status)
    if isinstance(bundle, Mapping):
        return str(bundle.get("status", ""))
    return ""


def _metric(
    metrics: Mapping[str, float],
    name: str,
    *,
    default: float = 0.0,
) -> float:
    return _float(metrics.get(name, default), default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
