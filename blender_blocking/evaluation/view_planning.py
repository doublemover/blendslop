"""Active view planning for silhouette-only reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    max_views: int = 3,
) -> tuple[ViewRequest, ...]:
    metrics = _metric_index(bundle)
    failures = _failure_codes(bundle)
    existing = {str(view).lower() for view in existing_views}
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
                metadata={"base_priority": base.priority, "bundle_status": status},
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
    max_views: int = 3,
) -> dict[str, object]:
    requests = suggest_next_views(
        bundle,
        existing_views=existing_views,
        max_views=max_views,
    )
    return {
        "schema_version": "active-view-plan-v1",
        "candidate_id": str(getattr(bundle, "candidate_id", "")) if not isinstance(bundle, Mapping) else str(bundle.get("candidate_id", "")),
        "requests": [request.to_dict() for request in requests],
    }


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
