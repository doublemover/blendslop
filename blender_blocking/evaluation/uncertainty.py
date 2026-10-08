"""Uncertainty metrics for silhouette-derived reconstruction targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ViewUncertaintyReport:
    view: str
    confidence_mean: float
    confidence_min: float
    boundary_uncertainty_mean: float
    foreground_entropy_mean: float
    profile_width_std_mean: float = 0.0
    profile_center_std_mean: float = 0.0

    @property
    def confidence_score(self) -> float:
        return _clamp01(self.confidence_mean)

    @property
    def stability_score(self) -> float:
        profile_instability = min(
            1.0,
            (self.profile_width_std_mean + self.profile_center_std_mean) / 16.0,
        )
        boundary_instability = min(1.0, self.boundary_uncertainty_mean)
        entropy_instability = min(1.0, self.foreground_entropy_mean)
        return _clamp01(1.0 - (0.45 * profile_instability + 0.35 * boundary_instability + 0.2 * entropy_instability))

    def to_dict(self) -> dict[str, object]:
        return {
            "view": self.view,
            "confidence_mean": self.confidence_mean,
            "confidence_min": self.confidence_min,
            "boundary_uncertainty_mean": self.boundary_uncertainty_mean,
            "foreground_entropy_mean": self.foreground_entropy_mean,
            "profile_width_std_mean": self.profile_width_std_mean,
            "profile_center_std_mean": self.profile_center_std_mean,
            "confidence_score": self.confidence_score,
            "stability_score": self.stability_score,
        }


@dataclass(frozen=True)
class UncertaintyReport:
    view_reports: tuple[ViewUncertaintyReport, ...] = ()
    profile_distribution: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @property
    def view_count(self) -> int:
        return len(self.view_reports)

    @property
    def confidence_mean(self) -> float:
        return _mean(report.confidence_mean for report in self.view_reports)

    @property
    def boundary_uncertainty_mean(self) -> float:
        return _mean(report.boundary_uncertainty_mean for report in self.view_reports)

    @property
    def foreground_entropy_mean(self) -> float:
        return _mean(report.foreground_entropy_mean for report in self.view_reports)

    @property
    def profile_width_std_mean(self) -> float:
        return _mean(report.profile_width_std_mean for report in self.view_reports)

    @property
    def profile_center_std_mean(self) -> float:
        return _mean(report.profile_center_std_mean for report in self.view_reports)

    @property
    def confidence_score(self) -> float:
        return _clamp01(self.confidence_mean)

    @property
    def stability_score(self) -> float:
        return _mean(report.stability_score for report in self.view_reports)

    @property
    def consistency_score(self) -> float:
        if not self.view_reports:
            return 0.0
        confidence = self.confidence_score
        stability = self.stability_score
        view_balance = 1.0 - min(
            1.0,
            float(np.std([report.confidence_score for report in self.view_reports])),
        )
        return _clamp01(0.45 * confidence + 0.4 * stability + 0.15 * view_balance)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "uncertainty_report_v1",
            "view_count": self.view_count,
            "confidence_mean": self.confidence_mean,
            "boundary_uncertainty_mean": self.boundary_uncertainty_mean,
            "foreground_entropy_mean": self.foreground_entropy_mean,
            "profile_width_std_mean": self.profile_width_std_mean,
            "profile_center_std_mean": self.profile_center_std_mean,
            "confidence_score": self.confidence_score,
            "stability_score": self.stability_score,
            "consistency_score": self.consistency_score,
            "views": [report.to_dict() for report in self.view_reports],
            "profile_distribution": {
                key: dict(value) for key, value in self.profile_distribution.items()
            },
        }


def uncertainty_report_from_target(target: Any) -> UncertaintyReport:
    """Build an uncertainty report from a ReconstructionTarget-like object."""

    extras = getattr(target, "extras", {}) or {}
    profile_distribution = extras.get("profile_band_distribution", {})
    if not isinstance(profile_distribution, Mapping):
        profile_distribution = {}
    view_reports = []
    for constraint in getattr(target, "constraints", ()) or ():
        view = str(getattr(constraint, "view", ""))
        uncertainty = getattr(constraint, "uncertainty", None)
        profile = profile_distribution.get(view, {})
        profile_payload = profile if isinstance(profile, Mapping) else {}
        view_reports.append(
            ViewUncertaintyReport(
                view=view,
                confidence_mean=_array_mean(getattr(uncertainty, "confidence", None), default=1.0),
                confidence_min=_array_min(getattr(uncertainty, "confidence", None), default=1.0),
                boundary_uncertainty_mean=_array_mean(
                    getattr(uncertainty, "boundary_uncertainty", None),
                    default=0.0,
                ),
                foreground_entropy_mean=_entropy_mean(
                    getattr(uncertainty, "foreground_prob", None)
                ),
                profile_width_std_mean=_float(
                    profile_payload.get("width_std_mean"),
                    0.0,
                ),
                profile_center_std_mean=_float(
                    profile_payload.get("center_std_mean"),
                    0.0,
                ),
            )
        )
    return UncertaintyReport(
        view_reports=tuple(view_reports),
        profile_distribution={
            str(key): {str(k): _float(v, 0.0) for k, v in value.items()}
            for key, value in profile_distribution.items()
            if isinstance(value, Mapping)
        },
    )


def uncertainty_metric_payload_from_target(target: Any) -> dict[str, object]:
    """Return JSON-ready target uncertainty metrics."""

    return uncertainty_report_from_target(target).to_dict()


def _array_mean(value: Any, *, default: float) -> float:
    if value is None:
        return float(default)
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return float(default)
    return _float(np.mean(array), default)


def _array_min(value: Any, *, default: float) -> float:
    if value is None:
        return float(default)
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return float(default)
    return _float(np.min(array), default)


def _entropy_mean(probability: Any) -> float:
    if probability is None:
        return 0.0
    prob = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    if prob.size == 0:
        return 0.0
    eps = 1e-9
    entropy = -(prob * np.log2(prob + eps) + (1.0 - prob) * np.log2(1.0 - prob + eps))
    return _clamp01(float(np.mean(entropy)))


def _mean(values: Sequence[float] | Any) -> float:
    finite = [_float(value, float("nan")) for value in values]
    finite = [value for value in finite if np.isfinite(value)]
    return float(np.mean(finite)) if finite else 0.0


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
