"""Experiment manifests, factors, and decision records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .schemas import json_safe


@dataclass(frozen=True)
class ExperimentFactor:
    name: str
    values: tuple[str, ...]
    default_value: str
    one_factor_at_a_time: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "values": list(self.values),
            "default_value": self.default_value,
            "one_factor_at_a_time": self.one_factor_at_a_time,
        }


@dataclass(frozen=True)
class ExperimentManifest:
    experiment_id: str
    title: str
    owner: str
    created_at_utc: str
    hypothesis: str
    expected_win: str
    suites: tuple[str, ...]
    modes: tuple[str, ...]
    factors: tuple[ExperimentFactor, ...]
    metrics: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    artifact_root: str
    status: str = "draft"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "owner": self.owner,
            "created_at_utc": self.created_at_utc,
            "hypothesis": self.hypothesis,
            "expected_win": self.expected_win,
            "suites": list(self.suites),
            "modes": list(self.modes),
            "factors": [factor.to_dict() for factor in self.factors],
            "metrics": list(self.metrics),
            "stop_conditions": list(self.stop_conditions),
            "artifact_root": self.artifact_root,
            "status": self.status,
            "metadata": json_safe(self.metadata),
        }


DEFAULT_FACTORS = (
    ExperimentFactor("silhouette_metric_bundle", ("area_only", "area_boundary", "area_boundary_sdf", "full_soft"), "area_boundary_sdf"),
    ExperimentFactor("geometry_metric_bundle", ("none", "surface_only", "volume_surface", "recoverability_dual"), "volume_surface"),
    ExperimentFactor("mesh_method", ("points", "marching_cubes", "lewiner", "dual_contouring", "flexicubes_research"), "lewiner"),
    ExperimentFactor("postprocess", ("none", "smooth_guarded", "poisson", "screened_poisson", "topology_repair"), "none"),
    ExperimentFactor("selection_objective", ("balanced", "fidelity", "editable", "printable", "fast_preview"), "balanced"),
    ExperimentFactor("uncertainty_model", ("none", "threshold_sweep", "profile_band", "bayesian_profile"), "profile_band"),
    ExperimentFactor("calibration", ("none", "bounds_only", "per_view_offset", "full_safe"), "bounds_only"),
)


@dataclass(frozen=True)
class ExperimentDecision:
    experiment_id: str
    decision: str
    summary: str
    accepted_factors: Mapping[str, str] = field(default_factory=dict)
    rejected_factors: Mapping[str, str] = field(default_factory=dict)
    followups: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "decision": self.decision,
            "summary": self.summary,
            "accepted_factors": dict(self.accepted_factors),
            "rejected_factors": dict(self.rejected_factors),
            "followups": list(self.followups),
        }
