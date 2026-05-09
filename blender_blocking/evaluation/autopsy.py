"""Autopsy-pack and suggested-action contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .schemas import EvaluationBundle, json_safe


@dataclass(frozen=True)
class SuggestedAction:
    action_id: str
    title: str
    reason: str
    expected_improvement: Mapping[str, float | str] = field(default_factory=dict)
    cost_estimate: str = "unknown"
    risk: str = "medium"
    command: tuple[str, ...] | None = None
    user_input_needed: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "title": self.title,
            "reason": self.reason,
            "expected_improvement": json_safe(self.expected_improvement),
            "cost_estimate": self.cost_estimate,
            "risk": self.risk,
            "command": list(self.command) if self.command else None,
            "user_input_needed": list(self.user_input_needed),
        }


ACTION_CATALOG = {
    "boundary_first_refinement": SuggestedAction(
        "boundary_first_refinement",
        "Run boundary-first refinement",
        "Boundary IoU or signed-distance evidence suggests contour mismatch.",
        {"silhouette.min_boundary_iou": "increase"},
        "medium",
        "medium",
    ),
    "safe_topology_repair": SuggestedAction(
        "safe_topology_repair",
        "Run safe topology repair",
        "Topology evidence indicates non-manifold or fragmented mesh output.",
        {"topology.non_manifold_edges": "decrease"},
        "medium",
        "medium",
    ),
    "add_active_view": SuggestedAction(
        "add_active_view",
        "Request another view",
        "Input ambiguity appears higher than backend-specific reconstruction error.",
        {"geometry.ambiguity_gap": "decrease"},
        "user_input",
        "low",
        user_input_needed=("additional silhouette view",),
    ),
}


@dataclass(frozen=True)
class AutopsyPack:
    candidate_id: str
    status: str
    failures: tuple[str, ...]
    suggested_actions: tuple[SuggestedAction, ...]
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    reproduce_command: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "failures": list(self.failures),
            "suggested_actions": [action.to_dict() for action in self.suggested_actions],
            "artifact_paths": dict(self.artifact_paths),
            "reproduce_command": list(self.reproduce_command),
        }


def autopsy_pack_from_bundle(bundle: EvaluationBundle) -> AutopsyPack:
    action_ids: list[str] = []
    for failure in bundle.failures:
        if "boundary" in failure.code:
            action_ids.append("boundary_first_refinement")
        if "topology" in failure.code:
            action_ids.append("safe_topology_repair")
        if "ambiguous" in failure.code or "calibration" in failure.code:
            action_ids.append("add_active_view")
    actions = tuple(ACTION_CATALOG[action_id] for action_id in dict.fromkeys(action_ids) if action_id in ACTION_CATALOG)
    return AutopsyPack(
        candidate_id=bundle.candidate_id,
        status=bundle.status,
        failures=tuple(failure.code for failure in bundle.failures),
        suggested_actions=actions,
        artifact_paths=bundle.artifacts,
    )

