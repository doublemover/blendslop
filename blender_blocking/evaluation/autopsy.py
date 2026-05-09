"""Autopsy-pack and suggested-action contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .schemas import EvaluationBundle, json_safe
from .view_planning import active_view_plan_payload
from .calibration import calibration_refinement_plan_payload
from .boundary_refinement import boundary_refinement_plan_payload

try:
    from metrics.topology import topology_repair_plan
except Exception:  # pragma: no cover - package import fallback
    from blender_blocking.metrics.topology import topology_repair_plan


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
    "run_calibration_sweep": SuggestedAction(
        "run_calibration_sweep",
        "Run calibration sweep",
        "Projection diagnostics suggest view role, bounds, scale, or framing mismatch.",
        {"diagnostics.visual_hull.axis_or_transform_suspect": "decrease"},
        "medium",
        "low",
        command=(
            "python",
            "blender_blocking/refinement_lab/cli.py",
            "plan",
            "--track",
            "visual-hull-transform",
            "--bounds-debug",
        ),
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
    active_view_plan: Mapping[str, object] = field(default_factory=dict)
    topology_repair_plan: Mapping[str, object] = field(default_factory=dict)
    calibration_plan: Mapping[str, object] = field(default_factory=dict)
    boundary_refinement_plan: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "failures": list(self.failures),
            "suggested_actions": [action.to_dict() for action in self.suggested_actions],
            "artifact_paths": dict(self.artifact_paths),
            "reproduce_command": list(self.reproduce_command),
            "active_view_plan": json_safe(self.active_view_plan),
            "topology_repair_plan": json_safe(self.topology_repair_plan),
            "calibration_plan": json_safe(self.calibration_plan),
            "boundary_refinement_plan": json_safe(self.boundary_refinement_plan),
        }


def autopsy_pack_from_bundle(bundle: EvaluationBundle) -> AutopsyPack:
    action_ids: list[str] = []
    for failure in bundle.failures:
        if "boundary" in failure.code:
            action_ids.append("boundary_first_refinement")
        if "topology" in failure.code:
            action_ids.append("safe_topology_repair")
        if (
            "axis_or_transform" in failure.code
            or "framing" in failure.code
            or "calibration" in failure.code
        ):
            action_ids.append("run_calibration_sweep")
        if (
            "ambiguous" in failure.code
            or "calibration" in failure.code
            or "recoverable_gap" in failure.code
            or failure.code in {"geometry_surface_fscore_low", "geometry_volume_iou_low"}
        ):
            action_ids.append("add_active_view")
    actions = tuple(ACTION_CATALOG[action_id] for action_id in dict.fromkeys(action_ids) if action_id in ACTION_CATALOG)
    active_view_plan = (
        active_view_plan_payload(
            bundle,
            existing_views=_existing_silhouette_views(bundle),
        )
        if "add_active_view" in action_ids
        else {}
    )
    repair_plan = (
        topology_repair_plan(_topology_metric_payload(bundle))
        if "safe_topology_repair" in action_ids
        else {}
    )
    calibration_plan = (
        calibration_refinement_plan_payload(bundle)
        if "run_calibration_sweep" in action_ids
        else {}
    )
    boundary_plan = (
        boundary_refinement_plan_payload(bundle)
        if "boundary_first_refinement" in action_ids
        else {}
    )
    return AutopsyPack(
        candidate_id=bundle.candidate_id,
        status=bundle.status,
        failures=tuple(failure.code for failure in bundle.failures),
        suggested_actions=actions,
        artifact_paths=bundle.artifacts,
        active_view_plan=active_view_plan,
        topology_repair_plan=repair_plan,
        calibration_plan=calibration_plan,
        boundary_refinement_plan=boundary_plan,
    )


def _existing_silhouette_views(bundle: EvaluationBundle) -> tuple[str, ...]:
    views: list[str] = []
    for name in bundle.metric_index():
        if not name.startswith("silhouette.per_view."):
            continue
        parts = name.split(".")
        if len(parts) >= 4 and parts[2]:
            views.append(parts[2])
    return tuple(dict.fromkeys(views))


def _topology_metric_payload(bundle: EvaluationBundle) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, metric in bundle.metric_index().items():
        if not name.startswith("topology."):
            continue
        payload[name.removeprefix("topology.")] = metric.value
    return payload
