"""Retopology and editable-mesh policy decisions.

The policy layer is intentionally Blender-free.  It turns mesh topology,
silhouette fit, and editability diagnostics into a deterministic decision that
downstream backends can emit in manifests or use to choose a repair/remesh path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from .topology import TopologyReport, mesh_topology_report, topology_repair_plan


@dataclass(frozen=True)
class RetopologyPolicy:
    """Thresholds for deciding whether a mesh is suitable for editing."""

    min_topology_score: float = 0.85
    max_boundary_edges: int = 0
    max_non_manifold_edges: int = 0
    max_degenerate_faces: int = 0
    max_loose_vertices: int = 0
    max_connected_components: int = 1
    min_editability_score: float = 0.45
    max_editable_faces: int = 50_000
    max_faces_for_manual_cleanup: int = 150_000
    min_faces_for_surface: int = 4
    min_required_iou: Optional[float] = None
    max_signed_distance_loss: Optional[float] = None
    prefer_primitives_below_editability: float = 0.35
    require_watertight: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "min_topology_score": self.min_topology_score,
            "max_boundary_edges": self.max_boundary_edges,
            "max_non_manifold_edges": self.max_non_manifold_edges,
            "max_degenerate_faces": self.max_degenerate_faces,
            "max_loose_vertices": self.max_loose_vertices,
            "max_connected_components": self.max_connected_components,
            "min_editability_score": self.min_editability_score,
            "max_editable_faces": self.max_editable_faces,
            "max_faces_for_manual_cleanup": self.max_faces_for_manual_cleanup,
            "min_faces_for_surface": self.min_faces_for_surface,
            "min_required_iou": self.min_required_iou,
            "max_signed_distance_loss": self.max_signed_distance_loss,
            "prefer_primitives_below_editability": (
                self.prefer_primitives_below_editability
            ),
            "require_watertight": self.require_watertight,
        }


@dataclass(frozen=True)
class RetopologyDecision:
    action: str
    severity: str
    reason: str
    blockers: tuple[str, ...]
    recommended_postprocess: str
    recommended_backend: str
    policy: RetopologyPolicy
    metrics: Mapping[str, Any]
    repair_plan: Mapping[str, Any]
    config_overrides: Mapping[str, Any]

    @property
    def accepted_for_editing(self) -> bool:
        return self.action in {"preserve", "safe_repair_optional"}

    @property
    def requires_human_or_research_step(self) -> bool:
        return self.action in {
            "retopology_required",
            "primitive_proxy_required",
            "reject_or_recapture",
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "severity": self.severity,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "recommended_postprocess": self.recommended_postprocess,
            "recommended_backend": self.recommended_backend,
            "accepted_for_editing": self.accepted_for_editing,
            "requires_human_or_research_step": self.requires_human_or_research_step,
            "policy": self.policy.to_dict(),
            "metrics": dict(self.metrics),
            "repair_plan": dict(self.repair_plan),
            "config_overrides": dict(self.config_overrides),
        }


def retopology_policy_from_config(
    config: Mapping[str, Any],
    *,
    prefix: str = "retopology",
) -> RetopologyPolicy:
    return RetopologyPolicy(
        min_topology_score=_float(config.get(f"{prefix}_min_topology_score"), 0.85),
        max_boundary_edges=_int(config.get(f"{prefix}_max_boundary_edges"), 0),
        max_non_manifold_edges=_int(config.get(f"{prefix}_max_non_manifold_edges"), 0),
        max_degenerate_faces=_int(config.get(f"{prefix}_max_degenerate_faces"), 0),
        max_loose_vertices=_int(config.get(f"{prefix}_max_loose_vertices"), 0),
        max_connected_components=_int(config.get(f"{prefix}_max_connected_components"), 1),
        min_editability_score=_float(config.get(f"{prefix}_min_editability_score"), 0.45),
        max_editable_faces=_int(config.get(f"{prefix}_max_editable_faces"), 50_000),
        max_faces_for_manual_cleanup=_int(
            config.get(f"{prefix}_max_faces_for_manual_cleanup"),
            150_000,
        ),
        min_faces_for_surface=_int(config.get(f"{prefix}_min_faces_for_surface"), 4),
        min_required_iou=_optional_float(config.get(f"{prefix}_min_required_iou")),
        max_signed_distance_loss=_optional_float(
            config.get(f"{prefix}_max_signed_distance_loss")
        ),
        prefer_primitives_below_editability=_float(
            config.get(f"{prefix}_prefer_primitives_below_editability"),
            0.35,
        ),
        require_watertight=_bool(config.get(f"{prefix}_require_watertight"), True),
    )


def evaluate_retopology_policy(
    topology: TopologyReport | Mapping[str, Any],
    *,
    editability_score: float = 0.0,
    min_required_iou: Optional[float] = None,
    signed_distance_loss: Optional[float] = None,
    policy: Optional[RetopologyPolicy] = None,
) -> RetopologyDecision:
    policy = policy or RetopologyPolicy()
    data = topology.to_dict() if hasattr(topology, "to_dict") else dict(topology)
    repair_plan = topology_repair_plan(data)
    topology_score = _float(data.get("topology_score"), 0.0)
    face_count = _int(data.get("face_count"), 0)
    watertight = _bool(data.get("watertight"), False)
    rejection_blockers: list[str] = []
    primitive_blockers: list[str] = []
    retopology_blockers: list[str] = []
    safe_repair_blockers: list[str] = []

    if face_count < int(policy.min_faces_for_surface):
        rejection_blockers.append(
            "mesh has too few faces to represent an editable surface"
        )
    if topology_score < float(policy.min_topology_score):
        safe_repair_blockers.append(
            f"topology_score {topology_score:.3f} below {policy.min_topology_score:.3f}"
        )
    _max_count_blocker(
        data,
        "degenerate_faces",
        policy.max_degenerate_faces,
        safe_repair_blockers,
    )
    _max_count_blocker(
        data,
        "loose_vertices",
        policy.max_loose_vertices,
        safe_repair_blockers,
    )
    _max_count_blocker(
        data,
        "connected_components",
        policy.max_connected_components,
        safe_repair_blockers,
    )
    _max_count_blocker(
        data,
        "non_manifold_edges",
        policy.max_non_manifold_edges,
        retopology_blockers,
    )
    _max_count_blocker(
        data,
        "boundary_edges",
        policy.max_boundary_edges,
        retopology_blockers,
    )
    if (
        bool(policy.require_watertight)
        and not watertight
        and _int(data.get("boundary_edges"), 0) <= int(policy.max_boundary_edges)
        and _int(data.get("non_manifold_edges"), 0) <= int(policy.max_non_manifold_edges)
        and _int(data.get("degenerate_faces"), 0) <= int(policy.max_degenerate_faces)
        and _int(data.get("loose_vertices"), 0) <= int(policy.max_loose_vertices)
    ):
        retopology_blockers.append("mesh is not watertight")
    if float(editability_score) < float(policy.prefer_primitives_below_editability):
        primitive_blockers.append(
            f"editability_score {float(editability_score):.3f} below primitive proxy "
            f"threshold {policy.prefer_primitives_below_editability:.3f}"
        )
    elif float(editability_score) < float(policy.min_editability_score):
        retopology_blockers.append(
            f"editability_score {float(editability_score):.3f} below "
            f"{policy.min_editability_score:.3f}"
        )
    if face_count > int(policy.max_faces_for_manual_cleanup):
        retopology_blockers.append(
            f"face_count {face_count} exceeds manual cleanup limit "
            f"{policy.max_faces_for_manual_cleanup}"
        )
    elif face_count > int(policy.max_editable_faces):
        safe_repair_blockers.append(
            f"face_count {face_count} exceeds editable target {policy.max_editable_faces}"
        )
    effective_min_iou = (
        min_required_iou
        if min_required_iou is not None
        else policy.min_required_iou
    )
    if effective_min_iou is not None and float(effective_min_iou) < 0.35:
        rejection_blockers.append(
            f"required silhouette IoU {float(effective_min_iou):.3f} is too low for retopology"
        )
    if (
        policy.max_signed_distance_loss is not None
        and signed_distance_loss is not None
        and float(signed_distance_loss) > float(policy.max_signed_distance_loss)
    ):
        rejection_blockers.append(
            f"signed_distance_loss {float(signed_distance_loss):.3f} exceeds "
            f"{float(policy.max_signed_distance_loss):.3f}"
        )

    blockers: list[str] = []
    blockers.extend(rejection_blockers)
    blockers.extend(primitive_blockers)
    blockers.extend(retopology_blockers)
    blockers.extend(safe_repair_blockers)
    action, postprocess, backend, severity = _classify_action(
        rejection_blockers=rejection_blockers,
        primitive_blockers=primitive_blockers,
        retopology_blockers=retopology_blockers,
        safe_repair_blockers=safe_repair_blockers,
        face_count=face_count,
        policy=policy,
        repair_plan=repair_plan,
    )
    metrics = {
        "topology_score": topology_score,
        "editability_score": float(editability_score),
        "face_count": face_count,
        "boundary_edges": _int(data.get("boundary_edges"), 0),
        "non_manifold_edges": _int(data.get("non_manifold_edges"), 0),
        "degenerate_faces": _int(data.get("degenerate_faces"), 0),
        "loose_vertices": _int(data.get("loose_vertices"), 0),
        "connected_components": _int(data.get("connected_components"), 0),
        "watertight": watertight,
        "min_required_iou": (
            None if effective_min_iou is None else float(effective_min_iou)
        ),
        "signed_distance_loss": (
            None if signed_distance_loss is None else float(signed_distance_loss)
        ),
        "max_signed_distance_loss": (
            None
            if policy.max_signed_distance_loss is None
            else float(policy.max_signed_distance_loss)
        ),
    }
    return RetopologyDecision(
        action=action,
        severity=severity,
        reason="accepted" if not blockers else "; ".join(blockers),
        blockers=tuple(blockers),
        recommended_postprocess=postprocess,
        recommended_backend=backend,
        policy=policy,
        metrics=metrics,
        repair_plan=repair_plan,
        config_overrides=_config_overrides(action, postprocess, policy),
    )


def retopology_decision_from_metrics(
    topology: TopologyReport | Mapping[str, Any],
    *,
    editability_score: float = 0.0,
    min_required_iou: Optional[float] = None,
    signed_distance_loss: Optional[float] = None,
    config: Optional[Mapping[str, Any]] = None,
    policy: Optional[RetopologyPolicy] = None,
) -> RetopologyDecision:
    """Evaluate policy from already-computed topology and score metrics."""
    resolved_policy = policy or retopology_policy_from_config(config or {})
    return evaluate_retopology_policy(
        topology,
        editability_score=editability_score,
        min_required_iou=min_required_iou,
        signed_distance_loss=signed_distance_loss,
        policy=resolved_policy,
    )


def retopology_decision_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    editability_score: float = 0.0,
    min_required_iou: Optional[float] = None,
    signed_distance_loss: Optional[float] = None,
    config: Optional[Mapping[str, Any]] = None,
    policy: Optional[RetopologyPolicy] = None,
) -> RetopologyDecision:
    """Compute topology and evaluate whether a mesh is editable enough."""
    topology = mesh_topology_report(vertices, faces)
    return retopology_decision_from_metrics(
        topology,
        editability_score=editability_score,
        min_required_iou=min_required_iou,
        signed_distance_loss=signed_distance_loss,
        config=config,
        policy=policy,
    )


def _classify_action(
    *,
    rejection_blockers: list[str],
    primitive_blockers: list[str],
    retopology_blockers: list[str],
    safe_repair_blockers: list[str],
    face_count: int,
    policy: RetopologyPolicy,
    repair_plan: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    if rejection_blockers:
        return "reject_or_recapture", "none", "calibration_or_view_recapture", "high"
    if primitive_blockers:
        return "primitive_proxy_required", "none", "shape_program_or_primitive_fit", "high"
    if retopology_blockers:
        if face_count > int(policy.max_faces_for_manual_cleanup):
            return "retopology_required", "external_retopology", "retopology", "high"
        if repair_plan.get("safe_automatic"):
            return "remesh_required", "topology_repair", "visual_hull_postprocess", "high"
        return "retopology_required", "external_retopology", "retopology", "high"
    if safe_repair_blockers:
        return "safe_repair_recommended", "topology_repair", "visual_hull_postprocess", "medium"
    return "preserve", "none", "current_mesh", "low"


def _config_overrides(
    action: str,
    postprocess: str,
    policy: RetopologyPolicy,
) -> dict[str, object]:
    if action == "preserve":
        return {}
    payload: dict[str, object] = {
        "postprocess": postprocess,
        "postprocess_required": action == "remesh_required",
        "repair_keep_largest_component": True,
        "guard_require_improvement": True,
        "guard_min_topology_delta": 0.0,
    }
    if action == "retopology_required":
        payload.update(
            {
                "retopology_required": True,
                "target_max_faces": policy.max_editable_faces,
                "target_style": "editable_quad_or_primitive_proxy",
            }
        )
    if action == "primitive_proxy_required":
        payload.update(
            {
                "recommended_mode": "shape_program,primitive_fit_refine",
                "target_style": "editable_parametric_proxy",
            }
        )
    if action == "reject_or_recapture":
        payload.update(
            {
                "recommended_mode": "calibration_diagnostics,view_planning",
                "target_style": "better_reference_or_calibrated_masks",
            }
        )
    return payload


def _max_count_blocker(
    data: Mapping[str, Any],
    key: str,
    limit: int,
    blockers: list[str],
) -> None:
    value = _int(data.get(key), 0)
    if value > int(limit):
        blockers.append(f"{key} {value} exceeds {int(limit)}")


def _float(value: Any, default: float) -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any, default: int) -> int:
    try:
        return int(default if value is None else value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)
