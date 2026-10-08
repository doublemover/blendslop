"""Guarded acceptance policy for topology-changing mesh postprocesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .topology import TopologyReport


@dataclass(frozen=True)
class MeshChangeGuardPolicy:
    min_topology_delta: float = 0.0
    max_boundary_edge_increase: int = 0
    max_non_manifold_edge_increase: int = 0
    max_component_increase: int = 0
    max_vertex_count_ratio: float = 4.0
    max_face_count_ratio: float = 4.0
    require_improvement: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "min_topology_delta": self.min_topology_delta,
            "max_boundary_edge_increase": self.max_boundary_edge_increase,
            "max_non_manifold_edge_increase": self.max_non_manifold_edge_increase,
            "max_component_increase": self.max_component_increase,
            "max_vertex_count_ratio": self.max_vertex_count_ratio,
            "max_face_count_ratio": self.max_face_count_ratio,
            "require_improvement": self.require_improvement,
        }


@dataclass(frozen=True)
class MeshChangeGuardDecision:
    accepted: bool
    reason: str
    topology_delta: float
    blockers: tuple[str, ...]
    policy: MeshChangeGuardPolicy

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "topology_delta": self.topology_delta,
            "blockers": list(self.blockers),
            "policy": self.policy.to_dict(),
        }


def guard_policy_from_config(
    config: Mapping[str, object],
    *,
    prefix: str = "guard",
) -> MeshChangeGuardPolicy:
    return MeshChangeGuardPolicy(
        min_topology_delta=_float(config.get(f"{prefix}_min_topology_delta"), 0.0),
        max_boundary_edge_increase=_int(
            config.get(f"{prefix}_max_boundary_edge_increase"),
            0,
        ),
        max_non_manifold_edge_increase=_int(
            config.get(f"{prefix}_max_non_manifold_edge_increase"),
            0,
        ),
        max_component_increase=_int(config.get(f"{prefix}_max_component_increase"), 0),
        max_vertex_count_ratio=_float(
            config.get(f"{prefix}_max_vertex_count_ratio"),
            4.0,
        ),
        max_face_count_ratio=_float(config.get(f"{prefix}_max_face_count_ratio"), 4.0),
        require_improvement=bool(config.get(f"{prefix}_require_improvement", False)),
    )


def evaluate_mesh_change(
    *,
    before: TopologyReport,
    after: TopologyReport,
    before_vertex_count: int,
    after_vertex_count: int,
    before_face_count: int,
    after_face_count: int,
    policy: MeshChangeGuardPolicy,
) -> MeshChangeGuardDecision:
    blockers: list[str] = []
    topology_delta = float(after.topology_score - before.topology_score)
    if topology_delta < float(policy.min_topology_delta):
        blockers.append(
            f"topology_delta {topology_delta:.6f} < {policy.min_topology_delta:.6f}"
        )
    if policy.require_improvement and topology_delta <= 0.0:
        blockers.append("topology improvement required")
    boundary_increase = int(after.boundary_edges - before.boundary_edges)
    if boundary_increase > int(policy.max_boundary_edge_increase):
        blockers.append(
            f"boundary_edges increased by {boundary_increase}"
        )
    non_manifold_increase = int(after.non_manifold_edges - before.non_manifold_edges)
    if non_manifold_increase > int(policy.max_non_manifold_edge_increase):
        blockers.append(
            f"non_manifold_edges increased by {non_manifold_increase}"
        )
    component_increase = int(after.connected_components - before.connected_components)
    if component_increase > int(policy.max_component_increase):
        blockers.append(f"connected_components increased by {component_increase}")
    vertex_ratio = _ratio(after_vertex_count, before_vertex_count)
    if vertex_ratio > float(policy.max_vertex_count_ratio):
        blockers.append(f"vertex count ratio {vertex_ratio:.3f} exceeds policy")
    face_ratio = _ratio(after_face_count, before_face_count)
    if face_ratio > float(policy.max_face_count_ratio):
        blockers.append(f"face count ratio {face_ratio:.3f} exceeds policy")
    accepted = not blockers
    return MeshChangeGuardDecision(
        accepted=accepted,
        reason="accepted" if accepted else "; ".join(blockers),
        topology_delta=topology_delta,
        blockers=tuple(blockers),
        policy=policy,
    )


def _ratio(after: int, before: int) -> float:
    if before <= 0:
        return 1.0 if after <= 0 else float("inf")
    return float(after) / float(before)


def _float(value: object, default: float) -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int) -> int:
    try:
        return int(default if value is None else value)
    except (TypeError, ValueError):
        return default
