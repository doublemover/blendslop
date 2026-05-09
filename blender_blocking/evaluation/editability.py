"""Blender-editability metric contracts and scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .schemas import json_safe


@dataclass(frozen=True)
class EditabilityReport:
    object_hierarchy_score: float = 0.0
    primitive_score: float = 0.0
    modifier_score: float = 0.0
    mesh_density_score: float = 0.0
    semantic_part_score: float = 0.0
    topology_score: float = 0.0
    export_roundtrip_score: float = 0.0
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def editable_reconstruction_index(self) -> float:
        weights = (
            (self.object_hierarchy_score, 0.15),
            (self.primitive_score, 0.20),
            (self.modifier_score, 0.10),
            (self.mesh_density_score, 0.15),
            (self.semantic_part_score, 0.15),
            (self.topology_score, 0.20),
            (self.export_roundtrip_score, 0.05),
        )
        return float(max(0.0, min(1.0, sum(value * weight for value, weight in weights))))

    def to_dict(self) -> dict[str, object]:
        return {
            "object_hierarchy_score": self.object_hierarchy_score,
            "primitive_score": self.primitive_score,
            "modifier_score": self.modifier_score,
            "mesh_density_score": self.mesh_density_score,
            "semantic_part_score": self.semantic_part_score,
            "topology_score": self.topology_score,
            "export_roundtrip_score": self.export_roundtrip_score,
            "editable_reconstruction_index": self.editable_reconstruction_index,
            "warnings": list(self.warnings),
            "metadata": json_safe(self.metadata),
        }


def report_from_candidate_metrics(metrics: Any) -> EditabilityReport:
    extras = getattr(metrics, "extras", {}) or {}
    topology_score = float(getattr(metrics, "topology_score", 0.0) or 0.0)
    primitive_score = 0.0
    modifier_score = 0.0
    hierarchy_score = 0.0
    if isinstance(extras, Mapping):
        primitive_score = float(extras.get("primitive_editability", 0.0) or 0.0)
        modifier_score = float(extras.get("modifier_editability", 0.0) or 0.0)
        hierarchy_score = float(extras.get("object_hierarchy_score", 0.0) or 0.0)
    mesh_density_score = 1.0 - min(1.0, float(getattr(metrics, "complexity_penalty", 0.0) or 0.0))
    return EditabilityReport(
        object_hierarchy_score=hierarchy_score,
        primitive_score=primitive_score,
        modifier_score=modifier_score,
        mesh_density_score=mesh_density_score,
        topology_score=topology_score,
    )

