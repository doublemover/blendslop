"""Shared reconstruction objective and loss result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ReconstructionLossWeights:
    area_iou: float = 1.0
    boundary_iou: float = 0.5
    silhouette_sdf: float = 0.25
    surface_chamfer: float = 0.25
    visual_hull_occupancy: float = 0.5
    primitive_count: float = 0.05
    overlap_penalty: float = 0.1
    topology_penalty: float = 0.5
    constraint_penalty: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "area_iou": self.area_iou,
            "boundary_iou": self.boundary_iou,
            "silhouette_sdf": self.silhouette_sdf,
            "surface_chamfer": self.surface_chamfer,
            "visual_hull_occupancy": self.visual_hull_occupancy,
            "primitive_count": self.primitive_count,
            "overlap_penalty": self.overlap_penalty,
            "topology_penalty": self.topology_penalty,
            "constraint_penalty": self.constraint_penalty,
        }


@dataclass(frozen=True)
class ReconstructionLossResult:
    total: float
    terms: Mapping[str, float] = field(default_factory=dict)
    per_view: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "terms": dict(self.terms),
            "per_view": dict(self.per_view),
            "warnings": list(self.warnings),
        }


def weighted_sum(
    terms: Mapping[str, float], weights: ReconstructionLossWeights
) -> ReconstructionLossResult:
    weight_map = weights.to_dict()
    total = 0.0
    for name, value in terms.items():
        total += float(value) * float(weight_map.get(name, 1.0))
    return ReconstructionLossResult(total=total, terms=dict(terms))
