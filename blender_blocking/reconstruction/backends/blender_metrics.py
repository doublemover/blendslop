"""Adapters from Blender mesh QA to backend-neutral candidate metrics."""

from __future__ import annotations

from typing import Any, Mapping

from ..types import Bounds3D, CandidateMetrics, MeshQualityReport


def _topology_score(quality: Any) -> float:
    score = 1.0
    score -= min(0.45, float(getattr(quality, "loose_vertices", 0)) * 0.05)
    score -= min(0.45, float(getattr(quality, "non_manifold_edges", 0)) * 0.03)
    score -= min(0.35, float(getattr(quality, "degenerate_faces", 0)) * 0.03)
    score -= min(0.25, float(getattr(quality, "zero_area_faces", 0)) * 0.03)
    score -= min(
        0.25, float(max(0, int(getattr(quality, "connected_components", 0)) - 1)) * 0.1
    )
    if not bool(getattr(quality, "bbox_sane", True)):
        score -= 0.25
    return max(0.0, min(1.0, score))


def _bounds_from_quality(quality: Any) -> Bounds3D | None:
    bbox = getattr(quality, "bbox", None)
    if bbox is None:
        return None
    return Bounds3D(
        min_x=float(getattr(bbox, "min_x", 0.0)),
        max_x=float(getattr(bbox, "max_x", 0.0)),
        min_y=float(getattr(bbox, "min_y", 0.0)),
        max_y=float(getattr(bbox, "max_y", 0.0)),
        min_z=float(getattr(bbox, "min_z", 0.0)),
        max_z=float(getattr(bbox, "max_z", 0.0)),
    )


def candidate_metrics_from_blender_object(
    obj: Any,
    *,
    editability_score: float,
    elapsed_s: float = 0.0,
    extras: Mapping[str, Any] | None = None,
) -> CandidateMetrics:
    """Collect real mesh counts/topology from a Blender object."""
    from integration.blender_ops.mesh_quality import collect_mesh_quality

    quality = collect_mesh_quality(obj)
    quality_dict = quality.to_dict()
    neutral_quality = MeshQualityReport(
        object_name=str(getattr(quality, "object_name", "")),
        vertices=int(getattr(quality, "vertices", 0)),
        edges=int(getattr(quality, "edges", 0)),
        faces=int(getattr(quality, "faces", 0)),
        loose_vertices=int(getattr(quality, "loose_vertices", 0)),
        non_manifold_edges=int(getattr(quality, "non_manifold_edges", 0)),
        boundary_edges=int(getattr(quality, "boundary_edges", 0)),
        connected_components=int(getattr(quality, "connected_components", 0)),
        degenerate_faces=int(getattr(quality, "degenerate_faces", 0)),
        zero_area_faces=int(getattr(quality, "zero_area_faces", 0)),
        bbox=_bounds_from_quality(quality),
        warnings=tuple(getattr(quality, "self_intersection_warnings", ())),
    )
    merged_extras = {
        "blender_mesh_quality": quality_dict,
        "watertight": bool(getattr(quality, "watertight", False)),
        "surface_area": getattr(quality, "surface_area", None),
        "volume": getattr(quality, "volume", None),
        "genus_estimate": getattr(quality, "genus_estimate", None),
    }
    if extras:
        merged_extras.update(extras)
    return CandidateMetrics(
        topology_score=_topology_score(quality),
        editability_score=float(editability_score),
        elapsed_s=float(elapsed_s),
        mesh_quality=neutral_quality,
        extras=merged_extras,
    )
