"""Blender-editability metric contracts and scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .schemas import json_safe
try:
    from blender_blocking.metrics.values import float_or as _float
except ImportError:  # pragma: no cover - script-style imports
    from metrics.values import float_or as _float


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
    if isinstance(extras, Mapping):
        explicit = extras.get("editability")
        if isinstance(explicit, Mapping):
            return report_from_mapping(explicit)
    topology_score = _clamped(float(getattr(metrics, "topology_score", 0.0) or 0.0))
    backend_editability = _clamped(
        float(getattr(metrics, "editability_score", 0.0) or 0.0)
    )
    primitive_score = 0.0
    modifier_score = 0.0
    hierarchy_score = 0.0
    semantic_score = 0.0
    export_score = 0.0
    warnings: list[str] = []
    metadata: dict[str, Any] = {}
    if isinstance(extras, Mapping):
        primitive_payload = extras.get("primitives") or extras.get("primitive_set")
        primitive_count = _count_primitives(primitive_payload)
        if primitive_count <= 0 and "primitive_count" in extras:
            primitive_count = max(0, int(_float(extras.get("primitive_count"))))
        primitive_score = _clamped(
            extras.get("primitive_editability", 1.0 if primitive_count else 0.0)
        )
        modifier_score = _clamped(extras.get("modifier_editability", 0.0))
        hierarchy_score = _clamped(extras.get("object_hierarchy_score", 0.0))
        semantic_score = _clamped(extras.get("semantic_part_score", 0.0))
        if primitive_count:
            hierarchy_score = max(hierarchy_score, min(1.0, 0.45 + 0.08 * primitive_count))
            semantic_score = max(semantic_score, min(1.0, 0.35 + 0.06 * primitive_count))
        export_score = _export_score(extras)
        mesh_counts = _mesh_counts(extras)
        metadata.update(
            {
                "primitive_count": primitive_count,
                "mesh_counts": mesh_counts,
                "backend_editability_score": backend_editability,
            }
        )
    else:
        mesh_counts = {}
    mesh_density_score = _mesh_density_score(
        mesh_counts,
        complexity_penalty=float(getattr(metrics, "complexity_penalty", 0.0) or 0.0),
    )
    mesh_quality = getattr(metrics, "mesh_quality", None)
    if mesh_quality is not None and hasattr(mesh_quality, "topology_score"):
        topology_score = max(topology_score, _clamped(mesh_quality.topology_score()))
        metadata["mesh_quality"] = mesh_quality.to_dict() if hasattr(mesh_quality, "to_dict") else {}
    if backend_editability > 0.0:
        primitive_score = max(primitive_score, backend_editability * 0.75)
        hierarchy_score = max(hierarchy_score, backend_editability * 0.55)
        semantic_score = max(semantic_score, backend_editability * 0.45)
    if topology_score <= 0.0:
        warnings.append("topology score missing or zero")
    if primitive_score <= 0.0:
        warnings.append("no primitive or parametric editability evidence")
    if mesh_density_score < 0.35:
        warnings.append("mesh density is likely too high for comfortable editing")
    if export_score <= 0.0:
        warnings.append("export round-trip editability evidence missing")
    return EditabilityReport(
        object_hierarchy_score=hierarchy_score,
        primitive_score=primitive_score,
        modifier_score=modifier_score,
        mesh_density_score=mesh_density_score,
        semantic_part_score=semantic_score,
        topology_score=topology_score,
        export_roundtrip_score=export_score,
        warnings=tuple(dict.fromkeys(warnings)),
        metadata=metadata,
    )


def report_from_mapping(payload: Mapping[str, Any]) -> EditabilityReport:
    return EditabilityReport(
        object_hierarchy_score=_float(payload.get("object_hierarchy_score")),
        primitive_score=_float(payload.get("primitive_score")),
        modifier_score=_float(payload.get("modifier_score")),
        mesh_density_score=_float(payload.get("mesh_density_score")),
        semantic_part_score=_float(payload.get("semantic_part_score")),
        topology_score=_float(payload.get("topology_score")),
        export_roundtrip_score=_float(payload.get("export_roundtrip_score")),
        warnings=tuple(str(item) for item in payload.get("warnings", ()) or ()),
        metadata=payload.get("metadata", {})
        if isinstance(payload.get("metadata", {}), Mapping)
        else {},
    )


def _clamped(value: Any, default: float = 0.0) -> float:
    parsed = _float(value, default)
    return max(0.0, min(1.0, parsed))


def _count_primitives(payload: Any) -> int:
    if payload is None:
        return 0
    if isinstance(payload, Mapping):
        for key in ("count", "primitive_count"):
            if key in payload:
                return max(0, int(_float(payload.get(key))))
        for key in ("items", "primitives"):
            value = payload.get(key)
            if isinstance(value, (list, tuple)):
                return len(value)
        return 1 if payload else 0
    if isinstance(payload, (list, tuple)):
        return len(payload)
    return 0


def _mesh_counts(extras: Mapping[str, Any]) -> dict[str, int]:
    for key in ("mesh", "mesh_metadata", "mesh_extraction"):
        payload = extras.get(key)
        if not isinstance(payload, Mapping):
            continue
        if "metrics" in payload and isinstance(payload.get("metrics"), Mapping):
            payload = payload["metrics"]  # type: ignore[index]
        vertices = _first_int(payload, ("vertex_count", "vertices", "output_vertex_count"))
        faces = _first_int(payload, ("face_count", "faces", "output_face_count"))
        if vertices or faces:
            return {"vertices": vertices, "faces": faces}
    topology = extras.get("topology")
    if isinstance(topology, Mapping):
        vertices = _first_int(topology, ("vertices", "vertex_count"))
        faces = _first_int(topology, ("faces", "face_count"))
        if vertices or faces:
            return {"vertices": vertices, "faces": faces}
    return {}


def _mesh_density_score(
    mesh_counts: Mapping[str, int],
    *,
    complexity_penalty: float,
) -> float:
    if mesh_counts:
        faces = max(0, int(mesh_counts.get("faces", 0)))
        vertices = max(0, int(mesh_counts.get("vertices", 0)))
        count = max(faces, vertices)
        if count <= 0:
            return 0.0
        if count <= 20_000:
            return 1.0
        if count >= 250_000:
            return 0.05
        return max(0.05, 1.0 - (count - 20_000) / 230_000)
    return 1.0 - min(1.0, max(0.0, complexity_penalty))


def _export_score(extras: Mapping[str, Any]) -> float:
    for key in ("export_qa", "asset_export", "export"):
        payload = extras.get(key)
        if not isinstance(payload, Mapping):
            continue
        if "qa_score" in payload:
            return _clamped(payload.get("qa_score"))
        status = str(payload.get("status", "")).strip().lower()
        reimport = str(payload.get("reimport_status", "")).strip().lower()
        if status in {"pass", "ok", "success"} and reimport in {"pass", "ok", "success"}:
            return 1.0
        if status in {"pass", "ok", "success"}:
            return 0.7
    return 0.0


def _first_int(payload: Mapping[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        if key not in payload:
            continue
        try:
            return max(0, int(payload.get(key) or 0))
        except (TypeError, ValueError):
            continue
    return 0
