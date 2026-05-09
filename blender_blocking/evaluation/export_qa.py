"""Export and asset-delivery QA contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExportQAReport:
    target: str
    status: str
    exported_path: str | None = None
    reimport_status: str | None = None
    object_count: int | None = None
    vertex_count: int | None = None
    face_count: int | None = None
    material_count: int | None = None
    bounds: tuple[float, float, float] | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "status": self.status,
            "exported_path": self.exported_path,
            "reimport_status": self.reimport_status,
            "object_count": self.object_count,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "material_count": self.material_count,
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

