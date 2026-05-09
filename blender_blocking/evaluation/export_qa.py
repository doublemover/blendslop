"""Export and asset-delivery QA contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .schemas import json_safe


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

    @property
    def status_ok(self) -> bool:
        return _ok_status(self.status)

    @property
    def reimport_ok(self) -> bool | None:
        if self.reimport_status is None:
            return None
        return _ok_status(self.reimport_status)

    @property
    def qa_score(self) -> float:
        status_score = 1.0 if self.status_ok else 0.0
        reimport_score = 1.0 if self.reimport_ok is True else 0.0 if self.reimport_ok is False else 0.5
        count_score = _count_score(
            self.object_count,
            self.vertex_count,
            self.face_count,
            self.material_count,
        )
        warning_penalty = min(0.25, 0.05 * len(self.warnings))
        error_penalty = min(1.0, 0.25 * len(self.errors))
        score = (0.35 * status_score) + (0.35 * reimport_score) + (0.30 * count_score)
        return float(max(0.0, min(1.0, score - warning_penalty - error_penalty)))

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
            "status_ok": self.status_ok,
            "reimport_ok": self.reimport_ok,
            "qa_score": self.qa_score,
        }


def report_from_mapping(payload: Mapping[str, Any]) -> ExportQAReport:
    """Normalize export/reimport metadata from a backend or external QA run."""
    bounds = _bounds(payload.get("bounds"))
    status = _status_text(payload.get("status"), "not_applicable")
    reimport_status = payload.get("reimport_status", payload.get("roundtrip_status"))
    return ExportQAReport(
        target=str(payload.get("target", payload.get("format", payload.get("name", "asset")))),
        status=status,
        exported_path=_string_or_none(payload.get("exported_path", payload.get("path"))),
        reimport_status=_status_text(reimport_status, "not_applicable")
        if reimport_status is not None
        else None,
        object_count=_int_or_none(payload.get("object_count", payload.get("objects"))),
        vertex_count=_int_or_none(payload.get("vertex_count", payload.get("vertices"))),
        face_count=_int_or_none(payload.get("face_count", payload.get("faces"))),
        material_count=_int_or_none(payload.get("material_count", payload.get("materials"))),
        bounds=bounds,
        warnings=_string_tuple(payload.get("warnings")),
        errors=_string_tuple(payload.get("errors")),
    )


def reports_from_payload(payload: Any) -> tuple[ExportQAReport, ...]:
    """Return one or more reports from common backend metadata shapes."""
    if isinstance(payload, ExportQAReport):
        return (payload,)
    if isinstance(payload, Mapping):
        reports = payload.get("reports")
        if isinstance(reports, Sequence) and not isinstance(reports, (str, bytes, bytearray)):
            return tuple(
                report_from_mapping(item)
                for item in reports
                if isinstance(item, Mapping)
            )
        targets = payload.get("targets")
        if isinstance(targets, Mapping):
            normalized = []
            for target, item in targets.items():
                if isinstance(item, Mapping):
                    merged = {"target": target, **dict(item)}
                    normalized.append(report_from_mapping(merged))
            return tuple(normalized)
        return (report_from_mapping(payload),)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return tuple(
            item if isinstance(item, ExportQAReport) else report_from_mapping(item)
            for item in payload
            if isinstance(item, (ExportQAReport, Mapping))
        )
    return ()


def aggregate_score(reports: Sequence[ExportQAReport]) -> float | None:
    if not reports:
        return None
    return float(sum(report.qa_score for report in reports) / len(reports))


def _ok_status(status: object) -> bool:
    return str(status or "").strip().lower() in {
        "pass",
        "success",
        "ok",
        "complete",
        "completed",
    }


def _status_text(value: Any, default: str) -> str:
    text = str(default if value is None else value).strip().lower()
    return text or default


def _count_score(*values: int | None) -> float:
    present = [value for value in values if value is not None]
    if not present:
        return 0.5
    positive = sum(1 for value in present if value > 0)
    return float(positive / len(present))


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bounds(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(json_safe(value)),)
