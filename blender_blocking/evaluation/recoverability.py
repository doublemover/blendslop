"""Recoverability metrics for silhouette-only reconstruction.

Silhouette inputs cannot determine every geometric detail.  This module keeps
two geometry reports separate:

* ``true``: distance to the synthetic or captured ground truth when available.
* ``recoverable``: distance to the visual-hull or other view-consistent envelope
  that should be reachable from silhouettes alone.

The gap between those reports is not a backend failure by itself.  It is the
evidence needed to decide whether a result is bad or whether the requested
shape detail is genuinely not identifiable from the provided views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .geometry import GeometryMetricReport, report_from_mapping as geometry_report_from_mapping
from .schemas import json_safe


@dataclass(frozen=True)
class RecoverabilityReport:
    """Geometry evidence split by true and silhouette-recoverable targets."""

    true_geometry: GeometryMetricReport | None = None
    recoverable_geometry: GeometryMetricReport | None = None
    ambiguity_gap_chamfer_l1: float | None = None
    ambiguity_gap_chamfer_l2: float | None = None
    ambiguity_gap_volume_iou: float | None = None
    source: str = "computed"
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "true_geometry": (
                self.true_geometry.to_dict() if self.true_geometry is not None else None
            ),
            "recoverable_geometry": (
                self.recoverable_geometry.to_dict()
                if self.recoverable_geometry is not None
                else None
            ),
            "ambiguity_gap_chamfer_l1": self.ambiguity_gap_chamfer_l1,
            "ambiguity_gap_chamfer_l2": self.ambiguity_gap_chamfer_l2,
            "ambiguity_gap_volume_iou": self.ambiguity_gap_volume_iou,
            "source": self.source,
            "warnings": list(self.warnings),
            "metadata": json_safe(self.metadata),
        }


def report_from_mapping(payload: Mapping[str, Any]) -> RecoverabilityReport:
    """Build a recoverability report from a flexible JSON-like payload."""

    true_payload = _mapping_or_none(
        payload.get("true_geometry", payload.get("true", payload.get("geometry_true")))
    )
    recoverable_payload = _mapping_or_none(
        payload.get(
            "recoverable_geometry",
            payload.get("recoverable", payload.get("visual_hull_envelope")),
        )
    )
    true_report = report_from_mapping_geometry(true_payload) if true_payload else None
    recoverable_report = (
        report_from_mapping_geometry(recoverable_payload)
        if recoverable_payload
        else None
    )
    warnings = [str(item) for item in payload.get("warnings", ()) or ()]
    if true_report is None and recoverable_report is None:
        warnings.append("recoverability payload did not include true or recoverable geometry")
    return RecoverabilityReport(
        true_geometry=true_report,
        recoverable_geometry=recoverable_report,
        ambiguity_gap_chamfer_l1=_coalesce_float(
            payload.get("ambiguity_gap_chamfer_l1"),
            _gap_chamfer(true_report, recoverable_report, "chamfer_l1"),
        ),
        ambiguity_gap_chamfer_l2=_coalesce_float(
            payload.get("ambiguity_gap_chamfer_l2"),
            payload.get("ambiguity_gap_chamfer"),
            _gap_chamfer(true_report, recoverable_report, "chamfer_l2"),
        ),
        ambiguity_gap_volume_iou=_coalesce_float(
            payload.get("ambiguity_gap_volume_iou"),
            payload.get("ambiguity_gap_volume"),
            _gap_volume(true_report, recoverable_report),
        ),
        source=str(payload.get("source", "computed")),
        warnings=tuple(dict.fromkeys(warnings)),
        metadata=payload.get("metadata", {})
        if isinstance(payload.get("metadata", {}), Mapping)
        else {},
    )


def report_from_mapping_geometry(payload: Mapping[str, Any]) -> GeometryMetricReport:
    return geometry_report_from_mapping(payload)


def _gap_chamfer(
    true_report: GeometryMetricReport | None,
    recoverable_report: GeometryMetricReport | None,
    attr: str,
) -> float | None:
    if true_report is None or recoverable_report is None:
        return None
    true_value = getattr(true_report, attr)
    recoverable_value = getattr(recoverable_report, attr)
    if true_value is None or recoverable_value is None:
        return None
    return max(0.0, float(true_value) - float(recoverable_value))


def _gap_volume(
    true_report: GeometryMetricReport | None,
    recoverable_report: GeometryMetricReport | None,
) -> float | None:
    if true_report is None or recoverable_report is None:
        return None
    true_value = true_report.volumetric_iou
    recoverable_value = recoverable_report.volumetric_iou
    if true_value is None or recoverable_value is None:
        return None
    return max(0.0, float(recoverable_value) - float(true_value))


def _coalesce_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None
