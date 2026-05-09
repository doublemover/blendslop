"""Camera/view calibration diagnostics contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .schemas import json_safe


@dataclass(frozen=True)
class ViewCalibrationReport:
    view_id: str
    role: str
    image_size: tuple[int, int]
    mask_bbox: Mapping[str, float] | None = None
    render_bbox: Mapping[str, float] | None = None
    centroid_delta_px: tuple[float, float] | None = None
    bbox_scale_ratio: tuple[float, float] | None = None
    orthographic_scale: float | None = None
    camera_pose: Mapping[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    status: str = "pass"

    def to_dict(self) -> dict[str, object]:
        return {
            "view_id": self.view_id,
            "role": self.role,
            "image_size": list(self.image_size),
            "mask_bbox": json_safe(self.mask_bbox),
            "render_bbox": json_safe(self.render_bbox),
            "centroid_delta_px": list(self.centroid_delta_px) if self.centroid_delta_px else None,
            "bbox_scale_ratio": list(self.bbox_scale_ratio) if self.bbox_scale_ratio else None,
            "orthographic_scale": self.orthographic_scale,
            "camera_pose": json_safe(self.camera_pose),
            "warnings": list(self.warnings),
            "status": self.status,
        }


@dataclass(frozen=True)
class CalibrationCandidate:
    global_scale: float = 1.0
    global_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    per_view_offset_px: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    bounds_padding: float = 0.0
    score_before: Mapping[str, float] = field(default_factory=dict)
    score_after: Mapping[str, float] = field(default_factory=dict)
    accepted: bool = False
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "global_scale": self.global_scale,
            "global_offset": list(self.global_offset),
            "per_view_offset_px": {key: list(value) for key, value in self.per_view_offset_px.items()},
            "bounds_padding": self.bounds_padding,
            "score_before": dict(self.score_before),
            "score_after": dict(self.score_after),
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }

