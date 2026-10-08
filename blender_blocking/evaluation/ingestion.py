"""Input and mask-ingestion evidence contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .schemas import json_safe


@dataclass(frozen=True)
class MaskVariant:
    variant_id: str
    threshold: float | None = None
    morphology: Mapping[str, Any] = field(default_factory=dict)
    area_ratio: float = 0.0
    bbox: Mapping[str, float] | None = None
    component_count: int = 0
    selected: bool = False
    score: float | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "threshold": self.threshold,
            "morphology": json_safe(self.morphology),
            "area_ratio": self.area_ratio,
            "bbox": json_safe(self.bbox),
            "component_count": self.component_count,
            "selected": self.selected,
            "score": self.score,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class UncertainMaskSummary:
    foreground_probability_path: str | None = None
    confidence_path: str | None = None
    entropy_path: str | None = None
    boundary_entropy: float | None = None
    area_variance: float | None = None
    bbox_variance: float | None = None
    component_count_variance: float | None = None
    centerline_variance: float | None = None
    profile_interval_variance: float | None = None
    variants: tuple[MaskVariant, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "foreground_probability_path": self.foreground_probability_path,
            "confidence_path": self.confidence_path,
            "entropy_path": self.entropy_path,
            "boundary_entropy": self.boundary_entropy,
            "area_variance": self.area_variance,
            "bbox_variance": self.bbox_variance,
            "component_count_variance": self.component_count_variance,
            "centerline_variance": self.centerline_variance,
            "profile_interval_variance": self.profile_interval_variance,
            "variants": [variant.to_dict() for variant in self.variants],
        }


@dataclass(frozen=True)
class TargetView:
    view_id: str
    role: str
    image_path: str | None = None
    mask_path: str | None = None
    confidence_path: str | None = None
    native_size: tuple[int, int] = (0, 0)
    canonical_size: tuple[int, int] = (0, 0)
    native_bbox: Mapping[str, float] | None = None
    canonical_bbox: Mapping[str, float] | None = None
    mask_area_ratio: float = 0.0
    extraction_method: str = "unknown"
    threshold: float | None = None
    morphology: Mapping[str, Any] = field(default_factory=dict)
    uncertainty_summary: UncertainMaskSummary | None = None
    canonicalization: Mapping[str, Any] = field(default_factory=dict)
    role_confidence: float | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "view_id": self.view_id,
            "role": self.role,
            "image_path": self.image_path,
            "mask_path": self.mask_path,
            "confidence_path": self.confidence_path,
            "native_size": list(self.native_size),
            "canonical_size": list(self.canonical_size),
            "native_bbox": json_safe(self.native_bbox),
            "canonical_bbox": json_safe(self.canonical_bbox),
            "mask_area_ratio": self.mask_area_ratio,
            "extraction_method": self.extraction_method,
            "threshold": self.threshold,
            "morphology": json_safe(self.morphology),
            "uncertainty_summary": json_safe(self.uncertainty_summary),
            "canonicalization": json_safe(self.canonicalization),
            "role_confidence": self.role_confidence,
            "warnings": list(self.warnings),
        }


def target_views_from_reconstruction_target(target: Any) -> tuple[TargetView, ...]:
    views = []
    for constraint in getattr(target, "constraints", ()) or ():
        mask = getattr(constraint, "mask", None)
        shape = getattr(getattr(mask, "mask", mask), "shape", (0, 0))
        height = int(shape[0]) if len(shape) >= 1 else 0
        width = int(shape[1]) if len(shape) >= 2 else 0
        diagnostics = dict(getattr(constraint, "diagnostics", {}) or {})
        selected = diagnostics.get("selected") if isinstance(diagnostics, Mapping) else {}
        canonical = diagnostics.get("canonical") if isinstance(diagnostics, Mapping) else {}
        bbox = getattr(constraint, "bbox", None)
        views.append(
            TargetView(
                view_id=str(getattr(constraint, "view", "")),
                role=str(getattr(constraint, "view", "")),
                native_size=(width, height),
                canonical_size=_canonical_size(canonical, width, height),
                native_bbox=_bbox_payload(bbox),
                canonical_bbox=_canonical_bbox(canonical),
                mask_area_ratio=float(selected.get("diagnostics", {}).get("area_frac", 0.0))
                if isinstance(selected, Mapping)
                else 0.0,
                extraction_method=str(selected.get("source", "unknown"))
                if isinstance(selected, Mapping)
                else "unknown",
                threshold=selected.get("threshold") if isinstance(selected, Mapping) else None,
                canonicalization=json_safe(canonical) if isinstance(canonical, Mapping) else {},
                warnings=tuple(str(item) for item in diagnostics.get("warnings", ()) or ()),
            )
        )
    return tuple(views)


def _bbox_payload(bbox: Any) -> dict[str, float] | None:
    if bbox is None:
        return None
    if hasattr(bbox, "to_dict"):
        return bbox.to_dict()
    if hasattr(bbox, "to_xyxy"):
        x0, y0, x1, y1 = bbox.to_xyxy()
        return {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)}
    return None


def _canonical_size(canonical: Any, width: int, height: int) -> tuple[int, int]:
    if not isinstance(canonical, Mapping):
        return (width, height)
    output_size = canonical.get("output_size")
    try:
        value = int(output_size)
    except (TypeError, ValueError):
        return (width, height)
    return (value, value)


def _canonical_bbox(canonical: Any) -> dict[str, float] | None:
    if not isinstance(canonical, Mapping):
        return None
    transform = canonical.get("transform", {})
    if not isinstance(transform, Mapping):
        return None
    width = transform.get("resized_width")
    height = transform.get("resized_height")
    x0 = transform.get("x_start")
    y0 = transform.get("y_start")
    try:
        return {
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x0) + float(width),
            "y1": float(y0) + float(height),
        }
    except (TypeError, ValueError):
        return None
