"""Risk register and metric-gaming controls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskRecord:
    risk_id: str
    title: str
    severity: str
    likelihood: str
    affected_specs: tuple[str, ...]
    controls: tuple[str, ...]
    tests: tuple[str, ...]
    status: str = "open"

    def to_dict(self) -> dict[str, object]:
        return {
            "risk_id": self.risk_id,
            "title": self.title,
            "severity": self.severity,
            "likelihood": self.likelihood,
            "affected_specs": list(self.affected_specs),
            "controls": list(self.controls),
            "tests": list(self.tests),
            "status": self.status,
        }


DEFAULT_RISK_REGISTER = (
    RiskRecord(
        "metric_gaming_blobby_silhouette",
        "High area IoU hides blobby contour output",
        "high",
        "high",
        ("02", "22", "28"),
        ("Boundary IoU", "signed-distance loss", "boundary overlays"),
        ("blobby_mask_high_area_low_boundary",),
    ),
    RiskRecord(
        "pretty_pixels_bad_asset",
        "Image metrics hide non-editable geometry",
        "critical",
        "medium",
        ("04", "05", "18", "27", "28"),
        ("editability index", "export QA", "proxy-only warnings"),
        ("proxy_only_no_mesh", "textured_plane_bad_geometry"),
    ),
    RiskRecord(
        "calibration_overfit",
        "Calibration moves hide backend reconstruction errors",
        "high",
        "medium",
        ("16", "28"),
        ("safe calibration bounds", "before/after calibration report"),
        ("excessive_calibration_shift",),
    ),
)


def risk_register_payload() -> dict[str, object]:
    return {risk.risk_id: risk.to_dict() for risk in DEFAULT_RISK_REGISTER}
