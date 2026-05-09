"""Camera/view calibration diagnostics contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .schemas import EvaluationBundle, json_safe


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


def calibration_refinement_plan_payload(bundle: EvaluationBundle) -> dict[str, object]:
    """Build a concrete safe calibration sweep plan from evaluation evidence."""
    metrics = bundle.metric_index()
    diagnostics = _visual_hull_diagnostics(bundle)
    min_iou = _metric_float(metrics, "silhouette.min_view_iou", 0.0)
    failed_view_count = int(
        _metric_float(metrics, "diagnostics.visual_hull.failed_view_count", 0.0)
    )
    top_like_failure_count = int(
        _metric_float(metrics, "diagnostics.visual_hull.top_like_failure_count", 0.0)
    )
    axis_suspect = bool(
        _metric_value(metrics, "diagnostics.visual_hull.axis_or_transform_suspect", False)
    )
    catastrophic = bool(
        _metric_value(metrics, "diagnostics.visual_hull.catastrophic_view_failure", False)
    )
    candidates = _calibration_candidates(
        min_iou=min_iou,
        failed_view_count=failed_view_count,
        axis_suspect=axis_suspect,
        catastrophic=catastrophic,
    )
    probes = [
        {
            "probe_id": "bounds_padding_sweep",
            "description": "Expand and contract target bounds before carving and projection.",
            "parameters": {"bounds_padding": [-0.04, -0.02, 0.0, 0.02, 0.04, 0.08]},
            "acceptance": {
                "min_iou_delta": 0.05,
                "must_not_reduce_topology_score": True,
            },
        },
        {
            "probe_id": "global_scale_sweep",
            "description": "Test conservative uniform scale corrections around the current target bounds.",
            "parameters": {"global_scale": [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]},
            "acceptance": {
                "min_iou_delta": 0.08 if catastrophic else 0.04,
                "max_complexity_delta": 0.05,
            },
        },
        {
            "probe_id": "per_view_offset_sweep",
            "description": "Shift projected masks by small image-space offsets to detect canonicalization/framing errors.",
            "parameters": {
                "offset_px": [-12, -8, -4, 0, 4, 8, 12],
                "views": list(_failed_views(diagnostics)),
            },
            "acceptance": {
                "failed_view_count_delta": -1,
                "min_boundary_iou_delta": 0.04,
            },
        },
    ]
    if axis_suspect or top_like_failure_count:
        probes.append(
            {
                "probe_id": "axis_role_permutation_sweep",
                "description": "Run the existing bounds-debug axis permutation search and compare front/side/top role assignments.",
                "parameters": {
                    "permutations": "xyz permutations",
                    "flips": "all sign flips",
                    "scale_modes": ["identity", "half", "double", "fit_bounds"],
                    "translation_modes": ["none", "center", "bottom"],
                },
                "acceptance": {
                    "average_iou_delta": 0.15,
                    "requires_manual_review": True,
                },
            }
        )
    return {
        "schema_version": "calibration_refinement_plan_v1",
        "candidate_id": bundle.candidate_id,
        "mode": bundle.mode,
        "trigger_metrics": {
            "silhouette.min_view_iou": min_iou,
            "failed_view_count": failed_view_count,
            "top_like_failure_count": top_like_failure_count,
            "axis_or_transform_suspect": axis_suspect,
            "catastrophic_view_failure": catastrophic,
        },
        "recommended_track": "visual-hull-transform",
        "recommended_command": [
            "python",
            "blender_blocking/refinement_lab/cli.py",
            "plan",
            "--track",
            "visual-hull-transform",
            "--bounds-debug",
        ],
        "candidates": [candidate.to_dict() for candidate in candidates],
        "probes": probes,
        "diagnostics": json_safe(diagnostics),
    }


def _calibration_candidates(
    *,
    min_iou: float,
    failed_view_count: int,
    axis_suspect: bool,
    catastrophic: bool,
) -> tuple[CalibrationCandidate, ...]:
    score_before = {
        "silhouette.min_view_iou": float(min_iou),
        "failed_view_count": float(failed_view_count),
    }
    candidates = [
        CalibrationCandidate(
            global_scale=1.0,
            bounds_padding=0.02,
            score_before=score_before,
            score_after={"expected_min_iou_delta": 0.03},
        ),
        CalibrationCandidate(
            global_scale=0.9,
            bounds_padding=0.0,
            score_before=score_before,
            score_after={"expected_min_iou_delta": 0.04},
        ),
        CalibrationCandidate(
            global_scale=1.1,
            bounds_padding=0.0,
            score_before=score_before,
            score_after={"expected_min_iou_delta": 0.04},
        ),
    ]
    if catastrophic or axis_suspect:
        candidates.extend(
            [
                CalibrationCandidate(
                    global_scale=0.5,
                    bounds_padding=0.0,
                    score_before=score_before,
                    score_after={"expected_min_iou_delta": 0.08},
                ),
                CalibrationCandidate(
                    global_scale=2.0,
                    bounds_padding=0.0,
                    score_before=score_before,
                    score_after={"expected_min_iou_delta": 0.08},
                ),
            ]
        )
    return tuple(candidates)


def _visual_hull_diagnostics(bundle: EvaluationBundle) -> Mapping[str, Any]:
    for group in bundle.metric_groups:
        if group.name != "diagnostics":
            continue
        metadata = group.metadata
        if isinstance(metadata, Mapping):
            diagnostics = metadata.get("visual_hull")
            if isinstance(diagnostics, Mapping):
                return diagnostics
    return {}


def _failed_views(diagnostics: Mapping[str, Any]) -> tuple[str, ...]:
    failed = diagnostics.get("failed_views", ())
    if not isinstance(failed, (list, tuple)):
        return ()
    return tuple(str(view) for view in failed)


def _metric_value(
    metrics: Mapping[str, Any],
    name: str,
    default: Any,
) -> Any:
    metric = metrics.get(name)
    if metric is None:
        return default
    return getattr(metric, "value", default)


def _metric_float(
    metrics: Mapping[str, Any],
    name: str,
    default: float,
) -> float:
    try:
        return float(_metric_value(metrics, name, default))
    except (TypeError, ValueError):
        return default
