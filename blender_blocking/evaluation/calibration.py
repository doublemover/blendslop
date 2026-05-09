"""Camera/view calibration diagnostics contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Mapping

import numpy as np

from metrics.silhouette import SilhouetteMetricResult, area_iou, silhouette_metric_result

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


@dataclass(frozen=True)
class MaskAlignmentCandidate:
    """One executable image-space calibration correction candidate."""

    view_id: str
    offset_px: tuple[int, int]
    score: float
    metric_before: Mapping[str, Any]
    metric_after: Mapping[str, Any]
    accepted: bool
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "view_id": self.view_id,
            "offset_px": list(self.offset_px),
            "score": self.score,
            "metric_before": json_safe(self.metric_before),
            "metric_after": json_safe(self.metric_after),
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class MaskAlignmentSweepReport:
    """Result of an executable per-view mask alignment sweep."""

    schema_version: str
    max_offset_px: int
    step_px: int
    candidates: tuple[MaskAlignmentCandidate, ...]
    best_by_view: Mapping[str, MaskAlignmentCandidate]
    aggregate_before: Mapping[str, float]
    aggregate_after: Mapping[str, float]
    accepted_count: int
    warnings: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        if self.accepted_count:
            return "improved"
        if self.candidates:
            return "no_improvement"
        return "skipped"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "max_offset_px": self.max_offset_px,
            "step_px": self.step_px,
            "accepted_count": self.accepted_count,
            "aggregate_before": dict(self.aggregate_before),
            "aggregate_after": dict(self.aggregate_after),
            "best_by_view": {
                key: value.to_dict() for key, value in self.best_by_view.items()
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ViewRolePermutationReport:
    """Detect likely front/side/top role swaps from mask overlap evidence."""

    reference_views: tuple[str, ...]
    candidate_views: tuple[str, ...]
    assignment: Mapping[str, str]
    identity_score: float
    best_score: float
    score_delta: float
    swapped: bool
    iou_matrix: Mapping[str, Mapping[str, float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_views": list(self.reference_views),
            "candidate_views": list(self.candidate_views),
            "assignment": dict(self.assignment),
            "identity_score": self.identity_score,
            "best_score": self.best_score,
            "score_delta": self.score_delta,
            "swapped": self.swapped,
            "iou_matrix": {
                key: dict(value) for key, value in self.iou_matrix.items()
            },
        }


@dataclass(frozen=True)
class CalibrationSweepReport:
    """Combined executable calibration diagnostics for a candidate run."""

    mask_alignment: MaskAlignmentSweepReport
    view_role_permutation: ViewRolePermutationReport

    @property
    def status(self) -> str:
        if self.view_role_permutation.swapped:
            return "view_roles_suspect"
        return self.mask_alignment.status

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "calibration_sweep_report_v1",
            "status": self.status,
            "mask_alignment": self.mask_alignment.to_dict(),
            "view_role_permutation": self.view_role_permutation.to_dict(),
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


def executable_calibration_sweep(
    reference_masks: Mapping[str, np.ndarray],
    candidate_masks: Mapping[str, np.ndarray],
    *,
    max_offset_px: int = 12,
    step_px: int = 4,
    min_area_iou_delta: float = 0.01,
    min_boundary_iou_delta: float = 0.01,
    max_signed_distance_loss_increase: float = 0.0,
    min_area_iou: float = 0.0,
    min_boundary_iou: float | None = None,
    max_signed_distance_loss: float | None = None,
) -> CalibrationSweepReport:
    """Run executable calibration probes over already-rendered masks.

    This does not pretend to solve Blender camera calibration by itself.  It
    gives refinement loops a cheap, deterministic test for the most common
    failure mode: reference/candidate masks that match after a bounded image
    translation or view-role permutation.
    """
    alignment = mask_alignment_sweep(
        reference_masks,
        candidate_masks,
        max_offset_px=max_offset_px,
        step_px=step_px,
        min_area_iou_delta=min_area_iou_delta,
        min_boundary_iou_delta=min_boundary_iou_delta,
        max_signed_distance_loss_increase=max_signed_distance_loss_increase,
        min_area_iou=min_area_iou,
        min_boundary_iou=min_boundary_iou,
        max_signed_distance_loss=max_signed_distance_loss,
    )
    permutation_report = detect_view_role_permutation(
        reference_masks,
        candidate_masks,
    )
    return CalibrationSweepReport(
        mask_alignment=alignment,
        view_role_permutation=permutation_report,
    )


def mask_alignment_sweep(
    reference_masks: Mapping[str, np.ndarray],
    candidate_masks: Mapping[str, np.ndarray],
    *,
    max_offset_px: int = 12,
    step_px: int = 4,
    min_area_iou_delta: float = 0.01,
    min_boundary_iou_delta: float = 0.01,
    max_signed_distance_loss_increase: float = 0.0,
    min_area_iou: float = 0.0,
    min_boundary_iou: float | None = None,
    max_signed_distance_loss: float | None = None,
) -> MaskAlignmentSweepReport:
    """Try bounded integer mask offsets and report best measurable corrections."""
    step_px = max(1, int(step_px))
    max_offset_px = max(0, int(max_offset_px))
    offsets = tuple(range(-max_offset_px, max_offset_px + 1, step_px))
    candidates: list[MaskAlignmentCandidate] = []
    best_by_view: dict[str, MaskAlignmentCandidate] = {}
    warnings: list[str] = []

    common_views = tuple(
        view for view in reference_masks.keys() if view in candidate_masks
    )
    missing_candidate = tuple(
        view for view in reference_masks.keys() if view not in candidate_masks
    )
    if missing_candidate:
        warnings.append(
            "missing candidate masks for views: " + ", ".join(missing_candidate)
        )

    for view in common_views:
        reference = _as_mask(reference_masks[view])
        candidate = _as_mask(candidate_masks[view])
        if reference.shape != candidate.shape:
            raise ValueError(f"mask shape mismatch for view {view!r}")
        before = _silhouette_metric(
            view,
            reference,
            candidate,
            min_area_iou=min_area_iou,
            min_boundary_iou=min_boundary_iou,
            max_signed_distance_loss=max_signed_distance_loss,
        )
        before_dict = before.to_dict()
        view_candidates: list[MaskAlignmentCandidate] = []
        for dy in offsets:
            for dx in offsets:
                shifted = shift_mask(candidate, dx=dx, dy=dy)
                after = _silhouette_metric(
                    view,
                    reference,
                    shifted,
                    min_area_iou=min_area_iou,
                    min_boundary_iou=min_boundary_iou,
                    max_signed_distance_loss=max_signed_distance_loss,
                )
                after_dict = after.to_dict()
                accepted, reason = _alignment_acceptance(
                    before,
                    after,
                    min_area_iou_delta=min_area_iou_delta,
                    min_boundary_iou_delta=min_boundary_iou_delta,
                    max_signed_distance_loss_increase=(
                        max_signed_distance_loss_increase
                    ),
                )
                view_candidates.append(
                    MaskAlignmentCandidate(
                        view_id=view,
                        offset_px=(int(dx), int(dy)),
                        score=_alignment_score(after),
                        metric_before=before_dict,
                        metric_after=after_dict,
                        accepted=accepted,
                        rejection_reason=reason,
                    )
                )
        best = max(view_candidates, key=lambda item: item.score)
        best_by_view[view] = best
        candidates.extend(view_candidates)

    accepted_count = sum(1 for item in best_by_view.values() if item.accepted)
    return MaskAlignmentSweepReport(
        schema_version="mask_alignment_sweep_v1",
        max_offset_px=max_offset_px,
        step_px=step_px,
        candidates=tuple(sorted(candidates, key=lambda item: item.score, reverse=True)),
        best_by_view=best_by_view,
        aggregate_before=_aggregate_metrics(
            candidate.metric_before for candidate in best_by_view.values()
        ),
        aggregate_after=_aggregate_metrics(
            candidate.metric_after for candidate in best_by_view.values()
        ),
        accepted_count=accepted_count,
        warnings=tuple(warnings),
    )


def detect_view_role_permutation(
    reference_masks: Mapping[str, np.ndarray],
    candidate_masks: Mapping[str, np.ndarray],
    *,
    max_views: int = 8,
) -> ViewRolePermutationReport:
    """Find whether assigning candidate masks to different view roles improves IoU."""
    reference_views = tuple(reference_masks.keys())
    candidate_views = tuple(candidate_masks.keys())
    common_views = tuple(view for view in reference_views if view in candidate_masks)
    limited_refs = reference_views[:max_views]
    limited_candidates = candidate_views[:max_views]
    iou_matrix: dict[str, dict[str, float]] = {}
    for ref_view in limited_refs:
        row: dict[str, float] = {}
        reference = _as_mask(reference_masks[ref_view])
        for candidate_view in limited_candidates:
            candidate = _as_mask(candidate_masks[candidate_view])
            if reference.shape != candidate.shape:
                row[candidate_view] = 0.0
                continue
            row[candidate_view] = area_iou(reference, candidate)[0]
        iou_matrix[ref_view] = row

    identity_score = _assignment_score(
        {view: view for view in common_views},
        iou_matrix,
    )
    best_assignment: dict[str, str] = {}
    best_score = -1.0
    for assignment in _candidate_assignments(limited_refs, limited_candidates):
        score = _assignment_score(assignment, iou_matrix)
        if score > best_score:
            best_assignment = dict(assignment)
            best_score = score
    if best_score < 0.0:
        best_score = 0.0
    score_delta = float(best_score - identity_score)
    return ViewRolePermutationReport(
        reference_views=reference_views,
        candidate_views=candidate_views,
        assignment=best_assignment,
        identity_score=float(identity_score),
        best_score=float(best_score),
        score_delta=score_delta,
        swapped=bool(score_delta > 0.05 and any(k != v for k, v in best_assignment.items())),
        iou_matrix=iou_matrix,
    )


def shift_mask(mask: np.ndarray, *, dx: int, dy: int) -> np.ndarray:
    """Translate a mask by integer pixels with zero-filled uncovered regions."""
    source = _as_mask(mask)
    output = np.zeros(source.shape, dtype=bool)
    height, width = source.shape[:2]
    abs_dx = abs(int(dx))
    abs_dy = abs(int(dy))
    if abs_dx >= width or abs_dy >= height:
        return output
    src_x0 = max(0, -int(dx))
    src_x1 = width - max(0, int(dx))
    dst_x0 = max(0, int(dx))
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    src_y0 = max(0, -int(dy))
    src_y1 = height - max(0, int(dy))
    dst_y0 = max(0, int(dy))
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    output[dst_y0:dst_y1, dst_x0:dst_x1] = source[src_y0:src_y1, src_x0:src_x1]
    return output


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


def _silhouette_metric(
    view: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    min_area_iou: float,
    min_boundary_iou: float | None,
    max_signed_distance_loss: float | None,
) -> SilhouetteMetricResult:
    return silhouette_metric_result(
        reference,
        candidate,
        view=view,
        min_area_iou=min_area_iou,
        min_boundary_iou=min_boundary_iou,
        max_signed_distance_loss=max_signed_distance_loss,
    )


def _alignment_acceptance(
    before: SilhouetteMetricResult,
    after: SilhouetteMetricResult,
    *,
    min_area_iou_delta: float,
    min_boundary_iou_delta: float,
    max_signed_distance_loss_increase: float,
) -> tuple[bool, str]:
    area_delta = float(after.area_iou - before.area_iou)
    boundary_before = float(before.boundary_iou or 0.0)
    boundary_after = float(after.boundary_iou or 0.0)
    boundary_delta = boundary_after - boundary_before
    sdf_before = float(before.signed_distance_loss or 0.0)
    sdf_after = float(after.signed_distance_loss or 0.0)
    sdf_delta = sdf_after - sdf_before
    if area_delta < float(min_area_iou_delta):
        return False, f"area_iou_delta {area_delta:.6f} below threshold"
    if boundary_delta < float(min_boundary_iou_delta):
        return False, f"boundary_iou_delta {boundary_delta:.6f} below threshold"
    if sdf_delta > float(max_signed_distance_loss_increase):
        return False, f"signed_distance_loss_delta {sdf_delta:.6f} above threshold"
    return True, ""


def _alignment_score(metric: SilhouetteMetricResult) -> float:
    boundary = float(metric.boundary_iou or 0.0)
    sdf = float(metric.signed_distance_loss or 0.0)
    return float(metric.area_iou + 0.5 * boundary - 0.25 * sdf)


def _aggregate_metrics(rows: Any) -> dict[str, float]:
    payloads = [dict(row) for row in rows]
    if not payloads:
        return {
            "area_iou": 0.0,
            "boundary_iou": 0.0,
            "signed_distance_loss": 0.0,
        }
    return {
        "area_iou": _mean(payload.get("area_iou") for payload in payloads),
        "boundary_iou": _mean(payload.get("boundary_iou") for payload in payloads),
        "signed_distance_loss": _mean(
            payload.get("signed_distance_loss") for payload in payloads
        ),
    }


def _candidate_assignments(
    reference_views: tuple[str, ...],
    candidate_views: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    if not reference_views or not candidate_views:
        return ()
    size = min(len(reference_views), len(candidate_views))
    assignments = []
    for permuted in permutations(candidate_views, size):
        assignments.append(
            {reference_views[index]: permuted[index] for index in range(size)}
        )
    return tuple(assignments)


def _assignment_score(
    assignment: Mapping[str, str],
    iou_matrix: Mapping[str, Mapping[str, float]],
) -> float:
    if not assignment:
        return 0.0
    values = []
    for reference_view, candidate_view in assignment.items():
        values.append(float(iou_matrix.get(reference_view, {}).get(candidate_view, 0.0)))
    return float(sum(values) / len(values)) if values else 0.0


def _mean(values: Any) -> float:
    parsed = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            pass
    return float(sum(parsed) / len(parsed)) if parsed else 0.0


def _as_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., -1]
    if array.ndim != 2:
        raise ValueError("masks must be 2D arrays or image-like arrays with channels")
    return array.astype(bool, copy=False)


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
