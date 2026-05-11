"""Build typed reconstruction targets from workflow images and constraints."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from constraints import (
    ConstraintSet,
    apply_constraints_to_masks,
    constraint_set_hash,
    constraint_set_to_payload,
    load_constraint_files,
    mask_extraction_hints,
)
from geometry.silhouette_pipeline import (
    build_uncertain_mask,
    canonicalize_silhouette,
    extract_silhouette_mask,
)

from .artifacts import hash_json, write_json
from .profile_bands import distributional_profile_bands, profile_distribution_summary
from .types import (
    Bounds2D,
    Bounds3D,
    ProfileBand,
    ProfileIntervalPx,
    ReconstructionTarget,
    UncertainProfileBand,
    ViewConstraint,
)


@dataclass(frozen=True)
class TargetBuildResult:
    """Target plus extracted intermediates needed for diagnostics."""

    target: ReconstructionTarget
    masks: Mapping[str, np.ndarray]
    confidences: Mapping[str, np.ndarray]
    probabilities: Mapping[str, np.ndarray]
    uncertainties: Mapping[str, Any]
    constraint_set: ConstraintSet
    constraint_reports: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    artifact_paths: Mapping[str, Path] = field(default_factory=dict)

    def to_manifest_fragment(self) -> dict[str, object]:
        return {
            "target": self.target.to_dict(),
            "views": sorted(self.masks),
            "probability_views": sorted(self.probabilities),
            "constraint_hash": constraint_set_hash(self.constraint_set),
            "constraint_reports": dict(self.constraint_reports),
            "warnings": list(self.warnings),
            "artifact_paths": {key: str(path) for key, path in self.artifact_paths.items()},
        }


def build_target_from_images(
    views: Mapping[str, np.ndarray],
    *,
    config: Any = None,
    constraint_files: Sequence[str | Path] = (),
    artifact_root: str | Path | None = None,
    bounds_minmax: Optional[tuple[Sequence[float], Sequence[float]]] = None,
    profile_samples: int = 100,
) -> TargetBuildResult:
    """Extract masks, apply constraints, and build a backend-neutral target."""
    if not views:
        raise ValueError("at least one view is required")

    warnings: list[str] = []
    root = Path(artifact_root) if artifact_root is not None else None
    constraint_set = (
        load_constraint_files(constraint_files) if constraint_files else ConstraintSet()
    )
    config_payload = config.to_dict() if hasattr(config, "to_dict") else {}
    extraction_config = getattr(config, "silhouette_extract_ref", None)
    canonical_config = getattr(config, "canonicalize", None)

    raw_masks: dict[str, np.ndarray] = {}
    uncertainties: dict[str, Any] = {}
    confidences: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    bboxes: dict[str, Bounds2D] = {}
    diagnostics: dict[str, Mapping[str, Any]] = {}

    for view, image in sorted(views.items()):
        overrides = mask_extraction_hints(constraint_set, view)
        selected = extract_silhouette_mask(image, extraction_config, **overrides)
        uncertain = build_uncertain_mask(image, extraction_config, **overrides)
        canonical = canonicalize_silhouette(selected.mask, canonical_config)
        raw_masks[view] = selected.mask
        uncertainties[view] = uncertain
        confidences[view] = uncertain.confidence
        probabilities[view] = uncertain.foreground_prob
        if selected.bbox is not None:
            bboxes[view] = Bounds2D.from_xyxy(selected.bbox.to_xyxy())
        diagnostics[view] = {
            "selected": selected.to_dict(),
            "canonical": canonical.to_dict(),
            "uncertainty": dict(uncertain.diagnostics),
        }
        if selected.bbox is None:
            warnings.append(f"{view} silhouette is empty")

    constraint_reports: dict[str, Mapping[str, Any]] = {}
    if not constraint_set.is_empty():
        constrained = apply_constraints_to_masks(
            raw_masks,
            constraint_set,
            confidences=confidences,
        )
        for view, result in constrained.items():
            raw_masks[view] = result.mask
            confidences[view] = result.confidence
            uncertainties[view] = _replace_uncertain_mask(
                uncertainties.get(view),
                hard_mask=result.mask,
                confidence=result.confidence,
                constraint_report=result.report,
            )
            probability = getattr(uncertainties[view], "foreground_prob", None)
            if probability is not None:
                probabilities[view] = probability
            constraint_reports[view] = result.report
            diagnostics[view] = {
                **dict(diagnostics.get(view, {})),
                "constraints": result.report,
                "uncertainty": dict(
                    getattr(uncertainties[view], "diagnostics", {}) or {}
                ),
            }
            bbox = _bbox_from_mask(result.mask)
            if bbox is not None:
                bboxes[view] = Bounds2D.from_xyxy(bbox)

    profile_bands = {
        view: tuple(
            distributional_profile_bands(
                mask,
                sample_count=profile_samples,
                view=view,
                probability=probabilities.get(view),
                confidence=confidences.get(view),
            )
        )
        for view, mask in raw_masks.items()
    }
    profile_distribution = {
        view: profile_distribution_summary(bands)
        for view, bands in profile_bands.items()
    }
    explicit_bounds = _bounds_from_minmax(bounds_minmax)
    inferred_bounds = _bounds_from_view_bboxes(bboxes, config)
    bounds = explicit_bounds or inferred_bounds
    bounds_source = (
        "explicit_minmax"
        if explicit_bounds is not None
        else "mask_bboxes"
        if inferred_bounds is not None
        else "missing"
    )
    constraints_payload = constraint_set_to_payload(constraint_set)
    target = ReconstructionTarget(
        constraints=_view_constraints(
            raw_masks,
            bboxes=bboxes,
            uncertainties=uncertainties,
            diagnostics=diagnostics,
        ),
        profile_bands=profile_bands,
        bounds=bounds,
        config_hash=hash_json(config_payload),
        constraint_hash=hash_json(constraints_payload),
        artifact_root=root,
        extras={
            "constraint_payload": constraints_payload,
            "view_diagnostics": diagnostics,
            "uncertainty_views": sorted(uncertainties),
            "probability_views": sorted(probabilities),
            "profile_samples": int(profile_samples),
            "profile_band_distribution": profile_distribution,
            "bounds_source": bounds_source,
        },
    )
    artifact_paths = _write_target_artifacts(
        root,
        target=target,
        masks=raw_masks,
        confidences=confidences,
        probabilities=probabilities,
        diagnostics=diagnostics,
        constraints_payload=constraints_payload,
    )
    return TargetBuildResult(
        target=target,
        masks=raw_masks,
        confidences=confidences,
        probabilities=probabilities,
        uncertainties=uncertainties,
        constraint_set=constraint_set,
        constraint_reports=constraint_reports,
        warnings=tuple(warnings),
        artifact_paths=artifact_paths,
    )


def mask_to_profile_bands(
    mask: np.ndarray,
    *,
    sample_count: int,
    view: str = "",
    probability: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
) -> tuple[ProfileBand, ...]:
    """Sample all foreground intervals per row instead of one width per height."""
    mask_bool = np.asarray(mask).astype(bool, copy=False)
    if mask_bool.ndim != 2:
        raise ValueError("mask must be 2D")
    probability_map = _optional_profile_map(probability, mask_bool.shape, "probability")
    confidence_map = _optional_profile_map(confidence, mask_bool.shape, "confidence")
    height, _width = mask_bool.shape
    if height == 0:
        return ()
    count = max(1, int(sample_count))
    if count == 1:
        rows = np.array([height // 2], dtype=np.int64)
    else:
        rows = np.linspace(0, height - 1, count).round().astype(np.int64)
    bands = []
    denom = max(1, height - 1)
    for row in rows:
        row_index = int(row)
        row_mask = mask_bool[row_index, :]
        probability_row = None if probability_map is None else probability_map[row_index, :]
        confidence_row = None if confidence_map is None else confidence_map[row_index, :]
        intervals = _row_intervals(row_mask, confidence=confidence_row)
        dominant = max(intervals, key=lambda item: item.width, default=None)
        total_width = float(sum(interval.width for interval in intervals))
        holes = _holes_between_intervals(intervals, confidence=confidence_row)
        moments = _row_moments(intervals)
        uncertainty_moments, center_std, width_std = _row_uncertainty_moments(
            row_mask,
            probability=probability_row,
            confidence=confidence_row,
        )
        moments.update(uncertainty_moments)
        band_kwargs = dict(
            t=float(1.0 - float(row) / denom),
            intervals=tuple(intervals),
            center_x=None if dominant is None else dominant.center,
            width_px=total_width,
            holes=tuple(holes),
            moments=moments,
            confidence=_row_band_confidence(row_mask, confidence_row),
            source_view=view,
        )
        if probability_map is not None or confidence_map is not None:
            bands.append(
                UncertainProfileBand(
                    **band_kwargs,
                    center_std=center_std,
                    width_std=width_std,
                )
            )
        else:
            bands.append(ProfileBand(**band_kwargs))
    return tuple(bands)


def _optional_profile_map(
    values: Optional[np.ndarray],
    shape: tuple[int, int],
    name: str,
) -> Optional[np.ndarray]:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} shape must match mask shape")
    return np.clip(array, 0.0, 1.0)


def _row_intervals(
    row: np.ndarray,
    *,
    confidence: Optional[np.ndarray] = None,
) -> tuple[ProfileIntervalPx, ...]:
    values = np.asarray(row).astype(bool, copy=False)
    if values.size == 0 or not values.any():
        return ()
    confidence_values = (
        None if confidence is None else np.asarray(confidence, dtype=np.float32)
    )
    if confidence_values is not None and confidence_values.shape != values.shape:
        raise ValueError("confidence row shape must match mask row shape")
    padded = np.pad(values.astype(np.int8), (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0]
    intervals = []
    for start, stop in zip(starts, stops):
        interval_confidence = 1.0
        if confidence_values is not None and stop > start:
            interval_confidence = float(np.mean(confidence_values[start:stop]))
        intervals.append(
            ProfileIntervalPx(
                float(start),
                float(stop),
                confidence=interval_confidence,
                source="mask",
            )
        )
    return tuple(intervals)


def _holes_between_intervals(
    intervals: Sequence[ProfileIntervalPx],
    *,
    confidence: Optional[np.ndarray] = None,
) -> tuple[ProfileIntervalPx, ...]:
    holes = []
    confidence_values = (
        None if confidence is None else np.asarray(confidence, dtype=np.float32)
    )
    for left, right in zip(intervals, intervals[1:]):
        if right.x0 > left.x1:
            start = int(round(left.x1))
            stop = int(round(right.x0))
            hole_confidence = 1.0
            if confidence_values is not None and stop > start:
                hole_confidence = float(np.mean(confidence_values[start:stop]))
            holes.append(
                ProfileIntervalPx(
                    float(left.x1),
                    float(right.x0),
                    confidence=hole_confidence,
                    source="hole",
                )
            )
    return tuple(holes)


def _row_moments(intervals: Sequence[ProfileIntervalPx]) -> dict[str, float]:
    if not intervals:
        return {"component_count": 0.0, "total_width": 0.0}
    widths = np.asarray([interval.width for interval in intervals], dtype=float)
    centers = np.asarray([interval.center for interval in intervals], dtype=float)
    total = float(widths.sum())
    weighted_center = float(np.average(centers, weights=widths)) if total else 0.0
    return {
        "component_count": float(len(intervals)),
        "total_width": total,
        "weighted_center_x": weighted_center,
    }


def _row_band_confidence(
    row_mask: np.ndarray,
    confidence: Optional[np.ndarray],
) -> float:
    mask = np.asarray(row_mask).astype(bool, copy=False)
    if not mask.any():
        return 0.0
    if confidence is None:
        return 1.0
    values = np.asarray(confidence, dtype=np.float32)
    return float(np.mean(values[mask])) if values.shape == mask.shape else 0.0


def _row_uncertainty_moments(
    row_mask: np.ndarray,
    *,
    probability: Optional[np.ndarray],
    confidence: Optional[np.ndarray],
) -> tuple[dict[str, float], float, float]:
    moments: dict[str, float] = {}
    mask = np.asarray(row_mask).astype(bool, copy=False)
    center_std = 0.0
    width_std = 0.0

    if probability is not None:
        prob = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
        mass = float(prob.sum())
        width_variance = float(np.sum(prob * (1.0 - prob)))
        width_std = float(np.sqrt(max(0.0, width_variance)))
        moments["probability_width_px"] = mass
        moments["probability_width_variance"] = width_variance
        if mass > 0.0:
            positions = np.arange(prob.size, dtype=np.float64) + 0.5
            center = float(np.average(positions, weights=prob))
            variance = float(np.average((positions - center) ** 2, weights=prob))
            center_std = float(np.sqrt(max(0.0, variance / max(mass, 1.0))))
            moments["probability_center_x"] = center
            moments["probability_center_variance"] = variance

    if confidence is not None:
        conf = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
        moments["row_confidence_mean"] = float(np.mean(conf)) if conf.size else 0.0
        if mask.any() and conf.shape == mask.shape:
            moments["foreground_confidence_mean"] = float(np.mean(conf[mask]))
        if probability is None and mask.any() and conf.shape == mask.shape:
            width_std = float(
                np.sqrt(max(0.0, np.sum((1.0 - conf[mask]) * conf[mask])))
            )

    return moments, center_std, width_std


def _replace_uncertain_mask(
    uncertain: Any,
    *,
    hard_mask: np.ndarray,
    confidence: np.ndarray,
    constraint_report: Optional[Mapping[str, Any]] = None,
) -> Any:
    if uncertain is None:
        return None
    hard = np.asarray(hard_mask).astype(bool, copy=True)
    conf = np.asarray(confidence, dtype=np.float32).copy()
    probability = np.asarray(
        getattr(uncertain, "foreground_prob", hard.astype(np.float32)),
        dtype=np.float32,
    ).copy()
    if probability.shape == hard.shape:
        previous_hard = np.asarray(
            getattr(uncertain, "hard_mask", hard),
        ).astype(bool, copy=False)
        if previous_hard.shape == hard.shape:
            probability[hard & ~previous_hard] = 1.0
            probability[~hard & previous_hard] = 0.0
    diagnostics = dict(getattr(uncertain, "diagnostics", {}) or {})
    if constraint_report is not None:
        diagnostics["constraint_report"] = dict(constraint_report)
    try:
        return replace(
            uncertain,
            foreground_prob=probability,
            hard_mask=hard,
            confidence=conf,
            diagnostics=diagnostics,
        )
    except TypeError:
        return uncertain


def _view_constraints(
    masks: Mapping[str, np.ndarray],
    *,
    bboxes: Mapping[str, Bounds2D],
    uncertainties: Mapping[str, Any],
    diagnostics: Mapping[str, Mapping[str, Any]],
) -> tuple[ViewConstraint, ...]:
    from .targets import make_axis_camera

    constraints = []
    for view, mask in sorted(masks.items()):
        height, width = mask.shape
        constraints.append(
            ViewConstraint(
                view=view,
                mask=mask,
                camera=make_axis_camera(view, (int(width), int(height))),
                bbox=bboxes.get(view),
                uncertainty=uncertainties.get(view),
                diagnostics=diagnostics.get(view, {}),
            )
        )
    return tuple(constraints)


def _bounds_from_minmax(
    bounds_minmax: Optional[tuple[Sequence[float], Sequence[float]]],
) -> Optional[Bounds3D]:
    if bounds_minmax is None:
        return None
    minimum, maximum = bounds_minmax
    return Bounds3D.from_min_max(minimum, maximum)


def _bounds_from_view_bboxes(
    bboxes: Mapping[str, Bounds2D],
    config: Any,
) -> Optional[Bounds3D]:
    if not bboxes:
        return None
    scale = _unit_scale(config)

    front = bboxes.get("front")
    side = bboxes.get("side")
    top = bboxes.get("top")

    width = float(front.width) * scale if front is not None else None
    depth = float(side.width) * scale if side is not None else None
    height_candidates = []
    if front is not None:
        height_candidates.append(float(front.height) * scale)
    if side is not None:
        height_candidates.append(float(side.height) * scale)
    height = max(height_candidates) if height_candidates else None

    if top is not None:
        top_width = float(top.width) * scale
        top_depth = float(top.height) * scale
        if width is None:
            width = top_width
        if depth is None:
            depth = top_depth

    if width is None and depth is not None:
        width = depth
    if depth is None and width is not None:
        depth = width
    if height is None and (width is not None or depth is not None):
        height = max(float(width or 0.0), float(depth or 0.0), scale)

    if width is None or depth is None or height is None:
        return None
    if width <= 0.0 or depth <= 0.0 or height <= 0.0:
        return None
    return Bounds3D(
        min_x=-float(width) * 0.5,
        max_x=float(width) * 0.5,
        min_y=-float(depth) * 0.5,
        max_y=float(depth) * 0.5,
        min_z=0.0,
        max_z=float(height),
    )


def _unit_scale(config: Any) -> float:
    reconstruction = getattr(config, "reconstruction", None)
    value = getattr(reconstruction, "unit_scale", 0.01)
    try:
        scale = float(value)
    except (TypeError, ValueError):
        scale = 0.01
    return scale if scale > 0.0 else 0.01


def _bbox_from_mask(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(np.asarray(mask).astype(bool, copy=False))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _write_target_artifacts(
    root: Optional[Path],
    *,
    target: ReconstructionTarget,
    masks: Mapping[str, np.ndarray],
    confidences: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, Mapping[str, Any]],
    constraints_payload: Mapping[str, Any],
) -> Mapping[str, Path]:
    if root is None:
        return {}
    target_dir = root / "t"
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["target"] = write_json(target_dir / "target.json", target.to_dict())
    paths["constraints"] = write_json(target_dir / "constraints.json", constraints_payload)
    paths["diagnostics"] = write_json(target_dir / "diag.json", diagnostics)
    for view, mask in masks.items():
        mask_path = target_dir / f"{view}-mask.npy"
        np.save(mask_path, np.asarray(mask).astype(bool, copy=False))
        paths[f"{view}_mask"] = mask_path
    for view, confidence in confidences.items():
        conf_path = target_dir / f"{view}-confidence.npy"
        np.save(conf_path, np.asarray(confidence, dtype=np.float32))
        paths[f"{view}_confidence"] = conf_path
    for view, probability in probabilities.items():
        prob_path = target_dir / f"{view}-probability.npy"
        np.save(prob_path, np.asarray(probability, dtype=np.float32))
        paths[f"{view}_probability"] = prob_path
    return paths
