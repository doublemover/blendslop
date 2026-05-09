"""Deterministic metric sensitivity probes for calibration baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from metrics.silhouette import silhouette_metric_result
except Exception:  # pragma: no cover - package import path
    from blender_blocking.metrics.silhouette import silhouette_metric_result

from .schemas import json_safe


@dataclass(frozen=True)
class SensitivityProbe:
    probe_id: str
    description: str
    metrics: Mapping[str, float | None]
    expected_direction: Mapping[str, str] = field(default_factory=dict)
    passed: bool = True
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "description": self.description,
            "metrics": json_safe(dict(self.metrics)),
            "expected_direction": dict(self.expected_direction),
            "passed": self.passed,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class MetricSensitivityReport:
    subject: str
    baseline_metrics: Mapping[str, float | None]
    probes: tuple[SensitivityProbe, ...]
    summary: Mapping[str, bool]
    warnings: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return all(probe.passed for probe in self.probes) and all(self.summary.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "baseline_metrics": json_safe(dict(self.baseline_metrics)),
            "probes": [probe.to_dict() for probe in self.probes],
            "summary": dict(self.summary),
            "warnings": list(self.warnings),
            "passed": self.passed,
        }


def silhouette_sensitivity_report(mask: Any) -> MetricSensitivityReport:
    """Generate standard silhouette perturbations and metric responses."""

    reference = _as_bool_mask(mask)
    baseline = _silhouette_metrics(reference, reference)
    perturbations = (
        (
            "shift_1px",
            "Translate the candidate by one pixel.",
            _translate(reference, dx=1, dy=0),
            {"area_iou": "decrease", "boundary_iou": "decrease", "signed_distance_loss": "increase"},
        ),
        (
            "shift_5px",
            "Translate the candidate by five pixels.",
            _translate(reference, dx=5, dy=0),
            {"area_iou": "decrease", "boundary_iou": "decrease", "signed_distance_loss": "increase"},
        ),
        (
            "erode_1px",
            "Remove a one-pixel shell from the candidate.",
            _binary_erosion(reference, radius=1),
            {"area_iou": "decrease", "boundary_iou": "decrease", "signed_distance_loss": "increase"},
        ),
        (
            "dilate_1px",
            "Add a one-pixel shell to the candidate.",
            _binary_dilation(reference, radius=1),
            {"area_iou": "decrease", "boundary_iou": "decrease", "signed_distance_loss": "increase"},
        ),
        (
            "false_island_far",
            "Add a small far-away false-positive island.",
            _add_false_island(reference),
            {"area_iou": "decrease", "boundary_iou": "decrease", "signed_distance_loss": "increase"},
        ),
    )
    probes = tuple(
        _probe_from_candidate(
            reference,
            baseline,
            probe_id=probe_id,
            description=description,
            candidate=candidate,
            expected_direction=expected_direction,
        )
        for probe_id, description, candidate, expected_direction in perturbations
    )
    by_id = {probe.probe_id: probe for probe in probes}
    summary = {
        "area_iou_shift_monotonic": _metric(by_id["shift_5px"], "area_iou")
        <= _metric(by_id["shift_1px"], "area_iou")
        <= _metric_mapping(baseline, "area_iou"),
        "boundary_iou_shift_monotonic": _metric(by_id["shift_5px"], "boundary_iou")
        <= _metric(by_id["shift_1px"], "boundary_iou")
        <= _metric_mapping(baseline, "boundary_iou"),
        "sdf_shift_monotonic": _metric(by_id["shift_5px"], "signed_distance_loss")
        >= _metric(by_id["shift_1px"], "signed_distance_loss")
        >= _metric_mapping(baseline, "signed_distance_loss"),
        "boundary_detects_shell_changes": (
            _drop(baseline, by_id["erode_1px"], "boundary_iou")
            >= 0.5 * _drop(baseline, by_id["erode_1px"], "area_iou")
        )
        and (
            _drop(baseline, by_id["dilate_1px"], "boundary_iou")
            >= 0.5 * _drop(baseline, by_id["dilate_1px"], "area_iou")
        ),
        "far_false_island_costs_sdf": _metric(by_id["false_island_far"], "signed_distance_loss")
        > 0.0,
    }
    warnings = tuple(
        f"sensitivity summary failed: {name}"
        for name, passed in summary.items()
        if not passed
    )
    return MetricSensitivityReport(
        subject="silhouette",
        baseline_metrics=baseline,
        probes=probes,
        summary=summary,
        warnings=warnings,
    )


def sensitivity_metric_table(report: MetricSensitivityReport) -> list[dict[str, object]]:
    """Return a flat table friendly to JSONL, Markdown, or console output."""

    rows = [
        {
            "probe_id": "baseline",
            "passed": True,
            **dict(report.baseline_metrics),
        }
    ]
    for probe in report.probes:
        rows.append(
            {
                "probe_id": probe.probe_id,
                "passed": probe.passed,
                **dict(probe.metrics),
            }
        )
    return rows


def _probe_from_candidate(
    reference: np.ndarray,
    baseline: Mapping[str, float | None],
    *,
    probe_id: str,
    description: str,
    candidate: np.ndarray,
    expected_direction: Mapping[str, str],
) -> SensitivityProbe:
    metrics = _silhouette_metrics(reference, candidate)
    warnings = []
    for metric_name, direction in expected_direction.items():
        baseline_value = _metric_mapping(baseline, metric_name)
        value = _metric_mapping(metrics, metric_name)
        if direction == "decrease" and value > baseline_value:
            warnings.append(f"{metric_name} increased from baseline")
        elif direction == "increase" and value < baseline_value:
            warnings.append(f"{metric_name} decreased from baseline")
    return SensitivityProbe(
        probe_id=probe_id,
        description=description,
        metrics=metrics,
        expected_direction=dict(expected_direction),
        passed=not warnings,
        warnings=tuple(warnings),
    )


def _silhouette_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | None]:
    result = silhouette_metric_result(
        reference,
        candidate,
        view="sensitivity",
        reference_probability=reference.astype(np.float32),
        candidate_probability=candidate.astype(np.float32),
    )
    payload = result.to_dict()
    return {
        "area_iou": _optional_float(payload.get("area_iou")),
        "boundary_iou": _optional_float(payload.get("boundary_iou")),
        "soft_iou": _optional_float(payload.get("soft_iou")),
        "signed_distance_loss": _optional_float(payload.get("signed_distance_loss")),
        "ref_area": _optional_float(payload.get("ref_area")),
        "render_area": _optional_float(payload.get("render_area")),
    }


def _translate(mask: np.ndarray, *, dx: int, dy: int) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    target = np.zeros_like(source)
    src_y0 = max(0, -dy)
    src_y1 = min(source.shape[0], source.shape[0] - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(source.shape[1], source.shape[1] - dx)
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        target[dst_y0:dst_y1, dst_x0:dst_x1] = source[src_y0:src_y1, src_x0:src_x1]
    return target


def _add_false_island(mask: np.ndarray, size: int = 3) -> np.ndarray:
    candidate = np.asarray(mask, dtype=bool).copy()
    height, width = candidate.shape
    side = max(1, min(int(size), height, width))
    corners = (
        (0, 0),
        (0, max(0, width - side)),
        (max(0, height - side), 0),
        (max(0, height - side), max(0, width - side)),
    )
    for y, x in corners:
        patch = candidate[y : y + side, x : x + side]
        if not patch.any():
            candidate[y : y + side, x : x + side] = True
            return candidate
    candidate[:side, :side] = True
    return candidate


def _binary_erosion(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(1, int(radius))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighbors = _neighbors(padded, result.shape)
        result = np.logical_and.reduce(neighbors)
    return result


def _binary_dilation(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(1, int(radius))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighbors = _neighbors(padded, result.shape)
        result = np.logical_or.reduce(neighbors)
    return result


def _neighbors(padded: np.ndarray, shape: Sequence[int]) -> list[np.ndarray]:
    height, width = int(shape[0]), int(shape[1])
    return [
        padded[dy : dy + height, dx : dx + width]
        for dy in range(3)
        for dx in range(3)
    ]


def _as_bool_mask(mask: Any) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("sensitivity masks must be 2D")
    if array.size == 0:
        raise ValueError("sensitivity masks cannot be empty")
    return array.astype(bool, copy=False)


def _drop(
    baseline: Mapping[str, float | None],
    probe: SensitivityProbe,
    metric_name: str,
) -> float:
    return _metric_mapping(baseline, metric_name) - _metric(probe, metric_name)


def _metric(probe: SensitivityProbe, metric_name: str) -> float:
    return _metric_mapping(probe.metrics, metric_name)


def _metric_mapping(metrics: Mapping[str, float | None], metric_name: str) -> float:
    value = metrics.get(metric_name)
    if value is None:
        return 0.0
    return float(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
