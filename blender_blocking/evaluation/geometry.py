"""Geometry metrics for synthetic and mesh-backed reconstruction evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GeometryMetricReport:
    chamfer_l1: float | None = None
    chamfer_l2: float | None = None
    fscore_tau: float | None = None
    fscore_tolerance: float | None = None
    volumetric_iou: float | None = None
    normal_consistency: float | None = None
    surface_coverage: float | None = None
    ambiguity_gap: float | None = None
    sample_count_ref: int = 0
    sample_count_candidate: int = 0
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "chamfer_l1": self.chamfer_l1,
            "chamfer_l2": self.chamfer_l2,
            "fscore_tau": self.fscore_tau,
            "fscore_tolerance": self.fscore_tolerance,
            "volumetric_iou": self.volumetric_iou,
            "normal_consistency": self.normal_consistency,
            "surface_coverage": self.surface_coverage,
            "ambiguity_gap": self.ambiguity_gap,
            "sample_count_ref": self.sample_count_ref,
            "sample_count_candidate": self.sample_count_candidate,
            "warnings": list(self.warnings),
        }


def surface_distance_report(
    reference_points: Any,
    candidate_points: Any,
    *,
    tolerance: float = 0.01,
    reference_normals: Any = None,
    candidate_normals: Any = None,
    max_points: int | None = 20000,
) -> GeometryMetricReport:
    """Compute Chamfer/F-score/normal metrics from point samples."""
    np = _np()
    ref = _points(reference_points, max_points=max_points)
    cand = _points(candidate_points, max_points=max_points)
    warnings: list[str] = []
    if ref.size == 0 or cand.size == 0:
        return GeometryMetricReport(
            sample_count_ref=int(ref.shape[0]),
            sample_count_candidate=int(cand.shape[0]),
            warnings=("empty point sample for surface metric",),
        )
    ref_to_cand = nearest_distances(ref, cand)
    cand_to_ref = nearest_distances(cand, ref)
    chamfer_l1 = float(np.mean(ref_to_cand) + np.mean(cand_to_ref))
    chamfer_l2 = float(np.mean(ref_to_cand ** 2) + np.mean(cand_to_ref ** 2))
    recall = float(np.mean(ref_to_cand <= tolerance))
    precision = float(np.mean(cand_to_ref <= tolerance))
    fscore = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
    normal = None
    if reference_normals is not None and candidate_normals is not None:
        normal = normal_consistency(
            reference_points=ref,
            candidate_points=cand,
            reference_normals=reference_normals,
            candidate_normals=candidate_normals,
            max_points=max_points,
        )
    coverage = float(recall)
    return GeometryMetricReport(
        chamfer_l1=chamfer_l1,
        chamfer_l2=chamfer_l2,
        fscore_tau=float(fscore),
        fscore_tolerance=float(tolerance),
        normal_consistency=normal,
        surface_coverage=coverage,
        sample_count_ref=int(ref.shape[0]),
        sample_count_candidate=int(cand.shape[0]),
        warnings=tuple(warnings),
    )


def chamfer_distance(
    reference_points: Any,
    candidate_points: Any,
    *,
    squared: bool = True,
    max_points: int | None = 20000,
) -> float:
    np = _np()
    ref = _points(reference_points, max_points=max_points)
    cand = _points(candidate_points, max_points=max_points)
    if ref.size == 0 or cand.size == 0:
        return float("inf")
    ref_to_cand = nearest_distances(ref, cand)
    cand_to_ref = nearest_distances(cand, ref)
    if squared:
        ref_to_cand = ref_to_cand ** 2
        cand_to_ref = cand_to_ref ** 2
    return float(np.mean(ref_to_cand) + np.mean(cand_to_ref))


def fscore_at_tolerance(
    reference_points: Any,
    candidate_points: Any,
    *,
    tolerance: float,
    max_points: int | None = 20000,
) -> float:
    np = _np()
    ref = _points(reference_points, max_points=max_points)
    cand = _points(candidate_points, max_points=max_points)
    if ref.size == 0 or cand.size == 0:
        return 0.0
    ref_to_cand = nearest_distances(ref, cand)
    cand_to_ref = nearest_distances(cand, ref)
    recall = float(np.mean(ref_to_cand <= tolerance))
    precision = float(np.mean(cand_to_ref <= tolerance))
    return 0.0 if precision + recall <= 0 else float(2.0 * precision * recall / (precision + recall))


def volumetric_iou(reference_occupancy: Any, candidate_occupancy: Any) -> float:
    np = _np()
    ref = np.asarray(reference_occupancy).astype(bool, copy=False)
    cand = np.asarray(candidate_occupancy).astype(bool, copy=False)
    if ref.shape != cand.shape:
        raise ValueError("reference and candidate occupancy grids must have matching shapes")
    intersection = int(np.logical_and(ref, cand).sum())
    union = int(np.logical_or(ref, cand).sum())
    return 1.0 if union == 0 else float(intersection / union)


def normal_consistency(
    *,
    reference_points: Any,
    candidate_points: Any,
    reference_normals: Any,
    candidate_normals: Any,
    max_points: int | None = 20000,
) -> float:
    np = _np()
    ref = _points(reference_points, max_points=max_points)
    cand = _points(candidate_points, max_points=max_points)
    ref_normals = _normalize_vectors(_points(reference_normals, max_points=max_points))
    cand_normals = _normalize_vectors(_points(candidate_normals, max_points=max_points))
    if ref.size == 0 or cand.size == 0 or ref_normals.size == 0 or cand_normals.size == 0:
        return 0.0
    indices = nearest_indices(ref, cand)
    count = min(len(indices), len(ref_normals))
    if count == 0:
        return 0.0
    clipped = np.clip(indices[:count], 0, max(0, len(cand_normals) - 1))
    aligned = cand_normals[clipped]
    return float(np.mean(np.abs(np.sum(ref_normals[:count] * aligned, axis=1))))


def nearest_distances(points: Any, reference: Any, *, chunk_size: int = 4096) -> Any:
    np = _np()
    pts = _points(points, max_points=None)
    ref = _points(reference, max_points=None)
    if pts.size == 0 or ref.size == 0:
        return np.asarray([], dtype=float)
    try:
        from scipy.spatial import cKDTree

        distances, _indices = cKDTree(ref).query(pts, k=1)
        return distances.astype(float, copy=False)
    except Exception:
        distances = []
        for start in range(0, pts.shape[0], chunk_size):
            chunk = pts[start : start + chunk_size]
            delta = chunk[:, None, :] - ref[None, :, :]
            distances.append(np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1)))
        return np.concatenate(distances) if distances else np.asarray([], dtype=float)


def nearest_indices(points: Any, reference: Any, *, chunk_size: int = 4096) -> Any:
    np = _np()
    pts = _points(points, max_points=None)
    ref = _points(reference, max_points=None)
    if pts.size == 0 or ref.size == 0:
        return np.asarray([], dtype=int)
    try:
        from scipy.spatial import cKDTree

        _distances, indices = cKDTree(ref).query(pts, k=1)
        return indices.astype(int, copy=False)
    except Exception:
        indices = []
        for start in range(0, pts.shape[0], chunk_size):
            chunk = pts[start : start + chunk_size]
            delta = chunk[:, None, :] - ref[None, :, :]
            indices.append(np.argmin(np.sum(delta * delta, axis=2), axis=1))
        return np.concatenate(indices) if indices else np.asarray([], dtype=int)


def report_from_mapping(payload: Mapping[str, Any]) -> GeometryMetricReport:
    return GeometryMetricReport(
        chamfer_l1=_optional_float(payload.get("chamfer_l1")),
        chamfer_l2=_optional_float(payload.get("chamfer_l2", payload.get("chamfer"))),
        fscore_tau=_optional_float(payload.get("fscore_tau", payload.get("f_score"))),
        fscore_tolerance=_optional_float(payload.get("fscore_tolerance", payload.get("tau"))),
        volumetric_iou=_optional_float(payload.get("volumetric_iou")),
        normal_consistency=_optional_float(payload.get("normal_consistency")),
        surface_coverage=_optional_float(payload.get("surface_coverage")),
        ambiguity_gap=_optional_float(payload.get("ambiguity_gap")),
        sample_count_ref=int(payload.get("sample_count_ref", 0) or 0),
        sample_count_candidate=int(payload.get("sample_count_candidate", 0) or 0),
        warnings=tuple(str(item) for item in payload.get("warnings", ()) or ()),
    )


def _points(value: Any, *, max_points: int | None) -> Any:
    np = _np()
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.empty((0, 3), dtype=float)
    if array.ndim == 0:
        return np.empty((0, 3), dtype=float)
    if array.ndim == 1:
        if array.size % 3 == 0:
            array = array.reshape((-1, 3))
        else:
            array = array.reshape((1, -1))
    else:
        array = array.reshape((-1, array.shape[-1]))
    if array.shape[1] > 3:
        array = array[:, :3]
    if array.shape[1] < 3:
        pad = np.zeros((array.shape[0], 3 - array.shape[1]), dtype=float)
        array = np.concatenate([array, pad], axis=1)
    if max_points is not None and array.shape[0] > max_points:
        step = max(1, array.shape[0] // int(max_points))
        array = array[::step][: int(max_points)]
    return array


def _normalize_vectors(vectors: Any) -> Any:
    np = _np()
    arr = _points(vectors, max_points=None)
    norms = np.linalg.norm(arr, axis=1)
    valid = norms > 1e-12
    if not valid.any():
        return np.empty((0, 3), dtype=float)
    return arr[valid] / norms[valid, None]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _np() -> Any:
    import numpy as np

    return np
