"""Bounds, projection, and axis diagnostics for reconstruction candidates."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ExperimentResult, json_safe

try:
    from blender_blocking.config import BlockingConfig
    from blender_blocking.integration.image_processing.image_loader import load_image
    from blender_blocking.validation.silhouette_iou import (
        canonicalize_mask,
        compute_mask_iou,
        mask_from_image_array,
    )
except ImportError:  # pragma: no cover - direct blender_blocking/ execution
    from config import BlockingConfig
    from integration.image_processing.image_loader import load_image
    from validation.silhouette_iou import canonicalize_mask, compute_mask_iou, mask_from_image_array


VIEW_AXES = {
    "front": (0, 2),
    "side": (1, 2),
    "top": (0, 1),
}


def obj_vertices(path: Path) -> tuple[np.ndarray, int]:
    vertices = []
    malformed = 0
    if path is None or not Path(path).exists():
        return np.empty((0, 3), dtype=float), 0
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                malformed += 1
                continue
            try:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                malformed += 1
    return np.asarray(vertices, dtype=float), malformed


def obj_bounds(path: Path) -> dict[str, object]:
    vertices, malformed = obj_vertices(path)
    if vertices.size == 0:
        return {
            "path": Path(path).as_posix() if path else "",
            "exists": bool(path and Path(path).exists()),
            "vertex_count": 0,
            "malformed_vertices": malformed,
            "bounds": None,
            "size": [0.0, 0.0, 0.0],
            "center": [0.0, 0.0, 0.0],
        }
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    size = maximum - minimum
    center = (minimum + maximum) * 0.5
    return {
        "path": Path(path).as_posix(),
        "exists": True,
        "vertex_count": int(len(vertices)),
        "malformed_vertices": malformed,
        "bounds": _bounds_dict(minimum, maximum),
        "size": [float(v) for v in size],
        "center": [float(v) for v in center],
    }


def build_bounds_debug_report(
    result: ExperimentResult,
    *,
    output_dir: Path,
    reference_paths: Mapping[str, Path] | None = None,
    render_paths: Mapping[str, Path] | None = None,
    target_payload: Mapping[str, Any] | None = None,
    write_files: bool = True,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    reference_paths = reference_paths or result.reference_paths
    render_paths = render_paths or result.render_paths
    mesh_path = _mesh_path(result)
    mesh_summary = obj_bounds(mesh_path) if mesh_path else obj_bounds(Path(""))
    vertices, _malformed = obj_vertices(mesh_path) if mesh_path else (np.empty((0, 3)), 0)
    volume = _volume_summary(result)
    target = _target_summary(target_payload, reference_paths)
    reference_masks = {
        view: _mask_for_path(path)
        for view, path in reference_paths.items()
        if Path(path).exists()
    }
    render_masks = {
        view: _mask_for_path(path)
        for view, path in render_paths.items()
        if Path(path).exists()
    }
    render = {
        "paths": {view: Path(path).as_posix() for view, path in render_paths.items()},
        "mask_bboxes": {
            view: _mask_summary(mask)["bbox_xyxy"] for view, mask in render_masks.items()
        },
        "mask_areas": {
            view: _mask_summary(mask)["area"] for view, mask in render_masks.items()
        },
    }
    target_bounds = _bounds_from_target_or_volume(target, volume, mesh_summary)
    axis_search = _axis_permutation_search(vertices, reference_masks, target_bounds)
    comparisons = {
        "mesh_vs_target_bounds": _bounds_agreement(mesh_summary.get("bounds"), target_bounds),
        "mesh_vs_volume_bounds": _bounds_agreement(mesh_summary.get("bounds"), volume.get("bounds")),
        "render_bbox_vs_reference_bbox": _bbox_comparisons(reference_masks, render_masks),
    }
    diagnosis = _diagnose(axis_search, comparisons, target, mesh_summary, result)
    report = {
        "schema_version": "bounds_debug_v1",
        "case_id": result.case_id,
        "variant_id": result.variant_id,
        "mode": result.mode,
        "target": target,
        "volume": volume,
        "mesh": mesh_summary,
        "render": render,
        "comparisons": comparisons,
        "axis_permutation_search": axis_search,
        "diagnosis": diagnosis,
    }
    if write_files:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "bounds-debug.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        (output_dir / "axis-permutation-search.json").write_text(
            json.dumps(axis_search, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        (output_dir / "bounds-debug.md").write_text(
            _bounds_debug_markdown(report),
            encoding="utf-8",
        )
    return report


def project_world_points_to_view(
    points: np.ndarray,
    *,
    view: str,
    bounds: Mapping[str, Any] | None,
    output_size: tuple[int, int] = (256, 256),
    dilation_radius: int = 1,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.zeros((output_size[1], output_size[0]), dtype=bool)
    axes = VIEW_AXES.get(view)
    if axes is None:
        raise ValueError(f"unknown view for projection: {view}")
    minimum, maximum = _bounds_min_max(bounds, points)
    mins = minimum[list(axes)]
    maxs = maximum[list(axes)]
    span = np.maximum(maxs - mins, 1e-9)
    plane = points[:, list(axes)]
    norm = (plane - mins) / span
    width, height = output_size
    xs = np.rint(norm[:, 0] * (width - 1)).astype(np.int64)
    ys = np.rint((1.0 - norm[:, 1]) * (height - 1)).astype(np.int64)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    mask = np.zeros((height, width), dtype=bool)
    mask[ys[valid], xs[valid]] = True
    if dilation_radius > 0 and mask.any():
        try:
            import cv2

            size = int(dilation_radius) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        except Exception:
            mask = _numpy_dilate(mask, dilation_radius)
    return mask


def _axis_permutation_search(
    vertices: np.ndarray,
    reference_masks: Mapping[str, np.ndarray],
    bounds: Mapping[str, Any] | None,
) -> dict[str, object]:
    if vertices.size == 0 or not reference_masks:
        return {"baseline": {}, "best": {}, "candidates": []}
    sample = _sample_vertices(vertices, 12000)
    candidates = []
    baseline = _score_transform(
        sample,
        reference_masks,
        bounds,
        permutation=(0, 1, 2),
        flips=(1, 1, 1),
        scale_mode="identity",
        translation_mode="none",
    )
    candidates.append(baseline)
    for permutation in itertools.permutations((0, 1, 2)):
        for flips in itertools.product((-1, 1), repeat=3):
            for scale_mode in ("identity", "half", "double", "fit_bounds"):
                for translation_mode in ("none", "center", "bottom"):
                    if (
                        permutation == (0, 1, 2)
                        and flips == (1, 1, 1)
                        and scale_mode == "identity"
                        and translation_mode == "none"
                    ):
                        continue
                    candidates.append(
                        _score_transform(
                            sample,
                            reference_masks,
                            bounds,
                            permutation=permutation,
                            flips=flips,
                            scale_mode=scale_mode,
                            translation_mode=translation_mode,
                        )
                    )
    candidates = sorted(
        candidates,
        key=lambda item: (float(item.get("min_iou", 0.0)), float(item.get("average_iou", 0.0))),
        reverse=True,
    )
    return {
        "baseline": baseline,
        "best": candidates[0] if candidates else {},
        "candidates": candidates[:25],
    }


def _score_transform(
    vertices: np.ndarray,
    reference_masks: Mapping[str, np.ndarray],
    bounds: Mapping[str, Any] | None,
    *,
    permutation: tuple[int, int, int],
    flips: tuple[int, int, int],
    scale_mode: str,
    translation_mode: str,
) -> dict[str, object]:
    transformed = vertices[:, permutation] * np.asarray(flips, dtype=float)
    transformed = _apply_scale(transformed, bounds, scale_mode)
    transformed = _apply_translation(transformed, bounds, translation_mode)
    per_view = {}
    ious = []
    canonical_config = BlockingConfig().canonicalize
    for view, ref_mask in reference_masks.items():
        projected = project_world_points_to_view(
            transformed,
            view=view,
            bounds=bounds,
            output_size=(canonical_config.output_size, canonical_config.output_size),
        )
        anchor = "center" if view == "top" else canonical_config.anchor
        ref_canon = canonicalize_mask(
            ref_mask,
            output_size=canonical_config.output_size,
            padding_frac=canonical_config.padding_frac,
            anchor=anchor,
        )
        proj_canon = canonicalize_mask(
            projected,
            output_size=canonical_config.output_size,
            padding_frac=canonical_config.padding_frac,
            anchor=anchor,
        )
        iou = compute_mask_iou(ref_canon, proj_canon).iou
        per_view[view] = iou
        ious.append(iou)
    return {
        "permutation": "".join("xyz"[index] for index in permutation),
        "flips": list(flips),
        "scale_mode": scale_mode,
        "translation_mode": translation_mode,
        "average_iou": float(sum(ious) / len(ious)) if ious else 0.0,
        "min_iou": float(min(ious)) if ious else 0.0,
        "per_view": per_view,
    }


def _target_summary(
    target_payload: Mapping[str, Any] | None,
    reference_paths: Mapping[str, Path],
) -> dict[str, object]:
    target_payload = target_payload or {}
    target = target_payload.get("target", target_payload)
    constraints = target.get("constraints", ()) if isinstance(target, Mapping) else ()
    rows = []
    if isinstance(constraints, Sequence):
        for item in constraints:
            if isinstance(item, Mapping):
                mask = item.get("mask")
                bbox = item.get("bbox")
                rows.append(
                    {
                        "view": item.get("view", ""),
                        "mask_shape": list(getattr(mask, "shape", ()) or item.get("mask_shape", ())),
                        "bbox_xyxy": _bbox_payload(bbox),
                        "camera": json_safe(item.get("camera", {})),
                        "diagnostics": json_safe(item.get("diagnostics", {})),
                    }
                )
    if not rows:
        for view, path in reference_paths.items():
            mask = _mask_for_path(path)
            summary = _mask_summary(mask)
            rows.append(
                {
                    "view": view,
                    "mask_shape": list(mask.shape),
                    "bbox_xyxy": summary["bbox_xyxy"],
                    "bbox_area": summary["bbox_area"],
                    "mask_area": summary["area"],
                    "camera": {"view_name": view, "axis": {"front": "y", "side": "x", "top": "z"}.get(view, "custom")},
                    "diagnostics": {},
                }
            )
    return {
        "bounds": json_safe(target.get("bounds") if isinstance(target, Mapping) else None),
        "constraints": rows,
    }


def _volume_summary(result: ExperimentResult) -> dict[str, object]:
    metadata_path = _find_path(result, "volume_metadata")
    if metadata_path is None:
        volume = result.backend_result.get("volume_path") if isinstance(result.backend_result, Mapping) else None
        if volume:
            candidate = Path(str(volume)) / "volume.json"
            if candidate.exists():
                metadata_path = candidate
    if metadata_path is None or not metadata_path.exists():
        return {"metadata_path": "", "bounds": None, "transform": None, "shape": [], "active_voxels": 0}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"metadata_path": metadata_path.as_posix(), "error": str(exc)}
    return {
        "metadata_path": metadata_path.as_posix(),
        "bounds": payload.get("bounds"),
        "transform": payload.get("transform"),
        "shape": payload.get("shape", []),
        "active_voxels": payload.get("active_voxels", 0),
        "occupancy_fraction": payload.get("stats", {}).get("occupancy_fraction"),
    }


def _mask_for_path(path: Path) -> np.ndarray:
    image = load_image(str(path))
    return mask_from_image_array(image)


def _mask_summary(mask: np.ndarray) -> dict[str, object]:
    ys, xs = np.where(np.asarray(mask).astype(bool, copy=False))
    area = int(xs.size)
    if area == 0:
        bbox = None
        bbox_area = 0
    else:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        bbox_area = int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return {
        "shape": list(mask.shape),
        "area": area,
        "coverage": float(area / max(1, mask.size)),
        "bbox_xyxy": bbox,
        "bbox_area": bbox_area,
    }


def _bounds_agreement(mesh_bounds: Any, target_bounds: Any) -> dict[str, object]:
    mesh_min, mesh_max = _bounds_min_max(mesh_bounds, None)
    target_min, target_max = _bounds_min_max(target_bounds, None)
    if mesh_min is None or target_min is None:
        return {"agreement": 0.0, "reason": "missing_bounds"}
    mesh_size = np.maximum(mesh_max - mesh_min, 1e-9)
    target_size = np.maximum(target_max - target_min, 1e-9)
    ratios = np.minimum(mesh_size / target_size, target_size / mesh_size)
    center_delta = np.linalg.norm(((mesh_min + mesh_max) * 0.5) - ((target_min + target_max) * 0.5))
    target_diag = max(float(np.linalg.norm(target_size)), 1e-9)
    center_score = max(0.0, 1.0 - center_delta / target_diag)
    agreement = float(np.clip(np.mean(ratios) * center_score, 0.0, 1.0))
    return {
        "agreement": agreement,
        "axis_size_ratios": [float(v) for v in ratios],
        "center_delta": float(center_delta),
    }


def _bbox_comparisons(
    reference_masks: Mapping[str, np.ndarray],
    render_masks: Mapping[str, np.ndarray],
) -> dict[str, object]:
    comparisons = {}
    for view, ref in reference_masks.items():
        render = render_masks.get(view)
        if render is None:
            comparisons[view] = {"agreement": 0.0, "reason": "missing_render_mask"}
            continue
        comparisons[view] = {
            "reference": _mask_summary(ref),
            "render": _mask_summary(render),
        }
    return comparisons


def _diagnose(
    axis_search: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    target: Mapping[str, Any],
    mesh: Mapping[str, Any],
    result: ExperimentResult,
) -> list[str]:
    diagnosis = []
    baseline = axis_search.get("baseline", {}) if isinstance(axis_search, Mapping) else {}
    best = axis_search.get("best", {}) if isinstance(axis_search, Mapping) else {}
    if _float(best.get("average_iou")) - _float(baseline.get("average_iou")) > 0.25:
        diagnosis.append("likely_axis_order_or_flip_bug")
    mesh_target = comparisons.get("mesh_vs_target_bounds", {}) if isinstance(comparisons, Mapping) else {}
    if _float(mesh_target.get("agreement"), 1.0) < 0.25:
        diagnosis.append("likely_scale_or_bounds_bug")
    if result.mode == "visual_hull_voxel" and result.min_iou < 0.2 and int(mesh.get("vertex_count", 0) or 0) > 0:
        diagnosis.append("visual_hull_mesh_exists_but_render_iou_catastrophic")
    for constraint in target.get("constraints", ()):
        if isinstance(constraint, Mapping):
            coverage = _float(constraint.get("mask_area"), 1.0) / max(1.0, np.prod(constraint.get("mask_shape") or (1, 1)))
            if coverage < 0.001:
                diagnosis.append("likely_mask_selection_bug")
                break
    return sorted(set(diagnosis))


def _bounds_debug_markdown(report: Mapping[str, Any]) -> str:
    diagnosis = ", ".join(report.get("diagnosis", ())) or "no specific diagnosis"
    mesh = report.get("mesh", {})
    volume = report.get("volume", {})
    comparisons = report.get("comparisons", {})
    axis = report.get("axis_permutation_search", {})
    best = axis.get("best", {}) if isinstance(axis, Mapping) else {}
    lines = [
        "# Bounds Debug",
        "",
        f"Diagnosis: `{diagnosis}`",
        "",
        "## Mesh",
        "",
        f"- Path: `{mesh.get('path', '') if isinstance(mesh, Mapping) else ''}`",
        f"- Vertices: `{mesh.get('vertex_count', 0) if isinstance(mesh, Mapping) else 0}`",
        f"- Bounds: `{json.dumps(mesh.get('bounds'), default=str) if isinstance(mesh, Mapping) else ''}`",
        "",
        "## Volume",
        "",
        f"- Metadata: `{volume.get('metadata_path', '') if isinstance(volume, Mapping) else ''}`",
        f"- Bounds: `{json.dumps(volume.get('bounds'), default=str) if isinstance(volume, Mapping) else ''}`",
        "",
        "## Comparisons",
        "",
        f"- Mesh vs target: `{json.dumps(comparisons.get('mesh_vs_target_bounds', {}), default=str) if isinstance(comparisons, Mapping) else ''}`",
        f"- Mesh vs volume: `{json.dumps(comparisons.get('mesh_vs_volume_bounds', {}), default=str) if isinstance(comparisons, Mapping) else ''}`",
        "",
        "## Best Axis Diagnostic",
        "",
        f"`{json.dumps(best, default=str)}`",
        "",
        "## Suspected Files",
        "",
        "- `blender_blocking/reconstruction/point_cloud.py`",
        "- `blender_blocking/volume/meshing.py`",
        "- `blender_blocking/volume/contracts.py`",
        "- `blender_blocking/reconstruction/target_builder.py`",
        "- `blender_blocking/reconstruction/targets.py`",
        "- `blender_blocking/integration/blender_ops/camera_framing.py`",
        "- `blender_blocking/test_e2e_validation.py`",
    ]
    return "\n".join(lines) + "\n"


def _mesh_path(result: ExperimentResult) -> Path | None:
    path = result.artifacts.get("mesh_obj") or result.artifacts.get("mesh")
    if path:
        return Path(path)
    backend = result.backend_result if isinstance(result.backend_result, Mapping) else {}
    selected = backend.get("selected") if isinstance(backend, Mapping) else None
    source = selected if isinstance(selected, Mapping) else backend
    if isinstance(source, Mapping):
        mesh = source.get("mesh_path")
        if mesh:
            return Path(str(mesh))
        artifacts = source.get("artifacts", {})
        if isinstance(artifacts, Mapping) and artifacts.get("mesh_obj"):
            return Path(str(artifacts["mesh_obj"]))
    return None


def _find_path(result: ExperimentResult, key: str) -> Path | None:
    if key in result.artifacts:
        return Path(result.artifacts[key])
    return _find_path_in_mapping(result.backend_result, key)


def _find_path_in_mapping(value: Any, key: str) -> Path | None:
    if isinstance(value, Mapping):
        if key in value and value[key]:
            return Path(str(value[key]))
        for nested in value.values():
            found = _find_path_in_mapping(nested, key)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            found = _find_path_in_mapping(nested, key)
            if found is not None:
                return found
    return None


def _bounds_from_target_or_volume(
    target: Mapping[str, Any],
    volume: Mapping[str, Any],
    mesh: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if target.get("bounds"):
        return target.get("bounds")
    if volume.get("bounds"):
        return volume.get("bounds")
    return mesh.get("bounds")


def _bounds_min_max(bounds: Any, points: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if isinstance(bounds, Mapping):
        if all(key in bounds for key in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")):
            return (
                np.array([bounds["min_x"], bounds["min_y"], bounds["min_z"]], dtype=float),
                np.array([bounds["max_x"], bounds["max_y"], bounds["max_z"]], dtype=float),
            )
        if "minimum" in bounds and "maximum" in bounds:
            return np.asarray(bounds["minimum"], dtype=float), np.asarray(bounds["maximum"], dtype=float)
    if points is not None and points.size:
        return points.min(axis=0), points.max(axis=0)
    return None, None


def _bounds_dict(minimum: np.ndarray, maximum: np.ndarray) -> dict[str, float]:
    return {
        "min_x": float(minimum[0]),
        "max_x": float(maximum[0]),
        "min_y": float(minimum[1]),
        "max_y": float(maximum[1]),
        "min_z": float(minimum[2]),
        "max_z": float(maximum[2]),
    }


def _bbox_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if all(key in value for key in ("x0", "y0", "x1", "y1")):
            return [value["x0"], value["y0"], value["x1"], value["y1"]]
        return json_safe(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return value


def _apply_scale(points: np.ndarray, bounds: Mapping[str, Any] | None, mode: str) -> np.ndarray:
    if mode == "half":
        return points * 0.5
    if mode == "double":
        return points * 2.0
    if mode != "fit_bounds":
        return points
    target_min, target_max = _bounds_min_max(bounds, None)
    if target_min is None or points.size == 0:
        return points
    point_size = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-9)
    target_size = np.maximum(target_max - target_min, 1e-9)
    scale = float(np.min(target_size / point_size))
    return points * scale


def _apply_translation(points: np.ndarray, bounds: Mapping[str, Any] | None, mode: str) -> np.ndarray:
    target_min, target_max = _bounds_min_max(bounds, None)
    if mode == "none" or target_min is None or points.size == 0:
        return points
    point_min = points.min(axis=0)
    point_max = points.max(axis=0)
    if mode == "bottom":
        delta = np.array([0.0, 0.0, target_min[2] - point_min[2]], dtype=float)
        return points + delta
    if mode == "center":
        point_center = (point_min + point_max) * 0.5
        target_center = (target_min + target_max) * 0.5
        return points + (target_center - point_center)
    return points


def _sample_vertices(vertices: np.ndarray, max_vertices: int) -> np.ndarray:
    if len(vertices) <= max_vertices:
        return vertices
    indices = np.linspace(0, len(vertices) - 1, max_vertices).round().astype(np.int64)
    return vertices[indices]


def _numpy_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    output = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.zeros_like(mask)
            src_y0 = max(0, -dy)
            src_y1 = mask.shape[0] - max(0, dy)
            src_x0 = max(0, -dx)
            src_x1 = mask.shape[1] - max(0, dx)
            dst_y0 = max(0, dy)
            dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
            dst_x0 = max(0, dx)
            dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
            if dst_y1 > dst_y0 and dst_x1 > dst_x0:
                shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
            output |= shifted
    return output


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default
