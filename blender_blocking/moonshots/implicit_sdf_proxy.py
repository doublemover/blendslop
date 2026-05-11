"""Implicit SDF proxy moonshot for post-hull detail recovery."""

from __future__ import annotations

from .contracts import (
    MoonshotExperiment,
    MoonshotRequest,
    MoonshotResult,
    bundle_result,
    error_result,
    skipped_result,
    write_moonshot_artifact,
)
from .papers import POISSON, SCREENED_POISSON, SPACE_CARVING
from .support import (
    bounded,
    candidate_rows,
    per_view_boundary_iou,
    per_view_signed_distance_loss,
    row_metric,
    target_signals,
)


EXPERIMENT = MoonshotExperiment(
    experiment_id="implicit_sdf_proxy",
    title="Hybrid visual-hull SDF proxy and editable extraction",
    subsystem="volume",
    hypothesis=(
        "A calibrated SDF proxy can smooth voxel artifacts and preserve silhouette constraints "
        "before editable mesh or primitive extraction."
    ),
    expected_wins={
        "quality": "smoother surfaces, better normal consistency, fewer staircase artifacts",
        "performance": "sparse SDF chunks reuse the existing visual-hull chunk cache",
    },
    required_inputs=("visual_hull_volume", "silhouette_signed_distance_fields"),
    validation_metrics=("chamfer_l2", "normal_consistency", "volumetric_iou", "watertightness"),
    papers=(SPACE_CARVING, POISSON, SCREENED_POISSON),
)


def run(request: MoonshotRequest) -> MoonshotResult:
    try:
        signals = target_signals(request)
        rows = candidate_rows(request.candidate)
        if not rows and not any(bool(group.get("available")) for group in signals.values()):
            return skipped_result(
                request,
                reason="implicit SDF proxy needs visual-hull/candidate metrics or target signals",
                next_steps=("run after visual-hull target building or candidate evaluation",),
            )
        return _build_proxy(request, rows=rows, signals=signals)
    except Exception as exc:
        return error_result(request, error=f"{type(exc).__name__}: {exc}")


def _build_proxy(request: MoonshotRequest, *, rows: tuple, signals: dict | object) -> MoonshotResult:
    config = dict(request.config or {})
    resolution = max(8, min(128, int(config.get("sdf_resolution", 32) or 32)))
    max_resolution = max(resolution, min(192, int(config.get("sdf_max_resolution", resolution * 2) or resolution)))
    profile = signals["profile"]  # type: ignore[index]
    surface = signals["surface"]  # type: ignore[index]
    uncertainty = signals["uncertainty"]  # type: ignore[index]
    topology = signals["topology"]  # type: ignore[index]
    surface_points = int(surface.get("surface_point_count", 0) or 0)
    band_samples = int(profile.get("band_samples", 0) or 0)
    constraint_count = int(surface.get("constraint_count", 0) or 0)
    sparse_voxels = max(
        8,
        min(
            resolution ** 3,
            int((surface_points or band_samples * resolution or resolution * 4) * 1.5),
        ),
    )
    baseline_iou = _baseline_iou(rows)
    boundary_uncertainty = float(
        uncertainty.get("overall_boundary_uncertainty_mean", 0.0) or 0.0
    )
    topology_score = float(topology.get("score", 0.75) or 0.75)
    smoothing_gain = bounded(
        0.03
        + 0.08 * boundary_uncertainty
        + 0.04 * (1.0 - topology_score)
        + 0.015 * min(4, constraint_count),
        0.0,
        0.18,
    )
    volumetric_iou_proxy = bounded(baseline_iou + smoothing_gain, 0.0, 0.99)
    chamfer_delta = -bounded(0.02 + smoothing_gain * 0.8, 0.0, 0.18)
    view_fields = _view_field_manifests(rows, uncertainty)
    levels = _level_schedule(
        resolution=resolution,
        max_resolution=max_resolution,
        sparse_voxels=sparse_voxels,
        boundary_uncertainty=boundary_uncertainty,
    )
    extraction_plan = _extraction_plan(
        baseline_iou=baseline_iou,
        topology_score=topology_score,
        smoothing_gain=smoothing_gain,
        levels=levels,
        view_fields=view_fields,
    )
    sdf_payload = {
        "schema_version": "implicit_sdf_proxy_artifact_v1",
        "resolution": resolution,
        "max_resolution": max_resolution,
        "sparse_voxels": sparse_voxels,
        "source": "visual_hull_signal_proxy",
        "levels": levels,
        "view_fields": view_fields,
        "signed_distance_fields": {
            "available": bool(boundary_uncertainty > 0.0 or rows or view_fields),
            "boundary_uncertainty_mean": boundary_uncertainty,
        },
        "projection": {
            "constraint_count": constraint_count,
            "surface_point_count": surface_points,
            "profile_band_samples": band_samples,
            "uncertainty_weighted": bool(uncertainty.get("available")),
        },
    }
    mesh_payload = {
        "schema_version": "implicit_sdf_mesh_proxy_v1",
        "method": "sparse_sdf_guarded_marching_cubes_manifest",
        "estimated_vertices": max(8, int(sparse_voxels ** (2.0 / 3.0) * 6)),
        "estimated_faces": max(12, int(sparse_voxels ** (2.0 / 3.0) * 12)),
        "postprocess": "smooth_guarded",
        "editable_retopology_recommended": topology_score < 0.85,
        "extraction_plan": extraction_plan,
    }
    artifacts: dict[str, str] = {}
    sdf_path = write_moonshot_artifact(request, "sdf-proxy.json", sdf_payload)
    mesh_path = write_moonshot_artifact(request, "mesh-proxy.json", mesh_payload)
    levels_path = write_moonshot_artifact(request, "sdf-levels.json", {"levels": levels})
    extraction_path = write_moonshot_artifact(request, "extraction-plan.json", extraction_plan)
    if sdf_path:
        artifacts["sdf_proxy"] = sdf_path
    if mesh_path:
        artifacts["mesh_proxy"] = mesh_path
    if levels_path:
        artifacts["level_schedule"] = levels_path
    if extraction_path:
        artifacts["extraction_plan"] = extraction_path
    evidence = {
        "sdf_proxy": sdf_payload,
        "mesh_proxy": mesh_payload,
        "extraction_plan": extraction_plan,
        "baseline_iou": baseline_iou,
        "estimated_delta": {
            "volumetric_iou": volumetric_iou_proxy - baseline_iou,
            "chamfer_l2": chamfer_delta,
            "normal_consistency": smoothing_gain * 0.6,
        },
    }
    metrics = {
        "ran": 1.0,
        "sparse_voxels": float(sparse_voxels),
        "volumetric_iou_proxy": volumetric_iou_proxy,
        "volumetric_iou_delta": volumetric_iou_proxy - baseline_iou,
        "chamfer_l2_delta": chamfer_delta,
        "normal_consistency_delta": smoothing_gain * 0.6,
        "level_count": float(len(levels)),
        "view_field_count": float(len(view_fields)),
        "guard_count": float(len(extraction_plan.get("guards", ()))) if isinstance(extraction_plan, dict) else 0.0,
    }
    return bundle_result(
        request,
        status="ran",
        metrics=metrics,
        evidence=evidence,
        artifacts=artifacts,
        artifact_name="implicit-sdf-proxy.json",
        next_steps=(
            "extract a mesh only after SDF proxy deltas clear topology gates",
            "compare this manifest against synthetic SDF ground truth when available",
        ),
    )


def _baseline_iou(rows: tuple) -> float:
    if not rows:
        return 0.55
    values = [
        row_metric(
            row,
            "render.min_view_iou",
            "min_view_iou",
            "area_iou_min",
            "metric_result.area_iou_min",
            default=0.0,
        )
        for row in rows
    ]
    present = [value for value in values if value > 0.0]
    return bounded(sum(present) / len(present)) if present else 0.55


def _view_field_manifests(rows: tuple, uncertainty: dict) -> list[dict[str, object]]:
    details = uncertainty.get("view_details") if isinstance(uncertainty, dict) else {}
    details = details if isinstance(details, dict) else {}
    output = []
    for view in ("front", "side", "top"):
        boundary_values = [
            per_view_boundary_iou(row).get(view)
            for row in rows
            if per_view_boundary_iou(row).get(view) is not None
        ]
        sdf_values = [
            per_view_signed_distance_loss(row).get(view)
            for row in rows
            if per_view_signed_distance_loss(row).get(view) is not None
        ]
        boundary_iou = (
            sum(float(value) for value in boundary_values) / len(boundary_values)
            if boundary_values
            else None
        )
        signed_distance = (
            sum(float(value) for value in sdf_values) / len(sdf_values)
            if sdf_values
            else None
        )
        view_uncertainty = details.get(view, {}) if isinstance(details, dict) else {}
        boundary_uncertainty = (
            float(view_uncertainty.get("boundary_uncertainty_mean", 0.0) or 0.0)
            if isinstance(view_uncertainty, dict)
            else 0.0
        )
        pressure = bounded(
            (1.0 - float(boundary_iou if boundary_iou is not None else 0.75)) * 0.55
            + min(1.0, float(signed_distance or 0.0) * 20.0) * 0.30
            + boundary_uncertainty * 0.15
        )
        output.append(
            {
                "view": view,
                "boundary_iou": boundary_iou,
                "signed_distance_loss": signed_distance,
                "boundary_uncertainty": boundary_uncertainty,
                "weight": pressure,
                "field_role": "constraint" if pressure < 0.25 else "repair_pressure",
            }
        )
    return output


def _level_schedule(
    *,
    resolution: int,
    max_resolution: int,
    sparse_voxels: int,
    boundary_uncertainty: float,
) -> list[dict[str, object]]:
    levels = []
    current = resolution
    while current <= max_resolution and len(levels) < 4:
        scale = current / max(1, resolution)
        levels.append(
            {
                "level": len(levels),
                "resolution": current,
                "estimated_sparse_voxels": min(int(sparse_voxels * scale ** 2), current ** 3),
                "narrow_band_voxels": max(8, int(sparse_voxels * (0.35 + boundary_uncertainty))),
                "operation": "initialize" if not levels else "refine_boundary_band",
            }
        )
        current *= 2
    return levels


def _extraction_plan(
    *,
    baseline_iou: float,
    topology_score: float,
    smoothing_gain: float,
    levels: list[dict[str, object]],
    view_fields: list[dict[str, object]],
) -> dict[str, object]:
    worst_view = max(
        view_fields,
        key=lambda item: float(item.get("weight", 0.0) or 0.0),
        default={"view": "front", "weight": 0.0},
    )
    return {
        "schema_version": "implicit_sdf_extraction_plan_v1",
        "strategy": "coarse_to_fine_narrow_band_sdf",
        "level_count": len(levels),
        "worst_view": worst_view,
        "guards": [
            {
                "metric": "render.min_view_iou",
                "minimum": max(0.45, min(0.98, baseline_iou - 0.01)),
                "reason": "SDF smoothing must not erase silhouette support",
            },
            {
                "metric": "topology.score",
                "minimum": max(0.75, min(0.95, topology_score - 0.02)),
                "reason": "extracted mesh cannot trade topology for smoothness",
            },
            {
                "metric": "normal_consistency_delta",
                "minimum": max(0.0, smoothing_gain * 0.25),
                "reason": "proxy should improve normals enough to justify extraction",
            },
        ],
        "postprocess_order": [
            "signed_distance_fusion",
            "narrow_band_smoothing",
            "guarded_marching_cubes",
            "topology_qa",
            "render_qa",
        ],
    }
