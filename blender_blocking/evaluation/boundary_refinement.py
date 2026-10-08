"""Boundary-first refinement plans for silhouette reconstruction failures."""

from __future__ import annotations

from typing import Any, Mapping

from .schemas import EvaluationBundle, json_safe


def boundary_refinement_plan_payload(bundle: EvaluationBundle) -> dict[str, object]:
    """Build a concrete boundary/detail refinement sweep from bundle metrics."""
    metrics = bundle.metric_index()
    min_boundary = _metric_float(metrics, "silhouette.min_boundary_iou", 0.0)
    mean_boundary = _metric_float(metrics, "silhouette.mean_boundary_iou", min_boundary)
    min_iou = _metric_float(metrics, "silhouette.min_view_iou", 0.0)
    average_iou = _metric_float(metrics, "silhouette.average_iou", 0.0)
    sdf = _metric_float(metrics, "silhouette.mean_signed_distance_loss", 1.0)
    failures = [failure.to_dict() for failure in bundle.failures if "boundary" in failure.code]
    return {
        "schema_version": "boundary_refinement_plan_v1",
        "candidate_id": bundle.candidate_id,
        "mode": bundle.mode,
        "trigger_metrics": {
            "silhouette.min_boundary_iou": min_boundary,
            "silhouette.mean_boundary_iou": mean_boundary,
            "silhouette.min_view_iou": min_iou,
            "silhouette.average_iou": average_iou,
            "silhouette.mean_signed_distance_loss": sdf,
        },
        "recommended_track": "content-adaptive-patches",
        "recommended_command": [
            "python",
            "blender_blocking/refinement_lab/cli.py",
            "plan",
            "--track",
            "content-adaptive-patches",
            "--shape-residual-policy",
            "suggest_patches",
        ],
        "probes": _boundary_probes(sdf=sdf, min_boundary=min_boundary),
        "acceptance": {
            "min_boundary_iou_delta": 0.04,
            "mean_signed_distance_loss_delta": -0.03,
            "must_not_reduce_min_view_iou": True,
            "must_not_reduce_topology_score": True,
        },
        "failure_evidence": json_safe(failures),
    }


def _boundary_probes(*, sdf: float, min_boundary: float) -> list[dict[str, object]]:
    boundary_weight = 1.5 if min_boundary < 0.45 else 1.2
    sdf_weight = 1.6 if sdf > 0.12 else 1.1
    return [
        {
            "probe_id": "mask_threshold_sweep",
            "description": "Re-extract masks with stricter and looser alpha/luma thresholds.",
            "parameters": {
                "threshold": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
                "anti_alias_softening": [0.0, 0.5, 1.0],
            },
        },
        {
            "probe_id": "morphology_sweep",
            "description": "Search small open/close kernels to recover thin parts without blob growth.",
            "parameters": {
                "operation": ["none", "open", "close", "open_close", "close_open"],
                "kernel_px": [0, 1, 2, 3],
                "iterations": [1, 2],
            },
        },
        {
            "probe_id": "visual_hull_boundary_band_sweep",
            "description": "Adjust visual-hull boundary expansion around uncertain contour pixels.",
            "parameters": {
                "boundary_refine": [True],
                "boundary_dilate_px": [-1, 0, 1, 2, 3, 4],
                "occupancy_threshold": [0.45, 0.50, 0.55, 0.60],
            },
        },
        {
            "probe_id": "content_adaptive_patch_sweep",
            "description": "Focus local refinement on high residual, high uncertainty, and high boundary-error regions.",
            "parameters": {
                "patch_sizes": [[64], [96], [96, 160], [128, 192]],
                "max_patches": [4, 8, 12],
                "edge_weight_strength": [0.5, 1.0, 1.5],
                "alignment_mode": ["shift", "affine"],
            },
        },
        {
            "probe_id": "differentiable_boundary_weight_sweep",
            "description": "Increase soft silhouette boundary and signed-distance terms in CPU differentiable refinement.",
            "parameters": {
                "boundary_weight": [1.0, boundary_weight, 2.0],
                "signed_distance_weight": [1.0, sdf_weight, 2.0],
                "max_objective_evaluations": [64, 128, 256],
            },
        },
    ]


def _metric_float(
    metrics: Mapping[str, Any],
    name: str,
    default: float,
) -> float:
    metric = metrics.get(name)
    value = default if metric is None else getattr(metric, "value", default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
