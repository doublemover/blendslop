"""Rule-based failure taxonomy for evaluation bundles."""

from __future__ import annotations

from typing import Mapping

from .schemas import EvaluationBundle, FailureObservation


def classify_bundle_failures(
    bundle: EvaluationBundle,
) -> tuple[FailureObservation, ...]:
    metrics = bundle.metric_index()
    failures: list[FailureObservation] = []

    min_iou = _metric_value(metrics, "silhouette.min_view_iou")
    avg_iou = _metric_value(metrics, "silhouette.average_iou")
    required_views = _metric_value(metrics, "silhouette.required_view_count")
    failed_required = _metric_value(metrics, "silhouette.failed_required_view_count")
    missing_required = _metric_value(metrics, "silhouette.missing_required_metric_count")
    if (
        bundle.status != "skip"
        and min_iou is not None
        and min_iou <= 0.0
        and required_views is not None
        and required_views > 0
        and failed_required is not None
        and failed_required > 0
    ):
        failures.append(
            FailureObservation(
                code="silhouette_required_views_failed",
                severity="fail",
                subsystem="silhouette",
                evidence_metrics={
                    "silhouette.min_view_iou": min_iou,
                    "silhouette.average_iou": avg_iou,
                    "silhouette.required_view_count": required_views,
                    "silhouette.failed_required_view_count": failed_required,
                },
                likely_causes=(
                    "one or more required views have near-zero overlap with the reference silhouette",
                ),
                recommended_actions=(
                    "inspect per-view overlays and active view calibration",
                    "run boundary-first or signed-distance refinement before candidate selection",
                ),
            )
        )
    elif bundle.status != "skip" and (min_iou is None or min_iou <= 0.0):
        failures.append(
            FailureObservation(
                code="silhouette_required_metrics_missing",
                severity="fail",
                subsystem="silhouette",
                evidence_metrics={
                    "silhouette.min_view_iou": min_iou,
                    "silhouette.average_iou": avg_iou,
                    "silhouette.required_view_count": required_views,
                    "silhouette.missing_required_metric_count": missing_required,
                },
                likely_causes=(
                    "candidate reported success without required per-view silhouette metrics",
                ),
                recommended_actions=(
                    "ensure backend populates CandidateMetrics.per_view and area_iou_min",
                    "run backend-status validation with evaluation bundle output",
                ),
            )
        )
    if min_iou is not None and avg_iou is not None and min_iou < 0.5 <= avg_iou:
        failures.append(
            FailureObservation(
                code="silhouette_mean_hides_failed_view",
                severity="fail",
                subsystem="silhouette",
                evidence_metrics={
                    "silhouette.min_view_iou": min_iou,
                    "silhouette.average_iou": avg_iou,
                },
                likely_causes=("one required view is misaligned or missing detail",),
                recommended_actions=(
                    "inspect per-view overlays",
                    "run calibration diagnostics",
                ),
            )
        )
    boundary = _metric_value(metrics, "silhouette.min_boundary_iou")
    sdf = _metric_value(metrics, "silhouette.mean_signed_distance_loss")
    if missing_required is not None and missing_required > 0:
        failures.append(
            FailureObservation(
                code="silhouette_required_metric_fields_missing",
                severity="fail",
                subsystem="silhouette",
                evidence_metrics={
                    "silhouette.missing_required_metric_count": missing_required,
                    "silhouette.failed_required_view_count": failed_required,
                    "silhouette.mean_signed_distance_loss": sdf,
                    "silhouette.min_boundary_iou": boundary,
                },
                likely_causes=(
                    "backend emitted aggregate IoU but did not emit Boundary IoU or signed-distance loss for every required view",
                ),
                recommended_actions=(
                    "populate per-view area_iou, boundary_iou, signed_distance_loss, required, passed, and reason",
                    "use the shared silhouette metric adapter before building CandidateMetrics",
                ),
            )
        )
    if (
        boundary is not None
        and avg_iou is not None
        and avg_iou >= 0.85
        and boundary < 0.55
    ):
        failures.append(
            FailureObservation(
                code="silhouette_boundary_blobby",
                severity="fail",
                subsystem="silhouette",
                evidence_metrics={
                    "silhouette.average_iou": avg_iou,
                    "silhouette.min_boundary_iou": boundary,
                },
                likely_causes=("contour smoothing or missing thin features",),
                recommended_actions=(
                    "run boundary-first refinement",
                    "inspect boundary overlay",
                ),
            )
        )
    non_manifold = _metric_value(metrics, "topology.non_manifold_edges")
    if non_manifold is not None and non_manifold > 0:
        failures.append(
            FailureObservation(
                code="topology_non_manifold",
                severity="fail",
                subsystem="topology",
                evidence_metrics={"topology.non_manifold_edges": non_manifold},
                recommended_actions=(
                    "run safe topology repair",
                    "try alternate mesh extraction",
                ),
            )
        )
    fscore = _metric_value(metrics, "geometry.fscore_tau")
    volumetric_iou = _metric_value(metrics, "geometry.volumetric_iou")
    chamfer = _metric_value(metrics, "geometry.chamfer_l2")
    true_fscore = _metric_value(metrics, "geometry.true.fscore_tau")
    recoverable_fscore = _metric_value(metrics, "geometry.recoverable.fscore_tau")
    ambiguity_gap = _metric_value(metrics, "geometry.ambiguity_gap_chamfer_l2")
    if fscore is not None and fscore < 0.5:
        failures.append(
            FailureObservation(
                code="geometry_surface_fscore_low",
                severity="fail",
                subsystem="geometry",
                evidence_metrics={"geometry.fscore_tau": fscore},
                likely_causes=("surface samples miss ground truth within tolerance",),
                recommended_actions=(
                    "increase visual hull resolution",
                    "inspect synthetic ground-truth alignment",
                ),
            )
        )
    if volumetric_iou is not None and volumetric_iou < 0.45:
        failures.append(
            FailureObservation(
                code="geometry_volume_iou_low",
                severity="fail",
                subsystem="geometry",
                evidence_metrics={"geometry.volumetric_iou": volumetric_iou},
                likely_causes=(
                    "carved occupancy disagrees with synthetic ground truth",
                ),
                recommended_actions=(
                    "run sparse visual hull resolution climb",
                    "check camera bounds and occupancy threshold",
                ),
            )
        )
    if chamfer is not None and chamfer > 0.05:
        failures.append(
            FailureObservation(
                code="geometry_chamfer_high",
                severity="warn",
                subsystem="geometry",
                evidence_metrics={"geometry.chamfer_l2": chamfer},
                likely_causes=(
                    "surface is shifted, over-smoothed, or missing thin structures",
                ),
                recommended_actions=(
                    "run boundary-first refinement",
                    "inspect residual patches",
                ),
            )
        )
    if (
        true_fscore is not None
        and recoverable_fscore is not None
        and recoverable_fscore >= 0.75
        and true_fscore < 0.45
    ):
        failures.append(
            FailureObservation(
                code="geometry_true_recoverable_gap_large",
                severity="warn",
                subsystem="geometry",
                evidence_metrics={
                    "geometry.true.fscore_tau": true_fscore,
                    "geometry.recoverable.fscore_tau": recoverable_fscore,
                    "geometry.ambiguity_gap_chamfer_l2": ambiguity_gap,
                },
                likely_causes=(
                    "provided silhouettes underdetermine hidden concavity or internal detail",
                ),
                recommended_actions=(
                    "score this case against the recoverable envelope before treating it as a backend regression",
                    "request an additional discriminating view from the active view planner",
                ),
            )
        )
    psnr = _metric_value(metrics, "novel_view.psnr")
    ssim = _metric_value(metrics, "novel_view.ssim")
    lpips = _metric_value(metrics, "novel_view.lpips")
    if psnr is not None and psnr < 20.0:
        failures.append(
            FailureObservation(
                code="novel_view_psnr_low",
                severity="warn",
                subsystem="novel_view",
                evidence_metrics={"novel_view.psnr": psnr},
                likely_causes=("rendered appearance diverges from held-out view",),
                recommended_actions=("inspect novel-view render overlays",),
            )
        )
    if ssim is not None and ssim < 0.65:
        failures.append(
            FailureObservation(
                code="novel_view_ssim_low",
                severity="warn",
                subsystem="novel_view",
                evidence_metrics={"novel_view.ssim": ssim},
                recommended_actions=("inspect lighting/material/framing consistency",),
            )
        )
    if lpips is not None and lpips > 0.35:
        failures.append(
            FailureObservation(
                code="novel_view_lpips_high",
                severity="warn",
                subsystem="novel_view",
                evidence_metrics={"novel_view.lpips": lpips},
                recommended_actions=("inspect perceptual novel-view mismatch",),
            )
        )
    export_score = _metric_value(metrics, "export.qa_score")
    editability = _metric_value(metrics, "editability.editable_reconstruction_index")
    if editability is not None and editability < 0.35:
        failures.append(
            FailureObservation(
                code="editability_low",
                severity="warn",
                subsystem="editability",
                evidence_metrics={
                    "editability.editable_reconstruction_index": editability,
                    "topology.non_manifold_edges": non_manifold,
                    "export.qa_score": export_score,
                },
                likely_causes=(
                    "result is a dense or unstructured mesh rather than a clean editable Blender asset",
                ),
                recommended_actions=(
                    "prefer primitive, shape-program, or modifier-backed candidates when silhouette metrics tie",
                    "run mesh simplification and semantic part separation before export",
                ),
            )
        )
    if export_score is not None and export_score < 0.6:
        failures.append(
            FailureObservation(
                code="export_qa_low",
                severity="fail",
                subsystem="export",
                evidence_metrics={"export.qa_score": export_score},
                likely_causes=(
                    "asset export failed, reimport failed, or exported asset has empty geometry/material counts",
                ),
                recommended_actions=(
                    "inspect export QA report metadata",
                    "rerun Blender export/reimport smoke with the same candidate artifact",
                ),
            )
        )
    if bundle.status == "degraded":
        failures.append(
            FailureObservation(
                code="backend_degraded_output",
                severity="warn",
                subsystem="backend",
                evidence_metrics=bundle.degradation_state,
                recommended_actions=("inspect dependency and degradation metadata",),
            )
        )
    if bundle.errors:
        failures.append(
            FailureObservation(
                code="backend_contract_or_runtime_error",
                severity="fail",
                subsystem="backend",
                evidence_metrics={"error_count": len(bundle.errors)},
                likely_causes=tuple(bundle.errors),
                recommended_actions=("inspect candidate result errors",),
            )
        )
    return tuple(failures)


def _metric_value(metrics: Mapping[str, object], name: str) -> float | None:
    metric = metrics.get(name)
    if metric is None:
        return None
    value = getattr(metric, "value", None)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
