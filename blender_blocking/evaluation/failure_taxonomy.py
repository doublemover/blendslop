"""Rule-based failure taxonomy for evaluation bundles."""

from __future__ import annotations

from typing import Mapping

from .schemas import EvaluationBundle, FailureObservation


def classify_bundle_failures(bundle: EvaluationBundle) -> tuple[FailureObservation, ...]:
    metrics = bundle.metric_index()
    failures: list[FailureObservation] = []

    min_iou = _metric_value(metrics, "silhouette.min_view_iou")
    avg_iou = _metric_value(metrics, "silhouette.average_iou")
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
                recommended_actions=("inspect per-view overlays", "run calibration diagnostics"),
            )
        )
    boundary = _metric_value(metrics, "silhouette.min_boundary_iou")
    if boundary is not None and avg_iou is not None and avg_iou >= 0.85 and boundary < 0.55:
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
                recommended_actions=("run boundary-first refinement", "inspect boundary overlay"),
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
                recommended_actions=("run safe topology repair", "try alternate mesh extraction"),
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

