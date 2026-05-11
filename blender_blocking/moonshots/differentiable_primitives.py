"""Differentiable editable primitive fitting moonshot."""

from __future__ import annotations

from .contracts import (
    MoonshotExperiment,
    MoonshotRequest,
    MoonshotResult,
    bundle_result,
    error_result,
    unsupported_result,
)
from .papers import NEURAL_MESH_RENDERER, SOFT_RASTERIZER, SUPERQUADRICS
from .support import (
    bounded,
    candidate_rows,
    metric_extras,
    per_view_boundary_iou,
    per_view_signed_distance_loss,
    row_metric,
    target_signals,
)


EXPERIMENT = MoonshotExperiment(
    experiment_id="differentiable_primitives",
    title="Differentiable silhouette fitting for editable primitive programs",
    subsystem="differentiable",
    hypothesis=(
        "A CPU-compatible soft silhouette objective can refine primitive transforms "
        "and dimensions without requiring CUDA-only rasterization."
    ),
    expected_wins={
        "quality": "lower boundary loss after primitive initialization",
        "editability": "directly optimized primitives instead of post-hoc mesh cleanup",
    },
    required_inputs=("primitive_program", "silhouette_targets", "profile_bands"),
    optional_dependencies=("torch", "rocm-capable-array-backend"),
    validation_metrics=("boundary_iou", "signed_distance_loss", "objective_delta"),
    papers=(NEURAL_MESH_RENDERER, SOFT_RASTERIZER, SUPERQUADRICS),
)


def run(request: MoonshotRequest) -> MoonshotResult:
    try:
        rows = candidate_rows(request.candidate)
        primitive_count = _primitive_count(rows)
        if primitive_count <= 0:
            return unsupported_result(
                request,
                reason="no primitive program or primitive-fit payload was available for differentiable probing",
                next_steps=("run after primitive_fit_refine or shape_program emits editable primitives",),
            )
        baseline = max(
            (
                row_metric(
                    row,
                    "render.boundary_iou_mean",
                    "boundary_iou_mean",
                    "metric_result.boundary_iou_mean",
                    default=0.0,
                )
                for row in rows
            ),
            default=0.0,
        )
        signals = target_signals(request)
        boundary = _boundary_baseline(rows)
        schedule = _finite_difference_schedule(primitive_count, boundary)
        parameter_groups = _parameter_groups(rows, primitive_count)
        objective_terms = _objective_terms(rows, signals)
        expected_objective_delta = min(
            0.28,
            0.025
            + primitive_count * 0.006
            + boundary["pressure"] * 0.08
            + min(0.06, len(objective_terms) * 0.008),
        )
        evidence = {
            "primitive_count": primitive_count,
            "baseline_boundary_iou": baseline,
            "gradient_mode": "finite_difference_manifest",
            "accepted_parameters": min(primitive_count * 6, 48),
            "parameter_groups": parameter_groups,
            "objective_terms": objective_terms,
            "finite_difference_schedule": schedule,
            "trust_region": {
                "max_translation_fraction": 0.06,
                "max_scale_fraction": 0.10,
                "max_rotation_deg": 8.0,
                "reject_on_any_view_regression": True,
            },
            "dependency_status": {
                "torch": "optional",
                "nvdiffrast": "not_required_for_cpu_probe",
                "cpu_soft_silhouette": "required",
            },
        }
        return bundle_result(
            request,
            status="ran",
            metrics={
                "ran": 1.0,
                "primitive_count": float(primitive_count),
                "expected_objective_delta": expected_objective_delta,
                "expected_boundary_iou": min(1.0, baseline + expected_objective_delta),
                "expected_signed_distance_loss_delta": -bounded(
                    boundary["signed_distance_loss"] * 0.35 + expected_objective_delta * 0.08,
                    0.0,
                    0.12,
                ),
                "parameter_group_count": float(len(parameter_groups)),
                "objective_term_count": float(len(objective_terms)),
                "finite_difference_probe_count": float(
                    sum(int(item.get("probe_count", 0) or 0) for item in schedule)
                ),
            },
            evidence=evidence,
            artifact_name="differentiable-primitives.json",
            next_steps=(
                "run finite-difference crosschecks before accepting parameter updates",
                "gate updates on per-view boundary IoU and signed-distance loss",
            ),
        )
    except Exception as exc:
        return error_result(request, error=f"{type(exc).__name__}: {exc}")


def _primitive_count(rows: tuple) -> int:
    best = 0
    for row in rows:
        extras = metric_extras(row)
        count = extras.get("primitive_count")
        if count is None:
            primitives = extras.get("primitives")
            if isinstance(primitives, dict):
                count = primitives.get("count")
        if count is None:
            shape_program = extras.get("shape_program")
            if isinstance(shape_program, dict):
                count = len(shape_program.get("root_nodes", ()) or ())
        try:
            best = max(best, int(count or 0))
        except (TypeError, ValueError):
            pass
    return best


def _boundary_baseline(rows: tuple) -> dict[str, float]:
    boundary_values = []
    sdf_values = []
    for row in rows:
        boundary_values.extend(per_view_boundary_iou(row).values())
        sdf_values.extend(per_view_signed_distance_loss(row).values())
    boundary_iou = sum(boundary_values) / len(boundary_values) if boundary_values else 0.45
    signed_distance = sum(sdf_values) / len(sdf_values) if sdf_values else 0.0
    return {
        "boundary_iou": bounded(boundary_iou),
        "signed_distance_loss": max(0.0, signed_distance),
        "pressure": bounded((1.0 - boundary_iou) * 0.75 + min(1.0, signed_distance * 16.0) * 0.25),
    }


def _parameter_groups(rows: tuple, primitive_count: int) -> list[dict[str, object]]:
    extras = {}
    for row in rows:
        extras = dict(metric_extras(row))
        if extras:
            break
    primitive_payload = extras.get("primitives") if isinstance(extras.get("primitives"), dict) else {}
    names = primitive_payload.get("types") if isinstance(primitive_payload, dict) else None
    if not isinstance(names, (list, tuple)):
        names = ["primitive"] * primitive_count
    groups = []
    for index in range(min(primitive_count, 12)):
        primitive_type = str(names[index] if index < len(names) else "primitive")
        groups.append(
            {
                "group_id": f"primitive_{index:02d}",
                "primitive_type": primitive_type,
                "parameters": [
                    "translate_x",
                    "translate_y",
                    "translate_z",
                    "scale_x",
                    "scale_y",
                    "scale_z",
                ],
                "regularization": "keep_editable_bounds",
            }
        )
    return groups


def _objective_terms(rows: tuple, signals: dict) -> list[dict[str, object]]:
    terms = []
    boundary_by_view = {}
    sdf_by_view = {}
    for row in rows:
        for view, value in per_view_boundary_iou(row).items():
            boundary_by_view.setdefault(view, []).append(value)
        for view, value in per_view_signed_distance_loss(row).items():
            sdf_by_view.setdefault(view, []).append(value)
    for view in ("front", "side", "top"):
        boundary_values = boundary_by_view.get(view, [])
        sdf_values = sdf_by_view.get(view, [])
        boundary = sum(boundary_values) / len(boundary_values) if boundary_values else 0.45
        signed_distance = sum(sdf_values) / len(sdf_values) if sdf_values else 0.0
        terms.append(
            {
                "term": f"{view}_soft_silhouette",
                "view": view,
                "weight": bounded(0.8 + (1.0 - boundary) * 0.7 + min(0.5, signed_distance * 6.0), 0.2, 1.8),
                "baseline_boundary_iou": boundary,
                "baseline_signed_distance_loss": signed_distance,
            }
        )
    profile = signals.get("profile", {}) if isinstance(signals, dict) else {}
    if int(profile.get("hole_count", 0) or 0) > 0:
        terms.append(
            {
                "term": "hole_preservation",
                "view": "all",
                "weight": 0.55,
                "hole_count": int(profile.get("hole_count", 0) or 0),
            }
        )
    return terms


def _finite_difference_schedule(
    primitive_count: int,
    boundary: dict[str, float],
) -> list[dict[str, object]]:
    active_groups = min(primitive_count, 12)
    coarse_probe_count = active_groups * 6 * 2
    refine_probe_count = active_groups * 4 * 2 if boundary["pressure"] > 0.18 else active_groups * 2
    return [
        {
            "stage": "coarse_axis_aligned_probe",
            "step_scale": 0.08,
            "probe_count": coarse_probe_count,
            "acceptance": "accept only if objective and every required area IoU stay neutral or improve",
        },
        {
            "stage": "boundary_weighted_refine",
            "step_scale": 0.03,
            "probe_count": refine_probe_count,
            "acceptance": "accept only if boundary IoU improves on the weakest view",
        },
        {
            "stage": "trust_region_commit",
            "step_scale": 0.01,
            "probe_count": max(2, active_groups),
            "acceptance": "commit as advisory primitive deltas, not backend success",
        },
    ]
