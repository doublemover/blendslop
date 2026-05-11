from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def differentiable_config_payload(
    *,
    backend_choice: str,
    optional_dependency_policy: str,
    parsed_config: Mapping[str, Any],
    config_warnings: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    return {
        "backend": backend_choice,
        "optional_dependency_policy": optional_dependency_policy,
        "gradient_mode": str(parsed_config["gradient_mode"]),
        "finite_difference_epsilon": float(parsed_config["finite_difference_epsilon"]),
        "softness": float(parsed_config["softness"]),
        "min_variance": float(parsed_config["min_variance"]),
        "primitive_opacity_floor": float(parsed_config["primitive_opacity_floor"]),
        "silhouette_bounds_padding": float(parsed_config["silhouette_bounds_padding"]),
        "include_bounds_proxy": bool(parsed_config["include_bounds_proxy"]),
        "calibrate_silhouette_bounds": bool(parsed_config["calibrate_silhouette_bounds"]),
        "mesh_proxy_scale": float(parsed_config["mesh_proxy_scale"]),
        "visual_hull_resolution": int(parsed_config["visual_hull_resolution"]),
        "primitive_count": int(parsed_config["primitive_count"]),
        "target_point_count": int(parsed_config["target_point_count"]),
        "min_radius": float(parsed_config["min_radius"]),
        "covariance_floor": float(parsed_config["covariance_floor"]),
        "kmeans_iterations": int(parsed_config["kmeans_iterations"]),
        "chunk_size": parsed_config["chunk_size"],
        "optimization_steps": int(parsed_config["optimization_steps"]),
        "optimization_initial_step": float(parsed_config["optimization_initial_step"]),
        "optimization_step_decay": float(parsed_config["optimization_step_decay"]),
        "optimization_min_step": float(parsed_config["optimization_min_step"]),
        "max_objective_evaluations": parsed_config["max_objective_evaluations"],
        "max_runtime_s": parsed_config["max_runtime_s"],
        "loss_weights": dict(parsed_config["loss_weights_dict"]),
        "warnings": tuple(config_warnings),
    }


def write_differentiable_candidate_artifacts(
    *,
    root: Path | None,
    candidate_id: str,
    backend_name: str,
    backend_choice: str,
    optional_dependency_policy: str,
    optimized_primitives: tuple[Any, ...],
    mesh_proxy: Any,
    loss: Any,
    baseline_loss: Any,
    initial_loss: Any,
    point_meta: Mapping[str, Any],
    topology_payload: Mapping[str, Any],
    optimization_summary: Mapping[str, Any],
    optimization_history: list[dict[str, object]],
    objective_improvement_record: Mapping[str, Any],
    objective_history: list[dict[str, Any]],
    history_payload: list[dict[str, Any]],
    parsed_config: Mapping[str, Any],
    config_warnings: tuple[str, ...] | list[str],
    target_view_weights: Mapping[str, float],
    view_signal_details: Mapping[str, Any],
    target_signal_warnings: tuple[str, ...] | list[str],
) -> tuple[Path | None, Path | None, dict[str, Path]]:
    if root is None:
        return None, None, {}

    try:
        from reconstruction.artifacts import write_json
        from reconstruction.mesh_io import write_obj, write_primitive_set
    except ImportError:  # pragma: no cover - package import path
        from ..artifacts import write_json
        from ..mesh_io import write_obj, write_primitive_set

    artifacts: dict[str, Path] = {}
    primitive_path = write_primitive_set(
        root / "p" / "diff.json",
        optimized_primitives,
        metadata={
            "backend": backend_choice,
            "loss": loss.terms,
            "per_view": loss.per_view,
            "surface_points": point_meta,
            "topology": dict(topology_payload),
            "optimization": dict(optimization_summary),
        },
    )
    artifacts["primitive_json"] = primitive_path
    mesh_path = write_obj(
        root / "m" / "diff.obj",
        mesh_proxy,
        header=(f"candidate {candidate_id}", backend_name),
    )
    artifacts["mesh_obj"] = mesh_path
    config_payload = differentiable_config_payload(
        backend_choice=backend_choice,
        optional_dependency_policy=optional_dependency_policy,
        parsed_config=parsed_config,
        config_warnings=config_warnings,
    )
    objective_path = root / "h" / "objective.json"
    write_json(
        objective_path,
        {
            "candidate_id": candidate_id,
            "backend": backend_choice,
            "objective_improvement": dict(objective_improvement_record),
            "history": objective_history,
            "config": config_payload,
            "loss": {
                "baseline_total": float(baseline_loss.total),
                "initial_total": float(initial_loss.total),
                "final_total": float(loss.total),
                "baseline_terms": dict(baseline_loss.terms),
                "initial_terms": dict(initial_loss.terms),
                "final_terms": dict(loss.terms),
            },
            "optimization": dict(optimization_summary),
            "optimization_history": optimization_history,
            "view_signal_weights": dict(target_view_weights),
            "view_signal_details": dict(view_signal_details),
            "view_signal_warnings": tuple(target_signal_warnings),
        },
    )
    artifacts["objective_history"] = objective_path
    history_path = root / "h" / "history.json"
    write_json(
        history_path,
        {
            "candidate_id": candidate_id,
            "backend": backend_choice,
            "config": config_payload,
            "history": history_payload,
            "objective_improvement": dict(objective_improvement_record),
            "objective_history": objective_history,
            "optimization": dict(optimization_summary),
            "optimization_history": optimization_history,
            "loss": {
                "total": float(loss.total),
                "terms": dict(loss.terms),
                "warnings": tuple(loss.warnings),
            },
            "per_view": {name: dict(values) for name, values in loss.per_view.items()},
        },
    )
    artifacts["refinement_history"] = history_path
    return primitive_path, mesh_path, artifacts
