from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def resfit_history_records(history: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "iteration": record.iteration,
            "total": record.total,
            "terms": dict(record.terms),
            "accepted_moves": record.accepted_moves,
            "rejected_moves": getattr(record, "rejected_moves", 0),
            "reason": getattr(record, "reason", ""),
            "step_size": record.step_size,
        }
        for record in history
    ]


def build_resfit_mesh_proxy(result: Any) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    try:
        from reconstruction.mesh_io import combine_primitive_meshes, mesh_arrays_from_object
    except ImportError:  # pragma: no cover - package import path
        from ...reconstruction.mesh_io import combine_primitive_meshes, mesh_arrays_from_object

    mesh_proxy = combine_primitive_meshes(result.primitives, resolution=24)
    mesh_vertices, mesh_faces = mesh_arrays_from_object(mesh_proxy)
    mesh_metadata = {
        "source": "primitive_proxy",
        "vertex_count": int(len(mesh_vertices)),
        "face_count": int(len(mesh_faces)),
        "resolution": 24,
    }
    try:
        from metrics.topology import mesh_topology_report

        topology_payload = mesh_topology_report(mesh_vertices, mesh_faces).to_dict()
    except Exception as exc:
        topology_payload = {
            "topology_score": 0.0,
            "warnings": [f"topology report unavailable: {exc}"],
        }
    return mesh_proxy, mesh_vertices, mesh_faces, mesh_metadata, topology_payload


def write_resfit_artifacts(
    *,
    root: Path | None,
    candidate_id: str,
    backend_name: str,
    primitive_family: str,
    result: Any,
    mesh_proxy: Any,
    surface_meta: Mapping[str, Any],
    occupied_meta: Mapping[str, Any],
    signal_summary: Mapping[str, Any],
    pipeline_config: Any,
    max_runtime_s: float | None,
    max_objective_evaluations: int | None,
    initial_primitives: Sequence[Any] | None,
    history_records: Sequence[Mapping[str, Any]],
) -> tuple[Path | None, Path | None, dict[str, Any]]:
    if root is None:
        return None, None, {}

    try:
        from reconstruction.artifacts import write_json
        from reconstruction.mesh_io import write_obj, write_primitive_set
    except ImportError:  # pragma: no cover - package import path
        from ...reconstruction.artifacts import write_json
        from ...reconstruction.mesh_io import write_obj, write_primitive_set

    artifacts: dict[str, Any] = {}
    primitive_path = write_primitive_set(
        root / "p" / "primitive-fit.json",
        result.primitives,
        metadata={
            "family": primitive_family,
            "initial_loss": result.initial_loss.terms,
            "final_loss": result.final_loss.terms,
            "surface_points": surface_meta,
            "occupied_points": occupied_meta,
            "signal_summary": signal_summary,
        },
    )
    mesh_path = write_obj(
        root / "m" / "primitive-fit.obj",
        mesh_proxy,
        header=(f"candidate {candidate_id}", backend_name),
    )
    artifacts["primitive_json"] = primitive_path
    artifacts["mesh_obj"] = mesh_path
    objective_path = write_json(
        root / "h" / "objective.json",
        {
            "weights": {
                "surface_residual": pipeline_config.weights.surface_residual,
                "visual_hull_occupancy": pipeline_config.weights.visual_hull_occupancy,
                "primitive_count": pipeline_config.weights.primitive_count,
                "overlap_penalty": pipeline_config.weights.overlap_penalty,
                "silhouette": pipeline_config.weights.silhouette,
                "topology_penalty": pipeline_config.weights.topology_penalty,
                "constraint_penalty": pipeline_config.weights.constraint_penalty,
                "uncertainty_penalty": pipeline_config.weights.uncertainty_penalty,
            },
            "initial_loss": result.initial_loss.terms,
            "final_loss": result.final_loss.terms,
            "improvement": result.initial_loss.total - result.final_loss.total,
            "optimization": {
                "termination_reason": result.optimization_termination_reason,
                "objective_evaluations": result.objective_evaluations,
                "elapsed_s": result.optimizer_elapsed_s,
                "selected_attempt": result.selected_attempt,
                "attempts": list(result.attempts),
                "residual_proposals": list(result.residual_proposals),
                "max_runtime_s": max_runtime_s,
                "max_objective_evaluations": max_objective_evaluations,
            },
            "history": list(history_records),
            "signal_summary": signal_summary,
            "initial_primitives_from_profile": bool(initial_primitives),
        },
    )
    artifacts["objective_history"] = objective_path
    return primitive_path, mesh_path, artifacts
