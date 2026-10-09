from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifact_io import artifact_paths


def save_volume_artifacts(
    *,
    grid: Any,
    root: Path,
    candidate_id: str,
    source_views: tuple[str, ...],
    extra: Mapping[str, Any],
    artifacts: dict[str, Path],
    metrics: dict[str, Any],
    warnings: list[str],
) -> Path | None:
    try:
        from volume import save_volume

        paths = artifact_paths(root)
        volume_dir = paths["volume_dir"]
        metadata = save_volume(
            grid,
            volume_dir,
            source_candidate_id=candidate_id,
            source_views=source_views,
            extra=extra,
        )
        artifacts["volume_metadata"] = volume_dir / "volume.json"
        artifacts["volume_npz"] = volume_dir / "volume.npz"
        metrics["volume_metadata"] = metadata.to_dict()
        return volume_dir
    except Exception as exc:
        warnings.append(f"failed to save volume artifact: {exc}")
        return None


def save_sdf_volume_artifacts(
    *,
    sdf_grid: Any,
    root: Path,
    candidate_id: str,
    source_views: tuple[str, ...],
    sdf_report: Mapping[str, Any],
    artifacts: dict[str, Path],
    metrics: dict[str, Any],
    warnings: list[str],
) -> None:
    try:
        from volume import save_volume

        paths = artifact_paths(root)
        sdf_dir = paths["sdf_volume_dir"]
        sdf_metadata = save_volume(
            sdf_grid,
            sdf_dir,
            source_candidate_id=candidate_id,
            source_views=source_views,
            extra={
                "backend": "visual_hull_voxel",
                "source": "visual_hull_occupancy",
                "sdf_projection": dict(sdf_report),
            },
        )
        artifacts["sdf_volume_metadata"] = sdf_dir / "volume.json"
        artifacts["sdf_volume_npz"] = sdf_dir / "volume.npz"
        metrics["sdf_volume_metadata"] = sdf_metadata.to_dict()
    except Exception as exc:
        warnings.append(f"failed to save SDF volume artifact: {exc}")


def export_openvdb_artifact(
    *,
    grid: Any,
    root: Path,
    artifacts: dict[str, Path],
    metrics: dict[str, Any],
    warnings: list[str],
) -> None:
    try:
        from volume import export_to_openvdb

        openvdb_export_path = artifact_paths(root)["openvdb"]
        openvdb_export_status = export_to_openvdb(grid, openvdb_export_path)
        metrics["openvdb_export"] = openvdb_export_status.to_dict()
        if openvdb_export_status.status == "exported":
            artifacts["volume_openvdb"] = openvdb_export_path
        else:
            warnings.append(
                "OpenVDB artifact export did not complete: "
                f"{openvdb_export_status.message}"
            )
    except Exception as exc:
        warnings.append(f"failed to export OpenVDB artifact: {exc}")
        metrics["openvdb_export"] = {
            "available": False,
            "status": "failed",
            "message": str(exc),
            "error_type": type(exc).__name__,
        }


def write_visual_hull_mesh_artifact(
    *,
    root: Path,
    candidate_id: str,
    backend_name: str,
    vertices: Any,
    faces: Any,
    artifacts: dict[str, Path],
) -> Path:
    from reconstruction.backend_artifact_ownership import write_owned_backend_obj

    mesh_path = write_owned_backend_obj(
        artifact_paths(root)["mesh_obj"],
        {"vertices": vertices, "faces": faces},
        producer="visual_hull_mesh_export",
        header=(f"candidate {candidate_id}", backend_name),
        metadata={"candidate_id": candidate_id, "backend_name": backend_name},
        receipt_artifacts=artifacts,
    )
    artifacts["mesh_obj"] = mesh_path
    return mesh_path


def write_visual_hull_editable_proxy_artifacts(
    *,
    root: Path,
    program_payload: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    artifacts: dict[str, Path],
) -> tuple[Path, Path]:
    from reconstruction.artifacts import write_json

    program_path = write_json(
        root / "ep" / "shape-program.json",
        program_payload,
    )
    diagnostics_path = write_json(
        root / "ep" / "diagnostics.json",
        diagnostics,
    )
    artifacts["editable_proxy_shape_program"] = program_path
    artifacts["editable_proxy_diagnostics"] = diagnostics_path
    return program_path, diagnostics_path
