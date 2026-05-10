from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


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

        metadata = save_volume(
            grid,
            root / "volume",
            source_candidate_id=candidate_id,
            source_views=source_views,
            extra=extra,
        )
        artifacts["volume_metadata"] = root / "volume" / "volume.json"
        artifacts["volume_npz"] = root / "volume" / "volume.npz"
        metrics["volume_metadata"] = metadata.to_dict()
        return root / "volume"
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

        sdf_metadata = save_volume(
            sdf_grid,
            root / "sdf_volume",
            source_candidate_id=candidate_id,
            source_views=source_views,
            extra={
                "backend": "visual_hull_voxel",
                "source": "visual_hull_occupancy",
                "sdf_projection": dict(sdf_report),
            },
        )
        artifacts["sdf_volume_metadata"] = root / "sdf_volume" / "volume.json"
        artifacts["sdf_volume_npz"] = root / "sdf_volume" / "volume.npz"
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

        openvdb_export_path = root / "volume" / "volume.vdb"
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
    from reconstruction.mesh_io import write_obj

    mesh_path = write_obj(
        root / "mesh" / "visual_hull.obj",
        {"vertices": vertices, "faces": faces},
        header=(f"candidate {candidate_id}", backend_name),
    )
    artifacts["mesh_obj"] = mesh_path
    return mesh_path
