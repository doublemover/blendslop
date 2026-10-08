from __future__ import annotations

from typing import Any, Mapping

from .metrics import _openvdb_required, _retopology_required
from .postprocess import _mesh_required, _postprocess_required


def visual_hull_result_status(
    *,
    config: Mapping[str, Any],
    requested_backend: str,
    mesh_metrics: Mapping[str, Any],
    mesh_result: Any,
    postprocess_status: Mapping[str, Any],
    retopology_decision: Any,
    errors: list[str],
) -> tuple[str, bool]:
    status = "failed" if errors else "success"
    if _openvdb_required(config):
        openvdb_payload = mesh_metrics.get("openvdb")
        openvdb_export_payload = mesh_metrics.get("openvdb_export")
        if requested_backend == "openvdb" and isinstance(openvdb_payload, Mapping):
            if not bool(openvdb_payload.get("available")):
                status = "failed"
                errors.append(
                    str(
                        openvdb_payload.get(
                            "message",
                            "OpenVDB backend was required but bindings were unavailable",
                        )
                    )
                )
        if bool(config.get("export_openvdb")):
            if (
                not isinstance(openvdb_export_payload, Mapping)
                or openvdb_export_payload.get("status") != "exported"
            ):
                status = "failed"
                if isinstance(openvdb_export_payload, Mapping):
                    errors.append(
                        str(
                            openvdb_export_payload.get(
                                "message",
                                "OpenVDB export was required but did not complete",
                            )
                        )
                    )
                else:
                    errors.append(
                        "OpenVDB export was required but no export status was produced"
                    )
    if postprocess_status.get("status") == "failed" and _postprocess_required(config):
        status = "failed"
        errors.append(str(postprocess_status.get("message", "mesh postprocess failed")))
    mesh_extraction_payload = mesh_metrics.get("mesh_extraction")
    if (
        isinstance(mesh_extraction_payload, Mapping)
        and mesh_extraction_payload.get("status") == "failed"
    ):
        message = str(
            mesh_extraction_payload.get("message", "mesh extraction failed")
        )
        if _mesh_required(config):
            status = "failed"
            errors.append(message)
        elif status == "success":
            status = "degraded"
    if (
        mesh_result is not None
        and not mesh_result.available
        and str(getattr(mesh_result, "method", "")) != "points"
    ):
        message = str(
            getattr(mesh_result, "message", "mesh extraction did not produce a mesh")
        )
        if _mesh_required(config):
            status = "failed"
            errors.append(message)
        elif status == "success":
            status = "degraded"
    mesh_artifact_payload = mesh_metrics.get("mesh_artifact_export")
    if isinstance(mesh_artifact_payload, Mapping):
        artifact_required = bool(
            mesh_artifact_payload.get("required_for_render")
            or config.get("require_mesh_artifact")
            or config.get("require_renderable_mesh")
            or config.get("fail_on_mesh_artifact_skip")
        )
        artifact_status = mesh_artifact_payload.get("status")
        if artifact_status == "failed":
            message = str(
                mesh_artifact_payload.get("message", "mesh artifact export failed")
            )
            if artifact_required:
                status = "failed"
                errors.append(message)
            elif status == "success":
                status = "degraded"
        elif artifact_status == "skipped" and artifact_required:
            status = "failed"
            errors.append(
                str(
                    mesh_artifact_payload.get(
                        "reason",
                        "mesh artifact export skipped",
                    )
                )
            )
    if (
        retopology_decision is not None
        and not retopology_decision.accepted_for_editing
        and _retopology_required(config)
    ):
        status = "failed"
        errors.append(str(retopology_decision.reason))
    return status, status == "degraded"
