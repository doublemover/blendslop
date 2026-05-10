from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from blender_blocking.utils.optional_deps import dependency_report, probe_dependency
except Exception:  # pragma: no cover - script-style imports
    from utils.optional_deps import dependency_report, probe_dependency


def _optional_dependency_status(module_name: str) -> dict[str, Any]:
    return probe_dependency(module_name).to_dict()

def _visual_hull_dependency_report() -> dict[str, Any]:
    report: dict[str, Any] = dependency_report(("skimage", "open3d", "openvdb"))
    try:
        from volume import detect_openvdb

        openvdb_status = detect_openvdb().to_dict()
        report["openvdb"] = {
            **report.get("openvdb", {}),
            "binding_status": openvdb_status,
            "available": bool(openvdb_status.get("available")),
            "status": openvdb_status.get(
                "status",
                report.get("openvdb", {}).get("status"),
            ),
            "message": openvdb_status.get(
                "message",
                report.get("openvdb", {}).get("error", ""),
            ),
        }
    except Exception as exc:
        report["openvdb"] = {
            **report.get("openvdb", {}),
            "module_name": "openvdb",
            "available": False,
            "status": "probe_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return report
