from __future__ import annotations

from typing import Any


def mesh_method_status(mesh_result: Any) -> dict[str, Any]:
    return {
        "available": bool(getattr(mesh_result, "available", False)),
        "status": getattr(mesh_result, "status", "unknown"),
        "method": getattr(mesh_result, "method", "unknown"),
        "requested_method": getattr(mesh_result, "requested_method", getattr(mesh_result, "method", "unknown")),
    }
