"""Optional OpenVDB adapter with lazy dependency imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .contracts import Bounds3D, VoxelTransform
from .sparse_hash import SparseHashVolumeGrid


@dataclass(frozen=True)
class OpenVDBStatus:
    available: bool
    status: str
    message: str
    module_name: Optional[str] = None
    module_version: Optional[str] = None
    module_file: Optional[str] = None
    variant: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OpenVDBStatus":
        return cls(
            available=bool(data.get("available", False)),
            status=str(data.get("status", "unknown")),
            message=str(data.get("message", "")),
            module_name=_optional_str(data.get("module_name")),
            module_version=_optional_str(data.get("module_version")),
            module_file=_optional_str(data.get("module_file")),
            variant=_optional_str(data.get("variant")),
            details=dict(data.get("details", {})),
        )

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "status": self.status,
            "message": self.message,
            "module_name": self.module_name,
            "module_version": self.module_version,
            "module_file": self.module_file,
            "variant": self.variant,
            "details": dict(self.details),
        }


def detect_openvdb() -> OpenVDBStatus:
    module, attempts = _import_openvdb()
    if module is None:
        return OpenVDBStatus(
            available=False,
            status="missing",
            message="OpenVDB Python bindings are not installed",
            details={"import_attempts": attempts},
        )

    metadata = _describe_module(module)
    return OpenVDBStatus(
        available=True,
        status="detected",
        message="OpenVDB Python bindings available",
        module_name=metadata["module_name"],
        module_version=metadata["module_version"],
        module_file=metadata["module_file"],
        variant=metadata["variant"],
        details={**metadata, "import_attempts": attempts},
    )


def export_to_openvdb(grid: SparseHashVolumeGrid, path: str | Path) -> OpenVDBStatus:
    module, attempts = _import_openvdb()
    if module is None:
        return OpenVDBStatus(
            available=False,
            status="missing",
            message="OpenVDB export skipped because Python bindings are unavailable",
            details={"import_attempts": attempts},
        )

    metadata = _describe_module(module)
    _ = grid  # placeholder for future implementations where grid variants diverge.
    _ = path
    return OpenVDBStatus(
        available=False,
        status="unsupported",
        message=(
            "OpenVDB bindings were detected, but direct export is not implemented "
            "for this binding variant; use NPZ interchange"
        ),
        module_name=metadata["module_name"],
        module_version=metadata["module_version"],
        module_file=metadata["module_file"],
        variant=metadata["variant"],
        details={"operation": "export_to_openvdb", "import_attempts": attempts},
    )


def import_from_openvdb(
    path: str | Path,
    *,
    bounds: Bounds3D,
    transform: VoxelTransform,
    value_type: str = "occupancy_bool",
    chunk_size: int = 32,
    default_value: Any = False,
) -> SparseHashVolumeGrid | OpenVDBStatus:
    module, attempts = _import_openvdb()
    if module is None:
        return OpenVDBStatus(
            available=False,
            status="missing",
            message="OpenVDB import skipped because Python bindings are unavailable",
            details={"import_attempts": attempts},
        )

    metadata = _describe_module(module)
    return OpenVDBStatus(
        available=False,
        status="unsupported",
        module_name=metadata["module_name"],
        module_version=metadata["module_version"],
        module_file=metadata["module_file"],
        variant=metadata["variant"],
        message=(
            "OpenVDB bindings were detected, but direct import is not implemented "
            "for this binding variant; use NPZ interchange"
        ),
        details={"operation": "import_from_openvdb", "import_attempts": attempts},
    )


class OpenVDBVolumeGrid(SparseHashVolumeGrid):
    """Sparse grid adapter used when OpenVDB bindings are not present."""

    backend = "openvdb"

    def __init__(
        self,
        *args: Any,
        openvdb_status: OpenVDBStatus | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(openvdb_status, OpenVDBStatus):
            self.openvdb_status = openvdb_status
        elif isinstance(openvdb_status, Mapping):
            self.openvdb_status = OpenVDBStatus.from_dict(openvdb_status)
        else:
            self.openvdb_status = OpenVDBStatus(
                available=False,
                status="npz_interchange",
                message=(
                    "OpenVDBVolumeGrid is backed by sparse hash chunks and NPZ "
                    "interchange; no direct OpenVDB grid is attached"
                ),
                variant="sparse_hash_interchange",
                details={
                    "storage_backend": "sparse_hash",
                    "direct_openvdb_grid": False,
                },
            )

    def openvdb_metadata(self) -> dict[str, Any]:
        metadata = self.openvdb_status.to_dict()
        metadata["storage_backend"] = "sparse_hash"
        metadata["serialization"] = "npz_interchange"
        metadata["direct_openvdb_grid"] = False
        return metadata

    @classmethod
    def unavailable(
        cls,
        *,
        shape: tuple[int, int, int],
        bounds: Bounds3D,
        transform: Optional[VoxelTransform] = None,
        chunk_size: int = 32,
    ) -> "OpenVDBVolumeGrid":
        return cls(
            chunks={},
            shape=shape,
            bounds=bounds,
            transform=transform,
            value_type="occupancy_bool",
            dtype=np.dtype(bool),
            default_value=False,
            chunk_size=chunk_size,
            openvdb_status=OpenVDBStatus(
                available=False,
                status="unavailable",
                message=(
                    "OpenVDB grid requested, but this adapter is storing an empty "
                    "sparse NPZ interchange grid"
                ),
                variant="sparse_hash_interchange",
                details={
                    "storage_backend": "sparse_hash",
                    "direct_openvdb_grid": False,
                },
            ),
        )


def _import_openvdb() -> tuple[Any | None, tuple[dict[str, Any], ...]]:
    attempts: list[dict[str, Any]] = []
    for module_name in ("pyopenvdb", "openvdb"):
        try:
            module = __import__(module_name)
            return module, tuple(attempts)
        except Exception as exc:
            attempts.append(
                {
                    "module_name": module_name,
                    "status": "import_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
    return None, tuple(attempts)


def _describe_module(module: Any) -> dict[str, Any]:
    return {
        "module_name": getattr(module, "__name__", None),
        "module_version": getattr(module, "__version__", None),
        "module_file": str(getattr(module, "__file__", "")) or None,
        "variant": getattr(module, "__name__", "openvdb"),
    }


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)
