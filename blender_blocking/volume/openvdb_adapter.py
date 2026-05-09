"""Optional OpenVDB adapter with lazy dependency imports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .contracts import Bounds3D, VoxelTransform
from .sparse_hash import SparseHashVolumeGrid


@dataclass(frozen=True)
class OpenVDBStatus:
    available: bool
    message: str
    module_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "message": self.message,
            "module_name": self.module_name,
        }


def detect_openvdb() -> OpenVDBStatus:
    module = _import_openvdb()
    if module is None:
        return OpenVDBStatus(
            available=False,
            message="OpenVDB Python bindings are not installed",
        )
    return OpenVDBStatus(
        available=True,
        message="OpenVDB Python bindings available",
        module_name=getattr(module, "__name__", "openvdb"),
    )


def export_to_openvdb(grid: SparseHashVolumeGrid, path: str | Path) -> OpenVDBStatus:
    module = _import_openvdb()
    if module is None:
        return OpenVDBStatus(
            available=False,
            message="OpenVDB export skipped because Python bindings are unavailable",
        )

    return OpenVDBStatus(
        available=False,
        module_name=getattr(module, "__name__", "openvdb"),
        message=(
            "OpenVDB bindings were detected, but direct export is not implemented "
            "for this binding variant; use NPZ interchange"
        ),
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
    module = _import_openvdb()
    if module is None:
        return OpenVDBStatus(
            available=False,
            message="OpenVDB import skipped because Python bindings are unavailable",
        )

    return OpenVDBStatus(
        available=False,
        module_name=getattr(module, "__name__", "openvdb"),
        message=(
            "OpenVDB bindings were detected, but direct import is not implemented "
            "for this binding variant; use NPZ interchange"
        ),
    )


class OpenVDBVolumeGrid(SparseHashVolumeGrid):
    """Sparse grid adapter used when OpenVDB bindings are not present."""

    backend = "openvdb"

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
        )


def _import_openvdb() -> Any:
    for module_name in ("pyopenvdb", "openvdb"):
        try:
            return __import__(module_name)
        except Exception:
            continue
    return None
