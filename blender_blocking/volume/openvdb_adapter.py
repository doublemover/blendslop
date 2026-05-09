"""Optional OpenVDB adapter with lazy dependency imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from .contracts import Bounds3D, VoxelTransform
from .sparse_hash import SparseHashVolumeGrid

try:
    from blender_blocking.utils.optional_deps import probe_dependency
except Exception:  # pragma: no cover - script-style imports
    from utils.optional_deps import probe_dependency


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
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        vdb_grid, active_voxels = _build_openvdb_grid(module, grid)
        _write_openvdb_grid(module, output, vdb_grid)
    except Exception as exc:
        return OpenVDBStatus(
            available=False,
            status="unsupported",
            message=f"OpenVDB export failed for this binding variant: {exc}",
            module_name=metadata["module_name"],
            module_version=metadata["module_version"],
            module_file=metadata["module_file"],
            variant=metadata["variant"],
            details={
                "operation": "export_to_openvdb",
                "import_attempts": attempts,
                "path": str(output),
                "error_type": type(exc).__name__,
                "active_voxels_attempted": int(getattr(grid.stats(), "active_voxels", 0)),
            },
        )

    return OpenVDBStatus(
        available=True,
        status="exported",
        message="OpenVDB grid exported through installed Python bindings",
        module_name=metadata["module_name"],
        module_version=metadata["module_version"],
        module_file=metadata["module_file"],
        variant=metadata["variant"],
        details={
            "operation": "export_to_openvdb",
            "import_attempts": attempts,
            "path": str(output),
            "grid_name": "occupancy",
            "active_voxels": int(active_voxels),
            "value_type": grid.value_type,
            "shape": list(grid.shape),
            "bounds": grid.bounds.to_dict(),
            "transform": grid.transform.to_dict(),
        },
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
    source = Path(path)
    try:
        vdb_grid = _read_openvdb_grid(module, source)
        dense, active_values, skipped_values = _dense_from_openvdb_grid(
            vdb_grid,
            transform=transform,
            value_type=value_type,
            default_value=default_value,
        )
    except Exception as exc:
        return OpenVDBStatus(
            available=False,
            status="unsupported",
            module_name=metadata["module_name"],
            module_version=metadata["module_version"],
            module_file=metadata["module_file"],
            variant=metadata["variant"],
            message=f"OpenVDB import failed for this binding variant: {exc}",
            details={
                "operation": "import_from_openvdb",
                "import_attempts": attempts,
                "path": str(source),
                "error_type": type(exc).__name__,
            },
        )

    status = OpenVDBStatus(
        available=True,
        status="imported",
        module_name=metadata["module_name"],
        module_version=metadata["module_version"],
        module_file=metadata["module_file"],
        variant=metadata["variant"],
        message="OpenVDB grid imported through installed Python bindings",
        details={
            "operation": "import_from_openvdb",
            "import_attempts": attempts,
            "path": str(source),
            "active_values": int(active_values),
            "skipped_out_of_bounds_values": int(skipped_values),
            "value_type": value_type,
            "shape": list(transform.shape),
            "bounds": bounds.to_dict(),
            "transform": transform.to_dict(),
        },
    )
    imported = OpenVDBVolumeGrid.from_dense(
        dense,
        bounds,
        transform=transform,
        value_type=value_type,
        default_value=default_value,
        chunk_size=chunk_size,
    )
    imported.openvdb_status = status
    return imported


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
    dependency = probe_dependency("openvdb", cache=False)
    attempts = tuple(dict(attempt) for attempt in dependency.attempts)
    if dependency.available:
        return dependency.module, attempts
    return None, attempts


def _describe_module(module: Any) -> dict[str, Any]:
    return {
        "module_name": getattr(module, "__name__", None),
        "module_version": getattr(module, "__version__", None),
        "module_file": str(getattr(module, "__file__", "")) or None,
        "variant": getattr(module, "__name__", "openvdb"),
    }


def _build_openvdb_grid(
    module: Any,
    grid: SparseHashVolumeGrid,
) -> tuple[Any, int]:
    grid_class = _select_grid_class(module, grid)
    background = _openvdb_background_value(grid)
    try:
        vdb_grid = grid_class(background=background)
    except TypeError:
        try:
            vdb_grid = grid_class(background)
        except TypeError:
            vdb_grid = grid_class()
    _set_grid_name(vdb_grid, "occupancy")
    _set_grid_transform(module, vdb_grid, grid.transform)

    active_voxels = 0
    for chunk in grid.iter_active_chunks():
        valid_slices = tuple(slice(0, int(size)) for size in chunk.valid_shape)
        valid_data = np.asarray(chunk.data[valid_slices])
        active_mask = valid_data != grid.default_value
        for relative_index in np.argwhere(active_mask):
            global_index = tuple(
                int(chunk.origin_index[axis] + relative_index[axis])
                for axis in range(3)
            )
            value = valid_data[tuple(int(v) for v in relative_index)]
            _set_openvdb_value(vdb_grid, global_index, _openvdb_value(value, grid))
            active_voxels += 1
    return vdb_grid, active_voxels


def _select_grid_class(module: Any, grid: SparseHashVolumeGrid) -> Any:
    if np.dtype(grid.dtype) == np.dtype(bool) and hasattr(module, "BoolGrid"):
        return getattr(module, "BoolGrid")
    if hasattr(module, "FloatGrid"):
        return getattr(module, "FloatGrid")
    if hasattr(module, "BoolGrid"):
        return getattr(module, "BoolGrid")
    raise RuntimeError("binding exposes neither FloatGrid nor BoolGrid")


def _openvdb_background_value(grid: SparseHashVolumeGrid) -> bool | float:
    if np.dtype(grid.dtype) == np.dtype(bool):
        return bool(grid.default_value)
    try:
        return float(grid.default_value)
    except (TypeError, ValueError):
        return 0.0


def _openvdb_value(value: Any, grid: SparseHashVolumeGrid) -> bool | float:
    if np.dtype(grid.dtype) == np.dtype(bool):
        return bool(value)
    return float(value)


def _set_grid_name(vdb_grid: Any, name: str) -> None:
    if hasattr(vdb_grid, "setName"):
        vdb_grid.setName(name)
        return
    try:
        setattr(vdb_grid, "name", name)
    except Exception:
        pass


def _set_grid_transform(module: Any, vdb_grid: Any, transform: VoxelTransform) -> None:
    create_linear = getattr(module, "createLinearTransform", None)
    if create_linear is None:
        return
    voxel_size = float(np.mean(np.asarray(transform.voxel_size, dtype=float)))
    try:
        vdb_transform = create_linear(voxelSize=voxel_size)
    except TypeError:
        vdb_transform = create_linear(voxel_size)
    if hasattr(vdb_grid, "setTransform"):
        vdb_grid.setTransform(vdb_transform)
        return
    try:
        setattr(vdb_grid, "transform", vdb_transform)
    except Exception:
        pass


def _set_openvdb_value(vdb_grid: Any, coord: tuple[int, int, int], value: Any) -> None:
    targets = []
    if hasattr(vdb_grid, "getAccessor"):
        targets.append(vdb_grid.getAccessor())
    targets.append(vdb_grid)
    for target in targets:
        for method_name in ("setValueOn", "setValue"):
            method = getattr(target, method_name, None)
            if method is None:
                continue
            method(coord, value)
            return
    raise RuntimeError("binding exposes no setValueOn/setValue accessor")


def _write_openvdb_grid(module: Any, output: Path, vdb_grid: Any) -> None:
    write = getattr(module, "write", None)
    if write is None:
        raise RuntimeError("binding exposes no write() function")
    try:
        write(str(output), grids=[vdb_grid])
        return
    except TypeError:
        pass
    try:
        write(str(output), [vdb_grid])
        return
    except TypeError:
        pass
    write(str(output), vdb_grid)


def _read_openvdb_grid(module: Any, source: Path) -> Any:
    read = getattr(module, "read", None)
    if read is None:
        raise RuntimeError("binding exposes no read() function")
    grids = read(str(source))
    if isinstance(grids, Mapping):
        if "occupancy" in grids:
            return grids["occupancy"]
        if grids:
            return next(iter(grids.values()))
    if isinstance(grids, (list, tuple)):
        if not grids:
            raise RuntimeError("OpenVDB file contained no grids")
        return grids[0]
    if grids is None:
        raise RuntimeError("OpenVDB read returned no grid")
    return grids


def _dense_from_openvdb_grid(
    vdb_grid: Any,
    *,
    transform: VoxelTransform,
    value_type: str,
    default_value: Any,
) -> tuple[np.ndarray, int, int]:
    dtype = bool if value_type == "occupancy_bool" else np.float32
    dense = np.full(transform.shape, default_value, dtype=dtype)
    active = 0
    skipped = 0
    shape = tuple(int(v) for v in transform.shape)
    for coord, value in _iter_openvdb_active_values(vdb_grid):
        ix, iy, iz = coord
        if ix < 0 or iy < 0 or iz < 0 or ix >= shape[0] or iy >= shape[1] or iz >= shape[2]:
            skipped += 1
            continue
        dense[ix, iy, iz] = bool(value) if dtype is bool else float(value)
        active += 1
    return dense, active, skipped


def _iter_openvdb_active_values(vdb_grid: Any) -> Iterable[tuple[tuple[int, int, int], Any]]:
    active_voxels = getattr(vdb_grid, "active_voxels", None)
    if isinstance(active_voxels, Mapping):
        for coord, value in active_voxels.items():
            yield _coord_tuple(coord), value
        return

    iter_active = getattr(vdb_grid, "iter_active_values", None)
    if callable(iter_active):
        for item in iter_active():
            coord, value = _split_coord_value(item)
            yield coord, value
        return

    items = getattr(vdb_grid, "items", None)
    if callable(items):
        for item in items():
            coord, value = _split_coord_value(item)
            yield coord, value
        return

    iter_on_values = getattr(vdb_grid, "iterOnValues", None)
    if callable(iter_on_values):
        yielded = False
        for item in iter_on_values():
            for coord, value in _coords_values_from_openvdb_item(item):
                yielded = True
                yield coord, value
        if yielded:
            return

    iter_on = getattr(vdb_grid, "iterOn", None)
    if callable(iter_on):
        yielded = False
        for item in iter_on():
            for coord, value in _coords_values_from_openvdb_item(item):
                yielded = True
                yield coord, value
        if yielded:
            return

    raise RuntimeError("binding exposes no supported active-value iterator")


def _split_coord_value(item: Any) -> tuple[tuple[int, int, int], Any]:
    if isinstance(item, Mapping):
        return _coord_tuple(item.get("coord", item.get("index"))), item.get("value", True)
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return _coord_tuple(item[0]), item[1]
    if isinstance(item, (list, tuple)) and len(item) >= 4:
        return _coord_tuple(item[:3]), item[3]
    coord = getattr(item, "coord", None) or getattr(item, "index", None)
    value = getattr(item, "value", True)
    if callable(value):
        value = value()
    return _coord_tuple(coord), value


def _coords_values_from_openvdb_item(item: Any) -> Iterable[tuple[tuple[int, int, int], Any]]:
    value = getattr(item, "value", True)
    if callable(value):
        value = value()
    minimum = _call_or_value(getattr(item, "min", None))
    maximum = _call_or_value(getattr(item, "max", None))
    if minimum is None:
        coord = getattr(item, "coord", None) or getattr(item, "index", None)
        if coord is not None:
            yield _coord_tuple(coord), value
        return
    lo = _coord_tuple(minimum)
    hi = _coord_tuple(maximum) if maximum is not None else lo
    for ix in range(lo[0], hi[0] + 1):
        for iy in range(lo[1], hi[1] + 1):
            for iz in range(lo[2], hi[2] + 1):
                yield (ix, iy, iz), value


def _call_or_value(value: Any) -> Any:
    if callable(value):
        try:
            return value()
        except TypeError:
            return value
    return value


def _coord_tuple(value: Any) -> tuple[int, int, int]:
    if value is None:
        raise RuntimeError("OpenVDB iterator item did not include coordinates")
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) < 3:
            raise RuntimeError(f"coordinate has fewer than three components: {value!r}")
        return (int(value[0]), int(value[1]), int(value[2]))
    attrs = []
    for name in ("x", "y", "z"):
        attr = getattr(value, name, None)
        attrs.append(attr() if callable(attr) else attr)
    if all(component is not None for component in attrs):
        return (int(attrs[0]), int(attrs[1]), int(attrs[2]))
    try:
        return (int(value[0]), int(value[1]), int(value[2]))
    except Exception as exc:
        raise RuntimeError(f"unsupported coordinate object: {value!r}") from exc


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)
