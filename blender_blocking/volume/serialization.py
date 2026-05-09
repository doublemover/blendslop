"""Volume metadata JSON and NPZ serialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .chunks import ChunkedVolumeGrid
from .contracts import ChunkKey, VolumeGrid, VolumeMetadata
from .dense import DenseVolumeGrid
from .sparse_hash import SparseHashVolumeGrid


FORMAT_VERSION = 1


def stable_json_hash(data: Mapping[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def save_volume(
    grid: VolumeGrid,
    directory: str | Path,
    *,
    source_candidate_id: Optional[str] = None,
    source_masks: tuple[str, ...] = (),
    source_views: tuple[str, ...] = (),
    generation_seed: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> VolumeMetadata:
    """Write volume.json and volume.npz into directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / "volume.npz"
    json_path = directory / "volume.json"

    backend = getattr(grid, "backend", grid.__class__.__name__)
    arrays, payload_extra = _arrays_for_grid(grid, backend)
    np.savez_compressed(npz_path, **arrays)

    metadata_extra = {**payload_extra, **dict(extra or {})}
    metadata = _metadata_for_grid(
        grid,
        backend=backend,
        source_candidate_id=source_candidate_id,
        source_masks=source_masks,
        source_views=source_views,
        generation_seed=generation_seed,
        extra=_normalize_metadata(metadata_extra),
    )
    metadata_dict = metadata.to_dict()
    metadata_dict["hashes"] = {
        "metadata_without_hashes": stable_json_hash(
            {**metadata_dict, "hashes": {}}
        ),
        "npz_sha256": file_sha256(npz_path),
    }
    json_path.write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return VolumeMetadata.from_dict(metadata_dict)


def load_volume(
    directory: str | Path, *, validate_hashes: bool = True
) -> VolumeGrid:
    directory = Path(directory)
    json_path = directory / "volume.json"
    npz_path = directory / "volume.npz"
    metadata_dict = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = VolumeMetadata.from_dict(metadata_dict)

    if validate_hashes:
        expected_metadata = metadata.hashes.get("metadata_without_hashes")
        if expected_metadata:
            actual_metadata = stable_json_hash({**metadata_dict, "hashes": {}})
            if actual_metadata != expected_metadata:
                raise ValueError("volume.json hash does not match metadata contents")
        expected = metadata.hashes.get("npz_sha256")
        if expected and file_sha256(npz_path) != expected:
            raise ValueError("volume.npz hash does not match metadata")

    with np.load(npz_path, allow_pickle=False) as npz:
        if metadata.backend == "dense":
            return DenseVolumeGrid(
                npz["data"],
                metadata.bounds,
                transform=metadata.transform,
                value_type=metadata.value_type,
                default_value=metadata.default_value,
                chunk_size=metadata.chunk_size,
            )
        if metadata.backend == "chunked":
            return _load_chunked(metadata, npz, sparse=False)
        if metadata.backend in {"sparse_hash", "openvdb"}:
            return _load_chunked(metadata, npz, sparse=True)

    raise ValueError(f"unsupported volume backend: {metadata.backend}")


def _metadata_for_grid(
    grid: VolumeGrid,
    *,
    backend: str,
    source_candidate_id: Optional[str],
    source_masks: tuple[str, ...],
    source_views: tuple[str, ...],
    generation_seed: Optional[int],
    extra: Mapping[str, Any],
) -> VolumeMetadata:
    dense_shape = tuple(int(v) for v in grid.transform.shape)
    dtype = str(_grid_dtype(grid))
    return VolumeMetadata(
        format_version=FORMAT_VERSION,
        backend=backend,
        value_type=grid.value_type,
        dtype=dtype,
        bounds=grid.bounds,
        transform=grid.transform,
        shape=dense_shape,
        chunk_size=grid.chunk_size,
        default_value=grid.default_value,
        active_voxels=grid.active_voxel_count(),
        source_candidate_id=source_candidate_id,
        source_masks=source_masks,
        source_views=source_views,
        generation_seed=generation_seed,
        hashes={},
        extra=extra,
    )


def _arrays_for_grid(
    grid: VolumeGrid, backend: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if backend == "dense":
        dense = grid.to_dense()
        return {"data": dense, "data_sha256": np.array(array_sha256(dense))}, {}

    keys = []
    chunks = []
    for chunk in grid.iter_active_chunks():
        keys.append(chunk.key.to_tuple())
        chunks.append(chunk.data)

    key_array = np.asarray(keys, dtype=np.int64).reshape((-1, 3))
    dtype = _grid_dtype(grid)
    payload_extra: dict[str, Any] = {}
    if backend == "openvdb":
        openvdb_metadata = getattr(grid, "openvdb_metadata", None)
        if callable(openvdb_metadata):
            payload_extra["openvdb"] = openvdb_metadata()
        else:
            status = getattr(grid, "openvdb_status", None)
            if status is not None:
                payload_extra["openvdb"] = status
        payload_extra["storage_backend"] = "sparse_hash"
        payload_extra["serialization"] = "npz_interchange"

    if backend in {"sparse_hash", "openvdb"} and dtype == np.dtype(bool):
        if chunks:
            chunk_stack = np.asarray(
                [
                    np.packbits(np.asarray(chunk, dtype=bool).reshape(-1))
                    for chunk in chunks
                ],
                dtype=np.uint8,
            )
        else:
            packed_size = (grid.chunk_size**3 + 7) // 8
            chunk_stack = np.empty((0, packed_size), dtype=np.uint8)
        return {
            "keys": key_array,
            "packed_chunks": chunk_stack,
            "packed": np.array(True),
        }, {**payload_extra, "packed_bool_chunks": True}

    if chunks:
        chunk_array = np.asarray(chunks, dtype=dtype)
    else:
        chunk_array = np.empty(
            (0, grid.chunk_size, grid.chunk_size, grid.chunk_size),
            dtype=dtype,
        )
    return {
        "keys": key_array,
        "chunks": chunk_array,
        "packed": np.array(False),
    }, payload_extra


def _normalize_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _normalize_metadata_value(item)
        for key, item in dict(value).items()
    }


def _normalize_metadata_value(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return _normalize_metadata_value(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _normalize_metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_metadata_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.hex()
    return value


def _grid_dtype(grid: VolumeGrid) -> np.dtype:
    dtype = getattr(grid, "dtype", None)
    if dtype is not None:
        return np.dtype(dtype)
    data = getattr(grid, "data", None)
    if data is not None:
        return np.asarray(data).dtype
    for chunk in grid.iter_active_chunks():
        return chunk.data.dtype
    return np.asarray(grid.default_value).dtype


def _load_chunked(
    metadata: VolumeMetadata, npz: Mapping[str, np.ndarray], *, sparse: bool
) -> VolumeGrid:
    keys = [ChunkKey.from_iterable(row) for row in npz["keys"]]
    chunks: dict[ChunkKey, np.ndarray] = {}
    if bool(npz["packed"]):
        packed_chunks = npz["packed_chunks"]
        for key, packed in zip(keys, packed_chunks):
            chunks[key] = SparseHashVolumeGrid.unpack_bool_chunk(
                packed, metadata.chunk_size
            )
    else:
        for key, chunk in zip(keys, npz["chunks"]):
            chunks[key] = np.asarray(chunk, dtype=np.dtype(metadata.dtype))

    if metadata.backend == "openvdb":
        from .openvdb_adapter import OpenVDBVolumeGrid

        return OpenVDBVolumeGrid(
            chunks=chunks,
            shape=metadata.shape,
            bounds=metadata.bounds,
            transform=metadata.transform,
            value_type=metadata.value_type,
            dtype=np.dtype(metadata.dtype),
            default_value=metadata.default_value,
            chunk_size=metadata.chunk_size,
            openvdb_status=metadata.extra.get("openvdb"),
        )

    grid_cls = SparseHashVolumeGrid if sparse else ChunkedVolumeGrid
    return grid_cls(
        chunks=chunks,
        shape=metadata.shape,
        bounds=metadata.bounds,
        transform=metadata.transform,
        value_type=metadata.value_type,
        dtype=np.dtype(metadata.dtype),
        default_value=metadata.default_value,
        chunk_size=metadata.chunk_size,
    )
