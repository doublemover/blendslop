"""Deterministic on-disk cache for sparse/chunked volume reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional

import numpy as np

from .serialization import stable_json_hash


@dataclass(frozen=True)
class ChunkCacheKey:
    namespace: str
    digest: str

    @classmethod
    def from_payload(cls, namespace: str, payload: Mapping[str, Any]) -> "ChunkCacheKey":
        namespace_slug = _safe_slug(namespace)
        digest = stable_json_hash({"namespace": namespace_slug, "payload": _json_safe(payload)})
        return cls(namespace=namespace_slug, digest=digest)

    @property
    def filename(self) -> str:
        return f"{self.digest}.npz"

    def to_dict(self) -> dict[str, str]:
        return {"namespace": self.namespace, "digest": self.digest}


@dataclass(frozen=True)
class ChunkCacheStats:
    directory: str
    namespace: str
    enabled: bool
    hits: int = 0
    misses: int = 0
    writes: int = 0
    read_errors: int = 0
    write_errors: int = 0
    bytes_read: int = 0
    bytes_written: int = 0

    @property
    def requests(self) -> int:
        return int(self.hits + self.misses + self.read_errors)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return float(self.hits / total) if total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "directory": self.directory,
            "namespace": self.namespace,
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "read_errors": self.read_errors,
            "write_errors": self.write_errors,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "requests": self.requests,
            "hit_rate": self.hit_rate,
        }


class VolumeChunkCache:
    """Small deterministic NPZ cache for independently carved volume chunks."""

    def __init__(
        self,
        directory: str | Path,
        *,
        namespace: str = "volume",
        read: bool = True,
        write: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.namespace = _safe_slug(namespace)
        self.read_enabled = bool(read)
        self.write_enabled = bool(write)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.read_errors = 0
        self.write_errors = 0
        self.bytes_read = 0
        self.bytes_written = 0

    @property
    def enabled(self) -> bool:
        return self.read_enabled or self.write_enabled

    def path_for_key(self, key: ChunkCacheKey) -> Path:
        if key.namespace != self.namespace:
            raise ValueError("cache key namespace does not match cache namespace")
        return self.directory / key.namespace / key.filename

    def load(
        self,
        key: ChunkCacheKey,
        *,
        expected_shape: tuple[int, int, int],
        expected_dtype: Any = bool,
    ) -> Optional[np.ndarray]:
        if not self.read_enabled:
            self.misses += 1
            return None
        path = self.path_for_key(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            with np.load(path, allow_pickle=False) as npz:
                data = np.asarray(npz["data"], dtype=np.dtype(expected_dtype))
                metadata = json.loads(str(npz["metadata"].item()))
            if tuple(data.shape) != tuple(expected_shape):
                self.read_errors += 1
                return None
            if metadata.get("digest") != key.digest:
                self.read_errors += 1
                return None
            self.hits += 1
            self.bytes_read += int(path.stat().st_size)
            return data.copy()
        except Exception:
            self.read_errors += 1
            return None

    def store(
        self,
        key: ChunkCacheKey,
        data: np.ndarray,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        if not self.write_enabled:
            return False
        path = self.path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "namespace": key.namespace,
            "digest": key.digest,
            "shape": list(np.asarray(data).shape),
            "dtype": str(np.asarray(data).dtype),
            **dict(metadata or {}),
        }
        try:
            np.savez_compressed(
                path,
                data=np.asarray(data),
                metadata=np.array(json.dumps(_json_safe(payload), sort_keys=True)),
            )
            self.writes += 1
            self.bytes_written += int(path.stat().st_size)
            return True
        except Exception:
            self.write_errors += 1
            return False

    def stats(self) -> ChunkCacheStats:
        return ChunkCacheStats(
            directory=self.directory.as_posix(),
            namespace=self.namespace,
            enabled=self.enabled,
            hits=self.hits,
            misses=self.misses,
            writes=self.writes,
            read_errors=self.read_errors,
            write_errors=self.write_errors,
            bytes_read=self.bytes_read,
            bytes_written=self.bytes_written,
        )


def chunk_cache_key(namespace: str, payload: Mapping[str, Any]) -> ChunkCacheKey:
    return ChunkCacheKey.from_payload(namespace, payload)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "volume"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
