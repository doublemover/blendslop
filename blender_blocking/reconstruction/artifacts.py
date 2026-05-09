"""Artifact helpers for reconstruction runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def stable_json_dumps(payload: Mapping[str, Any]) -> str:
    """Serialize JSON deterministically for hashes and manifests."""
    return json.dumps(payload, sort_keys=True, indent=2, default=str)


def ensure_artifact_root(path: str | Path) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(stable_json_dumps(payload) + "\n", encoding="utf-8")
    return output


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(payload: Mapping[str, Any]) -> str:
    return hash_bytes(stable_json_dumps(payload).encode("utf-8"))
