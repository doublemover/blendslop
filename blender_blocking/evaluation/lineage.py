"""Run lineage, hashing, and artifact provenance helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .schemas import json_safe


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_json_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hash_bytes(encoded)


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_revision(cwd: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def dirty_worktree(cwd: str | Path | None = None) -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except Exception:
        return None


@dataclass(frozen=True)
class ArtifactRecord:
    key: str
    path: str
    sha256: str | None = None
    media_type: str = "application/octet-stream"
    size_bytes: int | None = None
    generated: bool = True
    committed_allowed: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "path": self.path,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "generated": self.generated,
            "committed_allowed": self.committed_allowed,
            "description": self.description,
        }


@dataclass(frozen=True)
class RunLineage:
    schema_version: str
    run_id: str
    parent_run_id: str | None = None
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    repo_revision: str | None = None
    dirty_worktree: bool | None = None
    command: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    input_hashes: Mapping[str, str] = field(default_factory=dict)
    config_hashes: Mapping[str, str] = field(default_factory=dict)
    random_seeds: Mapping[str, int] = field(default_factory=dict)
    artifacts: tuple[ArtifactRecord, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "created_at_utc": self.created_at_utc,
            "repo_revision": self.repo_revision,
            "dirty_worktree": self.dirty_worktree,
            "command": list(self.command),
            "environment": dict(self.environment),
            "input_hashes": dict(self.input_hashes),
            "config_hashes": dict(self.config_hashes),
            "random_seeds": dict(self.random_seeds),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

