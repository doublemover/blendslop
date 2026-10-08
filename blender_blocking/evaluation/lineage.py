"""Run lineage, hashing, and artifact provenance helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
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


def hash_path(path: str | Path) -> str | None:
    """Hash a file or directory tree in a stable, content-addressed way."""
    path = Path(path)
    if not path.exists():
        return None
    if path.is_file():
        return hash_file(path)
    if not path.is_dir():
        return None
    records = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        rel = child.relative_to(path).as_posix()
        records.append({"path": rel, "sha256": hash_file(child)})
    return hash_json_payload({"directory": path.name, "files": records})


def path_size_bytes(path: str | Path) -> int | None:
    path = Path(path)
    if not path.exists():
        return None
    if path.is_file():
        return int(path.stat().st_size)
    if path.is_dir():
        return int(sum(child.stat().st_size for child in path.rglob("*") if child.is_file()))
    return None


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


def capture_environment() -> dict[str, str]:
    """Capture a compact reproducibility environment block."""
    environment = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    try:
        import bpy  # type: ignore

        environment["blender_version"] = str(bpy.app.version_string)
    except Exception:
        environment["blender_version"] = ""
    return environment


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

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        key: str,
        root: str | Path | None = None,
        generated: bool = True,
        committed_allowed: bool = False,
        description: str = "",
    ) -> "ArtifactRecord":
        path_obj = Path(path)
        display_path = _display_path(path_obj, root)
        return cls(
            key=key,
            path=display_path,
            sha256=hash_path(path_obj),
            media_type=_media_type(path_obj),
            size_bytes=path_size_bytes(path_obj),
            generated=generated,
            committed_allowed=committed_allowed,
            description=description,
        )


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


def write_run_lineage(lineage: RunLineage, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(lineage.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def write_reproduce_script(
    command: tuple[str, ...],
    path: str | Path,
    *,
    cwd: str | Path | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    command_line = _powershell_command(command)
    lines = []
    if cwd is not None:
        lines.append(f"Set-Location -LiteralPath {_powershell_literal(str(Path(cwd)))}")
    lines.append(command_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _display_path(path: Path, root: str | Path | None) -> str:
    if root is None:
        return path.as_posix()
    try:
        return path.resolve(strict=False).relative_to(Path(root).resolve(strict=False)).as_posix()
    except ValueError:
        return path.resolve(strict=False).as_posix()


def _media_type(path: Path) -> str:
    if path.is_dir():
        return "inode/directory"
    suffix = path.suffix.lower()
    return {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".md": "text/markdown",
        ".html": "text/html",
        ".txt": "text/plain",
        ".ps1": "text/x-powershell",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".obj": "model/obj",
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".npz": "application/octet-stream",
    }.get(suffix, "application/octet-stream")


def _powershell_command(command: tuple[str, ...]) -> str:
    return " ".join(_powershell_literal(part) for part in command)


def _powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
