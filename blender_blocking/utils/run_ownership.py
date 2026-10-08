"""Read-only reclamation planning for explicitly manifest-owned run artifacts.

This module never deletes files, creates ownership for old outputs, releases a
lease, or treats a dry-run receipt as permission to mutate. Producers must opt
in to the proposed manifest/lease contract before any lifecycle adoption.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

CATEGORIES = {"final_output", "diagnostic", "disposable", "shared_input"}
MANIFEST_NAME = "run-ownership.json"
LEASE_NAME = "run-lease.json"


def _plain_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in {"..", "."} for part in path.parts):
        raise ValueError("artifact must be a contained relative file path")
    target = root / path
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise ValueError("symlink/junction artifact paths are refused")
    if not target.resolve().is_relative_to(root):
        raise ValueError("artifact escapes its run root")
    return target


def plan_run_reclamation(run_root, *, max_hash_bytes=268435456):
    """Return eligible disposable bytes only; every mutation remains unimplemented.

    Active/unknown lease, missing ownership, content drift, unknown files and
    link escapes conservatively block the complete plan. A future executor must
    recheck paths, lease identity and content immediately before each mutation.
    """
    supplied = Path(run_root).absolute()
    root = supplied.resolve()
    result = {"protocol": "run-ownership-dry-run-v1", "run_root": str(root),
              "mode": "read_only", "status": "blocked", "mutation_supported": False, "eligible": [],
              "eligible_bytes": 0, "retained_bytes": 0, "blockers": [], "unknown_files": []}
    if (not root.is_dir() or root == Path(root.anchor) or supplied.is_symlink() or
            getattr(supplied, "is_junction", lambda: False)() or supplied != root):
        result["blockers"].append("run root is missing, filesystem root, or a redirected path")
        return result
    if isinstance(max_hash_bytes, bool) or not isinstance(max_hash_bytes, int) or max_hash_bytes < 1:
        raise ValueError("hash verification bound must be a positive integer")
    try:
        manifest_path = _plain_path(root, MANIFEST_NAME)
        lease_path = _plain_path(root, LEASE_NAME)
        if manifest_path.stat().st_size > 1048576 or lease_path.stat().st_size > 65536:
            raise ValueError("ownership metadata exceeds its bounded size")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lease = json.loads(lease_path.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != 1 or manifest.get("run_root") != str(root) or
                not isinstance(manifest.get("run_id"), str) or not manifest["run_id"] or
                not manifest.get("owner_token") or not isinstance(manifest.get("artifacts"), list) or
                len(manifest["artifacts"]) > 4096):
            raise ValueError("invalid explicit run ownership manifest")
        if (lease.get("run_id") != manifest["run_id"] or lease.get("owner_token") != manifest["owner_token"] or
                lease.get("status") != "released" or manifest.get("state") not in {"succeeded", "failed", "cancelled"}):
            raise ValueError("active, unknown, or mismatched run lease; crashed runs require separate recovery")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        result["blockers"].append(str(exc))
        return result
    normalize_key = lambda name: name.casefold() if os.name == "nt" else name
    known = {normalize_key(MANIFEST_NAME), normalize_key(LEASE_NAME)}
    verified_bytes = 0
    for entry in manifest["artifacts"]:
        try:
            relative, category = entry["path"], entry["category"]
            if category not in CATEGORIES:
                raise ValueError("unknown category or duplicate/control-file artifact")
            path = _plain_path(root, relative)
            relative = path.relative_to(root).as_posix()
            key = normalize_key(relative)
            if key in known:
                raise ValueError("duplicate/control-file artifact after path normalization")
            known.add(key)
            if not path.is_file():
                raise ValueError("owned artifact missing or not a regular file: " + relative)
            size = path.stat().st_size
            if (isinstance(entry["bytes"], bool) or not isinstance(entry["bytes"], int) or
                    size != entry["bytes"] or verified_bytes + size > max_hash_bytes):
                raise ValueError("artifact bytes changed or exceed hash bound: " + relative)
            verified_bytes += size
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1048576), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError("artifact hash changed: " + relative)
            row = {"path": relative, "bytes": size, "sha256": digest.hexdigest()}
            if category == "disposable":
                result["eligible"].append(row)
                result["eligible_bytes"] += size
            else:
                result["retained_bytes"] += size
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["blockers"].append(str(exc))
    # Enumerate without following directory links. Unknown entries cannot become
    # owned just because a proposed root happens to contain them.
    seen_entries = 0
    def inventory(directory, depth=0):
        nonlocal seen_entries
        if depth > 32:
            result["blockers"].append("inventory depth bound exceeded")
            return
        for path in directory.iterdir():
            seen_entries += 1
            if seen_entries > 8192:
                raise OSError("inventory entry bound exceeded")
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                result["blockers"].append("linked/redirected entry: " + relative)
            elif path.is_dir():
                inventory(path, depth + 1)
            elif normalize_key(relative) not in known:
                result["unknown_files"].append(relative)
    try:
        inventory(root)
    except OSError as exc:
        result["blockers"].append(str(exc))
    if result["unknown_files"]:
        result["blockers"].append("unknown files present; ownership is not inferred")
    if result["blockers"]:
        result["eligible"] = []
        result["eligible_bytes"] = 0
    result["status"] = "blocked" if result["blockers"] else "dry_run_ready"
    return result
