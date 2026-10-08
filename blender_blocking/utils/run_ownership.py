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



class OwnedRun:
    """Producer-owned fresh scratch, durable lease/receipt, and no reclamation.

    Ownership is created only for a fresh generated directory. Existing inputs,
    caches and final outputs stay external; producers register generated files
    explicitly. Closing releases the lease after the producer joins its children.
    """
    def __init__(self, parent, *, producer, max_generated_bytes=16777216, shared_inputs=None):
        import tempfile, uuid
        if (isinstance(max_generated_bytes, bool) or not isinstance(max_generated_bytes, int) or
                max_generated_bytes <= 0 or not isinstance(producer, str) or not producer):
            raise ValueError("owned producer requires a name and positive byte bound")
        parent = Path(parent).absolute()
        if parent != parent.resolve() or parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
            raise ValueError("owned scratch parent must not be redirected")
        parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="owned-", dir=parent)).resolve()
        self.run_id, self.owner_token = uuid.uuid4().hex, uuid.uuid4().hex
        self.max_generated_bytes = max_generated_bytes
        self.records = {}
        self.state, self.error, self.closed = "active", None, False
        self.auxiliary_errors = []
        self.manifest = {"schema_version": 1, "run_root": str(self.root), "run_id": self.run_id,
                         "owner_token": self.owner_token, "producer": producer, "pid": os.getpid(),
                         "state": "active", "artifacts": [], "max_generated_bytes": max_generated_bytes,
                         "shared_inputs": dict(shared_inputs or {}), "reclamation": "unimplemented; all files retained"}
        self._write_metadata("active")

    def _atomic_json(self, path, value):
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _write_metadata(self, lease_status):
        self.manifest.update(state=self.state, error=self.error, artifacts=list(self.records.values()),
                             auxiliary_errors=list(self.auxiliary_errors))
        self._atomic_json(self.root / MANIFEST_NAME, self.manifest)
        self._atomic_json(self.root / LEASE_NAME, {"run_id": self.run_id, "owner_token": self.owner_token,
                          "pid": os.getpid(), "status": lease_status})

    def reserve_bytes(self, count):
        if self.closed or isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("invalid closed-owner/write reservation")
        if sum(row["bytes"] for row in self.records.values()) + count > self.max_generated_bytes:
            raise ValueError("owned generated byte budget exceeded before write")

    def register_file(self, relative, category):
        if self.closed or category not in CATEGORIES:
            raise ValueError("closed owner or unknown artifact category")
        path = _plain_path(self.root, relative)
        relative = path.relative_to(self.root).as_posix()
        if relative.casefold() in {MANIFEST_NAME, LEASE_NAME}:
            raise ValueError("ownership control files cannot be registered as producer artifacts")
        if not path.is_file():
            raise ValueError("owned generated artifact must be a regular file")
        key = relative.casefold() if os.name == "nt" else relative
        if len(self.records) >= 4096 and key not in self.records:
            raise ValueError("owned artifact record bound exceeded")
        size = path.stat().st_size
        used = sum(row["bytes"] for existing_key, row in self.records.items() if existing_key != key)
        if used + size > self.max_generated_bytes:
            raise ValueError("owned generated byte budget exceeded")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1048576), b""):
                digest.update(chunk)
        self.records[key] = {"path": relative, "category": category, "bytes": size, "sha256": digest.hexdigest()}
        self._write_metadata("active")

    def mark_failed(self, reason):
        self.state, self.error = "failed", str(reason)[:4096]

    def close(self, *, error=None, state="succeeded"):
        if self.closed:
            return
        if error is not None:
            self.state, self.error = "cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed", repr(error)[:4096]
        elif self.state == "active":
            if state not in {"succeeded", "failed", "cancelled"}:
                raise ValueError("invalid completed owned-run state")
            self.state = state
        self._write_metadata("released")
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        try:
            self.close(error=error)
        except Exception as publication_error:
            if error is None:
                raise
            self.auxiliary_errors.append(repr(publication_error))
            if hasattr(error, "add_note"):
                error.add_note("Owned-run receipt publication also failed: " + repr(publication_error))
        return False
