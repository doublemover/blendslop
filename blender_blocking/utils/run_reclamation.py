"""Bounded read-only plans for explicitly bound independent run owners.

Nested runs remain separate owners. This module never adopts files or leases,
releases an active run, deletes artifacts, or proves process/descendant exit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

from .run_ownership import LEASE_NAME, MANIFEST_NAME, _plain_path, plan_run_reclamation


@dataclass(frozen=True)
class ReclamationOwner:
    """Caller-retained identity and exact metadata snapshot for one run."""

    run_root: str
    run_id: str
    owner_token: str
    manifest_sha256: str
    lease_sha256: str


def _key(path):
    return os.path.normcase(str(path))


def _read_metadata(path, limit):
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("ownership metadata exceeds its bounded size")
    return data


def _metadata(owner, root):
    manifest_bytes = _read_metadata(_plain_path(root, MANIFEST_NAME), 1048576)
    lease_bytes = _read_metadata(_plain_path(root, LEASE_NAME), 65536)
    if (hashlib.sha256(manifest_bytes).hexdigest() != owner.manifest_sha256
            or hashlib.sha256(lease_bytes).hexdigest() != owner.lease_sha256):
        raise ValueError("ownership snapshot changed")
    manifest, lease = json.loads(manifest_bytes), json.loads(lease_bytes)
    if (manifest.get("run_id") != owner.run_id or lease.get("run_id") != owner.run_id
            or manifest.get("owner_token") != owner.owner_token
            or lease.get("owner_token") != owner.owner_token):
        raise ValueError("explicit ownership identity/token mismatch")
    return manifest, lease


def plan_run_reclamation_group(owners, *, max_hash_bytes=268435456):
    """Compose up to sixteen independently verified owners without mutation.

    A parent's otherwise unknown files are covered only when their exact paths
    belong to another explicitly bound descendant owner in this group. The flat
    plans are preserved; active leases, extra files, drift, co-ownership and any
    other blocker invalidate the entire plan. Registered bytes share one hash
    allowance. Future mutation would require a separate authorized executor and
    fresh path, lease, content and process checks; this plan cannot authorize it.
    """
    if (not isinstance(owners, (list, tuple)) or not 1 <= len(owners) <= 16
            or not all(isinstance(owner, ReclamationOwner) for owner in owners)):
        raise ValueError("one to sixteen explicit ReclamationOwner snapshots required")
    if (type(max_hash_bytes) is not int or not 1 <= max_hash_bytes <= 268435456):
        raise ValueError("hash allowance must be a positive integer at most 256 MiB")
    for owner in owners:
        if not all(isinstance(value, str) and value for value in
                   (owner.run_root, owner.run_id, owner.owner_token)):
            raise ValueError("explicit root/run ID/owner token required")
        for digest in (owner.manifest_sha256, owner.lease_sha256):
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("exact lowercase SHA256 metadata snapshots required")

    result = {"protocol": "explicit-run-group-reclamation-plan-v1", "mode": "read_only",
              "status": "blocked", "mutation_supported": False,
              "lease_release_supported": False, "process_exit_verified": False,
              "owners": [], "eligible": [], "eligible_bytes": 0,
              "retained_bytes": 0, "verified_registered_bytes": 0, "blockers": []}
    records = []
    roots = set()
    claimed = {}
    try:
        for owner in owners:
            supplied = Path(owner.run_root)
            root = supplied.resolve()
            if (not supplied.is_absolute() or supplied != root or not root.is_dir()
                    or root == Path(root.anchor) or root.is_symlink()
                    or getattr(root, "is_junction", lambda: False)()):
                raise ValueError("owner root is missing, nonabsolute or redirected")
            if _key(root) in roots:
                raise ValueError("duplicate explicit owner root")
            roots.add(_key(root))
            manifest, lease = _metadata(owner, root)
            entries = manifest.get("artifacts")
            if (not isinstance(entries, list) or len(entries) > 4096
                    or any(not isinstance(entry, dict) or type(entry.get("bytes")) is not int
                           or entry["bytes"] < 0 for entry in entries)):
                raise ValueError("invalid bounded registered artifact inventory")
            declared = sum(entry["bytes"] for entry in entries)
            remaining = max_hash_bytes - result["verified_registered_bytes"]
            if declared > remaining:
                raise ValueError("combined registered artifacts exceed hash allowance")
            flat = plan_run_reclamation(root, max_hash_bytes=max(1, remaining))
            # Do not relax any existing blocker except exact files belonging to
            # explicit descendant owners, verified together below.
            other = [reason for reason in flat["blockers"]
                     if reason != "unknown files present; ownership is not inferred"]
            if other:
                raise ValueError(str(root) + ": " + "; ".join(other))
            result["verified_registered_bytes"] += declared
            record = {"run_root": str(root), "run_id": owner.run_id,
                      "owner_token_matched": True,
                      "snapshot": {"manifest_sha256": owner.manifest_sha256,
                                   "lease_sha256": owner.lease_sha256},
                      "flat_plan": flat, "nested_owned_files": []}
            result["owners"].append(record)
            paths = [(_plain_path(root, MANIFEST_NAME), None),
                     (_plain_path(root, LEASE_NAME), None)]
            paths += [(_plain_path(root, entry["path"]), entry) for entry in entries]
            for path, entry in paths:
                if _key(path) in claimed:
                    raise ValueError("file claimed by more than one owner: " + str(path))
                claimed[_key(path)] = root
            records.append((owner, root, entries, flat, record))

        for owner, root, entries, flat, record in records:
            for relative in flat["unknown_files"]:
                path = _plain_path(root, relative)
                other_root = claimed.get(_key(path))
                if (other_root is None or other_root == root
                        or not other_root.is_relative_to(root)):
                    raise ValueError("unknown file remains unowned: " + str(path))
                record["nested_owned_files"].append(relative)
            for entry in entries:
                if entry["category"] == "disposable":
                    result["eligible"].append({"run_root": str(root), "run_id": owner.run_id,
                                               "path": entry["path"], "bytes": entry["bytes"],
                                               "sha256": entry["sha256"]})
                    result["eligible_bytes"] += entry["bytes"]
                else:
                    result["retained_bytes"] += entry["bytes"]
        # Bind the group to the same caller snapshots after all hash/inventory
        # work; a child becoming active or changing metadata blocks all owners.
        for owner, root, _, _, _ in records:
            _metadata(owner, root)
        result["status"] = "dry_run_ready"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        result["blockers"].append(str(exc))
        result["eligible"] = []
        result["eligible_bytes"] = 0
    return result
