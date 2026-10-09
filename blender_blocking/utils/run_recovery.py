"""Read-only, identity-bound recovery decisions; no lease release or adoption."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .run_ownership import MANIFEST_NAME, LEASE_NAME, _plain_path, plan_run_reclamation


def _bounded_read(path, limit):
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError("ownership metadata exceeds its bounded size")
    return value


def inspect_run_recovery(run_root, *, expected_run_id, expected_owner_token,
                         expected_snapshot=None, max_hash_bytes=268435456):
    """Inspect only explicitly identified ownership and retain active leases.

    A reported PID, caller boolean or disappearance is not join evidence.
    Snapshot hashes can bind a second inspection to the exact earlier metadata;
    neither inspection executes recovery, adopts unknown files or proves exit.
    """
    if not all(isinstance(v, str) and v for v in (expected_run_id, expected_owner_token)):
        raise ValueError("recovery inspection requires explicit run ID and owner token")
    if isinstance(max_hash_bytes, bool) or not isinstance(max_hash_bytes, int) or max_hash_bytes < 1:
        raise ValueError("hash verification bound must be a positive integer")
    supplied = Path(run_root).absolute()
    root = supplied.resolve()
    result = {"protocol": "owned-recovery-inspection-v1", "mode": "read_only",
              "run_root": str(root), "status": "blocked", "decision": "refuse_unverifiable_ownership",
              "recovery_execution_supported": False, "lease_release_supported": False,
              "process_identity_verified": False, "descendant_joins_verified": False,
              "registered_artifacts_verified": False, "snapshot": None, "blockers": []}
    try:
        if (not root.is_dir() or root == Path(root.anchor) or supplied != root or
                supplied.is_symlink() or getattr(supplied, "is_junction", lambda: False)()):
            raise ValueError("run root is missing, filesystem root or redirected")
        manifest_path, lease_path = _plain_path(root, MANIFEST_NAME), _plain_path(root, LEASE_NAME)
        manifest_bytes = _bounded_read(manifest_path, 1048576)
        lease_bytes = _bounded_read(lease_path, 65536)
        manifest, lease = json.loads(manifest_bytes), json.loads(lease_bytes)
        if (manifest.get("schema_version") != 1 or manifest.get("run_root") != str(root) or
                not isinstance(manifest.get("artifacts"), list) or len(manifest["artifacts"]) > 4096 or
                manifest.get("state") not in {"active", "succeeded", "failed", "cancelled"}):
            raise ValueError("invalid explicit ownership manifest")
        if (manifest.get("run_id") != expected_run_id or manifest.get("owner_token") != expected_owner_token or
                lease.get("run_id") != expected_run_id or lease.get("owner_token") != expected_owner_token or
                lease.get("pid") != manifest.get("pid") or lease.get("status") not in {"active", "released"}):
            raise ValueError("ownership identity/token/lease mismatch")
        snapshot = {"run_root": str(root), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "lease_sha256": hashlib.sha256(lease_bytes).hexdigest()}
        result.update(snapshot=snapshot, run_id=expected_run_id, owner_token_matched=True,
                      reported_pid=manifest.get("pid"), lease_status=lease["status"],
                      producer_state=manifest["state"], producer=manifest.get("producer"))
        if expected_snapshot is not None and dict(expected_snapshot) != snapshot:
            raise ValueError("ownership snapshot changed since the explicit prior inspection")
        if lease["status"] == "active":
            result["decision"] = "retain_active_lease"
            result["blockers"].append("active lease: original process start identity and complete descendant join evidence unavailable")
            result["required_recovery_evidence"] = ["original supervisor-bound process identity and actual join",
                                                   "complete ownership-bound descendant inventory and joins",
                                                   "separate explicit recovery decision preserving failure history"]
        else:
            plan = plan_run_reclamation(root, max_hash_bytes=max_hash_bytes)
            result["ownership_audit"] = plan
            if plan["status"] == "dry_run_ready":
                result.update(status="complete", decision="no_recovery_needed", registered_artifacts_verified=True)
            else:
                result["decision"] = "retain_unverified_released_run"
                result["blockers"].extend(plan["blockers"])
        if (_bounded_read(manifest_path, 1048576) != manifest_bytes or
                _bounded_read(lease_path, 65536) != lease_bytes):
            raise ValueError("ownership metadata changed during read-only inspection")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        result.update(status="blocked", decision="refuse_unverifiable_ownership", registered_artifacts_verified=False)
        result["blockers"].append(str(exc))
    return result
